"""Multi-start N-Queens inference.

Run from the repository root::

    python scripts/evaluate.py nqueens --board-size 8

For every sample the solver is started from ``num_starts`` random
feasible boards (:func:`ccem.projections.random_feasible_boards`), each
run of projected Adam over the Birkhoff polytope is decoded greedily
(:func:`ccem.tasks.nqueens.board.decode_board`) and one start is kept
per sample. ``--start-selection constraint`` (the rule behind the
paper's numbers) prefers a valid decoded board, then fewer conflicts,
then more queens and finally a lower relaxed energy;
``--start-selection energy`` keeps the start with the lowest relaxed
energy, as described in the paper text.

The RNG is seeded once, then the model is built and, per batch and per
start, a random feasible board and (with a positive
``tie_break_weight``) a random tie-break matrix are drawn in that order.
The report is printed to stdout and ``--output-json PATH`` writes its
numbers as one JSON file. Every other config entry (batch size, step
sizes, projection rounds, ...) is set with ``--set``.
"""

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from ccem.projections import board_projection, random_feasible_boards
from ccem.solvers import SolverConfig, run_projected_solver, with_tie_break
from ccem.tasks.nqueens.board import (
    board_to_line,
    decode_board,
    evaluate_board,
    load_solution_set,
    relaxed_board_sharpness,
)
from ccem.tasks.nqueens.energy import (
    NQueensComposedEnergy,
    load_composed_energy,
)
from ccem.utils import (
    apply_overrides,
    as_device,
    average_metrics,
    format_paths,
    load_yaml,
    parse_set_overrides,
    save_json,
    set_seed,
)

StartSelection = Literal['constraint', 'energy']
START_SELECTION_TEXT: Mapping[StartSelection, str] = {
    'constraint': 'valid -> fewer conflicts -> more queens -> lower relaxed '
    'energy',
    'energy': 'lower relaxed energy',
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI flags; every explicit flag overrides the config, ``--set``
    last.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate the CCEM N-Queens model.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/nqueens/picnn.yaml',
        help='YAML config (default: %(default)s)',
    )
    parser.add_argument(
        '--board-size',
        type=int,
        default=None,
        help='N; also fills {board_size} in the config paths',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None, help='checkpoint to evaluate'
    )
    parser.add_argument(
        '--test-path',
        type=str,
        default=None,
        help='known-solution set ({N}_queens_all.txt)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='cpu or cuda (config device)',
    )
    parser.add_argument(
        '--seed', type=int, default=None, help='RNG seed (config seed)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='boards to solve (eval.num_samples)',
    )
    parser.add_argument(
        '--num-starts',
        type=int,
        default=None,
        help='random starts per board (solver.num_starts)',
    )
    parser.add_argument(
        '--solver-steps',
        type=int,
        default=None,
        help='projected-Adam steps per start (solver.steps)',
    )
    parser.add_argument(
        '--start-selection',
        type=str,
        choices=('constraint', 'energy'),
        default='constraint',
        help='start kept per sample; constraint: '
        f'{START_SELECTION_TEXT["constraint"]} (paper numbers); '
        f'energy: {START_SELECTION_TEXT["energy"]}',
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='write the report numbers as JSON to this path',
    )
    parser.add_argument(
        '--set',
        action='append',
        default=None,
        metavar='SECTION.KEY=VALUE',
        help='override any config entry (repeatable)',
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load the YAML, apply the flags and fill the ``{board_size}``
    paths.
    """
    cfg = load_yaml(args.config)
    apply_overrides(
        cfg,
        {
            'board_size': args.board_size,
            'checkpoint': args.checkpoint,
            'test_path': args.test_path,
            'device': args.device,
            'seed': args.seed,
            'eval.num_samples': args.num_samples,
            'solver.steps': args.solver_steps,
            'solver.num_starts': args.num_starts,
        },
    )
    apply_overrides(cfg, parse_set_overrides(args.set))
    format_paths(
        cfg, ('checkpoint', 'test_path'), board_size=int(cfg['board_size'])
    )
    return cfg


@dataclass(frozen=True)
class SolveOutcome:
    """The start kept for every sample.

    Attributes:
        decoded: Hard ``[N, N]`` boards, one per sample.
        relaxed_energies: Relaxed energy of the kept start per sample.
        batch_sharpness: :func:`relaxed_board_sharpness` of the kept
            relaxed boards, one entry per batch (the report averages
            over batches).
    """

    decoded: list[Tensor]
    relaxed_energies: list[float]
    batch_sharpness: list[dict[str, float]]


def start_score(
    decoded: Tensor, energy: float, selection: StartSelection
) -> tuple[float, ...]:
    """Lexicographic score of one start; a larger score wins.

    ``'constraint'`` scores ``(valid, -conflicts, queens, -energy)``
    from the decoded board, ``'energy'`` scores ``(-energy,)`` only.
    """
    if selection == 'energy':
        return (-energy,)
    result = evaluate_board(decoded)
    conflicts = (
        result.num_row_conflicts
        + result.num_col_conflicts
        + result.num_diag_conflicts
    )
    return (
        float(result.is_valid),
        -float(conflicts),
        float(result.num_queens),
        -energy,
    )


def solve_samples(
    composer: NQueensComposedEnergy,
    *,
    num_samples: int,
    batch_size: int,
    solver: SolverConfig,
    simplex_iters: int,
    device: torch.device,
    selection: StartSelection,
) -> SolveOutcome:
    """Run the multi-start protocol and keep one start per sample.

    Starts are drawn per batch; within a batch every start is solved for
    all samples at once and compared against the best start seen so far
    with :func:`start_score` (ties keep the earlier start).
    """
    if num_samples < 1 or batch_size < 1:
        raise ValueError('num_samples and batch_size must be positive')
    board_size = composer.board_size
    projection = board_projection(simplex_iters)
    num_batches = math.ceil(num_samples / batch_size)
    outcome = SolveOutcome([], [], [])
    progress = tqdm(
        total=num_batches * solver.num_starts,
        desc='Solving',
        unit='start',
        dynamic_ncols=True,
    )
    try:
        for batch_idx in range(num_batches):
            current = min(batch_size, num_samples - batch_idx * batch_size)
            best_score: list[Optional[tuple[float, ...]]] = [None] * current
            best_decoded: list[Tensor] = [torch.empty(0)] * current
            best_relaxed: list[Tensor] = [torch.empty(0)] * current
            best_energy = [0.0] * current
            for _ in range(solver.num_starts):
                init = random_feasible_boards(
                    current, board_size, device, simplex_iters
                )
                tie_break = None
                if solver.tie_break_weight > 0.0:
                    tie_break = torch.empty_like(init).uniform_(-1.0, 1.0)
                energy_fn = with_tie_break(
                    composer.total_energy, tie_break, solver.tie_break_weight
                )
                solved, _ = run_projected_solver(
                    energy_fn, init, projection, solver
                )
                energies = composer.total_energy(solved).detach().cpu()
                solved_cpu = solved.detach().cpu()
                progress.update(1)
                for sample_idx in range(current):
                    relaxed = solved_cpu[sample_idx]
                    decoded = decode_board(relaxed)
                    energy = float(energies[sample_idx].item())
                    score = start_score(decoded, energy, selection)
                    previous = best_score[sample_idx]
                    if previous is not None and score <= previous:
                        continue
                    best_score[sample_idx] = score
                    best_decoded[sample_idx] = decoded
                    best_relaxed[sample_idx] = relaxed.clone()
                    best_energy[sample_idx] = energy
            outcome.decoded.extend(best_decoded)
            outcome.relaxed_energies.extend(best_energy)
            outcome.batch_sharpness.append(
                relaxed_board_sharpness(torch.stack(best_relaxed, dim=0))
            )
            progress.set_postfix(
                samples_done=len(outcome.decoded), refresh=False
            )
    finally:
        progress.close()
    return outcome


@dataclass(frozen=True)
class EvalSummary:
    """The settings and numbers of the report, in the order of the
    JSON summary.
    """

    board_size: int
    seed: int
    checkpoint: str
    test_path: str
    num_starts: int
    solver_steps: int
    start_selection: str
    samples: int
    valid_solutions: int
    unique_valid_solutions: int
    solutions_in_solution_set: int
    avg_queens: float
    std_queens: float
    avg_row_conflicts: float
    avg_column_conflicts: float
    avg_diagonal_conflicts: float
    avg_relaxed_energy: float
    std_relaxed_energy: float
    row_entropy: float
    col_entropy: float
    row_gap: float
    confident_rows: float
    row_sum_err: float
    col_sum_err: float


def summarize(
    outcome: SolveOutcome,
    solution_set: set[str],
    *,
    board_size: int,
    seed: int,
    checkpoint: str,
    test_path: str,
    solver: SolverConfig,
    selection: StartSelection,
) -> EvalSummary:
    """Aggregate the kept boards into the report statistics."""
    results = [evaluate_board(board) for board in outcome.decoded]
    lines = [board_to_line(board) for board in outcome.decoded]
    unique_valid = {
        line
        for line, result in zip(lines, results, strict=True)
        if result.is_valid
    }
    queens = [result.num_queens for result in results]
    return EvalSummary(
        board_size=board_size,
        seed=seed,
        checkpoint=checkpoint,
        test_path=test_path,
        num_starts=solver.num_starts,
        solver_steps=solver.steps,
        start_selection=selection,
        samples=len(results),
        valid_solutions=sum(result.is_valid for result in results),
        unique_valid_solutions=len(unique_valid),
        solutions_in_solution_set=sum(line in solution_set for line in lines),
        avg_queens=float(np.mean(queens)),
        std_queens=float(np.std(queens)),
        avg_row_conflicts=float(
            np.mean([r.num_row_conflicts for r in results])
        ),
        avg_column_conflicts=float(
            np.mean([r.num_col_conflicts for r in results])
        ),
        avg_diagonal_conflicts=float(
            np.mean([r.num_diag_conflicts for r in results])
        ),
        avg_relaxed_energy=float(np.mean(outcome.relaxed_energies)),
        std_relaxed_energy=float(np.std(outcome.relaxed_energies)),
        **average_metrics(outcome.batch_sharpness),
    )


def format_report(
    summary: EvalSummary,
    *,
    solver: SolverConfig,
    simplex_iters: int,
    selection: StartSelection,
) -> list[str]:
    """The lines of the printed report."""
    step_size_end = (
        solver.step_size
        if solver.step_size_final is None
        else solver.step_size_final
    )
    return [
        f'Samples: {summary.samples}',
        f'Birkhoff projection rounds: {simplex_iters}',
        f'Start selection: {START_SELECTION_TEXT[selection]}',
        f'Solver: {solver.optimizer}, starts={solver.num_starts}, '
        f'schedule={solver.schedule}',
        f'Step size: start={solver.step_size:.6g}, end={step_size_end:.6g}',
        f'Tie-break weight: {solver.tie_break_weight:.6g}',
        f'Valid solutions: {summary.valid_solutions}',
        f'Unique valid solutions: {summary.unique_valid_solutions}',
        f'Solutions present in solution set: '
        f'{summary.solutions_in_solution_set}',
        f'Avg. queens placed: {summary.avg_queens:.4f} +/- '
        f'{summary.std_queens:.4f}',
        f'Avg. row conflicts: {summary.avg_row_conflicts:.4f}',
        f'Avg. column conflicts: {summary.avg_column_conflicts:.4f}',
        f'Avg. diagonal conflicts: {summary.avg_diagonal_conflicts:.4f}',
        f'Avg. relaxed energy: {summary.avg_relaxed_energy:.4f} '
        f'+/- {summary.std_relaxed_energy:.4f}',
        f'Relaxed sharpness: row_H={summary.row_entropy:.4f} '
        f'col_H={summary.col_entropy:.4f} gap={summary.row_gap:.4f} '
        f'conf_rows={summary.confident_rows:.4f}',
        f'Relaxed sum errors: row={summary.row_sum_err:.6f} '
        f'col={summary.col_sum_err:.6f}',
    ]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of ``python scripts/evaluate.py nqueens`` (also
    ``python -m ccem.tasks.nqueens.evaluate``).
    """
    args = parse_args(argv)
    cfg = build_config(args)
    selection: StartSelection = args.start_selection
    seed = int(cfg.get('seed', 0))
    set_seed(seed)
    device = as_device(cfg.get('device'))
    print(f'Board size: {cfg["board_size"]}')
    print(f'Seed: {seed}')
    print(f'Checkpoint: {cfg["checkpoint"]}')
    print(f'Test set: {cfg["test_path"]}')

    composer, payload = load_composed_energy(cfg['checkpoint'], device)
    if int(payload['board_size']) != int(cfg['board_size']):
        raise ValueError(
            f'Checkpoint board_size={payload["board_size"]} does not match '
            f'configured board_size={cfg["board_size"]}'
        )
    solver = SolverConfig.from_mapping(cfg['solver'], num_starts=8)
    simplex_iters = int(cfg['solver'].get('simplex_iters', 8))
    outcome = solve_samples(
        composer,
        num_samples=int(cfg['eval']['num_samples']),
        batch_size=int(cfg['eval']['batch_size']),
        solver=solver,
        simplex_iters=simplex_iters,
        device=device,
        selection=selection,
    )
    summary = summarize(
        outcome,
        load_solution_set(cfg['test_path']),
        board_size=composer.board_size,
        seed=seed,
        checkpoint=str(cfg['checkpoint']),
        test_path=str(cfg['test_path']),
        solver=solver,
        selection=selection,
    )
    for line in format_report(
        summary,
        solver=solver,
        simplex_iters=simplex_iters,
        selection=selection,
    ):
        print(line)
    if args.output_json is not None:
        save_json(asdict(summary), args.output_json)
        print(f'Saved metrics to: {args.output_json}')


if __name__ == '__main__':
    main()
