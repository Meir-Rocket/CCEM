"""Two-phase training of the N-Queens factor energy.

Run from the repository root::

    python scripts/train.py nqueens --board-size 8 --output <path>

Phase 1 (*local*, :func:`local_epoch`) fits the shared
:class:`~ccem.tasks.nqueens.energy.ConstraintEnergyModel` on synthetic
line examples with the InfoNCE objective of
:func:`~ccem.tasks.nqueens.energy.train_energy_step`: for every valid
line the energies of ``negatives_per_positive`` invalid lines are pushed
above it. The validation set re-samples negatives from the same
synthetic factor set and only guards against degenerate epochs; the
lowest validation loss is checkpointed.

Phase 2 (*board fine-tuning*, :func:`finetune_epoch`) reloads the best
Phase-1 weights, composes the factor over whole boards
(:class:`~ccem.tasks.nqueens.energy.NQueensComposedEnergy`) and
back-propagates through the unrolled projected solver on real solution
boards. With ``S = board_num_starts`` random feasible starts solved
under frozen weights, the lowest-energy start is re-solved for
``T = board_unroll_steps`` differentiable steps (step sizes from
``board_unroll_step_size`` to ``board_unroll_step_size_final`` along
``board_step_schedule``, plus the random tie-break ``lambda <y, R>``
with ``lambda = board_tie_break_weight``). The loss on the solved board
``y`` against the target ``y*`` is

    alpha * MSE(y, y*)
    + beta  * softplus(E(y*) - E(y) + rho)
    + gamma * mean_h softplus(E(y*) - E(h) + rho_h)

with ``alpha = board_mse_weight``, ``beta = board_rank_weight``,
``rho = board_rank_margin``, ``gamma = board_hard_negative_weight``,
``rho_h = board_hard_negative_margin`` and the
``H = board_hard_negative_count`` hard negatives ``h`` of lowest mean
energy among the ``S`` warm-start solutions, the last ``H`` iterates of
the differentiable solve and one convex mix of the target with the best
start. Phase 2 keeps the Phase-1 AdamW optimizer and checkpoints the
epoch with the lowest training loss. Every ``training.*`` key without a
flag of its own is set with ``--set``.
"""

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ccem.projections import board_projection, random_feasible_boards
from ccem.solvers import SolverConfig, run_projected_solver, with_tie_break
from ccem.tasks.nqueens.board import relaxed_board_sharpness
from ccem.tasks.nqueens.data import NQueensBoardDataset
from ccem.tasks.nqueens.energy import (
    ConstraintBatch,
    ConstraintEnergyModel,
    NQueensComposedEnergy,
    NQueensConstraintDataset,
    build_factor_model,
    collate_constraint_batch,
    train_energy_step,
)
from ccem.utils import (
    apply_overrides,
    as_device,
    average_metrics,
    format_paths,
    load_checkpoint,
    load_yaml,
    parse_set_overrides,
    set_seed,
)

MIX_ALPHA_MIN = 0.35
MIX_ALPHA_RANGE = 0.30
"""The convex-mix hard negative uses ``alpha ~ U[0.35, 0.65)`` per
board.
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI flags; every explicit flag overrides the config, ``--set``
    last.
    """
    parser = argparse.ArgumentParser(
        description='Train the CCEM N-Queens model.'
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
        '--device',
        type=str,
        default=None,
        help='cpu or cuda (config device)',
    )
    parser.add_argument(
        '--seed', type=int, default=None, help='RNG seed (config seed)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='checkpoint to write (config output_path)',
    )
    parser.add_argument(
        '--local-epochs',
        type=int,
        default=None,
        help='Phase-1 epochs (training.local_epochs)',
    )
    parser.add_argument(
        '--board-finetune-epochs',
        type=int,
        default=None,
        help='Phase-2 epochs, 0 skips the phase '
        '(training.board_finetune_epochs)',
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
            'device': args.device,
            'seed': args.seed,
            'output_path': args.output,
            'training.local_epochs': args.local_epochs,
            'training.board_finetune_epochs': args.board_finetune_epochs,
        },
    )
    apply_overrides(cfg, parse_set_overrides(args.set))
    format_paths(
        cfg,
        ('board_train_path', 'output_path'),
        board_size=int(cfg['board_size']),
    )
    return cfg


@dataclass(frozen=True)
class FinetuneSettings:
    """Phase-2 settings (``training.board_*`` keys, see module
    docstring).

    Attributes:
        solver: The unrolled solve: ``T`` steps, step-size schedule,
            optimizer, ``S`` starts, Adam constants and tie-break
            weight.
        simplex_iters: Birkhoff projection rounds
            (``solver.simplex_iters``).
        mse_weight: ``alpha``.
        rank_weight: ``beta``.
        rank_margin: ``rho``.
        hard_negative_weight: ``gamma``.
        hard_negative_margin: ``rho_h``.
        hard_negative_count: ``H``; ``0`` disables hard-negative mining.
    """

    solver: SolverConfig
    simplex_iters: int
    mse_weight: float
    rank_weight: float
    rank_margin: float
    hard_negative_weight: float
    hard_negative_margin: float
    hard_negative_count: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> 'FinetuneSettings':
        """Read the ``training`` block, falling back to ``solver`` then
        defaults.

        The solver keys fall back as ``training.board_<x>`` ->
        ``solver.<x>`` -> default (optimizer ``adam``, 8 starts,
        constant schedule ending at the initial step size); the Adam
        constants are read from ``solver.adam_*`` (0.9/0.999/1e-8 when
        absent). The loss weights default to ``alpha=1, beta=0.25,
        rho=0.05, gamma=0.25, rho_h=0.05, H=4`` and the tie-break
        weight to ``0``.
        """
        training = cfg['training']
        solver_cfg = cfg.get('solver', {})

        def fallback(train_key: str, solver_key: str, default: Any) -> Any:
            return training.get(train_key, solver_cfg.get(solver_key, default))

        step_size = float(training['board_unroll_step_size'])
        solver = SolverConfig(
            steps=int(training['board_unroll_steps']),
            step_size=step_size,
            step_size_final=float(
                fallback(
                    'board_unroll_step_size_final',
                    'step_size_final',
                    step_size,
                )
            ),
            schedule=str(
                fallback('board_step_schedule', 'step_schedule', 'constant')
            ),
            optimizer=str(
                fallback('board_solver_optimizer', 'optimizer', 'adam')
            ),
            num_starts=max(
                int(fallback('board_num_starts', 'num_starts', 8)), 1
            ),
            adam_beta1=float(solver_cfg.get('adam_beta1', 0.9)),
            adam_beta2=float(solver_cfg.get('adam_beta2', 0.999)),
            adam_eps=float(solver_cfg.get('adam_eps', 1.0e-8)),
            tie_break_weight=float(
                training.get('board_tie_break_weight', 0.0)
            ),
        )
        return cls(
            solver=solver,
            simplex_iters=int(solver_cfg.get('simplex_iters', 8)),
            mse_weight=float(training.get('board_mse_weight', 1.0)),
            rank_weight=float(training.get('board_rank_weight', 0.25)),
            rank_margin=float(training.get('board_rank_margin', 0.05)),
            hard_negative_weight=float(
                training.get('board_hard_negative_weight', 0.25)
            ),
            hard_negative_margin=float(
                training.get('board_hard_negative_margin', 0.05)
            ),
            hard_negative_count=max(
                int(training.get('board_hard_negative_count', 4)), 0
            ),
        )


def local_epoch(
    loader: DataLoader[ConstraintBatch],
    model: ConstraintEnergyModel,
    optimizer: Optional[AdamW],
    device: torch.device,
) -> dict[str, float]:
    """One Phase-1 pass; ``optimizer=None`` evaluates without updating.

    Returns:
        The epoch means of the :func:`train_energy_step` metrics.
    """
    is_train = optimizer is not None
    model.train(is_train)
    metrics_all: list[dict[str, float]] = []
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            loss, metrics = train_energy_step(model, batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            metrics_all.append(metrics)
    return average_metrics(metrics_all)


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    """Disable parameter gradients inside the block (iterate
    gradients stay).
    """
    for param in module.parameters():
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param in module.parameters():
            param.requires_grad_(True)


def random_tie_break(like: Tensor, weight: float) -> Optional[Tensor]:
    """Draw ``U[-1, 1)`` tie-break matrices when ``weight > 0`` (RNG
    order).
    """
    if weight <= 0.0:
        return None
    return torch.empty_like(like).uniform_(-1.0, 1.0)


def select_warm_start(
    composer: NQueensComposedEnergy, boards: Tensor, settings: FinetuneSettings
) -> tuple[Tensor, list[Tensor]]:
    """Solve ``S`` random starts per board under frozen weights.

    Returns:
        The lowest-energy solved start per board ``[B, N, N]``
        (detached) and the ``S`` detached solved starts ``[B, N, N]``
        each (hard-negative candidates).
    """
    batch_size, board_size = int(boards.size(0)), int(boards.size(1))
    starts = settings.solver.num_starts
    warm_init = random_feasible_boards(
        batch_size * starts, board_size, boards.device, settings.simplex_iters
    )
    tie_break = random_tie_break(warm_init, settings.solver.tie_break_weight)
    energy_fn = with_tie_break(
        composer.total_energy, tie_break, settings.solver.tie_break_weight
    )
    with frozen_parameters(composer):
        warm_solved, _ = run_projected_solver(
            energy_fn,
            warm_init,
            board_projection(settings.simplex_iters),
            settings.solver,
        )
        warm_solved = warm_solved.view(
            starts, batch_size, board_size, board_size
        )
        warm_energies = (
            composer.total_energy(
                warm_solved.reshape(
                    starts * batch_size, board_size, board_size
                ).detach()
            )
            .view(starts, batch_size)
            .detach()
        )
    best_start_idx = torch.argmin(warm_energies, dim=0)
    batch_indices = torch.arange(batch_size, device=boards.device)
    best_start = warm_solved[best_start_idx, batch_indices].detach()
    candidates = [
        warm_solved[start_idx].detach() for start_idx in range(starts)
    ]
    return best_start, candidates


def hard_negative_loss(
    composer: NQueensComposedEnergy,
    target: Tensor,
    true_energy: Tensor,
    best_start: Tensor,
    candidates: list[Tensor],
    settings: FinetuneSettings,
) -> Tensor:
    """Mean rank-margin loss over the ``H`` lowest-energy candidates.

    ``candidates`` holds the warm-start solutions and the recorded
    solver iterates; a convex mix
    ``alpha * target + (1 - alpha) * best_start`` with
    ``alpha ~ U[0.35, 0.65)`` per board is appended. Candidates are
    ranked by their batch-mean energy and the ``H`` lowest give
    ``softplus(E(target) - E(h) + rho_h)``, averaged over the batch and
    over ``h``. The candidate energies carry gradients to the
    parameters.
    """
    mix_alpha = MIX_ALPHA_MIN + MIX_ALPHA_RANGE * torch.rand(
        target.size(0), 1, 1, device=target.device, dtype=target.dtype
    )
    mix_candidate = (
        mix_alpha * target + (1.0 - mix_alpha) * best_start
    ).clamp(0.0, 1.0)
    energies = [
        composer.total_energy(state) for state in [*candidates, mix_candidate]
    ]
    keys = torch.stack([energy.mean().detach() for energy in energies])
    top_idx = torch.argsort(keys)[
        : min(settings.hard_negative_count, len(energies))
    ]
    terms = [
        f.softplus(
            true_energy - energies[idx] + settings.hard_negative_margin
        ).mean()
        for idx in top_idx.tolist()
    ]
    return torch.stack(terms).mean()


def finetune_epoch(
    loader: DataLoader[Tensor],
    composer: NQueensComposedEnergy,
    optimizer: AdamW,
    device: torch.device,
    settings: FinetuneSettings,
) -> dict[str, float]:
    """One Phase-2 pass over the solution boards (see module docstring).

    Per batch the RNG is consumed as: warm starts, warm tie-break, final
    tie-break, mix coefficients (the last only with ``H > 0``). The
    differentiable solve records its last ``H`` iterates (detached
    copies) as hard-negative candidates. Batches with a non-finite loss
    are skipped.

    Returns:
        Epoch means of ``loss``, ``mse``, ``rank``, ``hard``,
        ``true_energy``, ``solved_energy`` and the
        :func:`relaxed_board_sharpness` statistics of the solved boards.
    """
    composer.train(True)
    projection = board_projection(settings.simplex_iters)
    mine_hard = settings.hard_negative_count > 0
    metrics_all: list[dict[str, float]] = []
    for board_batch in loader:
        target = board_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        best_start, candidates = select_warm_start(composer, target, settings)
        tie_break = random_tie_break(
            best_start, settings.solver.tie_break_weight
        )
        energy_fn = with_tie_break(
            composer.total_energy, tie_break, settings.solver.tie_break_weight
        )
        solved, history = run_projected_solver(
            energy_fn,
            best_start,
            projection,
            settings.solver,
            create_graph=True,
            history_last=settings.hard_negative_count,
        )
        mse_loss = f.mse_loss(solved, target)
        true_energy = composer.total_energy(target)
        solved_energy = composer.total_energy(solved)
        rank_loss = f.softplus(
            true_energy - solved_energy + settings.rank_margin
        ).mean()
        hard_loss = torch.zeros((), device=device, dtype=mse_loss.dtype)
        if mine_hard:
            hard_loss = hard_negative_loss(
                composer,
                target,
                true_energy,
                best_start,
                candidates + history,
                settings,
            )
        loss = (
            settings.mse_weight * mse_loss
            + settings.rank_weight * rank_loss
            + settings.hard_negative_weight * hard_loss
        )
        if not torch.isfinite(loss):
            continue
        # Tensor.backward is untyped in the torch stubs.
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        metrics_all.append(
            {
                'loss': loss.item(),
                'mse': mse_loss.item(),
                'rank': rank_loss.item(),
                'hard': hard_loss.item(),
                'true_energy': true_energy.mean().item(),
                'solved_energy': solved_energy.mean().item(),
                **relaxed_board_sharpness(solved.detach()),
            }
        )
    return average_metrics(metrics_all)


def save_checkpoint(
    path: Path, cfg: Mapping[str, Any], model: ConstraintEnergyModel
) -> None:
    """Write the payload read by
    :func:`ccem.tasks.nqueens.energy.load_composed_energy`.
    """
    torch.save(
        {
            'config': dict(cfg),
            'model_state': model.state_dict(),
            'board_size': int(cfg['board_size']),
            'factor_types': list(cfg['factor_types']),
            'model_type': 'picnn',
        },
        path,
    )


def run_local_phase(
    cfg: Mapping[str, Any],
    model: ConstraintEnergyModel,
    optimizer: AdamW,
    device: torch.device,
    output_path: Path,
) -> None:
    """Phase 1: InfoNCE epochs, checkpointing the lowest validation
    loss.
    """
    training = cfg['training']

    def constraint_loader(*, shuffle: bool) -> DataLoader[ConstraintBatch]:
        dataset = NQueensConstraintDataset(
            board_size=int(cfg['board_size']),
            factor_types=cfg['factor_types'],
            negatives_per_positive=int(training['negatives_per_positive']),
        )
        return DataLoader(
            dataset,
            batch_size=int(training['batch_size']),
            shuffle=shuffle,
            collate_fn=collate_constraint_batch,
        )

    train_loader = constraint_loader(shuffle=True)
    valid_loader = constraint_loader(shuffle=False)
    best_val = float('inf')
    for epoch in tqdm(range(int(training['local_epochs']))):
        train_metrics = local_epoch(train_loader, model, optimizer, device)
        valid_metrics = local_epoch(valid_loader, model, None, device)
        tqdm.write(
            f'[local {epoch:04d}] '
            f'train_loss={train_metrics["loss"]:.4f} '
            f'train_margin={train_metrics["margin"]:.4f} '
            f'val_loss={valid_metrics["loss"]:.4f} '
            f'val_margin={valid_metrics["margin"]:.4f}'
        )
        if valid_metrics['loss'] < best_val:
            best_val = valid_metrics['loss']
            save_checkpoint(output_path, cfg, model)


def run_board_phase(
    cfg: Mapping[str, Any],
    model: ConstraintEnergyModel,
    optimizer: AdamW,
    device: torch.device,
    output_path: Path,
) -> None:
    """Phase 2: unrolled fine-tuning from the best Phase-1 weights."""
    training = cfg['training']
    if int(training.get('local_epochs', 0)) > 0 and output_path.exists():
        model.load_state_dict(
            load_checkpoint(output_path, device)['model_state']
        )
        tqdm.write(f'[board] Loaded best local checkpoint from {output_path}')
    board_loader: DataLoader[Tensor] = DataLoader(
        NQueensBoardDataset(
            cfg['board_train_path'],
            augment_symmetries=bool(
                training.get('board_augment_symmetries', False)
            ),
        ),
        batch_size=int(training['board_batch_size']),
        shuffle=True,
    )
    composer = NQueensComposedEnergy(
        model, int(cfg['board_size']), cfg['factor_types']
    ).to(device)
    settings = FinetuneSettings.from_config(cfg)
    best_loss = float('inf')
    best_epoch = -1
    for epoch in tqdm(range(int(training['board_finetune_epochs']))):
        metrics = finetune_epoch(
            board_loader, composer, optimizer, device, settings
        )
        tqdm.write(
            f'[full {epoch:04d}] '
            f'loss={metrics.get("loss", 0.0):.6f} '
            f'mse={metrics.get("mse", 0.0):.6f} '
            f'rank={metrics.get("rank", 0.0):.6f} '
            f'hard={metrics.get("hard", 0.0):.6f} '
            f'row_H={metrics.get("row_entropy", 0.0):.4f} '
            f'col_H={metrics.get("col_entropy", 0.0):.4f} '
            f'gap={metrics.get("row_gap", 0.0):.4f} '
            f'conf_rows={metrics.get("confident_rows", 0.0):.4f}'
        )
        loss = metrics.get('loss', float('inf'))
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            save_checkpoint(output_path, cfg, model)
            tqdm.write(
                f'[full {epoch:04d}] New best board loss: {best_loss:.6f}'
            )
    if best_epoch >= 0:
        tqdm.write(
            f'[board] Best checkpoint epoch={best_epoch:04d} '
            f'loss={best_loss:.6f}'
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of ``python scripts/train.py nqueens`` (also
    ``python -m ccem.tasks.nqueens.train``).
    """
    args = parse_args(argv)
    cfg = build_config(args)
    set_seed(int(cfg.get('seed', 0)))
    device = as_device(cfg.get('device'))
    model = build_factor_model(cfg['model'], int(cfg['board_size'])).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg['training']['lr']),
        weight_decay=float(cfg['training'].get('weight_decay', 0.0)),
    )
    output_path = Path(cfg['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f'Board size: {cfg["board_size"]}')
    tqdm.write(f'Train boards: {cfg["board_train_path"]}')
    tqdm.write(f'Saving checkpoint to: {output_path}')

    run_local_phase(cfg, model, optimizer, device, output_path)
    if int(cfg['training'].get('board_finetune_epochs', 0)) > 0:
        run_board_phase(cfg, model, optimizer, device, output_path)


if __name__ == '__main__':
    main()
