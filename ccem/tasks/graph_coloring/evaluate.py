"""Evaluate the graph coloring factor: multi-start relaxation,
energy-selected start, greedy decode.

Run from the repository root::

    python scripts/evaluate.py graph_coloring \\
        --test-path data/graph_coloring/erdos_renyi.json

Protocol for every (graph, trial) pair: ``S`` random starts on the
product of per-node simplices, ``T`` projected solver steps on the
composed energy plus a small random linear tie-break, selection of the
start with the lowest base relaxed energy, constraint-aware greedy
decoding restricted to the stored color budget
(``GraphInstance.chromatic_number``) and validity = no monochromatic
edge. The report is printed; ``--output-json PATH`` additionally
writes one JSON summary.
"""

import argparse
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from tqdm import tqdm

from ccem.solvers import SolverConfig
from ccem.tasks.graph_coloring.energy import (
    EdgeColorEnergyModel,
    GraphColoringComposedEnergy,
    load_factor_model,
    random_node_assignments,
    solve_graph,
)
from ccem.tasks.graph_coloring.graphs import (
    GraphInstance,
    count_conflicting_edges,
    greedy_color_decode,
    read_graphs,
)
from ccem.utils import (
    apply_overrides,
    as_device,
    load_yaml,
    parse_set_overrides,
    save_json,
    set_seed,
)


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Outcome of one multi-start trial on one graph.

    Attributes:
        graph_index: Position of the graph in the evaluated list.
        graph_name: ``GraphInstance.name``.
        num_nodes: Number of nodes of the graph.
        num_edges: Number of edges (factors) of the graph.
        target_colors: Color budget used for decoding
            (``chromatic_number``).
        valid: ``True`` when the decoded coloring has no monochromatic
            edge.
        conflicting_edges: Number of monochromatic edges after decoding.
    """

    graph_index: int
    graph_name: str
    num_nodes: int
    num_edges: int
    target_colors: int
    valid: bool
    conflicting_edges: int


@dataclass(frozen=True, slots=True)
class GraphAggregate:
    """Per-graph means over its trials (the unit of the reported
    statistics).
    """

    graph_index: int
    graph_name: str
    num_nodes: int
    num_edges: int
    target_colors: int
    trials: int
    valid_trials: int
    valid_rate_percent: float
    conflicting_edges_mean: float


@dataclass(frozen=True, slots=True)
class GraphSummary:
    """Mean and sample std (ddof=1) of the per-graph results across
    graphs.

    The per-graph trial means are aggregated first; the standard
    deviations are taken across graphs and are ``0.0`` when only one
    graph was evaluated.

    Attributes:
        num_graphs: Number of evaluated graphs.
        trials_per_graph: Trials per graph (identical for every graph).
        valid_trials: Number of valid decodings over all trials.
        total_trials: Number of trials over all graphs.
        valid_rate_percent_mean: Mean per-graph valid-coloring rate in
            percent.
        valid_rate_percent_std: Sample std of the per-graph rates in
            percent.
        conflicting_edges_mean: Mean of the per-graph mean conflict
            counts.
        conflicting_edges_std: Sample std of the per-graph mean
            conflicts.
        per_graph: The per-graph aggregates in graph order.
    """

    num_graphs: int
    trials_per_graph: int
    valid_trials: int
    total_trials: int
    valid_rate_percent_mean: float
    valid_rate_percent_std: float
    conflicting_edges_mean: float
    conflicting_edges_std: float
    per_graph: tuple[GraphAggregate, ...]


def evaluate_graphs(
    model: EdgeColorEnergyModel,
    graphs: Sequence[GraphInstance],
    *,
    num_samples: int,
    solver: SolverConfig,
    device: torch.device,
) -> list[TrialResult]:
    """Run the multi-start trials protocol on every graph.

    The caller owns the random state: :func:`main` calls
    :func:`ccem.utils.set_seed` and then loads the factor, whose
    parameter initialisation consumes generator draws before the first
    trial, so seed first and load ``model`` second to reproduce a run.
    Per graph and per trial, the ``solver.num_starts`` starts are drawn
    with :func:`~.energy.random_node_assignments` and, when
    ``solver.tie_break_weight > 0``, the tie-break matrix with
    ``uniform_(-1, 1)``. The start with the lowest base relaxed energy
    is decoded with :func:`~.graphs.greedy_color_decode` over its first
    ``chromatic_number`` columns (argmax fallback on dead ends, so the
    coloring is always complete and within budget) and scored with
    :func:`~.graphs.count_conflicting_edges`.

    Args:
        model: Frozen factor model (defines the palette size ``K``).
        graphs: Instances to evaluate; each needs
            ``1 <= chromatic_number <= K``.
        num_samples: Independent trials per graph.
        solver: Solver settings; ``num_starts`` starts per trial.
        device: Device of ``model``; the relaxations run there.

    Returns:
        One :class:`TrialResult` per (graph, trial), graphs in order and
        trials in order within a graph.

    Raises:
        ValueError: If a color budget is outside ``1 .. K``.
    """
    factor_colors = int(model.num_colors)
    for graph in graphs:
        if not 1 <= int(graph.chromatic_number) <= factor_colors:
            raise ValueError(
                f'{graph.name}: color budget {graph.chromatic_number} is '
                f'outside the checkpoint palette 1..{factor_colors}'
            )
    trials: list[TrialResult] = []
    progress = tqdm(
        total=len(graphs) * int(num_samples),
        desc='Solving',
        unit='trial',
        dynamic_ncols=True,
    )
    for graph_index, graph in enumerate(graphs):
        target_colors = int(graph.chromatic_number)
        composer = GraphColoringComposedEnergy(model, graph).to(device=device)
        composer.eval()
        for _ in range(int(num_samples)):
            init = random_node_assignments(
                solver.num_starts, graph.num_nodes, factor_colors, device
            )
            tie_break: Optional[torch.Tensor] = None
            if solver.tie_break_weight > 0.0:
                tie_break = torch.empty_like(init).uniform_(-1.0, 1.0)
            solved = solve_graph(composer, init, solver, tie_break=tie_break)
            with torch.no_grad():
                energies = composer.total_energy(solved)
            best = int(energies.argmin().item())
            soft = solved[best].detach().cpu()[:, :target_colors]
            coloring = greedy_color_decode(graph, soft)
            conflicts = count_conflicting_edges(graph, coloring)
            trials.append(
                TrialResult(
                    graph_index=graph_index,
                    graph_name=graph.name,
                    num_nodes=graph.num_nodes,
                    num_edges=graph.num_edges(),
                    target_colors=target_colors,
                    valid=conflicts == 0,
                    conflicting_edges=conflicts,
                )
            )
            progress.update(1)
    progress.close()
    return trials


def summarize(trials: Sequence[TrialResult]) -> GraphSummary:
    """Aggregate trials into the reported statistics.

    Trials are grouped by ``graph_index``; the valid rate and the mean
    conflict count of each graph are averaged across graphs and their
    sample standard deviation (ddof=1, ``0.0`` for a single graph) is
    reported.

    Raises:
        ValueError: If ``trials`` is empty or the graphs have different
            trial counts.
    """
    if not trials:
        raise ValueError('no trials to summarize')
    by_graph: dict[int, list[TrialResult]] = defaultdict(list)
    for trial in trials:
        by_graph[trial.graph_index].append(trial)
    counts = {len(items) for items in by_graph.values()}
    if len(counts) != 1:
        raise ValueError('every graph must have the same number of trials')
    per_graph: list[GraphAggregate] = []
    for graph_index in sorted(by_graph):
        items = by_graph[graph_index]
        first = items[0]
        valid_trials = sum(int(item.valid) for item in items)
        per_graph.append(
            GraphAggregate(
                graph_index=graph_index,
                graph_name=first.graph_name,
                num_nodes=first.num_nodes,
                num_edges=first.num_edges,
                target_colors=first.target_colors,
                trials=len(items),
                valid_trials=valid_trials,
                valid_rate_percent=100.0 * valid_trials / len(items),
                conflicting_edges_mean=statistics.fmean(
                    item.conflicting_edges for item in items
                ),
            )
        )
    valid_rates = [item.valid_rate_percent for item in per_graph]
    conflict_means = [item.conflicting_edges_mean for item in per_graph]
    return GraphSummary(
        num_graphs=len(per_graph),
        trials_per_graph=counts.pop(),
        valid_trials=sum(item.valid_trials for item in per_graph),
        total_trials=len(trials),
        valid_rate_percent_mean=statistics.fmean(valid_rates),
        valid_rate_percent_std=_sample_std(valid_rates),
        conflicting_edges_mean=statistics.fmean(conflict_means),
        conflicting_edges_std=_sample_std(conflict_means),
        per_graph=tuple(per_graph),
    )


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI flags; every explicit flag overrides the config, ``--set``
    last.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate the CCEM graph coloring factor.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/graph_coloring/picnn.yaml',
        help='YAML config (default: %(default)s)',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None, help='checkpoint to evaluate'
    )
    parser.add_argument(
        '--test-path',
        type=str,
        default=None,
        help='graph instance file (data/graph_coloring/*.json)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='cpu or cuda (config device)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='RNG seed (config seed)',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='trials per graph (eval.num_samples)',
    )
    parser.add_argument(
        '--num-starts',
        type=int,
        default=None,
        help='random starts per trial (solver.num_starts)',
    )
    parser.add_argument(
        '--solver-steps',
        type=int,
        default=None,
        help='projected solver steps per start (solver.steps)',
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='write the summary to this JSON file',
    )
    parser.add_argument(
        '--set',
        action='append',
        default=None,
        metavar='SECTION.KEY=VALUE',
        help='override any config entry (repeatable)',
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of ``python scripts/evaluate.py graph_coloring``."""
    args = parse_args(argv)
    cfg = load_yaml(args.config)
    apply_overrides(
        cfg,
        {
            'checkpoint': args.checkpoint,
            'test_path': args.test_path,
            'device': args.device,
            'seed': args.seed,
            'eval.num_samples': args.num_samples,
            'solver.num_starts': args.num_starts,
            'solver.steps': args.solver_steps,
        },
    )
    apply_overrides(cfg, parse_set_overrides(args.set))
    for key in ('checkpoint', 'test_path'):
        if not cfg.get(key):
            raise ValueError(
                f'{key} must be set in the config or on the command line'
            )
    eval_cfg: dict[str, Any] = cfg.get('eval', {})
    seed = int(cfg.get('seed', 0))
    device = as_device(cfg.get('device'))
    solver = SolverConfig.from_mapping(cfg.get('solver', {}))
    num_samples = max(1, int(eval_cfg.get('num_samples', 1)))

    graphs = read_graphs(cfg['test_path'])
    if not graphs:
        raise ValueError(f'no graphs found in {cfg["test_path"]}')
    # Seed before loading: the factor's parameter initialisation is part
    # of the generator stream of a run.
    set_seed(seed)
    model = load_factor_model(cfg['checkpoint'], device)
    print(f'Checkpoint: {cfg["checkpoint"]} (K={model.num_colors})')
    print(f'Graphs: {len(graphs)} from {cfg["test_path"]}')
    print(
        f'Solver: {solver.optimizer}, starts={solver.num_starts}, '
        f'steps={solver.steps}, schedule={solver.schedule}'
    )
    print(f'Trials per graph: {num_samples} (seed {seed})')

    started = time.perf_counter()
    trials = evaluate_graphs(
        model, graphs, num_samples=num_samples, solver=solver, device=device
    )
    elapsed = time.perf_counter() - started
    summary = summarize(trials)
    for item in summary.per_graph:
        print(
            f'{item.graph_name:<24s} V={item.num_nodes:>4d} '
            f'E={item.num_edges:>5d} '
            f'chi={item.target_colors:>2d} '
            f'valid={item.valid_trials}/{item.trials} '
            f'avg_conflicts={item.conflicting_edges_mean:.2f}'
        )
    print(
        f'Valid colorings: {summary.valid_trials}/{summary.total_trials} '
        f'({summary.valid_rate_percent_mean:.2f} +/- '
        f'{summary.valid_rate_percent_std:.2f} % across graphs)'
    )
    print(
        f'Avg. conflicting edges: {summary.conflicting_edges_mean:.4f} +/- '
        f'{summary.conflicting_edges_std:.4f}'
    )
    print(f'Elapsed: {elapsed:.1f} s')

    if args.output_json is not None:
        payload: dict[str, Any] = {
            'checkpoint': str(cfg['checkpoint']),
            'test_path': str(cfg['test_path']),
            'seed': seed,
            'device': str(device),
            'factor_colors': int(model.num_colors),
            'solver': asdict(solver),
            'elapsed_seconds': elapsed,
            **asdict(summary),
        }
        save_json(payload, args.output_json)
        print(f'Saved metrics to: {args.output_json}')


if __name__ == '__main__':
    main()
