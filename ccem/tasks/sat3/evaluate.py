"""Decode 3-SAT formulas with a trained clause factor and count the
satisfied clauses.

Run from the repository root::

    python scripts/evaluate.py sat3 --config configs/sat3/picnn.yaml

The clause factor and the composed energy are built first; the RNG is
seeded right before the decoding loop, whose only random draws are the
solver starts of
:func:`ccem.tasks.sat3.energy.decode_sat3_assignments` (one draw per
batch), so the reported numbers are reproducible for a given seed and
batch size. ``--output-json PATH`` additionally writes one JSON summary
of the settings and the metrics.
"""

import argparse
from collections.abc import Sequence
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from ccem.tasks.sat3.dataset import (
    SAT3FormulaDataset,
    SAT3FormulaSample,
    collate_formula_batch,
)
from ccem.tasks.sat3.energy import (
    SAT3ComposedEnergy,
    SAT3SolverConfig,
    decode_sat3_assignments,
    load_clause_model,
)
from ccem.tasks.sat3.metrics import SAT3EvalResult, evaluate_batch
from ccem.utils import (
    apply_overrides,
    as_device,
    load_yaml,
    parse_set_overrides,
    save_json,
    set_seed,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI flags; every explicit flag overrides the config, ``--set``
    last.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate the CCEM 3-SAT clause factor.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/sat3/picnn.yaml',
        help='YAML config (default: %(default)s)',
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None, help='checkpoint to evaluate'
    )
    parser.add_argument(
        '--test-path', type=str, default=None, help='test formulas (.npz)'
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
        '--num-formulas',
        type=int,
        default=None,
        help='decode only the first N formulas (eval.num_formulas; 0 = all)',
    )
    parser.add_argument(
        '--num-starts',
        type=int,
        default=None,
        help='random starts per formula (solver.num_starts)',
    )
    parser.add_argument(
        '--solver-steps',
        type=int,
        default=None,
        help='projected-gradient steps per start (solver.steps)',
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


def evaluate_loader(
    loader: DataLoader[SAT3FormulaSample],
    composer: SAT3ComposedEnergy,
    solver: SAT3SolverConfig,
    device: torch.device,
    *,
    description: str = '3-SAT eval',
) -> SAT3EvalResult:
    """Decode every batch of ``loader`` and accumulate the exact
    metrics.
    """
    total = SAT3EvalResult()
    progress = tqdm(loader, desc=description, unit='batch', dynamic_ncols=True)
    for batch in progress:
        clause_var_ids = batch['clause_var_ids'].to(device)
        clause_signs = batch['clause_signs'].to(device)
        clause_mask = batch['clause_mask'].to(device)
        pred_values = decode_sat3_assignments(
            composer,
            clause_var_ids,
            clause_signs,
            clause_mask,
            solver,
            num_variables=batch['num_variables'].to(device),
        )
        total = total + evaluate_batch(
            pred_values, clause_var_ids, clause_signs, clause_mask
        )
        progress.set_postfix(
            formulas=total.num_formulas,
            sat=f'{total.sat_formula_fraction:.3f}',
            refresh=False,
        )
    return total


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of ``python scripts/evaluate.py sat3``."""
    args = parse_args(argv)
    cfg = load_yaml(args.config)
    apply_overrides(
        cfg,
        {
            'checkpoint': args.checkpoint,
            'test_path': args.test_path,
            'device': args.device,
            'seed': args.seed,
            'eval.num_formulas': args.num_formulas,
            'solver.num_starts': args.num_starts,
            'solver.steps': args.solver_steps,
        },
    )
    apply_overrides(cfg, parse_set_overrides(args.set))
    for key in ('test_path', 'checkpoint'):
        if not cfg.get(key):
            raise ValueError(
                f'{key} must be set in the config or on the command line'
            )
    eval_cfg: dict[str, Any] = cfg.get('eval', {})
    seed = int(cfg.get('seed', 0))
    device = as_device(str(cfg.get('device', 'cpu')))

    full_dataset = SAT3FormulaDataset(cfg['test_path'])
    dataset: Dataset[SAT3FormulaSample] = full_dataset
    requested = int(eval_cfg.get('num_formulas', 0))
    if requested > 0:
        dataset = Subset(
            full_dataset, list(range(min(requested, len(full_dataset))))
        )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(eval_cfg.get('batch_size', 16))),
        shuffle=False,
        collate_fn=collate_formula_batch,
    )

    # The factor is built before seeding; the seed covers the starts.
    model = load_clause_model(cfg['checkpoint'], device)
    composer = SAT3ComposedEnergy(model, full_dataset.num_variables).to(device)
    composer.eval()
    solver = SAT3SolverConfig.from_mapping(cfg.get('solver', {}))

    set_seed(seed)
    total = evaluate_loader(loader, composer, solver, device)
    if total.num_formulas == 0:
        raise RuntimeError('Evaluation dataset is empty')

    print(f'Formulas: {total.num_formulas}')
    print(f'Clause satisfaction rate: {total.clause_satisfaction_rate:.4f}')
    print(f'SAT formula fraction: {total.sat_formula_fraction:.4f}')
    print(f'Avg unsatisfied clauses: {total.avg_unsat_clauses:.4f}')

    if args.output_json is not None:
        summary: dict[str, Any] = {
            'checkpoint': str(cfg['checkpoint']),
            'test_path': str(cfg['test_path']),
            'seed': seed,
            'solver_steps': solver.continuous.steps,
            'solver_step_size': solver.continuous.step_size,
            'num_starts': solver.num_starts,
            'discrete_steps': solver.discrete_steps,
            **total.as_dict(),
        }
        save_json(summary, args.output_json)
        print(f'Saved metrics to: {args.output_json}')


if __name__ == '__main__':
    main()
