"""Generate planted-solution random 3-SAT splits.

Run from the repository root::

    python -m ccem.tasks.sat3.generate_data --output-dir data/sat3

Protocol: for every formula a uniform assignment is planted, then each
clause picks three distinct variables and redraws its three polarities
until the planted assignment satisfies it (after
``--max-clause-resample`` failed draws one literal is forced to be
true). Training and validation formulas have ``n`` uniform in
``[10, 20]`` variables and ``round(4.258 n)`` clauses (the
phase-transition ratio); the test split has 100 formulas with 20
variables and 91 clauses. One NumPy generator is consumed sequentially
(train, valid, test), so the default ``--seed 0`` with the default
sizes reproduces every array of the shipped ``data/sat3`` files bit
for bit. Arrays are padded to ``--pad-num-variables`` /
``--pad-num-clauses`` (see :mod:`ccem.tasks.sat3.dataset` for the
schema).
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ccem.tasks.sat3.dataset import (
    LITERALS_PER_CLAUSE,
    FloatArray,
    FormulaArrays,
    IntArray,
    clauses_satisfied,
)


@dataclass(frozen=True)
class DatasetSpec:
    """Sizes and generation settings of one dataset build."""

    train_size: int = 4000
    valid_size: int = 1000
    test_size: int = 100
    train_min_variables: int = 10
    train_max_variables: int = 20
    phase_alpha: float = 4.258
    test_num_variables: int = 20
    test_num_clauses: int = 91
    pad_num_variables: int = 50
    pad_num_clauses: int = 218
    seed: int = 0
    max_clause_resample: int = 16

    def __post_init__(self) -> None:
        if (
            self.train_min_variables < 1
            or self.train_max_variables < self.train_min_variables
        ):
            raise ValueError(
                'train variable range must satisfy 1 <= min <= max'
            )
        for num_variables in (
            self.train_max_variables,
            self.test_num_variables,
        ):
            if num_variables > self.pad_num_variables:
                raise ValueError(
                    'num_variables cannot exceed pad_num_variables'
                )
        max_train_clauses = phase_clause_count(
            self.train_max_variables, self.phase_alpha
        )
        for num_clauses in (max_train_clauses, self.test_num_clauses):
            if num_clauses > self.pad_num_clauses:
                raise ValueError('num_clauses cannot exceed pad_num_clauses')


@dataclass(frozen=True)
class Formula:
    """One padded formula with its planted assignment."""

    clause_var_ids: IntArray
    clause_signs: IntArray
    clause_mask: FloatArray
    solution_values: IntArray
    num_variables: int
    num_clauses: int


def phase_clause_count(num_variables: int, alpha: float) -> int:
    """``max(1, round(alpha * n))`` clauses for ``n`` variables."""
    return int(max(1, round(float(alpha) * float(num_variables))))


def _generate_clause(
    assignment: IntArray,
    num_variables: int,
    rng: np.random.Generator,
    max_clause_resample: int,
) -> tuple[IntArray, IntArray]:
    """Draw a clause over three distinct variables satisfied by
    ``assignment``.
    """
    var_ids = rng.choice(
        num_variables, size=LITERALS_PER_CLAUSE, replace=False
    ).astype(np.int64)
    polarities = np.asarray([-1, 1], dtype=np.int64)
    for _ in range(max(int(max_clause_resample), 1)):
        signs = rng.choice(polarities, size=LITERALS_PER_CLAUSE, replace=True)
        if clauses_satisfied(assignment[var_ids], signs):
            return var_ids, signs
    signs = rng.choice(polarities, size=LITERALS_PER_CLAUSE, replace=True)
    forced_idx = int(rng.integers(0, LITERALS_PER_CLAUSE))
    signs[forced_idx] = (
        1 if int(assignment[int(var_ids[forced_idx])]) == 1 else -1
    )
    return var_ids, signs


def generate_formula(
    num_variables: int,
    num_clauses: int,
    rng: np.random.Generator,
    spec: DatasetSpec,
) -> Formula:
    """Plant a uniform assignment and draw ``num_clauses`` satisfied
    clauses.
    """
    assignment = rng.integers(0, 2, size=(num_variables,), dtype=np.int64)
    clause_var_ids = np.zeros(
        (spec.pad_num_clauses, LITERALS_PER_CLAUSE), dtype=np.int64
    )
    clause_signs = np.ones(
        (spec.pad_num_clauses, LITERALS_PER_CLAUSE), dtype=np.int64
    )
    clause_mask = np.zeros((spec.pad_num_clauses,), dtype=np.float32)
    for clause_idx in range(num_clauses):
        var_ids, signs = _generate_clause(
            assignment, num_variables, rng, spec.max_clause_resample
        )
        clause_var_ids[clause_idx] = var_ids
        clause_signs[clause_idx] = signs
        clause_mask[clause_idx] = 1.0
    solution_values = np.zeros((spec.pad_num_variables,), dtype=np.int64)
    solution_values[:num_variables] = assignment
    return Formula(
        clause_var_ids=clause_var_ids,
        clause_signs=clause_signs,
        clause_mask=clause_mask,
        solution_values=solution_values,
        num_variables=num_variables,
        num_clauses=num_clauses,
    )


def stack_formulas(formulas: list[Formula]) -> FormulaArrays:
    """Stack per-formula arrays into one split in the ``.npz``
    schema.
    """
    return FormulaArrays(
        clause_var_ids=np.stack([item.clause_var_ids for item in formulas]),
        clause_signs=np.stack([item.clause_signs for item in formulas]),
        clause_mask=np.stack([item.clause_mask for item in formulas]),
        solution_values=np.stack([item.solution_values for item in formulas]),
        num_variables=np.asarray(
            [item.num_variables for item in formulas], dtype=np.int64
        ),
        num_clauses=np.asarray(
            [item.num_clauses for item in formulas], dtype=np.int64
        ),
    )


def build_mixed_split(
    split_size: int, rng: np.random.Generator, spec: DatasetSpec
) -> FormulaArrays:
    """Formulas with ``n ~ U{min..max}`` variables at the
    phase-transition ratio.
    """
    formulas = []
    for _ in range(split_size):
        num_variables = int(
            rng.integers(
                spec.train_min_variables, spec.train_max_variables + 1
            )
        )
        num_clauses = phase_clause_count(num_variables, spec.phase_alpha)
        formulas.append(
            generate_formula(num_variables, num_clauses, rng, spec)
        )
    return stack_formulas(formulas)


def build_fixed_split(
    split_size: int,
    num_variables: int,
    num_clauses: int,
    rng: np.random.Generator,
    spec: DatasetSpec,
) -> FormulaArrays:
    """``split_size`` formulas with fixed ``num_variables`` /
    ``num_clauses``.
    """
    return stack_formulas(
        [
            generate_formula(num_variables, num_clauses, rng, spec)
            for _ in range(split_size)
        ]
    )


def build_dataset(output_dir: str | Path, spec: DatasetSpec) -> None:
    """Write ``train.npz``, ``valid.npz`` and ``test.npz`` under
    ``output_dir``.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    splits = {
        'train.npz': build_mixed_split(spec.train_size, rng, spec),
        'valid.npz': build_mixed_split(spec.valid_size, rng, spec),
        'test.npz': build_fixed_split(
            spec.test_size,
            spec.test_num_variables,
            spec.test_num_clauses,
            rng,
            spec,
        ),
    }
    for name, payload in splits.items():
        path = root / name
        np.savez_compressed(path, **payload)
        print(
            f'Saved {name} ({int(payload["num_variables"].shape[0])} '
            f'formulas) to: {path}'
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI flags; the defaults reproduce the shipped ``data/sat3``
    splits.
    """
    defaults = DatasetSpec()
    parser = argparse.ArgumentParser(
        description=(
            'Generate the planted random 3-SAT train/valid/test splits.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/sat3',
        help='directory of train.npz, valid.npz and test.npz',
    )
    parser.add_argument(
        '--train-size',
        type=int,
        default=defaults.train_size,
        help='training formulas',
    )
    parser.add_argument(
        '--valid-size',
        type=int,
        default=defaults.valid_size,
        help='validation formulas',
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=defaults.test_size,
        help='test formulas',
    )
    parser.add_argument(
        '--train-min-variables',
        type=int,
        default=defaults.train_min_variables,
        help='smallest variable count of the train/valid formulas',
    )
    parser.add_argument(
        '--train-max-variables',
        type=int,
        default=defaults.train_max_variables,
        help='largest variable count of the train/valid formulas',
    )
    parser.add_argument(
        '--phase-alpha',
        type=float,
        default=defaults.phase_alpha,
        help='clauses per variable of the train/valid formulas',
    )
    parser.add_argument(
        '--test-num-variables',
        type=int,
        default=defaults.test_num_variables,
        help='variables of every test formula',
    )
    parser.add_argument(
        '--test-num-clauses',
        type=int,
        default=defaults.test_num_clauses,
        help='clauses of every test formula',
    )
    parser.add_argument(
        '--pad-num-variables',
        type=int,
        default=defaults.pad_num_variables,
        help='variable padding of the stored arrays',
    )
    parser.add_argument(
        '--pad-num-clauses',
        type=int,
        default=defaults.pad_num_clauses,
        help='clause padding of the stored arrays',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=defaults.seed,
        help='NumPy seed; 0 reproduces the shipped files',
    )
    parser.add_argument(
        '--max-clause-resample',
        type=int,
        default=defaults.max_clause_resample,
        help='polarity redraws before one literal is forced true',
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of ``python -m ccem.tasks.sat3.generate_data``."""
    args = vars(parse_args(argv))
    output_dir = args.pop('output_dir')
    build_dataset(output_dir, DatasetSpec(**args))


if __name__ == '__main__':
    main()
