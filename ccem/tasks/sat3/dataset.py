"""3-SAT formula datasets and the ``.npz`` schema shared with the
generator (:mod:`ccem.tasks.sat3.generate_data`).

A file holds ``B`` planted-solution formulas padded to ``M`` clauses and
``N_pad`` variables (``numpy.savez_compressed``):

* ``clause_var_ids`` int64 ``[B, M, 3]``: 0-based ids of the three
  distinct variables of each clause (all zero in padded clauses);
* ``clause_signs`` int64 ``[B, M, 3]`` in ``{-1, +1}``: literal
  polarity, ``+1`` for ``x`` and ``-1`` for ``not x`` (all ``+1`` in
  padded clauses);
* ``clause_mask`` float32 ``[B, M]``: 1 for active clauses, 0 for
  padding;
* ``num_variables``, ``num_clauses`` int64 ``[B]``: active sizes;
* ``solution_values`` int64 ``[B, N_pad]`` in ``{0, 1}`` (optional): a
  planted assignment satisfying every active clause, zero beyond
  ``num_variables``. It is a training witness only; evaluation accepts
  any satisfying assignment.

Unknown keys are ignored. Relaxed assignments are one 2-simplex row per
variable, ``[B, N, 2]`` with class 1 meaning "true", so the local input
of a clause factor is the ``[3, 2]`` block gathered at its three
variables.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as f
from torch import Tensor
from torch.utils.data import Dataset

LITERALS_PER_CLAUSE = 3
BINARY_VALUES = 2

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class FormulaArrays(TypedDict):
    """One stacked split in the ``.npz`` schema of this module."""

    clause_var_ids: IntArray
    clause_signs: IntArray
    clause_mask: FloatArray
    solution_values: IntArray
    num_variables: IntArray
    num_clauses: IntArray


@dataclass(frozen=True)
class SAT3FormulaSample:
    """One padded formula: clause tensors and its active variable
    count.
    """

    clause_var_ids: Tensor
    clause_signs: Tensor
    clause_mask: Tensor
    num_variables: int


@dataclass(frozen=True)
class SAT3ClauseSample:
    """A clause with a satisfying and the falsifying local assignment.

    ``positive_assignment``/``negative_assignment`` are one-hot
    ``[3, 2]`` blocks (class 1 = true).
    """

    clause_signs: Tensor
    positive_assignment: Tensor
    negative_assignment: Tensor


def check_clause_tensors(clause_var_ids: Tensor, clause_signs: Tensor) -> None:
    """Raise ``ValueError`` unless both tensors are ``[B, M, 3]`` and
    alike.
    """
    if (
        clause_var_ids.ndim != 3
        or clause_var_ids.shape[-1] != LITERALS_PER_CLAUSE
    ):
        raise ValueError('clause_var_ids must have shape [B, M, 3]')
    if clause_signs.shape != clause_var_ids.shape:
        raise ValueError(
            'clause_signs must have the same shape as clause_var_ids'
        )


def gather_clause_values(values: Tensor, clause_var_ids: Tensor) -> Tensor:
    """Read per-variable values at the three variables of every clause.

    Args:
        values: ``[B, N, ...]`` per-variable values (integers ``[B, N]``
            or relaxed rows ``[B, N, 2]``).
        clause_var_ids: ``[B, M, 3]`` variable ids.

    Returns:
        ``[B, M, 3, ...]`` with the trailing dimensions of ``values``.
    """
    batch_ids = torch.arange(values.shape[0], device=values.device)[
        :, None, None
    ]
    return values[batch_ids, clause_var_ids.long()]


def assignment_onehot(
    values: Tensor, dtype: torch.dtype = torch.float32
) -> Tensor:
    """One-hot encode ``{0, 1}`` values as ``[..., 2]`` rows (class 1
    = true).
    """
    return f.one_hot(values.long(), num_classes=BINARY_VALUES).to(dtype=dtype)


def clauses_satisfied(
    literal_values: IntArray, clause_signs: IntArray
) -> BoolArray:
    """Whether clauses hold given the ``{0, 1}`` values of their
    literals.

    A literal with sign ``+1`` is true when its variable is 1 and one
    with sign ``-1`` when it is 0; a clause holds when any literal is
    true. Both arrays are ``[..., 3]`` and the result drops the last
    axis.
    """
    literal_is_true = np.where(
        clause_signs > 0, literal_values == 1, literal_values == 0
    )
    return np.asarray(literal_is_true.any(axis=-1), dtype=np.bool_)


def _load_split(path: str | Path) -> tuple[FormulaArrays, bool]:
    """Read the schema keys of ``path``; the flag says whether a
    witness exists.
    """
    with np.load(path, allow_pickle=False) as data:
        required = (
            'clause_var_ids',
            'clause_signs',
            'clause_mask',
            'num_variables',
            'num_clauses',
        )
        missing = sorted(set(required).difference(data.files))
        if missing:
            raise ValueError(f'Dataset file is missing keys: {missing}')
        has_witness = 'solution_values' in data
        arrays = FormulaArrays(
            clause_var_ids=data['clause_var_ids'].astype(np.int64),
            clause_signs=data['clause_signs'].astype(np.int64),
            clause_mask=data['clause_mask'].astype(np.float32),
            solution_values=(
                data['solution_values'].astype(np.int64)
                if has_witness
                else np.zeros(
                    (data['clause_var_ids'].shape[0], 0), dtype=np.int64
                )
            ),
            num_variables=data['num_variables'].astype(np.int64).reshape(-1),
            num_clauses=data['num_clauses'].astype(np.int64).reshape(-1),
        )
    return arrays, has_witness


def _validate_split(arrays: FormulaArrays, *, has_witness: bool) -> None:
    """Check the structural invariants of the schema in the module
    docstring.
    """
    clause_var_ids = arrays['clause_var_ids']
    clause_signs = arrays['clause_signs']
    clause_mask = arrays['clause_mask']
    num_variables = arrays['num_variables']
    num_clauses = arrays['num_clauses']
    num_formulas = clause_var_ids.shape[0]
    if (
        clause_var_ids.ndim != 3
        or clause_var_ids.shape[-1] != LITERALS_PER_CLAUSE
    ):
        raise ValueError('clause_var_ids must have shape [B, M, 3]')
    if clause_signs.shape != clause_var_ids.shape:
        raise ValueError(
            'clause_signs must have the same shape as clause_var_ids'
        )
    if clause_mask.shape != clause_var_ids.shape[:2]:
        raise ValueError('clause_mask must have shape [B, M]')
    if (
        num_variables.shape[0] != num_formulas
        or num_clauses.shape[0] != num_formulas
    ):
        raise ValueError(
            'num_variables and num_clauses must contain one value per formula'
        )
    if np.any(clause_var_ids < 0):
        raise ValueError('clause_var_ids must be non-negative')
    if np.any((clause_signs != -1) & (clause_signs != 1)):
        raise ValueError('clause_signs must contain only -1 and +1')
    if np.any(num_variables <= 0):
        raise ValueError('Every formula must contain at least one variable')
    if np.any(num_clauses <= 0):
        raise ValueError('Every formula must contain at least one clause')

    active = clause_mask > 0.5
    if np.any(
        active[..., None] & (clause_var_ids >= num_variables[:, None, None])
    ):
        raise ValueError(
            'An active clause references an out-of-range variable'
        )
    repeated_variable = (
        (clause_var_ids[..., 0] == clause_var_ids[..., 1])
        | (clause_var_ids[..., 0] == clause_var_ids[..., 2])
        | (clause_var_ids[..., 1] == clause_var_ids[..., 2])
    )
    if np.any(active & repeated_variable):
        raise ValueError(
            'Every active 3-SAT clause must use three distinct variables'
        )
    if not has_witness:
        return
    solution_values = arrays['solution_values']
    if solution_values.ndim != 2 or solution_values.shape[0] != num_formulas:
        raise ValueError('solution_values must have shape [B, N]')
    if np.any((solution_values != 0) & (solution_values != 1)):
        raise ValueError('solution_values must contain only 0 and 1')
    if solution_values.shape[1] < int(np.max(num_variables)):
        raise ValueError('solution_values is shorter than num_variables')
    batch_ids = np.arange(num_formulas)[:, None, None]
    literal_values = solution_values[batch_ids, clause_var_ids]
    if np.any(active & ~clauses_satisfied(literal_values, clause_signs)):
        raise ValueError(
            'Stored solution does not satisfy every active clause'
        )


class SAT3FormulaDataset(Dataset[SAT3FormulaSample]):
    """Whole formulas of one ``.npz`` split, validated at load time.

    Attributes:
        clause_var_ids: Stacked ``[B, M, 3]`` variable ids.
        clause_signs: Stacked ``[B, M, 3]`` literal polarities.
        clause_mask: Stacked ``[B, M]`` active-clause masks.
        solution_values: ``[B, N_pad]`` planted witnesses or ``None``.
        num_variables_per_formula: ``[B]`` active variable counts.
        num_formulas: Number of formulas ``B`` in the file.
        num_variables: Largest active variable count in the file; the
            composed energy is instantiated with this width.
    """

    def __init__(self, file_path: str | Path) -> None:
        arrays, has_witness = _load_split(file_path)
        _validate_split(arrays, has_witness=has_witness)
        self.clause_var_ids = torch.from_numpy(arrays['clause_var_ids'])
        self.clause_signs = torch.from_numpy(arrays['clause_signs'])
        self.clause_mask = torch.from_numpy(arrays['clause_mask'])
        self.solution_values: Optional[Tensor] = (
            torch.from_numpy(arrays['solution_values'])
            if has_witness
            else None
        )
        self.num_variables_per_formula = torch.from_numpy(
            arrays['num_variables']
        )
        self.num_formulas = int(self.clause_var_ids.shape[0])
        self.num_variables = int(np.max(arrays['num_variables']))

    def __len__(self) -> int:
        return self.num_formulas

    def __getitem__(self, index: int) -> SAT3FormulaSample:
        return SAT3FormulaSample(
            clause_var_ids=self.clause_var_ids[index],
            clause_signs=self.clause_signs[index],
            clause_mask=self.clause_mask[index],
            num_variables=int(self.num_variables_per_formula[index].item()),
        )


class SAT3ClauseDataset(Dataset[SAT3ClauseSample]):
    """Local training pairs for Phase 1: one item per active clause.

    The positive example is the witness restricted to the clause's
    variables; the negative one is the unique falsifying local
    assignment (every literal false, i.e. value 0 where the sign is
    ``+1`` and 1 where it is ``-1``).
    """

    def __init__(self, file_path: str | Path) -> None:
        self.formulas = SAT3FormulaDataset(file_path)
        if self.formulas.solution_values is None:
            raise ValueError(
                'Clause training requires solution_values in the dataset'
            )
        self.num_formulas = len(self.formulas)

        active = self.formulas.clause_mask > 0.5
        formula_ids = torch.arange(self.num_formulas)[:, None].expand_as(
            active
        )[active]
        clause_var_ids = self.formulas.clause_var_ids[active]
        self.clause_signs = self.formulas.clause_signs[active]
        positive_values = self.formulas.solution_values[
            formula_ids[:, None], clause_var_ids
        ]
        negative_values = (self.clause_signs < 0).long()
        self.positive_assignments = assignment_onehot(positive_values)
        self.negative_assignments = assignment_onehot(negative_values)

    def __len__(self) -> int:
        return int(self.clause_signs.shape[0])

    def __getitem__(self, index: int) -> SAT3ClauseSample:
        return SAT3ClauseSample(
            clause_signs=self.clause_signs[index],
            positive_assignment=self.positive_assignments[index],
            negative_assignment=self.negative_assignments[index],
        )


def collate_clause_batch(items: list[SAT3ClauseSample]) -> dict[str, Tensor]:
    """Stack clause samples into ``clause_signs`` and the two
    assignments.
    """
    return {
        'clause_signs': torch.stack(
            [item.clause_signs for item in items]
        ).long(),
        'positive_assignment': torch.stack(
            [item.positive_assignment for item in items]
        ).float(),
        'negative_assignment': torch.stack(
            [item.negative_assignment for item in items]
        ).float(),
    }


def collate_formula_batch(items: list[SAT3FormulaSample]) -> dict[str, Tensor]:
    """Stack the clause tensors and active variable counts; stored
    witnesses are never batched.
    """
    return {
        'clause_var_ids': torch.stack(
            [item.clause_var_ids for item in items]
        ).long(),
        'clause_signs': torch.stack(
            [item.clause_signs for item in items]
        ).long(),
        'clause_mask': torch.stack(
            [item.clause_mask for item in items]
        ).float(),
        'num_variables': torch.tensor(
            [item.num_variables for item in items], dtype=torch.long
        ),
    }
