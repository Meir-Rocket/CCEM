"""Exact 3-SAT metrics on rounded assignments."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ccem.tasks.sat3.dataset import check_clause_tensors, gather_clause_values


@dataclass(frozen=True)
class SAT3EvalResult:
    """Additive clause/formula counts; ``SAT3EvalResult()`` is the
    identity.
    """

    num_formulas: int = 0
    num_clauses: int = 0
    num_satisfied_clauses: int = 0
    num_satisfied_formulas: int = 0

    @property
    def clause_satisfaction_rate(self) -> float:
        """Satisfied active clauses over all active clauses (the
        paper's metric).
        """
        if self.num_clauses == 0:
            return 0.0
        return self.num_satisfied_clauses / self.num_clauses

    @property
    def sat_formula_fraction(self) -> float:
        """Fraction of formulas whose decoded assignment satisfies
        every clause.
        """
        if self.num_formulas == 0:
            return 0.0
        return self.num_satisfied_formulas / self.num_formulas

    @property
    def avg_unsat_clauses(self) -> float:
        """Mean number of violated clauses per formula."""
        if self.num_formulas == 0:
            return 0.0
        return (
            self.num_clauses - self.num_satisfied_clauses
        ) / self.num_formulas

    def __add__(self, other: 'SAT3EvalResult') -> 'SAT3EvalResult':
        return SAT3EvalResult(
            num_formulas=self.num_formulas + other.num_formulas,
            num_clauses=self.num_clauses + other.num_clauses,
            num_satisfied_clauses=self.num_satisfied_clauses
            + other.num_satisfied_clauses,
            num_satisfied_formulas=(
                self.num_satisfied_formulas + other.num_satisfied_formulas
            ),
        )

    def as_dict(self) -> dict[str, float]:
        """Counts and derived rates as a flat mapping (for the JSON
        summary).
        """
        return {
            'num_formulas': self.num_formulas,
            'num_clauses': self.num_clauses,
            'num_satisfied_clauses': self.num_satisfied_clauses,
            'num_satisfied_formulas': self.num_satisfied_formulas,
            'clause_satisfaction_rate': self.clause_satisfaction_rate,
            'sat_formula_fraction': self.sat_formula_fraction,
            'avg_unsat_clauses': self.avg_unsat_clauses,
        }


def clause_satisfaction(
    pred_values: Tensor, clause_var_ids: Tensor, clause_signs: Tensor
) -> Tensor:
    """Boolean ``[B, M]`` tensor: which clauses the ``{0, 1}``
    assignment satisfies.

    Args:
        pred_values: ``[B, N]`` integer assignment.
        clause_var_ids: ``[B, M, 3]`` variable ids.
        clause_signs: ``[B, M, 3]`` literal polarities in ``{-1, +1}``.
    """
    if pred_values.ndim != 2:
        raise ValueError('pred_values must have shape [B, N]')
    check_clause_tensors(clause_var_ids, clause_signs)
    literal_values = gather_clause_values(pred_values, clause_var_ids)
    literal_is_true = torch.where(
        clause_signs > 0, literal_values == 1, literal_values == 0
    )
    return literal_is_true.any(dim=-1)


def evaluate_batch(
    pred_values: Tensor,
    clause_var_ids: Tensor,
    clause_signs: Tensor,
    clause_mask: Tensor,
) -> SAT3EvalResult:
    """Count satisfied active clauses and fully satisfied formulas of
    a batch.

    Args:
        pred_values: ``[B, N]`` integer assignment.
        clause_var_ids: ``[B, M, 3]`` variable ids.
        clause_signs: ``[B, M, 3]`` literal polarities in ``{-1, +1}``.
        clause_mask: ``[B, M]`` active-clause mask; padded clauses are
            not counted.
    """
    satisfied = clause_satisfaction(pred_values, clause_var_ids, clause_signs)
    if clause_mask.shape != satisfied.shape:
        raise ValueError('clause_mask must have shape [B, M]')
    active = clause_mask > 0.5
    unsatisfied_per_formula = ((~satisfied) & active).sum(dim=1)
    return SAT3EvalResult(
        num_formulas=int(pred_values.shape[0]),
        num_clauses=int(active.sum().item()),
        num_satisfied_clauses=int((satisfied & active).sum().item()),
        num_satisfied_formulas=int(
            (unsatisfied_per_formula == 0).sum().item()
        ),
    )
