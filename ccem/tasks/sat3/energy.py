"""Clause factor, composed 3-SAT energy and the projected solve/decode.

One PICNN factor ``E_c(y_c | s_c)`` scores the ``[3, 2]`` relaxed local
assignment ``y_c`` of a clause given its literal polarities ``s_c``. It
is trained so that the unique falsifying local assignment has high
energy and the seven satisfying ones low energy (low energy =
satisfied). The composed energy of a formula is the sum of the factor
over its active clauses, each evaluated on the probability rows gathered
at the clause's variables (the paper's composed energy with a single
factor type). Inference minimises it with projected gradient descent
over the product of per-variable 2-simplices, rounds by ``argmax``,
refines the rounded assignment by coordinate descent on the learned
energy and keeps the lowest-energy start.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ccem.icnn import PICNN
from ccem.projections import project_simplex
from ccem.solvers import EnergyFn, SolverConfig, run_projected_solver
from ccem.tasks.sat3.dataset import (
    BINARY_VALUES,
    LITERALS_PER_CLAUSE,
    assignment_onehot,
    check_clause_tensors,
    gather_clause_values,
)
from ccem.utils import load_checkpoint

ENERGY_SEMANTICS = 'sat3_clause_examples_v2'
"""Training-objective tag stored in checkpoints; the loader requires
equality.
"""


def build_clause_descriptor(clause_signs: Tensor) -> Tensor:
    """PICNN context of a clause: its polarities as ``[..., 3]`` in
    ``{0, 1}``.

    ``1`` marks a positive literal. The shipped checkpoint was trained
    on this exact encoding (``model.u_layers.0.weight`` has three input
    columns).
    """
    if clause_signs.shape[-1] != LITERALS_PER_CLAUSE:
        raise ValueError('Last dimension must contain exactly three literals')
    return (clause_signs > 0).float()


def random_sat3_assignments(
    batch_size: int, num_variables: int, device: torch.device
) -> Tensor:
    """Uniform random starts ``[B, N, 2]``: ``p ~ U[0, 1)`` and rows
    ``(1 - p, p)``.

    Draws from the torch RNG; callers seed before drawing starts.
    """
    probability_one = torch.rand(
        int(batch_size), int(num_variables), 1, device=device
    )
    return torch.cat((1.0 - probability_one, probability_one), dim=-1)


def literal_true_probabilities(
    assign_probs: Tensor, clause_var_ids: Tensor, clause_signs: Tensor
) -> Tensor:
    """Probability ``[B, M, 3]`` that each literal is true under relaxed
    rows.

    A positive literal reads class 1 of its variable's row, a negative
    literal class 0 (used by the Phase-2 soft-clause objective).
    """
    local = gather_clause_values(assign_probs, clause_var_ids)
    return torch.where(clause_signs > 0, local[..., 1], local[..., 0])


class SAT3ClauseEnergyModel(nn.Module):
    """PICNN clause factor conditioned only on the three literal
    polarities.

    The child ``model`` is ``PICNN(context_dim=3, y_dim=6,
    hidden_dims=[hidden_dim] * depth, activation)``. Its name and the
    PICNN children are fixed by the shipped checkpoint (state-dict
    prefix ``model.``).
    """

    def __init__(
        self, hidden_dim: int, depth: int, activation: str = 'relu'
    ) -> None:
        super().__init__()
        self.model = PICNN(
            context_dim=LITERALS_PER_CLAUSE,
            y_dim=LITERALS_PER_CLAUSE * BINARY_VALUES,
            hidden_dims=[int(hidden_dim)] * max(int(depth), 1),
            activation=activation,
        )

    @classmethod
    def from_config(
        cls, model_cfg: Mapping[str, Any]
    ) -> 'SAT3ClauseEnergyModel':
        """Build from a ``model:`` block / checkpoint ``model_config``
        mapping with the keys ``hidden_dim``, ``depth`` and
        ``activation``.
        """
        return cls(
            hidden_dim=int(model_cfg['hidden_dim']),
            depth=int(model_cfg['depth']),
            activation=str(model_cfg['activation']),
        )

    def forward(self, clause_signs: Tensor, y_clause: Tensor) -> Tensor:
        """Energy ``[...]`` of local assignments ``y_clause``
        ``[..., 3, 2]``.
        """
        if y_clause.ndim < 3 or y_clause.shape[-2:] != (
            LITERALS_PER_CLAUSE,
            BINARY_VALUES,
        ):
            raise ValueError('y_clause must have shape [..., 3, 2]')
        if clause_signs.shape != y_clause.shape[:-1]:
            raise ValueError(
                'clause_signs must match the literal dimensions of y_clause'
            )
        descriptor = build_clause_descriptor(clause_signs).to(
            device=y_clause.device, dtype=y_clause.dtype
        )
        energy: Tensor = self.model(
            descriptor.reshape(-1, LITERALS_PER_CLAUSE),
            y_clause.reshape(-1, LITERALS_PER_CLAUSE * BINARY_VALUES),
        )
        return energy.reshape(y_clause.shape[:-2])


class SAT3ComposedEnergy(nn.Module):
    """Composed energy: the clause factor summed over the active
    clauses.

    Args:
        factor_model: The shared clause factor.
        num_variables: Width ``N`` of the assignments ``[B, N, 2]`` (the
            largest variable count of the dataset; unused variables are
            simply never referenced by a clause).
    """

    def __init__(
        self, factor_model: SAT3ClauseEnergyModel, num_variables: int
    ) -> None:
        super().__init__()
        self.factor_model = factor_model
        self.num_variables = int(num_variables)

    def factor_energies(
        self,
        assign_probs: Tensor,
        clause_var_ids: Tensor,
        clause_signs: Tensor,
        clause_mask: Tensor,
    ) -> Tensor:
        """Per-clause energies ``[B, M]``, zeroed on padded clauses by
        the ``[B, M]`` mask.
        """
        if assign_probs.ndim != 3 or assign_probs.shape[-1] != BINARY_VALUES:
            raise ValueError('assign_probs must have shape [B, N, 2]')
        if assign_probs.shape[1] != self.num_variables:
            raise ValueError(
                f'assign_probs must contain {self.num_variables} variables'
            )
        check_clause_tensors(clause_var_ids, clause_signs)
        if clause_var_ids.shape[0] != assign_probs.shape[0]:
            raise ValueError(
                'Assignments and clauses must have the same batch size'
            )
        if clause_mask.shape != clause_var_ids.shape[:2]:
            raise ValueError('clause_mask must have shape [B, M]')
        energies: Tensor = self.factor_model(
            clause_signs=clause_signs,
            y_clause=gather_clause_values(assign_probs, clause_var_ids),
        )
        return energies * clause_mask.to(
            device=energies.device, dtype=energies.dtype
        )

    def total_energy(
        self,
        assign_probs: Tensor,
        clause_var_ids: Tensor,
        clause_signs: Tensor,
        clause_mask: Tensor,
    ) -> Tensor:
        """Formula energies ``[B]``: the sum of
        :meth:`factor_energies`.
        """
        return self.factor_energies(
            assign_probs, clause_var_ids, clause_signs, clause_mask
        ).sum(dim=-1)

    def energy_fn(
        self, clause_var_ids: Tensor, clause_signs: Tensor, clause_mask: Tensor
    ) -> EnergyFn:
        """Close over one batch of formulas:
        ``assign_probs -> total_energy``.
        """

        def energy(assign_probs: Tensor) -> Tensor:
            return self.total_energy(
                assign_probs, clause_var_ids, clause_signs, clause_mask
            )

        return energy


@dataclass(frozen=True)
class SAT3SolverConfig:
    """The continuous solve plus the discrete refinement of the 3-SAT
    decoder.

    Attributes:
        continuous: Projected-gradient-descent settings (``optimizer``
            must be ``pgd`` and no tie-break term is supported).
        discrete_steps: Maximum coordinate-descent flips per start.
        discrete_tolerance: A flip must lower the energy by more than
            this.
    """

    continuous: SolverConfig
    discrete_steps: int = 0
    discrete_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.continuous.optimizer != 'pgd':
            raise ValueError(
                '3-SAT inference supports solver.optimizer=pgd only'
            )
        if self.continuous.tie_break_weight != 0.0:
            raise ValueError(
                '3-SAT inference does not support a tie-break term'
            )
        if self.discrete_steps < 0:
            raise ValueError('solver.discrete_steps must be non-negative')
        if (
            not math.isfinite(self.discrete_tolerance)
            or self.discrete_tolerance < 0.0
        ):
            raise ValueError(
                'solver.discrete_tolerance must be finite and non-negative'
            )

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> 'SAT3SolverConfig':
        """Build from the YAML ``solver:`` block.

        The continuous part is
        :meth:`ccem.solvers.SolverConfig.from_mapping` with
        ``optimizer`` defaulting to ``pgd``; ``discrete_steps`` and
        ``discrete_tolerance`` default to 0 and ``1e-6``.
        """
        return cls(
            continuous=SolverConfig.from_mapping(cfg, optimizer='pgd'),
            discrete_steps=int(cfg.get('discrete_steps', 0)),
            discrete_tolerance=float(cfg.get('discrete_tolerance', 1.0e-6)),
        )

    @property
    def num_starts(self) -> int:
        """Random restarts per formula."""
        return self.continuous.num_starts


def load_clause_model(
    checkpoint_path: str | Path, device: torch.device
) -> SAT3ClauseEnergyModel:
    """Load a 3-SAT checkpoint strictly and return the factor in eval
    mode on ``device``.

    The payload must carry ``task == 'sat3'``, ``energy_semantics ==
    ENERGY_SEMANTICS``, ``model_state`` and ``model_config``.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f'3-SAT checkpoint does not exist: {path}. '
            'Run python scripts/train.py sat3 first.'
        )
    payload = load_checkpoint(path, device)
    if payload.get('task') != 'sat3':
        raise ValueError('The checkpoint is not a 3-SAT checkpoint')
    if payload.get('energy_semantics') != ENERGY_SEMANTICS:
        raise ValueError(
            'This checkpoint uses an incompatible 3-SAT training objective. '
            'Train a new clause factor with python scripts/train.py sat3.'
        )
    for key in ('model_state', 'model_config'):
        if key not in payload:
            raise ValueError(f'Checkpoint is missing {key}')
    model = SAT3ClauseEnergyModel.from_config(payload['model_config']).to(
        device
    )
    model.load_state_dict(payload['model_state'])
    model.eval()
    return model


def solve_sat3_assignments(
    composer: SAT3ComposedEnergy,
    init: Tensor,
    clause_var_ids: Tensor,
    clause_signs: Tensor,
    clause_mask: Tensor,
    config: SolverConfig,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Minimise the composed energy from ``init`` with projected
    gradient descent.

    Runs :func:`ccem.solvers.run_projected_solver` on
    :meth:`SAT3ComposedEnergy.energy_fn` with the Euclidean projection
    of every variable row onto the simplex; the result is detached
    unless ``create_graph`` (Phase-2 unrolling).
    """
    energy_fn = composer.energy_fn(clause_var_ids, clause_signs, clause_mask)
    with torch.enable_grad():
        solved, _ = run_projected_solver(
            energy_fn,
            init,
            project_simplex,
            config,
            create_graph=create_graph,
        )
    return solved if create_graph else solved.detach()


def _expand_starts(tensor: Tensor, starts: int) -> Tensor:
    """Repeat a batch ``starts`` times in start-major layout
    ``[S * B, ...]``.
    """
    if starts == 1:
        return tensor
    expanded = tensor.unsqueeze(0).expand(starts, *tensor.shape)
    return expanded.reshape(starts * tensor.shape[0], *tensor.shape[1:])


def _discrete_coordinate_descent(
    composer: SAT3ComposedEnergy,
    values: Tensor,
    clause_var_ids: Tensor,
    clause_signs: Tensor,
    clause_mask: Tensor,
    num_variables_per_formula: Tensor,
    steps: int,
    tolerance: float,
) -> Tensor:
    """Greedy single-flip descent on the learned energy of rounded
    assignments.

    Each step evaluates, for every variable, the energy change of
    flipping it (summed over the clauses it occurs in) and flips the
    best variable of each formula whose change is below ``-tolerance``;
    it stops when no formula improves. Only the learned energy is used,
    never the clause truth values.
    """
    values = values.clone()
    if steps <= 0:
        return values
    batch_size = int(values.shape[0])
    device = values.device
    model_dtype = next(composer.parameters()).dtype
    variable_ids = torch.arange(composer.num_variables, device=device)[None, :]
    valid_variables = variable_ids < num_variables_per_formula[:, None]
    active_mask = clause_mask.to(device=device, dtype=model_dtype)

    with torch.no_grad():
        for _ in range(int(steps)):
            local_assignments = gather_clause_values(
                assignment_onehot(values, dtype=model_dtype), clause_var_ids
            )
            base_factor_energy = composer.factor_model(
                clause_signs=clause_signs, y_clause=local_assignments
            )
            energy_delta = torch.zeros(
                batch_size,
                composer.num_variables,
                device=device,
                dtype=base_factor_energy.dtype,
            )
            for literal_position in range(LITERALS_PER_CLAUSE):
                candidate_local = local_assignments.clone()
                candidate_local[..., literal_position, :] = (
                    1.0 - candidate_local[..., literal_position, :]
                )
                candidate_factor_energy = composer.factor_model(
                    clause_signs=clause_signs, y_clause=candidate_local
                )
                contribution = (
                    candidate_factor_energy - base_factor_energy
                ) * active_mask
                energy_delta.scatter_add_(
                    dim=1,
                    index=clause_var_ids[..., literal_position].long(),
                    src=contribution,
                )
            energy_delta = energy_delta.masked_fill(
                ~valid_variables, torch.inf
            )
            best_delta, best_variable = energy_delta.min(dim=1)
            improve = best_delta < -float(tolerance)
            if not bool(improve.any()):
                break
            rows = torch.nonzero(improve, as_tuple=False).squeeze(-1)
            columns = best_variable[rows]
            values[rows, columns] = 1 - values[rows, columns]
    return values


def decode_sat3_assignments(
    composer: SAT3ComposedEnergy,
    clause_var_ids: Tensor,
    clause_signs: Tensor,
    clause_mask: Tensor,
    solver: SAT3SolverConfig,
    num_variables: Tensor,
) -> Tensor:
    """Decode ``{0, 1}`` assignments ``[B, N]`` for a batch of formulas.

    Four stages: (1) draw ``S`` random starts per formula
    (:func:`random_sat3_assignments`, one RNG draw for the whole
    expanded batch) and run the continuous solve; (2) round every start
    by ``argmax``; (3) refine each rounded start by coordinate descent
    on the learned energy (``solver.discrete_steps`` flips at most); (4)
    keep, per formula, the start whose one-hot assignment has the lowest
    composed energy.

    Args:
        composer: Composed energy with ``num_variables == N``.
        clause_var_ids: ``[B, M, 3]``.
        clause_signs: ``[B, M, 3]``.
        clause_mask: ``[B, M]``.
        solver: Continuous and discrete settings.
        num_variables: ``[B]`` active variable counts (variables beyond
            them are never flipped).
    """
    starts = solver.num_starts
    batch_size = int(clause_var_ids.shape[0])
    if num_variables.shape != (batch_size,):
        raise ValueError('num_variables must have shape [B]')
    num_variables = num_variables.to(
        device=clause_var_ids.device, dtype=torch.long
    )

    expanded_var_ids = _expand_starts(clause_var_ids, starts)
    expanded_signs = _expand_starts(clause_signs, starts)
    expanded_mask = _expand_starts(clause_mask, starts)
    init = random_sat3_assignments(
        starts * batch_size, composer.num_variables, clause_var_ids.device
    )
    continuous = solve_sat3_assignments(
        composer,
        init,
        expanded_var_ids,
        expanded_signs,
        expanded_mask,
        solver.continuous,
    )
    values = _discrete_coordinate_descent(
        composer,
        continuous.argmax(dim=-1).long(),
        expanded_var_ids,
        expanded_signs,
        expanded_mask,
        _expand_starts(num_variables, starts),
        solver.discrete_steps,
        solver.discrete_tolerance,
    )
    if starts == 1:
        return values

    with torch.no_grad():
        energies = composer.total_energy(
            assignment_onehot(values, dtype=next(composer.parameters()).dtype),
            expanded_var_ids,
            expanded_signs,
            expanded_mask,
        )
    values_by_start = values.reshape(
        starts, batch_size, composer.num_variables
    )
    best_start = energies.reshape(starts, batch_size).argmin(dim=0)
    return values_by_start[
        best_start, torch.arange(batch_size, device=values.device)
    ]
