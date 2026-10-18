"""Two-phase training of the 3-SAT clause factor.

Run from the repository root::

    python scripts/train.py sat3 --config configs/sat3/picnn.yaml

Phase 1 (*local*) fits
:class:`~ccem.tasks.sat3.energy.SAT3ClauseEnergyModel` on clause-level
pairs: the witness restricted to a clause (energy regressed to 0)
against the unique falsifying local assignment (energy regressed to 1),
plus a hinge keeping the two energies ``margin`` apart. The checkpoint
with the lowest validation loss is kept.

Phase 2 (*formula fine-tuning*) composes the factor over whole formulas,
solves every training formula from a random start with projected
gradient descent and back-propagates through the last
``formula_backprop_steps`` iterations (truncated unrolling, no witness
needed). The objective on the relaxed solution ``p`` is, with
``q_c = prod_{l in c} (1 - p_l)`` the probability that clause ``c`` is
false and ``m_c = max_{l in c} p_l`` its best literal,

    soft_clause_weight * mean_c q_c
    + clause_margin_weight * mean_c relu(clause_confidence_margin - m_c)
    + worst_clause_weight * max_c q_c,

averaged over the batch. Validation decodes the validation formulas
exactly (single start, fixed seed) and the epoch with the best
``(sat_formula_fraction, -avg_unsat_clauses)`` is saved.
"""

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as f
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ccem.solvers import SolverConfig
from ccem.tasks.sat3.dataset import (
    SAT3ClauseDataset,
    SAT3ClauseSample,
    SAT3FormulaDataset,
    SAT3FormulaSample,
    collate_clause_batch,
    collate_formula_batch,
)
from ccem.tasks.sat3.energy import (
    ENERGY_SEMANTICS,
    SAT3ClauseEnergyModel,
    SAT3ComposedEnergy,
    SAT3SolverConfig,
    literal_true_probabilities,
    random_sat3_assignments,
    solve_sat3_assignments,
)
from ccem.tasks.sat3.evaluate import evaluate_loader
from ccem.tasks.sat3.metrics import SAT3EvalResult
from ccem.utils import (
    apply_overrides,
    as_device,
    load_checkpoint,
    load_yaml,
    parse_set_overrides,
    set_seed,
)

MODEL_KEYS = ('hidden_dim', 'depth', 'activation')


@dataclass(frozen=True)
class LocalPhaseConfig:
    """Phase-1 settings (YAML ``training:`` keys of the same name).

    Attributes:
        epochs: Number of passes over the clause pairs.
        batch_size: Clause pairs per step.
        lr, weight_decay: Adam settings.
        positive_weight: Weight of ``E(pos)^2``.
        negative_weight: Weight of ``(E(neg) - 1)^2``.
        margin_weight: Weight of ``relu(margin - (E(neg) - E(pos)))``.
        margin: Required energy gap between the two local assignments.
        log_every: Print the epoch summary every this many epochs.
    """

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    positive_weight: float
    negative_weight: float
    margin_weight: float
    margin: float
    log_every: int

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> 'LocalPhaseConfig':
        """Read the ``training:`` block (defaults for missing keys)."""
        return cls(
            epochs=max(1, int(cfg.get('epochs', 50))),
            batch_size=max(1, int(cfg.get('batch_size', 2048))),
            lr=float(cfg.get('lr', 1.0e-3)),
            weight_decay=float(cfg.get('weight_decay', 0.0)),
            positive_weight=float(cfg.get('positive_weight', 1.0)),
            negative_weight=float(cfg.get('negative_weight', 1.0)),
            margin_weight=float(cfg.get('margin_weight', 0.25)),
            margin=float(cfg.get('margin', 1.0)),
            log_every=max(1, int(cfg.get('log_every', 5))),
        )


@dataclass(frozen=True)
class FormulaPhaseConfig:
    """Phase-2 settings (YAML ``training.formula_*`` keys and
    ``solver:``).

    Attributes:
        epochs: Fine-tuning epochs; 0 disables Phase 2.
        batch_size: Formulas per training step.
        valid_batch_size: Formulas per validation step.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        grad_clip: Gradient-norm clip (``error_if_nonfinite``);
            ``grad_clip <= 0`` disables clipping.
        backprop_steps: Unrolled solver iterations at the end of the
            horizon.
        soft_clause_weight: Weight of the mean false probability of the
            clauses (module docstring).
        clause_margin_weight: Weight of the confidence hinge.
        clause_confidence_margin: Target of the confidence hinge, in
            ``[0.5, 1]``.
        worst_clause_weight: Weight of the largest false probability.
        early_stopping_patience: Stop after this many epochs without a
            better validation key; 0 disables early stopping.
        validation_seed: Seed of the validation solver starts.
        solver: The ``solver:`` block with ``num_starts`` forced to 1;
            the unrolled solve uses its ``steps`` and ``step_size`` as
            constant-step PGD, the validation decode uses all of it,
            including the discrete refinement.
    """

    epochs: int
    batch_size: int
    valid_batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    backprop_steps: int
    soft_clause_weight: float
    clause_margin_weight: float
    clause_confidence_margin: float
    worst_clause_weight: float
    early_stopping_patience: int
    validation_seed: int
    solver: SAT3SolverConfig

    def __post_init__(self) -> None:
        if self.soft_clause_weight < 0.0:
            raise ValueError('Soft-clause weight must be non-negative')
        if self.clause_margin_weight < 0.0:
            raise ValueError('Clause-margin weight must be non-negative')
        if not 0.5 <= self.clause_confidence_margin <= 1.0:
            raise ValueError('Clause confidence margin must be in [0.5, 1]')
        if self.worst_clause_weight < 0.0:
            raise ValueError('Worst-clause weight must be non-negative')
        if self.epochs <= 0:
            return
        if self.solver.continuous.steps <= 0:
            raise ValueError(
                'Formula fine-tuning requires a positive solver horizon'
            )
        if not 1 <= self.backprop_steps <= self.solver.continuous.steps:
            raise ValueError(
                'formula_backprop_steps must be in [1, solver.steps]'
            )
        if self.solver.continuous.step_size <= 0.0:
            raise ValueError('Formula solver step size must be positive')

    @classmethod
    def from_mapping(
        cls,
        training_cfg: Mapping[str, Any],
        solver_cfg: Mapping[str, Any],
        seed: int,
    ) -> 'FormulaPhaseConfig':
        """Read ``training:`` and ``solver:``; training uses one start.

        The ``solver:`` block is the inference setting; its
        ``num_starts`` is replaced by 1 for the unrolled solve and the
        validation decode, so the same file trains and evaluates.
        """
        batch_size = max(1, int(training_cfg.get('formula_batch_size', 32)))
        solver = SAT3SolverConfig.from_mapping(solver_cfg)
        continuous = replace(solver.continuous, num_starts=1)
        return cls(
            epochs=max(0, int(training_cfg.get('formula_finetune_epochs', 0))),
            batch_size=batch_size,
            valid_batch_size=max(
                1,
                int(training_cfg.get('formula_valid_batch_size', batch_size)),
            ),
            lr=float(training_cfg.get('formula_lr', 1.0e-4)),
            weight_decay=float(training_cfg.get('formula_weight_decay', 0.0)),
            grad_clip=float(training_cfg.get('formula_grad_clip', 1.0)),
            backprop_steps=max(
                1, int(training_cfg.get('formula_backprop_steps', 32))
            ),
            soft_clause_weight=float(
                training_cfg.get('formula_soft_clause_weight', 1.0)
            ),
            clause_margin_weight=float(
                training_cfg.get('formula_clause_margin_weight', 0.25)
            ),
            clause_confidence_margin=float(
                training_cfg.get('formula_clause_confidence_margin', 0.75)
            ),
            worst_clause_weight=float(
                training_cfg.get('formula_worst_clause_weight', 0.25)
            ),
            early_stopping_patience=max(
                0, int(training_cfg.get('formula_early_stopping_patience', 5))
            ),
            validation_seed=int(
                training_cfg.get('formula_validation_seed', seed + 10_000)
            ),
            solver=replace(solver, continuous=continuous),
        )


@dataclass(frozen=True)
class SAT3TrainConfig:
    """The resolved training configuration of one run.

    Attributes:
        seed: Global seed (model initialisation, shuffling, solver
            starts).
        device: Device name passed to :func:`ccem.utils.as_device`.
        train_path, valid_path: ``.npz`` splits with planted witnesses.
        output_path: Checkpoint file (overwritten whenever the best
            improves).
        model: The ``model:`` block (``hidden_dim``, ``depth``,
            ``activation``), stored verbatim as the payload
            ``model_config``.
        local: Phase-1 settings.
        formula: Phase-2 settings.
    """

    seed: int
    device: str
    train_path: str
    valid_path: str
    output_path: Path
    model: dict[str, Any]
    local: LocalPhaseConfig
    formula: FormulaPhaseConfig

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> 'SAT3TrainConfig':
        """Resolve a loaded YAML mapping (after CLI overrides)."""
        for key in ('train_path', 'valid_path'):
            if not cfg.get(key):
                raise ValueError(
                    f'{key} must be set in the config or with --set'
                )
        seed = int(cfg.get('seed', 0))
        model_cfg: Mapping[str, Any] = cfg.get('model', {})
        training_cfg: Mapping[str, Any] = cfg.get('training', {})
        return cls(
            seed=seed,
            device=str(cfg.get('device', 'cpu')),
            train_path=str(cfg['train_path']),
            valid_path=str(cfg['valid_path']),
            output_path=Path(
                cfg.get('output_path', 'checkpoints/sat3/picnn_retrained.pt')
            ),
            model={
                key: model_cfg[key] for key in MODEL_KEYS if key in model_cfg
            },
            local=LocalPhaseConfig.from_mapping(training_cfg),
            formula=FormulaPhaseConfig.from_mapping(
                training_cfg, cfg.get('solver', {}), seed
            ),
        )


@dataclass(frozen=True)
class LocalPairTerms:
    """Phase-1 loss of one batch and the two ``[B]`` energy vectors it
    was built from.
    """

    loss: Tensor
    positive_energy: Tensor
    negative_energy: Tensor


def local_pair_objective(
    model: SAT3ClauseEnergyModel,
    clause_signs: Tensor,
    positive: Tensor,
    negative: Tensor,
    cfg: LocalPhaseConfig,
) -> LocalPairTerms:
    """Regress satisfying assignments to energy 0, falsifying ones to 1.

    The loss is::

        positive_weight * mean E(pos)^2
        + negative_weight * mean (E(neg) - 1)^2
        + margin_weight * mean relu(margin - (E(neg) - E(pos)))

    for one-hot local assignments ``positive``/``negative`` of shape
    ``[B, 3, 2]`` and polarities ``clause_signs`` ``[B, 3]``.
    """
    positive_energy = model(clause_signs=clause_signs, y_clause=positive)
    negative_energy = model(clause_signs=clause_signs, y_clause=negative)
    positive_mse = positive_energy.square().mean()
    negative_mse = (negative_energy - 1.0).square().mean()
    margin_loss = f.relu(
        cfg.margin - (negative_energy - positive_energy)
    ).mean()
    loss = (
        cfg.positive_weight * positive_mse
        + cfg.negative_weight * negative_mse
        + cfg.margin_weight * margin_loss
    )
    return LocalPairTerms(
        loss=loss,
        positive_energy=positive_energy,
        negative_energy=negative_energy,
    )


def _scalar(value: Tensor) -> float:
    return float(value.detach().cpu())


def run_local_epoch(
    loader: DataLoader[SAT3ClauseSample],
    model: SAT3ClauseEnergyModel,
    device: torch.device,
    cfg: LocalPhaseConfig,
    *,
    description: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, float]:
    """One Phase-1 pass; trains when ``optimizer`` is given, evaluates
    otherwise.

    Returns:
        Example-weighted means of the loss and the two energies plus
        ``separation = negative_energy - positive_energy``.
    """
    training = optimizer is not None
    model.train(training)
    totals: defaultdict[str, float] = defaultdict(float)
    num_examples = 0
    progress = tqdm(
        loader, desc=description, unit='batch', dynamic_ncols=True, leave=False
    )
    for batch in progress:
        clause_signs = batch['clause_signs'].to(device)
        positive = batch['positive_assignment'].to(device)
        negative = batch['negative_assignment'].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            terms = local_pair_objective(
                model, clause_signs, positive, negative, cfg
            )
            if optimizer is not None:
                # Tensor.backward is untyped in the torch stubs.
                terms.loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

        batch_size = int(clause_signs.shape[0])
        num_examples += batch_size
        batch_means = {
            'loss': _scalar(terms.loss),
            'positive_energy': _scalar(terms.positive_energy.mean()),
            'negative_energy': _scalar(terms.negative_energy.mean()),
        }
        for key, value in batch_means.items():
            totals[key] += value * batch_size
        separation = (
            batch_means['negative_energy'] - batch_means['positive_energy']
        )
        progress.set_postfix(
            loss=f'{batch_means["loss"]:.4f}',
            sep=f'{separation:.4f}',
            refresh=False,
        )

    metrics = {
        key: value / float(num_examples) for key, value in totals.items()
    }
    metrics['separation'] = (
        metrics['negative_energy'] - metrics['positive_energy']
    )
    return metrics


def unroll_formula_solver(
    composer: SAT3ComposedEnergy,
    clause_var_ids: Tensor,
    clause_signs: Tensor,
    clause_mask: Tensor,
    cfg: FormulaPhaseConfig,
) -> Tensor:
    """Solve from a random start; keep the graph of the last
    ``backprop_steps``.

    The horizon ``solver.steps`` is split into a detached warm-up and an
    unrolled tail, both constant-step PGD with ``solver.step_size``.

    Returns:
        The relaxed solution ``[B, N, 2]`` attached to the model
        parameters.
    """
    init = random_sat3_assignments(
        int(clause_var_ids.shape[0]),
        composer.num_variables,
        clause_var_ids.device,
    )
    horizon = cfg.solver.continuous
    warmup_steps = horizon.steps - cfg.backprop_steps
    current = init
    if warmup_steps > 0:
        current = solve_sat3_assignments(
            composer,
            current,
            clause_var_ids,
            clause_signs,
            clause_mask,
            SolverConfig(
                steps=warmup_steps,
                step_size=horizon.step_size,
                optimizer='pgd',
            ),
        )
    return solve_sat3_assignments(
        composer,
        current.detach(),
        clause_var_ids,
        clause_signs,
        clause_mask,
        SolverConfig(
            steps=cfg.backprop_steps,
            step_size=horizon.step_size,
            optimizer='pgd',
        ),
        create_graph=True,
    )


@dataclass(frozen=True)
class FormulaLossTerms:
    """Phase-2 scalar loss terms of one batch."""

    total: Tensor
    soft_clause: Tensor
    clause_margin: Tensor
    worst_clause: Tensor


def formula_objective(
    assign_probs: Tensor,
    clause_var_ids: Tensor,
    clause_signs: Tensor,
    clause_mask: Tensor,
    cfg: FormulaPhaseConfig,
) -> FormulaLossTerms:
    """The differentiable clause objective of the module docstring.

    Args:
        assign_probs: Relaxed solutions ``[B, N, 2]``.
        clause_var_ids, clause_signs: ``[B, M, 3]`` clause structure.
        clause_mask: ``[B, M]`` active-clause mask; means run over
            active clauses only (``clamp(min=1)`` guards empty
            formulas).
        cfg: Objective weights.
    """
    literal_true = literal_true_probabilities(
        assign_probs, clause_var_ids, clause_signs
    )
    active_clauses = clause_mask > 0.5
    active_clause_float = active_clauses.to(dtype=literal_true.dtype)
    false_probability = (1.0 - literal_true).prod(dim=-1) * active_clause_float
    clause_count = (
        active_clauses.sum(dim=1).clamp(min=1).to(dtype=literal_true.dtype)
    )

    soft_clause = (false_probability.sum(dim=1) / clause_count).mean()
    clause_margin_violation = (
        f.relu(cfg.clause_confidence_margin - literal_true.max(dim=-1).values)
        * active_clause_float
    )
    clause_margin = (clause_margin_violation.sum(dim=1) / clause_count).mean()
    worst_clause = (
        false_probability.masked_fill(~active_clauses, float('-inf'))
        .max(dim=1)
        .values
    ).mean()
    total = (
        cfg.soft_clause_weight * soft_clause
        + cfg.clause_margin_weight * clause_margin
        + cfg.worst_clause_weight * worst_clause
    )
    return FormulaLossTerms(
        total=total,
        soft_clause=soft_clause,
        clause_margin=clause_margin,
        worst_clause=worst_clause,
    )


def run_formula_epoch(
    loader: DataLoader[SAT3FormulaSample],
    composer: SAT3ComposedEnergy,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    cfg: FormulaPhaseConfig,
    *,
    description: str,
) -> dict[str, float]:
    """One Phase-2 pass: unroll, objective, clipped Adam step per batch.

    Returns:
        Formula-weighted means ``loss``, ``soft_clause_loss``,
        ``clause_margin_loss`` and ``worst_clause_loss``.
    """
    composer.train(True)
    totals: defaultdict[str, float] = defaultdict(float)
    num_formulas = 0
    progress = tqdm(
        loader, desc=description, unit='batch', dynamic_ncols=True, leave=False
    )
    for batch in progress:
        clause_var_ids = batch['clause_var_ids'].to(device)
        clause_signs = batch['clause_signs'].to(device)
        clause_mask = batch['clause_mask'].to(device)

        optimizer.zero_grad(set_to_none=True)
        solved = unroll_formula_solver(
            composer, clause_var_ids, clause_signs, clause_mask, cfg
        )
        terms = formula_objective(
            solved, clause_var_ids, clause_signs, clause_mask, cfg
        )
        if not torch.isfinite(terms.total):
            raise RuntimeError('Non-finite 3-SAT formula fine-tuning loss')
        terms.total.backward()  # type: ignore[no-untyped-call]
        if cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                composer.parameters(),
                max_norm=cfg.grad_clip,
                error_if_nonfinite=True,
            )
        optimizer.step()

        batch_size = int(clause_var_ids.shape[0])
        num_formulas += batch_size
        batch_means = {
            'loss': _scalar(terms.total),
            'soft_clause_loss': _scalar(terms.soft_clause),
            'clause_margin_loss': _scalar(terms.clause_margin),
            'worst_clause_loss': _scalar(terms.worst_clause),
        }
        for key, value in batch_means.items():
            totals[key] += value * batch_size
        progress.set_postfix(
            loss=f'{batch_means["loss"]:.4f}',
            soft=f'{batch_means["soft_clause_loss"]:.3f}',
            worst=f'{batch_means["worst_clause_loss"]:.3f}',
            refresh=False,
        )
    return {key: value / float(num_formulas) for key, value in totals.items()}


def validate_formulas(
    loader: DataLoader[SAT3FormulaSample],
    composer: SAT3ComposedEnergy,
    device: torch.device,
    cfg: FormulaPhaseConfig,
    *,
    description: str,
) -> SAT3EvalResult:
    """Exact single-start decode of the validation formulas under a
    fixed seed.

    The torch RNG is forked and seeded with ``cfg.validation_seed`` so
    the validation starts are identical in every epoch and the training
    stream is left untouched.
    """
    composer.eval()
    fork_devices: list[int] = []
    if device.type == 'cuda':
        fork_devices.append(
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(cfg.validation_seed)
        return evaluate_loader(
            loader, composer, cfg.solver, device, description=description
        )


def validation_key(result: SAT3EvalResult) -> tuple[float, float]:
    """Model-selection key: more satisfied formulas, then fewer
    violated clauses.
    """
    return (result.sat_formula_fraction, -result.avg_unsat_clauses)


def checkpoint_payload(
    model: SAT3ClauseEnergyModel,
    cfg: SAT3TrainConfig,
    *,
    best_validation_loss: float,
    epoch: int,
    stage: str = 'local',
    formula_epoch: int = 0,
    formula_validation: Optional[SAT3EvalResult] = None,
) -> dict[str, Any]:
    """The checkpoint dictionary (plain containers only,
    ``weights_only`` safe).

    ``task``, ``energy_semantics``, ``model_state`` and ``model_config``
    are what :func:`ccem.tasks.sat3.energy.load_clause_model` reads; the
    rest records provenance (``stage`` is ``local``,
    ``formula_baseline`` or ``formula_finetune``).
    """
    formula = cfg.formula
    return {
        'format_version': 2,
        'task': 'sat3',
        'energy_semantics': ENERGY_SEMANTICS,
        'model_state': model.state_dict(),
        'model_config': dict(cfg.model),
        'training_protocol': {
            'train_path': cfg.train_path,
            'valid_path': cfg.valid_path,
            'local_epochs': cfg.local.epochs,
            'formula_finetune_epochs': formula.epochs,
            'formula_solver_steps': formula.solver.continuous.steps,
            'formula_backprop_steps': formula.backprop_steps,
            'formula_step_size': formula.solver.continuous.step_size,
            'formula_solver_num_starts': formula.solver.num_starts,
            'formula_discrete_steps': formula.solver.discrete_steps,
            'formula_lr': formula.lr,
            'formula_weight_decay': formula.weight_decay,
            'formula_grad_clip': formula.grad_clip,
            'formula_soft_clause_weight': formula.soft_clause_weight,
            'formula_clause_margin_weight': formula.clause_margin_weight,
            'formula_clause_confidence_margin': (
                formula.clause_confidence_margin
            ),
            'formula_worst_clause_weight': formula.worst_clause_weight,
            'formula_early_stopping_patience': formula.early_stopping_patience,
            'formula_validation_seed': formula.validation_seed,
        },
        'best_validation_loss': float(best_validation_loss),
        'epoch': int(epoch),
        'stage': str(stage),
        'formula_epoch': int(formula_epoch),
        'formula_validation': (
            None
            if formula_validation is None
            else formula_validation.as_dict()
        ),
    }


def train_local_phase(
    model: SAT3ClauseEnergyModel,
    cfg: SAT3TrainConfig,
    device: torch.device,
    train_dataset: SAT3ClauseDataset,
    valid_dataset: SAT3ClauseDataset,
) -> tuple[float, int]:
    """Phase 1: train on clause pairs, saving the lowest-validation-loss
    model.

    Returns:
        The best validation loss and the epoch that reached it.
    """
    local = cfg.local
    train_loader = DataLoader(
        train_dataset,
        batch_size=local.batch_size,
        shuffle=True,
        collate_fn=collate_clause_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=local.batch_size,
        shuffle=False,
        collate_fn=collate_clause_batch,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=local.lr, weight_decay=local.weight_decay
    )
    best_validation_loss = float('inf')
    best_epoch = 0
    for epoch in range(1, local.epochs + 1):
        train_metrics = run_local_epoch(
            train_loader,
            model,
            device,
            local,
            description=f'Local train {epoch}/{local.epochs}',
            optimizer=optimizer,
        )
        valid_metrics = run_local_epoch(
            valid_loader,
            model,
            device,
            local,
            description=f'Local valid {epoch}/{local.epochs}',
        )
        if valid_metrics['loss'] < best_validation_loss:
            best_validation_loss = valid_metrics['loss']
            best_epoch = epoch
            torch.save(
                checkpoint_payload(
                    model,
                    cfg,
                    best_validation_loss=best_validation_loss,
                    epoch=epoch,
                ),
                cfg.output_path,
            )
        if epoch == 1 or epoch % local.log_every == 0 or epoch == local.epochs:
            tqdm.write(
                f'Local epoch {epoch:4d}/{local.epochs}: '
                f'train={train_metrics["loss"]:.6f}, '
                f'valid={valid_metrics["loss"]:.6f}, '
                f'train_sep={train_metrics["separation"]:.6f}, '
                f'valid_sep={valid_metrics["separation"]:.6f}'
            )
    return best_validation_loss, best_epoch


def _reload_saved_model(
    model: SAT3ClauseEnergyModel, path: Path, device: torch.device
) -> None:
    model.load_state_dict(load_checkpoint(path, device)['model_state'])


def _format_validation(result: SAT3EvalResult) -> str:
    return (
        f'valid_sat={result.sat_formula_fraction:.6f}, '
        f'valid_clause={result.clause_satisfaction_rate:.6f}, '
        f'valid_unsat={result.avg_unsat_clauses:.6f}'
    )


def train_formula_phase(
    model: SAT3ClauseEnergyModel,
    cfg: SAT3TrainConfig,
    device: torch.device,
    train_formulas: SAT3FormulaDataset,
    valid_formulas: SAT3FormulaDataset,
    *,
    best_validation_loss: float,
    best_local_epoch: int,
) -> None:
    """Phase 2: fine-tune through the unrolled solver with exact
    validation.

    Starts from the best Phase-1 weights saved at ``cfg.output_path``,
    records their validation as the ``formula_baseline`` stage and then
    saves every epoch that strictly improves :func:`validation_key`; the
    model ends up holding the saved weights.
    """
    formula = cfg.formula
    _reload_saved_model(model, cfg.output_path, device)
    tqdm.write(f'Loaded best local checkpoint from: {cfg.output_path}')
    tqdm.write(
        f'Formula checkpoint validation: all {len(valid_formulas)} formulas'
    )

    train_loader = DataLoader(
        train_formulas,
        batch_size=formula.batch_size,
        shuffle=True,
        collate_fn=collate_formula_batch,
    )
    valid_loader = DataLoader(
        valid_formulas,
        batch_size=formula.valid_batch_size,
        shuffle=False,
        collate_fn=collate_formula_batch,
    )
    train_composer = SAT3ComposedEnergy(
        model, train_formulas.num_variables
    ).to(device)
    valid_composer = SAT3ComposedEnergy(
        model, valid_formulas.num_variables
    ).to(device)

    best_validation = validate_formulas(
        valid_loader,
        valid_composer,
        device,
        formula,
        description='Formula baseline valid',
    )
    best_key = validation_key(best_validation)
    torch.save(
        checkpoint_payload(
            model,
            cfg,
            best_validation_loss=best_validation_loss,
            epoch=best_local_epoch,
            stage='formula_baseline',
            formula_validation=best_validation,
        ),
        cfg.output_path,
    )
    tqdm.write(f'Formula baseline: {_format_validation(best_validation)}')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=formula.lr, weight_decay=formula.weight_decay
    )
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, formula.epochs + 1):
        train_metrics = run_formula_epoch(
            train_loader,
            train_composer,
            device,
            optimizer,
            formula,
            description=f'Formula train {epoch}/{formula.epochs}',
        )
        validation = validate_formulas(
            valid_loader,
            valid_composer,
            device,
            formula,
            description=f'Formula valid {epoch}/{formula.epochs}',
        )
        key = validation_key(validation)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_validation = validation
            stale_epochs = 0
            torch.save(
                checkpoint_payload(
                    model,
                    cfg,
                    best_validation_loss=best_validation_loss,
                    epoch=best_local_epoch,
                    stage='formula_finetune',
                    formula_epoch=epoch,
                    formula_validation=validation,
                ),
                cfg.output_path,
            )
        else:
            stale_epochs += 1
        tqdm.write(
            f'Formula epoch {epoch:4d}/{formula.epochs}: '
            f'train={train_metrics["loss"]:.6f}, '
            f'soft_loss={train_metrics["soft_clause_loss"]:.6f}, '
            f'clause_margin={train_metrics["clause_margin_loss"]:.6f}, '
            f'worst_clause={train_metrics["worst_clause_loss"]:.6f}, '
            f'{_format_validation(validation)}'
        )
        if (
            formula.early_stopping_patience
            and stale_epochs >= formula.early_stopping_patience
        ):
            tqdm.write(
                f'Formula early stopping: {stale_epochs} epochs '
                'without validation improvement'
            )
            break

    _reload_saved_model(model, cfg.output_path, device)
    tqdm.write(
        f'Best formula checkpoint: epoch={best_epoch}, '
        f'{_format_validation(best_validation)}'
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI flags; every explicit flag overrides the config, ``--set``
    last.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Train the CCEM 3-SAT clause factor (local pairs, then formulas).'
        )
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/sat3/picnn.yaml',
        help='YAML config (default: %(default)s)',
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
        '--epochs', type=int, default=None, help='Phase-1 epochs'
    )
    parser.add_argument(
        '--formula-finetune-epochs',
        type=int,
        default=None,
        help='Phase-2 epochs (0 disables Phase 2)',
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
    """Entry point of ``python scripts/train.py sat3``."""
    args = parse_args(argv)
    raw_cfg = load_yaml(args.config)
    apply_overrides(
        raw_cfg,
        {
            'device': args.device,
            'seed': args.seed,
            'output_path': args.output,
            'training.epochs': args.epochs,
            'training.formula_finetune_epochs': args.formula_finetune_epochs,
        },
    )
    apply_overrides(raw_cfg, parse_set_overrides(args.set))
    cfg = SAT3TrainConfig.from_mapping(raw_cfg)

    set_seed(cfg.seed)
    device = as_device(cfg.device)
    train_dataset = SAT3ClauseDataset(cfg.train_path)
    valid_dataset = SAT3ClauseDataset(cfg.valid_path)
    model = SAT3ClauseEnergyModel.from_config(cfg.model).to(device)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    tqdm.write(f'Train formulas: {train_dataset.num_formulas}')
    tqdm.write(f'Train clause factors: {len(train_dataset)}')
    tqdm.write(f'Validation formulas: {valid_dataset.num_formulas}')
    tqdm.write(f'Validation clause factors: {len(valid_dataset)}')

    best_validation_loss, best_local_epoch = train_local_phase(
        model, cfg, device, train_dataset, valid_dataset
    )
    if cfg.formula.epochs > 0:
        train_formula_phase(
            model,
            cfg,
            device,
            train_dataset.formulas,
            valid_dataset.formulas,
            best_validation_loss=best_validation_loss,
            best_local_epoch=best_local_epoch,
        )
    tqdm.write(f'Saved best checkpoint to: {cfg.output_path}')


if __name__ == '__main__':
    main()
