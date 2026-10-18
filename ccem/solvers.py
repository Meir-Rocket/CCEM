"""Projected first-order solvers for the CCEM relaxations.

Inference minimises a composed convex energy ``E(y)`` over a product of
simplices (or the Birkhoff polytope) with projected gradient descent or
projected Adam; the same loops run with ``create_graph=True`` to unroll
the solver through the energy during end-to-end (Phase 2) training. The
solvers draw no random numbers: starts and tie-break matrices are
provided by the caller.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

EnergyFn = Callable[[Tensor], Tensor]
ProjectionFn = Callable[[Tensor], Tensor]

SCHEDULES = ('constant', 'linear', 'cosine')
OPTIMIZERS = ('pgd', 'adam')

_SOLVER_YAML_KEYS = {
    'steps': 'steps',
    'step_size': 'step_size',
    'step_size_final': 'step_size_final',
    'schedule': 'step_schedule',
    'optimizer': 'optimizer',
    'num_starts': 'num_starts',
    'adam_beta1': 'adam_beta1',
    'adam_beta2': 'adam_beta2',
    'adam_eps': 'adam_eps',
    'tie_break_weight': 'tie_break_weight',
}


def build_step_size_schedule(
    steps: int,
    start: float,
    end: float,
    schedule: str = 'constant',
) -> list[float]:
    """Per-step step sizes for ``steps`` solver iterations.

    With ``S = steps``, ``a = start``, ``b = end`` and
    ``i = 0 .. S-1``:

    * ``constant``: ``a`` at every step (also whenever ``a == b``);
    * ``linear``: ``a + (b - a) i / (S - 1)``;
    * ``cosine``: ``b + (a - b) (1 + cos(pi i / (S - 1))) / 2``.

    ``S = 1`` yields ``[b]`` for the two annealed modes and ``S <= 0``
    yields ``[]``. ``schedule`` is one of :data:`SCHEDULES`;
    :class:`SolverConfig` validates it.
    """
    if steps <= 0:
        return []
    if schedule == 'constant' or abs(start - end) <= 1.0e-12:
        return [start] * steps
    if steps == 1:
        return [end]
    if schedule == 'linear':
        return [
            start + (end - start) * float(idx) / float(steps - 1)
            for idx in range(steps)
        ]
    return [
        end
        + 0.5
        * (start - end)
        * (1.0 + math.cos(math.pi * float(idx) / float(steps - 1)))
        for idx in range(steps)
    ]


@dataclass(frozen=True)
class SolverConfig:
    """Settings of one projected solve, mirroring the YAML ``solver:``
    block.

    Attributes:
        steps: Number of solver iterations ``T``.
        step_size: Initial step size (the only one for a constant
            schedule).
        step_size_final: Final step size of the ``linear``/``cosine``
            schedules; ``None`` means equal to ``step_size``.
        schedule: ``'constant'``, ``'linear'`` or ``'cosine'``.
        optimizer: ``'adam'`` or ``'pgd'``; the name is stripped and
            lower-cased on construction.
        num_starts: Random restarts per instance (at least 1).
        adam_beta1: Adam first-moment decay.
        adam_beta2: Adam second-moment decay.
        adam_eps: Floor of the Adam denominator (clamp semantics).
        tie_break_weight: Weight of the random linear tie-break term
            drawn by the caller (see :func:`with_tie_break`); ``0``
            disables it.
    """

    steps: int
    step_size: float
    step_size_final: Optional[float] = None
    schedule: str = 'constant'
    optimizer: str = 'adam'
    num_starts: int = 1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    tie_break_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError('steps must be non-negative')
        if not math.isfinite(self.step_size) or self.step_size < 0.0:
            raise ValueError('step_size must be finite and non-negative')
        if self.step_size_final is not None and not math.isfinite(
            self.step_size_final
        ):
            raise ValueError('step_size_final must be finite')
        if self.num_starts < 1:
            raise ValueError('num_starts must be at least 1')
        schedule = str(self.schedule).strip().lower()
        if schedule not in SCHEDULES:
            raise ValueError(
                'schedule must be one of: constant, linear, cosine'
            )
        object.__setattr__(self, 'schedule', schedule)
        optimizer = str(self.optimizer).strip().lower()
        if optimizer not in OPTIMIZERS:
            raise ValueError(
                f'unsupported solver optimizer: {self.optimizer!r} '
                '(expected pgd or adam)'
            )
        object.__setattr__(self, 'optimizer', optimizer)

    @classmethod
    def from_mapping(
        cls, cfg: Mapping[str, Any], **defaults: Any
    ) -> 'SolverConfig':
        """Build from a YAML ``solver:`` block.

        Reads the keys ``steps``, ``step_size``, ``step_size_final``,
        ``step_schedule``, ``optimizer``, ``num_starts``,
        ``adam_beta1``, ``adam_beta2``, ``adam_eps`` and
        ``tie_break_weight``; other keys are ignored. Keyword
        ``defaults`` (field names) replace the dataclass defaults for
        keys missing from ``cfg`` so that a task can keep its own
        fallbacks, e.g. ``num_starts=8``.

        Raises:
            KeyError: If neither ``cfg`` nor ``defaults`` give ``steps``
                and ``step_size``.
        """
        raw: dict[str, Any] = dict(defaults)
        for name, key in _SOLVER_YAML_KEYS.items():
            if key in cfg:
                raw[name] = cfg[key]
        missing = [name for name in ('steps', 'step_size') if name not in raw]
        if missing:
            raise KeyError(f'solver config is missing {missing}')
        final = raw.get('step_size_final')
        return cls(
            steps=int(raw['steps']),
            step_size=float(raw['step_size']),
            step_size_final=None if final is None else float(final),
            schedule=str(raw.get('schedule', cls.schedule)),
            optimizer=str(raw.get('optimizer', cls.optimizer)),
            num_starts=int(raw.get('num_starts', cls.num_starts)),
            adam_beta1=float(raw.get('adam_beta1', cls.adam_beta1)),
            adam_beta2=float(raw.get('adam_beta2', cls.adam_beta2)),
            adam_eps=float(raw.get('adam_eps', cls.adam_eps)),
            tie_break_weight=float(
                raw.get('tie_break_weight', cls.tie_break_weight)
            ),
        )

    def step_sizes(self) -> list[float]:
        """The per-step schedule, see
        :func:`build_step_size_schedule`.
        """
        end = (
            self.step_size
            if self.step_size_final is None
            else self.step_size_final
        )
        return build_step_size_schedule(
            self.steps, self.step_size, end, self.schedule
        )


def with_tie_break(
    energy_fn: EnergyFn, tie_break: Optional[Tensor], weight: float
) -> EnergyFn:
    """Add the random linear tie-break ``weight * <y, tie_break>`` to an
    energy.

    The term breaks the symmetry between equivalent relaxed solutions so
    that different starts settle in different vertices. The caller draws
    ``tie_break`` (uniform in ``[-1, 1)``, shaped like ``y``) right
    after the corresponding start, which keeps the RNG consumption order
    of the paper runs; the inner product runs over all non-batch
    dimensions. With ``tie_break=None`` or ``weight <= 0`` the energy is
    returned unchanged.
    """
    if tie_break is None or weight <= 0.0:
        return energy_fn
    scale = float(weight)

    def tie_broken_energy(y: Tensor) -> Tensor:
        return energy_fn(y) + scale * (y * tie_break).sum(
            dim=tuple(range(1, y.ndim))
        )

    return tie_broken_energy


def _record_history(
    step_index: int, total_steps: int, history_last: int
) -> bool:
    """Whether the iterate produced by 0-based ``step_index`` is among
    the last ``history_last`` of ``total_steps`` iterates.
    """
    return history_last > 0 and step_index >= total_steps - history_last


def projected_gradient_descent(
    energy_fn: EnergyFn,
    init: Tensor,
    projection_fn: ProjectionFn,
    step_sizes: Sequence[float],
    *,
    create_graph: bool = False,
    history_last: int = 0,
) -> tuple[Tensor, list[Tensor]]:
    """Projected gradient descent
    ``y_{k+1} = P(y_k - eta_k grad E(y_k))``.

    One iteration is run per entry of ``step_sizes`` and ``eta_k`` is
    its ``k``-th entry. Gradients are sanitised with ``nan_to_num``
    (``+-inf`` to ``+-1e4``) and the proposal with ``+-1e6`` before
    projecting; both guards are part of the reference numerics.

    Args:
        energy_fn: Maps a batch of iterates ``[B, ...]`` to energies
            ``[B]``.
        init: Feasible starting iterates.
        projection_fn: Projection onto the feasible set.
        step_sizes: Per-step step sizes (see
            :meth:`SolverConfig.step_sizes`); their number is the
            number of iterations.
        create_graph: Keep the unrolled graph (Phase-2 training);
            otherwise every iterate is detached.
        history_last: Number of final iterates to record as detached
            copies; ``0`` records none.

    Returns:
        The final iterate and the recorded iterates (an empty list
        unless ``history_last > 0``).
    """
    y = init
    history: list[Tensor] = []
    total_steps = len(step_sizes)
    for step in range(total_steps):
        if not y.requires_grad:
            y = y.requires_grad_(True)
        energy = energy_fn(y)
        (grad,) = torch.autograd.grad(
            energy.sum(), y, create_graph=create_graph
        )
        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        next_y = y - float(step_sizes[step]) * grad
        next_y = torch.nan_to_num(next_y, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        y = projection_fn(next_y)
        if _record_history(step, total_steps, history_last):
            history.append(y.detach().clone())
        if not create_graph:
            y = y.detach()
    return y, history


def _adam_update(
    energy_fn: EnergyFn,
    state: Tensor,
    moment: Tensor,
    variance: Tensor,
    *,
    step: int,
    step_size: float,
    beta1: float,
    beta2: float,
    eps: float,
    differentiate: bool,
    projection_fn: ProjectionFn,
) -> tuple[Tensor, Tensor, Tensor]:
    """One bias-corrected Adam step (1-based ``step``) followed by
    projection.
    """
    if not state.requires_grad:
        state = state.requires_grad_(True)
    energy = energy_fn(state)
    gradient = torch.autograd.grad(
        energy.sum(), state, create_graph=differentiate
    )[0]
    gradient = torch.nan_to_num(gradient, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    moment = beta1 * moment + (1.0 - beta1) * gradient
    variance = beta2 * variance + (1.0 - beta2) * (gradient * gradient)
    moment = torch.nan_to_num(moment, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    variance = torch.nan_to_num(variance, nan=0.0, posinf=1.0e6, neginf=0.0)
    corrected_moment = moment / (1.0 - beta1**step)
    corrected_variance = variance / (1.0 - beta2**step)
    corrected_moment = torch.nan_to_num(
        corrected_moment, nan=0.0, posinf=1.0e6, neginf=-1.0e6
    )
    corrected_variance = torch.nan_to_num(
        corrected_variance, nan=0.0, posinf=1.0e6, neginf=0.0
    )
    if differentiate:
        # sqrt(0) has an infinite derivative; mask it so unrolled
        # training gets a zero gradient there (forward value unchanged).
        zero_variance = corrected_variance == 0
        root_input = torch.where(
            zero_variance,
            torch.ones_like(corrected_variance),
            corrected_variance,
        )
        denominator = torch.where(
            zero_variance,
            torch.zeros_like(corrected_variance),
            torch.sqrt(root_input),
        )
    else:
        denominator = torch.sqrt(corrected_variance)
    denominator = denominator.clamp(min=eps)
    denominator = torch.nan_to_num(
        denominator, nan=eps, posinf=1.0e6, neginf=eps
    )
    proposed = state - step_size * corrected_moment / denominator
    proposed = torch.nan_to_num(proposed, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    updated = projection_fn(proposed)
    return updated, moment, variance


def projected_adam(
    energy_fn: EnergyFn,
    init: Tensor,
    projection_fn: ProjectionFn,
    step_sizes: Sequence[float],
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
    create_graph: bool = False,
    history_last: int = 0,
) -> tuple[Tensor, list[Tensor]]:
    """Projected Adam with bias correction.

    For ``t = 1 .. len(step_sizes)`` with ``g`` the gradient of ``E``
    at ``y`` and ``eta_t`` the ``t``-th step size::

        m = b1 m + (1 - b1) g          v = b2 v + (1 - b2) g^2
        m_hat = m / (1 - b1^t)         v_hat = v / (1 - b2^t)
        D = max(sqrt(v_hat), eps)
        y = P(y - eta_t m_hat / D)

    Every intermediate quantity is sanitised with ``nan_to_num``,
    ``eps`` floors the denominator (clamp semantics) and, under
    ``create_graph``, the square root is taken through a mask so
    ``sqrt(0)`` has a zero rather than NaN gradient. All of this is
    reference numerics. The remaining arguments behave as in
    :func:`projected_gradient_descent`.

    Returns:
        The final iterate and the recorded iterates (an empty list
        unless ``history_last > 0``).
    """
    total_steps = len(step_sizes)
    y = init
    first_moment = torch.zeros_like(init)
    second_moment = torch.zeros_like(init)
    history: list[Tensor] = []
    beta1, beta2, eps = float(beta1), float(beta2), float(eps)
    for step in range(1, total_steps + 1):
        y, first_moment, second_moment = _adam_update(
            energy_fn,
            y,
            first_moment,
            second_moment,
            step=step,
            step_size=float(step_sizes[step - 1]),
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            differentiate=create_graph,
            projection_fn=projection_fn,
        )
        if _record_history(step - 1, total_steps, history_last):
            history.append(y.detach().clone())
        if not create_graph:
            y = y.detach()
            first_moment = first_moment.detach()
            second_moment = second_moment.detach()
    return y, history


def run_projected_solver(
    energy_fn: EnergyFn,
    init: Tensor,
    projection_fn: ProjectionFn,
    config: SolverConfig,
    *,
    create_graph: bool = False,
    history_last: int = 0,
) -> tuple[Tensor, list[Tensor]]:
    """Run the solver described by ``config``, dispatching on its
    optimizer.

    The step sizes are ``config.step_sizes()``, so ``config.steps``
    iterations are performed; ``create_graph`` and ``history_last`` are
    passed through to :func:`projected_adam` or
    :func:`projected_gradient_descent`.
    """
    step_sizes = config.step_sizes()
    if config.optimizer == 'adam':
        return projected_adam(
            energy_fn,
            init,
            projection_fn,
            step_sizes,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
            create_graph=create_graph,
            history_last=history_last,
        )
    return projected_gradient_descent(
        energy_fn,
        init,
        projection_fn,
        step_sizes,
        create_graph=create_graph,
        history_last=history_last,
    )
