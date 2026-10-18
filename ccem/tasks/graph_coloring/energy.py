"""Per-edge convex factor and its composition over a graph.

The factor ``E_theta([y_u, y_v])`` scores the concatenated color vectors
of an edge's endpoints; the composed energy of a relaxed coloring ``Y``
in ``[N, K]`` is ``E(Y) = sum_{(u, v) in edges} E_theta([Y_u, Y_v])``,
which is convex in ``Y`` because every factor is convex in its joint
input.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn

from ccem.icnn import PICNN
from ccem.projections import project_simplex
from ccem.solvers import SolverConfig, run_projected_solver, with_tie_break
from ccem.tasks.graph_coloring.graphs import GraphInstance
from ccem.utils import load_checkpoint


def random_node_assignments(
    batch_size: int, num_nodes: int, num_colors: int, device: torch.device
) -> Tensor:
    """Uniform noise ``[B, N, K]`` projected onto the per-node simplex.

    Consumes exactly one ``torch.rand`` call of shape ``[B, N, K]`` from
    the global generator; the evaluation protocol draws the tie-break
    matrix right after it, so the order matters.
    """
    raw = torch.rand(batch_size, num_nodes, num_colors, device=device)
    return project_simplex(raw)


class EdgeColorEnergyModel(nn.Module):
    """Convex energy of one edge's joint color vector ``[y_u, y_v]``.

    The input has width ``2K`` laid out as ``concat(y_u, y_v)``. The
    backbone is a ``PICNN(context_dim=1, y_dim=2K)`` fed a constant
    context of ones, so the energy is convex in the whole joint input.
    It is the child ``net`` (checkpoint key prefix ``net.``), built
    with ``hidden_dims`` (:func:`build_factor_model` passes
    ``[hidden_dim] * depth`` from the ``model:`` block).

    Args:
        num_colors: Palette size ``K``.
        hidden_dims: Hidden layer widths.
        activation: ``'relu'`` (shipped checkpoint) or ``'softplus'``.
    """

    def __init__(
        self,
        num_colors: int,
        hidden_dims: Sequence[int],
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        self.num_colors = int(num_colors)
        self.net = PICNN(1, 2 * self.num_colors, hidden_dims, activation)

    def forward(self, joint: Tensor) -> Tensor:
        """Energies ``[B]`` of joint vectors ``[B, 2K]``."""
        energy: Tensor = self.net(joint.new_ones((joint.size(0), 1)), joint)
        return energy


def build_factor_model(
    model_cfg: Mapping[str, Any], num_colors: int
) -> EdgeColorEnergyModel:
    """Instantiate the factor from a YAML ``model:`` block.

    Reads ``hidden_dim``, ``depth`` and ``activation`` (default
    ``'relu'``) and builds ``hidden_dims = [hidden_dim] * depth``, the
    construction baked into the shipped checkpoint.
    """
    hidden_dims = [int(model_cfg['hidden_dim'])] * int(model_cfg['depth'])
    return EdgeColorEnergyModel(
        num_colors, hidden_dims, str(model_cfg.get('activation', 'relu'))
    )


def load_factor_model(
    checkpoint_path: str | Path, device: torch.device
) -> EdgeColorEnergyModel:
    """Load a frozen factor from a checkpoint payload.

    The payload holds ``config`` (the training YAML), ``model_state``,
    ``num_colors`` and ``model_type`` (always ``'picnn'``). The weights
    are loaded strictly, frozen and switched to eval mode.

    Raises:
        ValueError: If the payload's ``model_type`` is not ``'picnn'``.
    """
    payload = load_checkpoint(checkpoint_path, device)
    if payload['model_type'] != 'picnn':
        raise ValueError(
            f'{checkpoint_path}: expected model_type picnn, got '
            f'{payload["model_type"]!r}'
        )
    model = build_factor_model(
        payload['config']['model'], int(payload['num_colors'])
    ).to(device=device)
    model.load_state_dict(payload['model_state'])
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


class GraphColoringComposedEnergy(nn.Module):
    """Sum of the edge factor over every edge of one graph.

    ``total_energy`` gathers the endpoint rows of each edge,
    concatenates them into ``[B * E, 2K]`` joint vectors, scores them
    with the factor and sums per batch element. A graph without edges
    has zero energy.

    Args:
        factor_model: The per-edge factor (defines ``num_colors``).
        graph: The instance whose edges are composed.
    """

    def __init__(
        self, factor_model: EdgeColorEnergyModel, graph: GraphInstance
    ) -> None:
        super().__init__()
        self.factor_model = factor_model
        self.num_colors = factor_model.num_colors
        self.num_nodes = graph.num_nodes
        self.register_buffer('edges', graph.edges.clone(), persistent=False)
        self.edges: Tensor

    def total_energy(self, soft_assignments: Tensor) -> Tensor:
        """Composed energy ``[B]`` of relaxed colorings ``[B, N, K]``.

        Raises:
            ValueError: If the shape, the node count or the palette
                size does not match.
        """
        if soft_assignments.ndim != 3:
            raise ValueError('soft_assignments must have shape [B, N, K]')
        batch_size = int(soft_assignments.size(0))
        if int(soft_assignments.size(1)) != self.num_nodes:
            raise ValueError(
                f'expected {self.num_nodes} nodes but received '
                f'{soft_assignments.size(1)}'
            )
        if int(soft_assignments.size(2)) != self.num_colors:
            raise ValueError(
                'last dim of soft_assignments must equal '
                'factor_model.num_colors'
            )
        edge_count = int(self.edges.size(0))
        if edge_count == 0:
            return torch.zeros(
                batch_size,
                device=soft_assignments.device,
                dtype=soft_assignments.dtype,
            )
        left_features = soft_assignments.index_select(
            dim=1, index=self.edges[:, 0]
        )
        right_features = soft_assignments.index_select(
            dim=1, index=self.edges[:, 1]
        )
        joint = torch.cat([left_features, right_features], dim=-1)
        joint_flat = joint.reshape(
            batch_size * edge_count, 2 * self.num_colors
        )
        energy_flat: Tensor = self.factor_model(joint_flat)
        return energy_flat.view(batch_size, edge_count).sum(dim=-1)


def solve_graph(
    composer: GraphColoringComposedEnergy,
    init: Tensor,
    config: SolverConfig,
    *,
    tie_break: Optional[Tensor],
) -> Tensor:
    """Minimise the composed energy from ``init`` with the configured
    solver.

    Every iterate is projected onto the product of per-node simplices
    with :func:`ccem.projections.project_simplex`. The optional
    ``tie_break`` matrix (drawn by the caller, shaped like ``init``)
    adds ``config.tie_break_weight * <Y, tie_break>`` to the objective;
    energies reported afterwards should use ``composer.total_energy``
    without it.

    Returns:
        The final iterates ``[B, N, K]``.
    """
    energy_fn = with_tie_break(
        composer.total_energy, tie_break, config.tie_break_weight
    )
    return run_projected_solver(energy_fn, init, project_simplex, config)[0]


def train_edge_energy_step(
    model: EdgeColorEnergyModel,
    batch: Mapping[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    """InfoNCE objective on one batch of
    :class:`~.data.EdgeColorDataset`.

    The logits are ``[-E(pos), -E(neg_1), ..., -E(neg_M)]`` and the loss
    is the cross-entropy with the positive as the target class, so each
    valid edge is pushed below its negatives.

    Returns:
        The loss and the batch metrics ``loss``, ``pos_energy``,
        ``neg_energy`` and ``margin`` (``neg_energy - pos_energy``).
    """
    positive = batch['positive'].to(device=device)
    negative = batch['negative'].to(device=device)
    pos_energy = model(positive)
    batch_size, negatives_per_positive, joint_dim = negative.shape
    neg_flat = negative.view(batch_size * negatives_per_positive, joint_dim)
    neg_energy = model(neg_flat).view(batch_size, negatives_per_positive)
    logits = torch.cat([-pos_energy.unsqueeze(1), -neg_energy], dim=1)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    loss = nn.CrossEntropyLoss()(logits, targets)
    metrics = {
        'loss': float(loss.detach().cpu()),
        'pos_energy': float(pos_energy.mean().detach().cpu()),
        'neg_energy': float(neg_energy.mean().detach().cpu()),
        'margin': float(
            (neg_energy.mean() - pos_energy.mean()).detach().cpu()
        ),
    }
    return loss, metrics
