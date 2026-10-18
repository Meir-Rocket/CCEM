"""Factor energies and their composition for N-Queens.

One shared PICNN scores every line of the board. A factor is a row,
column, diagonal or anti-diagonal; its input is the ``N`` board cells on
that line, left-aligned and zero-padded (``y``), together with a context
made of a 6-feature descriptor and the padding mask::

    context[0:4] one-hot kind (row=0, col=1, diag=2, anti=3)
    context[4]   active cells / N
    context[5]   offset / (N - 1)        (diagonal label, see board.py)
    PICNN input: u = cat([context, mask]) (6 + N), y = cells * mask (N)

The PICNN is convex in ``y`` for fixed ``u`` and the composed energy is
the sum over all factors, so it is convex in the relaxed board. Phase 1
trains the factor model with InfoNCE on synthetic line examples
(:class:`NQueensConstraintDataset`); Phase 2 and inference use
:class:`NQueensComposedEnergy`. The layout above, the factor order (row,
col, diag, anti) and ``context_dim = 6 + N`` are baked into the shipped
checkpoints.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.utils.data import Dataset

from ccem.icnn import PICNN
from ccem.tasks.nqueens.board import (
    Cell,
    LinePositions,
    anti_diagonal_positions,
    diagonal_positions,
)
from ccem.utils import load_checkpoint

CONTEXT_FEATURES: Final = 6
FACTOR_KINDS: Final = ('row', 'col', 'diag', 'anti')
DIAGONAL_KINDS: Final = frozenset({'diag', 'anti'})


def factor_layout(board_size: int, kind: str) -> LinePositions:
    """``(offset, cells)`` of every factor of one kind, in checkpoint
    order.

    Rows and columns are labelled by their index and cover ``N`` cells;
    diagonals come from
    :func:`ccem.tasks.nqueens.board.diagonal_positions` and
    :func:`ccem.tasks.nqueens.board.anti_diagonal_positions`.
    """
    if kind == 'row':
        return [
            (i, [(i, c) for c in range(board_size)]) for i in range(board_size)
        ]
    if kind == 'col':
        return [
            (j, [(r, j) for r in range(board_size)]) for j in range(board_size)
        ]
    if kind == 'diag':
        return diagonal_positions(board_size)
    if kind == 'anti':
        return anti_diagonal_positions(board_size)
    raise ValueError(f'unknown factor kind: {kind!r}')


def build_constraint_context(
    board_size: int, kind: str, offset: int, active_length: int
) -> Tensor:
    """The 6-feature float32 descriptor of one factor (see module
    docstring).
    """
    context = torch.zeros(CONTEXT_FEATURES, dtype=torch.float32)
    context[FACTOR_KINDS.index(kind)] = 1.0
    context[4] = float(active_length) / float(board_size)
    context[5] = float(offset) / float(max(board_size - 1, 1))
    return context


def constraint_context_table(board_size: int, kind: str) -> Tensor:
    """Descriptors of all factors of ``kind`` stacked as ``[F, 6]``."""
    return torch.stack(
        [
            build_constraint_context(board_size, kind, offset, len(cells))
            for offset, cells in factor_layout(board_size, kind)
        ],
        dim=0,
    )


def position_index_and_mask(
    board_size: int, cells_per_factor: Sequence[Sequence[Cell]]
) -> tuple[Tensor, Tensor]:
    """Flat-board gather indices ``[F, N]`` (long) and padding masks
    ``[F, N]``.

    Factor ``f`` reads ``board_flat[index[f, j]]`` for
    ``j < len(cells)``; the remaining slots are zero-padded (mask 0,
    index 0).
    """
    count = len(cells_per_factor)
    index = torch.zeros(count, board_size, dtype=torch.long)
    mask = torch.zeros(count, board_size, dtype=torch.float32)
    for factor_idx, cells in enumerate(cells_per_factor):
        for local_idx, (row, col) in enumerate(cells):
            index[factor_idx, local_idx] = row * board_size + col
            mask[factor_idx, local_idx] = 1.0
    return index, mask


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """One positive factor example: a valid line and its context."""

    kind: str
    vector: Tensor
    mask: Tensor
    context: Tensor


@dataclass(frozen=True, slots=True)
class ConstraintBatch:
    """A Phase-1 batch of positives with their negatives.

    Attributes:
        positive: Valid lines ``[B, N]``.
        negative: ``K`` negatives per positive ``[B, K, N]``.
        context: Factor descriptors ``[B, 6]``.
        mask: Padding masks ``[B, N]``.
    """

    positive: Tensor
    negative: Tensor
    context: Tensor
    mask: Tensor


def _random_other_index(active_indices: Tensor, exclude: int) -> int:
    """A uniformly random active index different from ``exclude``."""
    if active_indices.numel() <= 1:
        return exclude
    candidates = active_indices[active_indices != exclude]
    return int(
        candidates[
            int(torch.randint(int(candidates.numel()), (1,)).item())
        ].item()
    )


def _sample_outside_simplex(
    sample: Tensor, active_indices: Tensor, anchor: int
) -> Tensor:
    """Row/column negative: a vector whose mass is not one.

    Three families, drawn uniformly: under-satisfied (one entry in
    ``[0.20, 0.55)``), overloaded with two peaks (``[0.75, 1.0)`` at the
    queen and ``[0.35, 0.90)`` elsewhere) and empty. The family is drawn
    first and the second index is drawn next in every family, even the
    two that do not use it, so the RNG stream matches the reference
    sampler.
    """
    draw = int(torch.randint(0, 3, (1,)).item())
    other_index = _random_other_index(active_indices, anchor)
    if draw == 0:
        sample[anchor] = 0.20 + 0.35 * float(torch.rand(1).item())
    elif draw == 1:
        sample[anchor] = 0.75 + 0.25 * float(torch.rand(1).item())
        sample[other_index] = 0.35 + 0.55 * float(torch.rand(1).item())
    return sample


def _sample_outside_subsimplex(
    sample: Tensor, active_indices: Tensor, mask: Tensor
) -> Tensor:
    """Diagonal negative: active mass above one (outside
    ``{x >= 0, sum <= 1}``).

    A length-one diagonal gets the value 1.25. Otherwise two random
    active indices are drawn and one of three families: a strong and a
    medium peak (``[0.85, 1.0)`` and ``[0.35, 0.90)``), two medium peaks
    (``[0.60, 0.95)`` each) or random weights renormalised to a total
    mass in ``[1.25, 1.80)`` and clipped to one. Whenever the active
    mass still falls short of 1.20 the two chosen entries are raised (by
    0.30 within the mixture family, then by 0.20 unconditionally).
    """
    active_count = int(active_indices.numel())
    if active_count == 1:
        sample[active_indices[0]] = torch.tensor(1.25, dtype=mask.dtype)
        return sample
    first_index = int(
        active_indices[int(torch.randint(active_count, (1,)).item())].item()
    )
    second_index = _random_other_index(active_indices, first_index)
    min_diag_sum = 1.20
    draw = int(torch.randint(0, 3, (1,)).item())
    if draw == 0:
        sample[first_index] = 0.85 + 0.15 * float(torch.rand(1).item())
        sample[second_index] = 0.35 + 0.55 * float(torch.rand(1).item())
    elif draw == 1:
        sample[first_index] = 0.60 + 0.35 * float(torch.rand(1).item())
        sample[second_index] = 0.60 + 0.35 * float(torch.rand(1).item())
    else:
        weights = torch.rand(active_count, dtype=mask.dtype)
        weights = weights / weights.sum().clamp_min(1.0e-6)
        total_mass = 1.25 + 0.55 * float(torch.rand(1).item())
        sample[active_indices] = weights * total_mass
        sample = sample.clamp(max=1.0)
        if float(sample[active_indices].sum().item()) <= min_diag_sum:
            sample[first_index] = min(1.0, float(sample[first_index]) + 0.30)
            sample[second_index] = min(1.0, float(sample[second_index]) + 0.30)
    sample = sample * mask
    if float(sample[active_indices].sum().item()) <= min_diag_sum:
        sample[first_index] = float(sample[first_index]) + 0.20
        sample[second_index] = float(sample[second_index]) + 0.20
    return sample


def sample_invalid_constraint(
    kind: str, mask: Tensor, positive: Tensor
) -> Tensor:
    """Draw one negative line for a positive of ``kind`` (Phase-1
    sampler).

    Row/column positives are one-hot, so negatives lie outside the
    simplex; diagonal positives are zero or one-hot, so negatives carry
    more than one unit of mass on the active cells. Only the global
    torch RNG is used.
    """
    active_indices = torch.where(mask > 0.5)[0]
    sample = torch.zeros(mask.numel(), dtype=mask.dtype)
    if kind in DIAGONAL_KINDS:
        return _sample_outside_subsimplex(sample, active_indices, mask)
    anchor = int(
        active_indices[
            int(torch.argmax(positive[active_indices]).item())
        ].item()
    )
    return _sample_outside_simplex(sample, active_indices, anchor)


class NQueensConstraintDataset(Dataset[ConstraintBatch]):
    """Synthetic Phase-1 examples: every valid line of every factor.

    Rows and columns contribute their ``N`` one-hot vectors; diagonals
    and anti-diagonals contribute the zero vector and the one-hot
    vectors on their active cells. Each item pairs a positive with
    ``negatives_per_positive`` freshly drawn negatives
    (:func:`sample_invalid_constraint`).

    Args:
        board_size: ``N``.
        factor_types: Subset of ``('row', 'col', 'diag', 'anti')``.
        negatives_per_positive: Negatives drawn per item.
    """

    def __init__(
        self,
        board_size: int,
        factor_types: Sequence[str],
        negatives_per_positive: int = 3,
    ) -> None:
        size = int(board_size)
        self.negatives_per_positive = int(negatives_per_positive)
        self.constraints: list[ConstraintSpec] = []
        for kind in FACTOR_KINDS:
            if kind not in factor_types:
                continue
            contexts = constraint_context_table(size, kind)
            for factor_idx, (_, cells) in enumerate(factor_layout(size, kind)):
                active_length = len(cells)
                mask = torch.zeros(size, dtype=torch.float32)
                mask[:active_length] = 1.0
                vectors = [torch.zeros(size)] if kind in DIAGONAL_KINDS else []
                vectors.extend(torch.eye(size)[:active_length])
                self.constraints.extend(
                    ConstraintSpec(kind, vector, mask, contexts[factor_idx])
                    for vector in vectors
                )

    def __len__(self) -> int:
        return len(self.constraints)

    def __getitem__(self, index: int) -> ConstraintBatch:
        spec = self.constraints[index]
        negatives = [
            sample_invalid_constraint(spec.kind, spec.mask, spec.vector)
            for _ in range(self.negatives_per_positive)
        ]
        return ConstraintBatch(
            positive=spec.vector,
            negative=torch.stack(negatives, dim=0),
            context=spec.context,
            mask=spec.mask,
        )


def collate_constraint_batch(
    items: Sequence[ConstraintBatch],
) -> ConstraintBatch:
    """Stack dataset items into one :class:`ConstraintBatch`."""
    return ConstraintBatch(
        positive=torch.stack([item.positive for item in items], dim=0),
        negative=torch.stack([item.negative for item in items], dim=0),
        context=torch.stack([item.context for item in items], dim=0),
        mask=torch.stack([item.mask for item in items], dim=0),
    )


class ConstraintEnergyModel(nn.Module):
    """The shared factor energy ``E(y | context, mask)``.

    ``net`` is a :class:`~ccem.icnn.PICNN`: convex in the line values
    ``y`` and conditioned on ``cat([context, mask])``. The attribute
    name ``net`` and the PICNN child names are the keys of the shipped
    ``model_state`` dictionaries.

    Args:
        board_size: ``N``; the context has ``6 + N`` features.
        hidden_dims: Hidden widths, ``[hidden_dim] * depth`` in the
            configs.
        activation: ``'softplus'`` (shipped checkpoints) or ``'relu'``.
    """

    net: PICNN

    def __init__(
        self, board_size: int, hidden_dims: Sequence[int], activation: str
    ) -> None:
        super().__init__()
        self.net = PICNN(
            CONTEXT_FEATURES + board_size, board_size, hidden_dims, activation
        )

    def forward(self, y: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        """Energies ``[B]`` of lines ``y [B, N]`` with
        ``context [B, 6]``, ``mask [B, N]``.
        """
        energy: Tensor = self.net(torch.cat([context, mask], dim=-1), y * mask)
        return energy


def build_factor_model(
    model_cfg: Mapping[str, Any], board_size: int
) -> ConstraintEnergyModel:
    """Build the factor model from a YAML ``model:`` block.

    Reads ``hidden_dim``, ``depth`` and ``activation`` (default
    ``'relu'``); ``hidden_dims = [hidden_dim] * depth``.
    """
    hidden_dims = [int(model_cfg['hidden_dim'])] * int(model_cfg['depth'])
    return ConstraintEnergyModel(
        board_size, hidden_dims, str(model_cfg.get('activation', 'relu'))
    )


@dataclass(frozen=True, slots=True)
class FactorInputs:
    """Factor-model inputs of one factor kind.

    Attributes:
        vectors: Padded line values ``[B, F, N]`` (keeps the board's
            graph).
        contexts: Factor descriptors ``[F, 6]``.
        masks: Padding masks ``[F, N]``.
    """

    vectors: Tensor
    contexts: Tensor
    masks: Tensor


class NQueensComposedEnergy(nn.Module):
    """Sum of the factor energies over all lines of a relaxed board.

    The factors are evaluated in the order row, col, diag, anti
    (restricted to ``factor_types``) in a single batched call of
    ``factor_model``. Rows are fed as the board, columns as its
    transpose and diagonals through a gather with their padding masks.
    The geometry buffers are rebuilt from :func:`factor_layout` at
    construction and are not part of the state dict.

    Args:
        factor_model: The shared line energy.
        board_size: ``N``.
        factor_types: Which factor kinds to compose.
    """

    def __init__(
        self,
        factor_model: ConstraintEnergyModel,
        board_size: int,
        factor_types: Sequence[str],
    ) -> None:
        super().__init__()
        self.factor_model = factor_model
        self.board_size = int(board_size)
        self.factor_types = [
            kind for kind in FACTOR_KINDS if kind in factor_types
        ]
        for kind in self.factor_types:
            self.register_buffer(
                f'{kind}_context',
                constraint_context_table(self.board_size, kind),
                persistent=False,
            )
            if kind in DIAGONAL_KINDS:
                cells = [
                    cells for _, cells in factor_layout(self.board_size, kind)
                ]
                index, mask = position_index_and_mask(self.board_size, cells)
                self.register_buffer(f'{kind}_index', index, persistent=False)
                self.register_buffer(f'{kind}_mask', mask, persistent=False)

    def factor_context(self, kind: str) -> Tensor:
        """Descriptors ``[F, 6]`` of the factors of ``kind``."""
        return self.get_buffer(f'{kind}_context')

    def factor_mask(self, kind: str) -> Tensor:
        """Padding masks ``[F, N]`` of the factors of ``kind`` (ones
        for rows and columns).
        """
        if kind in DIAGONAL_KINDS:
            return self.get_buffer(f'{kind}_mask')
        contexts = self.factor_context(kind)
        return torch.ones(
            contexts.size(0), self.board_size, device=contexts.device
        )

    def factor_inputs(self, board: Tensor) -> dict[str, FactorInputs]:
        """Per-kind factor-model inputs of boards ``[B, N, N]``.

        Contexts and masks are cast to the board's dtype and device;
        vectors keep the autograd graph of ``board``.

        Raises:
            ValueError: If ``board`` is not three-dimensional.
        """
        if board.ndim != 3:
            raise ValueError('board must have shape [B, N, N]')
        batch_size = int(board.size(0))
        # Created after the row/column branches so that autograd
        # accumulates the four gradient contributions in the reference
        # order.
        board_flat: Optional[Tensor] = None
        inputs: dict[str, FactorInputs] = {}
        for kind in self.factor_types:
            if kind == 'row':
                vectors = board
            elif kind == 'col':
                vectors = board.transpose(1, 2)
            else:
                if board_flat is None:
                    board_flat = board.reshape(
                        batch_size, self.board_size * self.board_size
                    )
                index = self.get_buffer(f'{kind}_index')
                mask = self.get_buffer(f'{kind}_mask').to(
                    device=board.device, dtype=board.dtype
                )
                flat_index = index.view(1, -1).expand(batch_size, -1)
                gathered = board_flat.gather(dim=1, index=flat_index)
                vectors = gathered.view(
                    batch_size, -1, self.board_size
                ) * mask.unsqueeze(0)
            inputs[kind] = FactorInputs(
                vectors=vectors,
                contexts=self.factor_context(kind).to(
                    device=board.device, dtype=board.dtype
                ),
                masks=self.factor_mask(kind).to(
                    device=board.device, dtype=board.dtype
                ),
            )
        return inputs

    def _all_factor_energies(self, board: Tensor) -> Tensor:
        """Energies ``[B, F_total]`` of every factor, in kind order."""
        inputs = list(self.factor_inputs(board).values())
        vectors = torch.cat([item.vectors for item in inputs], dim=1)
        contexts = torch.cat([item.contexts for item in inputs], dim=0)
        masks = torch.cat(
            [
                item.masks.unsqueeze(0).expand(vectors.size(0), -1, -1)
                for item in inputs
            ],
            dim=1,
        )
        batch_size, factor_count, vec_dim = vectors.shape
        context_dim = int(contexts.size(-1))
        energies: Tensor = self.factor_model(
            vectors.reshape(batch_size * factor_count, vec_dim),
            contexts.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .reshape(batch_size * factor_count, context_dim),
            masks.reshape(batch_size * factor_count, vec_dim),
        )
        return energies.view(batch_size, factor_count)

    def total_energy(self, board: Tensor) -> Tensor:
        """Composed energy ``[B]`` of boards ``[B, N, N]``.

        Raises:
            ValueError: If ``board`` is not three-dimensional.
        """
        return self._all_factor_energies(board).sum(dim=1)


def load_composed_energy(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[NQueensComposedEnergy, dict[str, Any]]:
    """Load a shipped checkpoint into a frozen, eval-mode composed
    energy.

    The payload holds ``config`` (the training YAML), ``model_state``,
    ``board_size``, ``factor_types`` and ``model_type`` (always
    ``'picnn'``); the factor model is rebuilt from ``config['model']``
    and loaded with ``strict=True``.

    Returns:
        The composer on ``device`` and the raw payload.

    Raises:
        ValueError: If the payload's ``model_type`` is not ``'picnn'``.
    """
    payload = load_checkpoint(checkpoint_path, device)
    if payload['model_type'] != 'picnn':
        raise ValueError(
            f'{checkpoint_path} holds model_type '
            f'{payload["model_type"]!r}; expected picnn'
        )
    board_size = int(payload['board_size'])
    model = build_factor_model(payload['config']['model'], board_size).to(
        device
    )
    model.load_state_dict(payload['model_state'], strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    composer = NQueensComposedEnergy(
        model, board_size, payload['factor_types']
    ).to(device)
    composer.eval()
    return composer, payload


def train_energy_step(
    model: ConstraintEnergyModel, batch: ConstraintBatch, device: torch.device
) -> tuple[Tensor, dict[str, float]]:
    """One Phase-1 InfoNCE step on a batch of factor examples.

    The logits are the negated energies of the positive (index 0) and of
    its ``K`` negatives, so the cross-entropy pushes the positive below
    every negative.

    Returns:
        The loss and the metrics ``loss``, ``pos_energy``,
        ``neg_energy`` and ``margin`` (mean negative minus mean positive
        energy).
    """
    positive = batch.positive.to(device)
    negative = batch.negative.to(device)
    context = batch.context.to(device)
    mask = batch.mask.to(device)

    pos_energy = model(positive, context, mask)
    batch_size, negatives_per_positive, vec_dim = negative.shape
    flat = batch_size * negatives_per_positive
    neg_energy = model(
        negative.view(flat, vec_dim),
        context.unsqueeze(1)
        .expand(-1, negatives_per_positive, -1)
        .reshape(flat, -1),
        mask.unsqueeze(1)
        .expand(-1, negatives_per_positive, -1)
        .reshape(flat, -1),
    ).view(batch_size, negatives_per_positive)
    logits = torch.cat([-pos_energy.unsqueeze(1), -neg_energy], dim=1)
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)
    loss = f.cross_entropy(logits, targets)
    metrics = {
        'loss': loss.item(),
        'pos_energy': pos_energy.mean().item(),
        'neg_energy': neg_energy.mean().item(),
        'margin': (neg_energy.mean() - pos_energy.mean()).item(),
    }
    return loss, metrics
