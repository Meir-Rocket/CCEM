"""Contrastive edge dataset for training the graph coloring factor.

``EdgeColorDataset`` draws the samples of the local (per-edge)
contrastive objective: a valid edge, i.e. two distinct one-hot colors,
against a few invalid or off-simplex joint color vectors. The random
graph generators live in :mod:`.generate_data`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as f
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class EdgeSample:
    """One positive edge vector ``[2K]`` and its negatives
    ``[M, 2K]``.
    """

    positive: Tensor
    negative: Tensor


def _joint_one_hot(left: int, right: int, num_colors: int) -> Tensor:
    """``concat(one_hot(left, K), one_hot(right, K))`` as a float
    ``[2K]`` vector.
    """
    return (
        f.one_hot(torch.tensor([left, right]), num_colors).reshape(-1).float()
    )


class EdgeColorDataset(Dataset[EdgeSample]):
    """Random contrastive edge samples for the per-edge factor.

    A sample's input is the joint color vector ``[y_u, y_v]`` of an
    edge, laid out as ``concat(one_hot(left, K), one_hot(right, K))``;
    this layout fixes the factor's input dimension ``2K`` baked into the
    checkpoints. Positives are one-hot pairs with two distinct colors.
    Each positive gets ``negatives_per_positive`` negatives drawn from
    three modes with the given (normalised) fractions: the same color on
    both sides, a one-hot pair with an extra spike of mass on one side
    (clamped to ``[0, 1.5]``) and an off-simplex pair that spreads equal
    mass over two colors per side. Samples are drawn from the global
    torch generator; ``index`` is ignored.

    Args:
        num_colors: Palette size ``K`` (at least 2).
        num_samples: Length of the dataset (samples per epoch).
        negatives_per_positive: Negatives ``M`` per positive (at least
            1).
        same_color_fraction: Weight of the same-color negatives.
        perturbation_fraction: Weight of the spike-perturbed negatives.
        swap_fraction: Weight of the off-simplex negatives.

    Raises:
        ValueError: If ``num_colors < 2`` or the fractions sum to zero.
    """

    def __init__(
        self,
        num_colors: int,
        num_samples: int,
        negatives_per_positive: int = 4,
        same_color_fraction: float = 0.5,
        perturbation_fraction: float = 0.3,
        swap_fraction: float = 0.2,
    ) -> None:
        if num_colors < 2:
            raise ValueError('num_colors must be >= 2')
        weights_sum = float(
            same_color_fraction + perturbation_fraction + swap_fraction
        )
        if weights_sum <= 0.0:
            raise ValueError(
                'negative-mode fractions must sum to a positive value'
            )
        self.num_colors = int(num_colors)
        self.num_samples = int(num_samples)
        self.negatives_per_positive = max(int(negatives_per_positive), 1)
        self.same_color_fraction = float(same_color_fraction) / weights_sum
        self.perturbation_fraction = float(perturbation_fraction) / weights_sum
        self.swap_fraction = float(swap_fraction) / weights_sum

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> EdgeSample:
        left_color = int(torch.randint(0, self.num_colors, (1,)).item())
        right_color = self._sample_other_color(exclude=left_color)
        positive = _joint_one_hot(left_color, right_color, self.num_colors)
        negatives = [
            self._sample_negative(left_color, right_color)
            for _ in range(self.negatives_per_positive)
        ]
        return EdgeSample(
            positive=positive, negative=torch.stack(negatives, dim=0)
        )

    def _sample_other_color(self, exclude: int) -> int:
        candidate = int(torch.randint(0, self.num_colors - 1, (1,)).item())
        return candidate if candidate < exclude else candidate + 1

    def _sample_negative(
        self, positive_left: int, positive_right: int
    ) -> Tensor:
        draw = float(torch.rand(1).item())
        if draw < self.same_color_fraction:
            color = int(torch.randint(0, self.num_colors, (1,)).item())
            return _joint_one_hot(color, color, self.num_colors)
        if draw < self.same_color_fraction + self.perturbation_fraction:
            return self._perturbed_one_hot(positive_left, positive_right)
        return self._swap_off_simplex(positive_left, positive_right)

    def _perturbed_one_hot(self, left_color: int, right_color: int) -> Tensor:
        sample = _joint_one_hot(left_color, right_color, self.num_colors)
        side_start = int(torch.randint(0, 2, (1,)).item()) * self.num_colors
        spike_index = side_start + int(
            torch.randint(0, self.num_colors, (1,)).item()
        )
        magnitude = 0.55 + 0.45 * float(torch.rand(1).item())
        sample[spike_index] = float(sample[spike_index]) + magnitude
        return sample.clamp(min=0.0, max=1.5)

    def _swap_off_simplex(self, left_color: int, right_color: int) -> Tensor:
        sample = torch.zeros(2 * self.num_colors, dtype=torch.float32)
        left_alt = self._sample_other_color(exclude=left_color)
        right_alt = self._sample_other_color(exclude=right_color)
        primary_left = 0.30 + 0.40 * float(torch.rand(1).item())
        primary_right = 0.30 + 0.40 * float(torch.rand(1).item())
        sample[left_color] = primary_left
        sample[left_alt] = primary_left
        sample[self.num_colors + right_color] = primary_right
        sample[self.num_colors + right_alt] = primary_right
        return sample


def collate_edge_samples(items: Sequence[EdgeSample]) -> dict[str, Tensor]:
    """Stack samples into
    ``{'positive': [B, 2K], 'negative': [B, M, 2K]}``.
    """
    return {
        'positive': torch.stack([item.positive for item in items], dim=0),
        'negative': torch.stack([item.negative for item in items], dim=0),
    }
