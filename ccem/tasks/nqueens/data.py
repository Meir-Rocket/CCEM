"""Board-level training data for Phase 2 (solver unrolling on full
boards).
"""

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ccem.tasks.nqueens.board import (
    board_to_line,
    load_board_lines,
    parse_board_line,
)


def dihedral_transforms(board: Tensor) -> list[Tensor]:
    """The distinct images of a board under the eight board symmetries.

    Produces ``rot90(k)`` and its horizontal flip for ``k = 0..3`` and
    drops duplicates, keeping the first occurrence. Every image of a
    solution is again a solution, which is what makes the augmentation
    sound.
    """
    if board.ndim != 2 or board.size(0) != board.size(1):
        raise ValueError('board must have shape [N, N]')
    unique: list[Tensor] = []
    seen: set[str] = set()
    for k in range(4):
        rotated = torch.rot90(board, k=k, dims=(0, 1))
        for transformed in (rotated, torch.flip(rotated, dims=(1,))):
            key = board_to_line(transformed)
            if key in seen:
                continue
            seen.add(key)
            unique.append(transformed.contiguous())
    return unique


class NQueensBoardDataset(Dataset[Tensor]):
    """Solution boards read from a text file (see
    :mod:`ccem.tasks.nqueens.board`).

    The paper trains Phase 2 on the single board of
    ``{N}_queens_train.txt``; with ``augment_symmetries`` the dataset
    holds its dihedral images instead (eight distinct boards for the
    shipped 8-queens training board).

    Args:
        file_path: Board file.
        augment_symmetries: Replace every board by its symmetry images.
    """

    def __init__(
        self, file_path: str | Path, *, augment_symmetries: bool = False
    ) -> None:
        boards = [
            parse_board_line(line) for line in load_board_lines(file_path)
        ]
        if not boards:
            raise ValueError(f'no boards found in {file_path}')
        if augment_symmetries:
            boards = [
                image
                for board in boards
                for image in dihedral_transforms(board)
            ]
        self.boards = boards

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, index: int) -> Tensor:
        return self.boards[index].clone()
