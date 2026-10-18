"""Enumerate all N-Queens solutions and write the solution-set and
training files.

For every board size the complete solution set is enumerated, sorted
and shuffled with ``random.Random(seed)``, then written as

* ``{N}_queens_all.txt``: every solution, the known-solution set that
  evaluation checks decoded boards against,
* ``{N}_queens_train.txt``: the first board of the shuffled set, the
  one board Phase 2 trains on.

With the default sizes and ``--seed 0`` the command reproduces the
shipped ``data/nqueens`` files byte for byte::

    python -m ccem.tasks.nqueens.generate_data --output-dir data/nqueens
"""

import argparse
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from ccem.tasks.nqueens.board import board_to_line, write_board_lines

PAPER_SIZES = (4, 5, 6, 7, 8, 9, 10)
"""The board sizes of the paper's N-Queens experiments."""


def enumerate_solutions(board_size: int) -> Iterator[list[int]]:
    """Yield every solution as the queen column of each row
    (depth-first).
    """

    def place(
        row: int, columns: list[int], used: tuple[set[int], set[int], set[int]]
    ) -> Iterator[list[int]]:
        if row == board_size:
            yield list(columns)
            return
        used_cols, used_diag, used_anti = used
        for col in range(board_size):
            if (
                col in used_cols
                or (row - col) in used_diag
                or (row + col) in used_anti
            ):
                continue
            used_cols.add(col)
            used_diag.add(row - col)
            used_anti.add(row + col)
            columns.append(col)
            yield from place(row + 1, columns, used)
            columns.pop()
            used_cols.remove(col)
            used_diag.remove(row - col)
            used_anti.remove(row + col)

    yield from place(0, [], (set(), set(), set()))


def columns_to_board(columns: Sequence[int]) -> Tensor:
    """Build the ``[N, N]`` float board with a queen at
    ``(row, columns[row])``.
    """
    size = len(columns)
    board = torch.zeros(size, size, dtype=torch.float32)
    for row, col in enumerate(columns):
        board[row, col] = 1.0
    return board


def all_solution_lines(board_size: int, seed: int = 0) -> list[str]:
    """All solutions as text lines, sorted and then shuffled with
    ``seed``.
    """
    lines = sorted(
        board_to_line(columns_to_board(columns))
        for columns in enumerate_solutions(board_size)
    )
    random.Random(seed).shuffle(lines)
    return lines


def write_solution_files(
    board_size: int, output_dir: str | Path, seed: int = 0
) -> None:
    """Write ``{N}_queens_all.txt`` and ``{N}_queens_train.txt`` for one
    size.
    """
    lines = all_solution_lines(board_size, seed)
    if not lines:
        raise ValueError(f'no {board_size}-queens solutions exist')
    directory = Path(output_dir)
    for name, content in (('all', lines), ('train', lines[:1])):
        write_board_lines(
            content, directory / f'{board_size}_queens_{name}.txt'
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI flags of the generator."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--board-size',
        type=int,
        nargs='+',
        default=list(PAPER_SIZES),
        help='board sizes to generate (default: 4 .. 10)',
    )
    parser.add_argument(
        '--output-dir',
        default='data/nqueens',
        help=(
            'directory for the {N}_queens_*.txt files (default: data/nqueens)'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='shuffle seed (0 reproduces the shipped files)',
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of ``python -m ccem.tasks.nqueens.generate_data``."""
    args = parse_args(argv)
    for size in args.board_size:
        write_solution_files(size, args.output_dir, seed=args.seed)
        print(f'Generated N-Queens files for N={size} in {args.output_dir}')


if __name__ == '__main__':
    main()
