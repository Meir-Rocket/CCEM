"""Board geometry, text format, decoding and conflict counting for
N-Queens.

On-disk format
    One board per line, ``N * N`` characters in row-major order, ``'Q'``
    for a queen and ``'.'`` for an empty cell, LF-terminated. ``N`` is
    inferred from the line length. Evaluation checks decoded boards
    against ``data/nqueens/{N}_queens_all.txt`` by exact string
    equality, so :func:`board_to_line` and :func:`parse_board_line`
    define that match.

Line (factor) conventions
    :func:`diagonal_positions` and :func:`anti_diagonal_positions`
    define the order and labelling of the diagonal factors. The learned
    factor energies read ``offset / (N - 1)`` and ``len(cells) / N``
    from these lists and the composed energy gathers cells in this
    order, so the conventions here are baked into every shipped
    checkpoint and must not change.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

QUEEN_CHAR = 'Q'
EMPTY_CHAR = '.'

Cell = tuple[int, int]
LinePositions = list[tuple[int, list[Cell]]]
"""``[(offset, [(row, col), ...]), ...]``: the cells of each diagonal
line.
"""


@dataclass(frozen=True, slots=True)
class BoardEvalResult:
    """Validity and conflict counts of a hard ``{0, 1}`` board.

    These are the counts behind the evaluation report: a row or column
    conflict is a row/column whose sum differs from one (empty lines
    count), a diagonal conflict is a diagonal or anti-diagonal whose sum
    exceeds one. ``is_valid`` additionally requires exactly ``N``
    queens.
    """

    is_valid: bool
    num_queens: int
    num_row_conflicts: int
    num_col_conflicts: int
    num_diag_conflicts: int


def _infer_board_size(line: str) -> int:
    length = len(line)
    size = int(math.sqrt(length))
    if size * size != length:
        raise ValueError(f'Invalid board line of length {length}')
    return size


def parse_board_line(line: str) -> Tensor:
    """Parse one text line into a float32 ``[N, N]`` board of zeros
    and ones.
    """
    line = line.strip()
    size = _infer_board_size(line)
    values = [1.0 if char == QUEEN_CHAR else 0.0 for char in line]
    return torch.tensor(values, dtype=torch.float32).view(size, size)


def board_to_line(board: Tensor) -> str:
    """Format a board as a text line; a cell is a queen when
    ``|x - 1| < 1e-6``.
    """
    flat: list[float] = board.reshape(-1).tolist()
    return ''.join(
        QUEEN_CHAR if abs(x - 1) < 1.0e-6 else EMPTY_CHAR for x in flat
    )


def load_board_lines(file_path: str | Path) -> list[str]:
    """Read the non-empty, stripped lines of a board file."""
    with open(file_path, encoding='utf-8') as handle:
        return [line.strip() for line in handle if line.strip()]


def load_solution_set(file_path: str | Path) -> set[str]:
    """Read a board file as a set of lines (the known-solution
    membership set).
    """
    return set(load_board_lines(file_path))


def write_board_lines(lines: Iterable[str], file_path: str | Path) -> None:
    """Write board lines LF-terminated, creating parent directories."""
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'w', encoding='utf-8') as handle:
        for line in lines:
            handle.write(line.strip())
            handle.write('\n')


def diagonal_positions(board_size: int) -> LinePositions:
    """Cells of every main diagonal, keyed by ``offset = col - row``.

    Offsets run from ``-(N - 1)`` to ``N - 1`` in ascending order and
    the cells of each diagonal are listed by ascending row,
    ``(row, row + offset)``. The factor context uses
    ``offset / (N - 1)`` and the composed energy places
    ``board[row, col]`` at the cell's position in this list, so the
    ordering is checkpoint-critical.
    """
    positions: LinePositions = []
    for offset in range(-(board_size - 1), board_size):
        cells = [
            (row, row + offset)
            for row in range(board_size)
            if 0 <= row + offset < board_size
        ]
        positions.append((offset, cells))
    return positions


def anti_diagonal_positions(board_size: int) -> LinePositions:
    """Cells of every anti-diagonal, mirrored from
    :func:`diagonal_positions`.

    Each anti-diagonal keeps the label of the main diagonal it mirrors,
    ``(row, col) -> (row, N - 1 - col)``, rather than a ``row + col``
    label. The label is not the standard one, but it is what the shipped
    factor energies were trained on, so it stays.
    """
    return [
        (offset, [(row, (board_size - 1) - col) for row, col in cells])
        for offset, cells in diagonal_positions(board_size)
    ]


def decode_board(heatmap: Tensor) -> Tensor:
    """Greedy row-by-row decoding of a relaxed ``[N, N]`` board.

    Rows are visited in descending order of their maximum and, within a
    row, columns in descending value; the first column that is not
    attacked by an already placed queen is chosen, and the row's argmax
    when every column is attacked. The two
    ``torch.argsort(descending=True)`` calls fix the tie-breaking of the
    reported numbers.
    """
    solution = torch.zeros_like(heatmap)
    used_cols: set[int] = set()
    used_diag: set[int] = set()
    used_anti: set[int] = set()
    row_order = torch.argsort(heatmap.max(dim=1).values, descending=True)
    for row in row_order.tolist():
        chosen: Optional[int] = None
        for col in torch.argsort(heatmap[row], descending=True).tolist():
            if (
                col in used_cols
                or (row - col) in used_diag
                or (row + col) in used_anti
            ):
                continue
            chosen = col
            break
        if chosen is None:
            chosen = int(torch.argmax(heatmap[row]).item())
        solution[row, chosen] = 1.0
        used_cols.add(chosen)
        used_diag.add(row - chosen)
        used_anti.add(row + chosen)
    return solution


def evaluate_board(board: Tensor) -> BoardEvalResult:
    """Count conflicts of a hard board (see
    :class:`BoardEvalResult`).
    """
    size = int(board.size(0))
    row_conflicts = int((board.sum(dim=1) != 1.0).sum().item())
    col_conflicts = int((board.sum(dim=0) != 1.0).sum().item())
    diag_conflicts = 0
    flipped = board.flip(dims=(1,))
    for offset in range(-(size - 1), size):
        if board.diagonal(offset=offset).sum().item() > 1.0:
            diag_conflicts += 1
        if flipped.diagonal(offset=offset).sum().item() > 1.0:
            diag_conflicts += 1
    num_queens = int(board.sum().item())
    is_valid = (
        row_conflicts == 0
        and col_conflicts == 0
        and diag_conflicts == 0
        and num_queens == size
    )
    return BoardEvalResult(
        is_valid=is_valid,
        num_queens=num_queens,
        num_row_conflicts=row_conflicts,
        num_col_conflicts=col_conflicts,
        num_diag_conflicts=diag_conflicts,
    )


def relaxed_board_sharpness(board: Tensor) -> dict[str, float]:
    """Sharpness statistics of relaxed boards ``[..., N, N]``.

    Returns the mean row and column entropies (``row_entropy``,
    ``col_entropy``), the mean gap between the two largest row entries
    (``row_gap``), the fraction of rows whose maximum exceeds 0.9
    (``confident_rows``) and the mean absolute row/column sum errors
    (``row_sum_err``, ``col_sum_err``). These are the
    ``Relaxed sharpness`` and ``Relaxed sum errors`` lines of the
    evaluation report.
    """
    clipped = board.clamp(min=1.0e-8, max=1.0 - 1.0e-8)
    plogp = clipped * torch.log(clipped)
    row_top2 = torch.topk(board, k=2, dim=-1).values
    return {
        'row_entropy': float(-plogp.sum(dim=-1).mean().detach().cpu()),
        'col_entropy': float(-plogp.sum(dim=-2).mean().detach().cpu()),
        'row_gap': float(
            (row_top2[..., 0] - row_top2[..., 1]).mean().detach().cpu()
        ),
        'confident_rows': float(
            (board.max(dim=-1).values > 0.9).float().mean().detach().cpu()
        ),
        'row_sum_err': float(
            (board.sum(dim=-1) - 1.0).abs().mean().detach().cpu()
        ),
        'col_sum_err': float(
            (board.sum(dim=-2) - 1.0).abs().mean().detach().cpu()
        ),
    }
