"""Projections onto the convex relaxations used by the CCEM solvers.

Discrete assignments are relaxed to probability vectors: one simplex per
variable (graph coloring, 3-SAT) or, for N-Queens, a board in the
Birkhoff polytope of doubly-stochastic matrices. The projections here
map solver iterates back onto those sets after every gradient step. All
guards and the order of operations are part of the reference numerics of
the paper runs.
"""

import functools
from collections.abc import Callable

import torch
from torch import Tensor


def project_simplex(v: Tensor) -> Tensor:
    """Euclidean projection of each last-axis vector onto the
    probability simplex.

    Sort-based algorithm (Duchi et al., 2008): with ``u`` the entries
    sorted in descending order and ``cssv_j = sum_{i <= j} u_i - 1``,
    the active-set size is ``rho = max{j : u_j - cssv_j / j > 0}`` and
    the result is ``max(v - cssv_rho / rho, 0)``. Three guards keep the
    solvers alive on degenerate iterates: non-finite inputs are replaced
    first (NaN to 0, +-inf to +-1e6), rows whose projection has
    non-positive or non-finite mass fall back to the uniform point, and
    the result is rescaled to sum to exactly one. The map is piecewise
    linear and differentiable almost everywhere, so it can sit inside an
    unrolled solver.

    Args:
        v: Tensor ``[..., K]``; every ``[K]`` slice is projected
            independently.

    Returns:
        A tensor shaped like ``v`` whose last-axis slices are
        non-negative and sum to one.
    """
    original_shape = v.shape
    flat = v.reshape(-1, original_shape[-1])
    flat = torch.nan_to_num(flat, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    u, _ = torch.sort(flat, dim=-1, descending=True)
    cssv = torch.cumsum(u, dim=-1) - 1.0
    ind = torch.arange(1, flat.size(-1) + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    rho = cond.sum(dim=-1) - 1
    rho = rho.clamp(min=0, max=flat.size(-1) - 1)
    theta = cssv.gather(dim=1, index=rho.unsqueeze(1)) / (
        rho.to(dtype=v.dtype).unsqueeze(1) + 1.0
    )
    w = torch.clamp(flat - theta, min=0.0)
    row_sum = w.sum(dim=-1, keepdim=True)
    invalid = (~torch.isfinite(row_sum)) | (row_sum <= 0.0)
    if torch.any(invalid):
        uniform = torch.full_like(w, fill_value=1.0 / float(flat.size(-1)))
        w = torch.where(invalid.expand_as(w), uniform, w)
    row_sum = w.sum(dim=-1, keepdim=True).clamp(min=1.0e-12)
    w = w * (1.0 / row_sum)
    return w.reshape(original_shape)


def project_birkhoff(board: Tensor, num_iters: int = 8) -> Tensor:
    """Alternating row/column simplex projections: the N-Queens
    relaxation.

    Each round projects every row onto the simplex and then every
    column. These are projections onto the convex sets
    ``{rows sum to 1, >= 0}`` and ``{columns sum to 1, >= 0}`` whose
    intersection is the Birkhoff polytope of doubly-stochastic matrices.
    The alternating scheme converges to a feasible point but is not the
    exact Euclidean projection (no Dykstra correction). The
    rows-then-columns order and the number of rounds are part of the
    reference numerics; the shipped N-Queens config uses
    ``simplex_iters=50``.

    Args:
        board: Relaxed boards ``[B, N, N]``.
        num_iters: Number of rounds; at least one is always performed.

    Returns:
        Projected boards with the input shape.

    Raises:
        ValueError: If ``board`` is not three-dimensional.
    """
    if board.ndim != 3:
        raise ValueError('board must have shape [B, N, N]')
    projected = board
    for _ in range(max(int(num_iters), 1)):
        projected = project_simplex(projected)
        projected = project_simplex(projected.transpose(-1, -2)).transpose(
            -1, -2
        )
    return projected


def board_projection(simplex_iters: int = 8) -> Callable[[Tensor], Tensor]:
    """Bind ``num_iters`` so solvers can call
    :func:`project_birkhoff` on boards.
    """
    return functools.partial(project_birkhoff, num_iters=int(simplex_iters))


def random_feasible_boards(
    batch_size: int,
    board_size: int,
    device: torch.device,
    simplex_iters: int = 8,
) -> Tensor:
    """Draw ``U[0, 1)`` boards and project them with
    :func:`project_birkhoff`.

    Consumes exactly one ``torch.rand`` call of shape ``[B, N, N]`` from
    the global generator; the evaluation protocol depends on that order.
    """
    return project_birkhoff(
        torch.rand(batch_size, board_size, board_size, device=device),
        num_iters=simplex_iters,
    )
