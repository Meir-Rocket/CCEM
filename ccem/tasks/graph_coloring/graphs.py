"""Graph instances, the JSON dataset format, the greedy decoder and the
conflict count.

``greedy_color_decode`` turns a relaxed assignment ``soft`` of shape
``[N, K]`` (one probability vector per node) into an integer coloring
and ``count_conflicting_edges`` scores it. The greedy chromatic upper
bound is the stored color budget of every shipped file except
``regular_expander_2.json`` (certified chi = 3); see
``GraphInstance.chromatic_number``.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GraphInstance:
    """Immutable graph coloring instance.

    Attributes:
        name: Instance name (also the JSON ``name`` key).
        num_nodes: Number of nodes ``N``; nodes are ``0 .. N-1``.
        edges: Long tensor ``[E, 2]`` in the canonical form of
            :func:`normalized_edges` (``u < v``, sorted, no duplicates
            or self-loops).
        chromatic_number: The stored color budget used at evaluation
            time: decoding is restricted to the first
            ``chromatic_number`` colors. For the random distributions
            this is the greedy upper bound of
            :func:`greedy_chromatic_upper_bound`, not an exact
            chromatic number; it is exact for complete graphs and a
            certified chi = 3 for the shipped
            ``regular_expander_2.json``.
    """

    name: str
    num_nodes: int
    edges: Tensor
    chromatic_number: int

    def num_edges(self) -> int:
        """Number of edges ``E``."""
        return int(self.edges.size(0))


def normalized_edges(edges: Iterable[Sequence[int]], num_nodes: int) -> Tensor:
    """Canonicalise an edge list into the stored ``[E, 2]`` long tensor.

    Self-loops are dropped, every pair is ordered as ``u < v``,
    duplicates are removed and the result is sorted lexicographically.
    This canonical form defines the edge order of every shipped JSON
    file.

    Raises:
        ValueError: If an item is not a pair or an endpoint is out of
            range.
    """
    deduped: set[tuple[int, int]] = set()
    for raw_pair in edges:
        if len(raw_pair) != 2:
            raise ValueError('edges must be (u, v) pairs')
        u, v = int(raw_pair[0]), int(raw_pair[1])
        if u == v:
            continue
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            raise ValueError(
                f'edge ({u}, {v}) out of range for {num_nodes} nodes'
            )
        if u > v:
            u, v = v, u
        deduped.add((u, v))
    if not deduped:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(sorted(deduped), dtype=torch.long)


def _adjacency_lists(num_nodes: int, edges: Tensor) -> list[list[int]]:
    neighbours: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges.tolist():
        neighbours[u].append(v)
        neighbours[v].append(u)
    return neighbours


def greedy_chromatic_upper_bound(num_nodes: int, edges: Tensor) -> int:
    """Greedy upper bound on the chromatic number (the stored color
    budget).

    Nodes are visited in order of decreasing degree (stable sort) and
    each receives the smallest color not used by an already colored
    neighbour. The tie-breaking is part of the dataset provenance: it
    produced the ``chromatic_number`` field of the shipped
    random-distribution files (except ``regular_expander_2.json``,
    whose certified chi = 3 is stored instead).

    Returns:
        The number of colors used, at least 1.
    """
    neighbours = _adjacency_lists(num_nodes, edges)
    degrees = [len(adjacent) for adjacent in neighbours]
    order = sorted(range(num_nodes), key=lambda node_id: -degrees[node_id])
    colors = [-1] * num_nodes
    for node_id in order:
        forbidden = {
            colors[neighbour]
            for neighbour in neighbours[node_id]
            if colors[neighbour] >= 0
        }
        chosen = 0
        while chosen in forbidden:
            chosen += 1
        colors[node_id] = chosen
    return max(colors) + 1


def greedy_color_decode(graph: GraphInstance, soft: Tensor) -> Tensor:
    """Constraint-aware greedy decoding of a relaxed assignment.

    Nodes are visited in descending order of their largest soft score.
    Each node takes its highest-scoring color that no already colored
    neighbour uses; when every color conflicts it keeps its argmax
    color (the conflict is then counted by
    :func:`count_conflicting_edges`), so every node receives a color in
    ``0 .. K-1``. The evaluator passes the first ``chromatic_number``
    columns of the relaxed solution, which restricts decoding to the
    stored color budget.

    Args:
        graph: The instance whose edges define the conflicts.
        soft: Relaxed assignment ``[N, K]``.

    Returns:
        Long tensor ``[N]`` of colors.

    Raises:
        ValueError: If ``soft`` is not ``[graph.num_nodes, K]``.
    """
    if soft.ndim != 2 or soft.size(0) != graph.num_nodes:
        raise ValueError('soft must have shape [num_nodes, num_colors]')
    neighbours = _adjacency_lists(graph.num_nodes, graph.edges)
    coloring = [-1] * graph.num_nodes
    node_order = torch.argsort(soft.max(dim=-1).values, descending=True)
    for node_id in node_order.tolist():
        forbidden = {
            coloring[neighbour]
            for neighbour in neighbours[node_id]
            if coloring[neighbour] >= 0
        }
        chosen: Optional[int] = None
        for color in torch.argsort(soft[node_id], descending=True).tolist():
            if color in forbidden:
                continue
            chosen = color
            break
        if chosen is None:
            chosen = int(torch.argmax(soft[node_id]).item())
        coloring[node_id] = chosen
    return torch.tensor(coloring, dtype=torch.long)


def count_conflicting_edges(graph: GraphInstance, coloring: Tensor) -> int:
    """Number of monochromatic edges of an integer coloring ``[N]``.

    Raises:
        ValueError: If ``coloring`` does not hold one color per node.
    """
    if coloring.ndim != 1 or coloring.size(0) != graph.num_nodes:
        raise ValueError('coloring must have shape [num_nodes]')
    if graph.edges.numel() == 0:
        return 0
    left_colors = coloring[graph.edges[:, 0]]
    right_colors = coloring[graph.edges[:, 1]]
    return int((left_colors == right_colors).sum().item())


def _graph_to_dict(graph: GraphInstance) -> dict[str, Any]:
    return {
        'name': graph.name,
        'num_nodes': graph.num_nodes,
        'chromatic_number': graph.chromatic_number,
        'edges': [[int(u), int(v)] for u, v in graph.edges.tolist()],
    }


def _graph_from_dict(payload: Mapping[str, Any]) -> GraphInstance:
    if 'num_nodes' not in payload or 'edges' not in payload:
        raise ValueError('payload must contain num_nodes and edges')
    num_nodes = int(payload['num_nodes'])
    return GraphInstance(
        name=str(payload.get('name', 'graph')),
        num_nodes=num_nodes,
        edges=normalized_edges(payload['edges'], num_nodes),
        chromatic_number=int(payload.get('chromatic_number', 0)),
    )


def write_graphs(
    graphs: Sequence[GraphInstance], file_path: str | Path
) -> None:
    """Write instances as a JSON list of
    ``{name, num_nodes, chromatic_number, edges}``.

    The serialisation (``indent=2``, no trailing newline) is the one
    that produced the shipped files under ``data/graph_coloring``.
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'w', encoding='utf-8') as handle:
        json.dump(
            [_graph_to_dict(graph) for graph in graphs], handle, indent=2
        )


def read_graphs(file_path: str | Path) -> list[GraphInstance]:
    """Read a JSON list written by :func:`write_graphs`.

    Raises:
        ValueError: If the file does not hold a JSON list of instances.
    """
    with open(file_path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError('graph file must contain a JSON list')
    return [_graph_from_dict(item) for item in payload]
