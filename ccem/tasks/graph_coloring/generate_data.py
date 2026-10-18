"""Random graph generators for graph coloring instances.

The generators produce the random instances of the distributions used
in the paper's graph coloring experiments (Erdos-Renyi, Holme-Kim,
regular expander, Paley and complete graphs). Except for complete
graphs, the stored color budget of an instance is the greedy upper
bound of :func:`~.graphs.greedy_chromatic_upper_bound`. The generators
consume the ``random.Random`` stream in a fixed order so that a seed
reproduces a dataset exactly.

Write a dataset from the repository root with::

    python -m ccem.tasks.graph_coloring.generate_data \\
        --distribution erdos_renyi --count 10 --seed 0 \\
        --output <file.json>

``--node-min``/``--node-max`` bound the node count (ignored by
``paley``, which cycles through fixed primes) and
``--density-min``/``--density-max`` bound the edge probability of
``erdos_renyi`` only; see :func:`generate_distribution`.
"""

import argparse
import math
import random
import warnings
from collections.abc import Callable, Sequence
from typing import Optional

from ccem.tasks.graph_coloring.graphs import (
    GraphInstance,
    greedy_chromatic_upper_bound,
    normalized_edges,
    write_graphs,
)

DEFAULT_NODE_RANGE = (10, 18)
DEFAULT_DENSITY_RANGE = (0.20, 0.45)
HOLME_KIM_ATTACH_FRACTION = 0.1
HOLME_KIM_TRIAD_RANGE = (0.4, 0.8)
REGULAR_EXPANDER_DEGREE_FRACTION = 0.4
PALEY_PRIMES = (5, 13, 17, 29, 37)


def _graph_with_greedy_budget(
    name: str, num_nodes: int, edges: Sequence[tuple[int, int]]
) -> GraphInstance:
    edge_tensor = normalized_edges(edges, num_nodes)
    return GraphInstance(
        name=name,
        num_nodes=num_nodes,
        edges=edge_tensor,
        chromatic_number=greedy_chromatic_upper_bound(num_nodes, edge_tensor),
    )


def erdos_renyi_graph(
    num_nodes: int,
    density: float,
    rng: random.Random,
    name: str = 'erdos_renyi',
) -> GraphInstance:
    """``G(n, p)`` graph: each pair ``(u, v)``, ``u < v``, is an edge
    with probability ``density``.

    Raises:
        ValueError: If ``num_nodes < 2``.
    """
    if num_nodes < 2:
        raise ValueError('num_nodes must be >= 2')
    edges = [
        (left, right)
        for left in range(num_nodes)
        for right in range(left + 1, num_nodes)
        if rng.random() < density
    ]
    return _graph_with_greedy_budget(name, num_nodes, edges)


def complete_graph(num_nodes: int, name: str = 'complete') -> GraphInstance:
    """Complete graph ``K_n`` with the exact chromatic number ``n``.

    Raises:
        ValueError: If ``num_nodes < 2``.
    """
    if num_nodes < 2:
        raise ValueError('num_nodes must be >= 2')
    edges = [
        (left, right)
        for left in range(num_nodes)
        for right in range(left + 1, num_nodes)
    ]
    return GraphInstance(
        name=name,
        num_nodes=num_nodes,
        edges=normalized_edges(edges, num_nodes),
        chromatic_number=num_nodes,
    )


def regular_expander_graph(
    num_nodes: int,
    degree: int,
    rng: random.Random,
    max_attempts: int = 256,
    name: str = 'regular_expander',
) -> GraphInstance:
    """Random regular graph from the configuration model, with an
    Erdos-Renyi fallback.

    Each attempt shuffles ``degree`` stubs per node and pairs them up;
    an attempt fails on a self-loop or a repeated edge. When
    ``max_attempts`` attempts fail (common for the dense degrees that
    :func:`generate_distribution` requests), a warning is emitted and an
    Erdos-Renyi graph of matching expected degree is returned under the
    same name. This fallback is part of the provenance of the shipped
    ``regular_expander.json`` instances.

    Raises:
        ValueError: If ``num_nodes < 2``, ``degree < 1`` or
            ``num_nodes * degree`` is odd.
    """
    if num_nodes < 2:
        raise ValueError('num_nodes must be >= 2')
    if degree < 1:
        raise ValueError('degree must be >= 1')
    if (num_nodes * degree) % 2 != 0:
        raise ValueError('num_nodes * degree must be even for a regular graph')
    target_degree = min(int(degree), num_nodes - 1)
    for _ in range(max_attempts):
        edges: set[tuple[int, int]] = set()
        stubs = [
            node_id
            for node_id in range(num_nodes)
            for _ in range(target_degree)
        ]
        rng.shuffle(stubs)
        valid = True
        while stubs:
            left = stubs.pop()
            right = stubs.pop()
            ordered = (left, right) if left < right else (right, left)
            if left == right or ordered in edges:
                valid = False
                break
            edges.add(ordered)
        if valid:
            return _graph_with_greedy_budget(name, num_nodes, sorted(edges))
    warnings.warn(
        f'configuration model failed {max_attempts} times for {name} '
        f'(n={num_nodes}, degree={target_degree}); using an Erdos-Renyi graph',
        stacklevel=2,
    )
    return erdos_renyi_graph(
        num_nodes, float(target_degree) / float(num_nodes - 1), rng, name=name
    )


def _preferential_target(
    rng: random.Random, weights: Sequence[float], weight_total: float
) -> int:
    """Roulette-wheel draw proportional to ``weights`` (uniform when
    all are zero).
    """
    if weight_total <= 0.0:
        return rng.randint(0, len(weights) - 1)
    pick = rng.random() * weight_total
    cumulative = 0.0
    for candidate, weight in enumerate(weights):
        cumulative += weight
        if pick <= cumulative:
            return candidate
    return len(weights) - 1


def holme_kim_graph(
    num_nodes: int,
    attach_count: int,
    triad_probability: float,
    rng: random.Random,
    name: str = 'holme_kim',
) -> GraphInstance:
    """Holme-Kim growth model: preferential attachment with triad
    closure.

    The first ``attach_count + 1`` nodes form a clique. Every later node
    makes up to ``attach_count`` attachments (at most
    ``8 * attach_count`` draws): a target is drawn proportionally to its
    degree and, with probability ``triad_probability``, one random
    neighbour of that target is attached as well (closing a triangle).
    The chromatic number is the greedy bound.

    Raises:
        ValueError: If ``attach_count < 1`` or
            ``num_nodes <= attach_count``.
    """
    if num_nodes < attach_count + 1:
        raise ValueError('num_nodes must exceed the attach_count seed')
    if attach_count < 1:
        raise ValueError('attach_count must be >= 1')
    edges: set[tuple[int, int]] = set()
    degree = [0] * num_nodes

    def attach(node_a: int, node_b: int) -> None:
        edges.add((min(node_a, node_b), max(node_a, node_b)))
        degree[node_a] += 1
        degree[node_b] += 1

    initial_nodes = attach_count + 1
    for left in range(initial_nodes):
        for right in range(left + 1, initial_nodes):
            attach(left, right)
    for new_node in range(initial_nodes, num_nodes):
        targets: set[int] = set()
        weights = [float(degree[node_id]) for node_id in range(new_node)]
        weight_total = sum(weights)
        attempts = 0
        while len(targets) < attach_count and attempts < attach_count * 8:
            attempts += 1
            target = _preferential_target(rng, weights, weight_total)
            if target in targets:
                continue
            targets.add(target)
            attach(target, new_node)
            if rng.random() < triad_probability:
                neighbours_of_target = [
                    other
                    for other in range(new_node)
                    if other != target
                    and (min(other, target), max(other, target)) in edges
                ]
                rng.shuffle(neighbours_of_target)
                for triad_target in neighbours_of_target:
                    if triad_target in targets:
                        continue
                    targets.add(triad_target)
                    attach(triad_target, new_node)
                    break
    return _graph_with_greedy_budget(name, num_nodes, sorted(edges))


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value < 4:
        return True
    if value % 2 == 0:
        return False
    return all(
        value % divisor != 0 for divisor in range(3, math.isqrt(value) + 1, 2)
    )


def paley_graph(prime: int, name: str = 'paley') -> GraphInstance:
    """Paley graph on ``Z_p``: ``u ~ v`` iff ``v - u`` is a non-zero
    square mod ``p``.

    The chromatic number is the greedy bound.

    Raises:
        ValueError: If ``prime`` is not a prime ``>= 5`` with
            ``p = 1 (mod 4)``.
    """
    if prime < 5:
        raise ValueError('prime must be >= 5')
    if prime % 4 != 1:
        raise ValueError('prime must satisfy p = 1 (mod 4)')
    if not _is_prime(prime):
        raise ValueError('prime argument must be an actual prime')
    squares = {(value * value) % prime for value in range(1, prime)}
    edges = [
        (left, right)
        for left in range(prime)
        for right in range(left + 1, prime)
        if ((right - left) % prime) in squares
    ]
    return _graph_with_greedy_budget(name, prime, edges)


def _sample_erdos_renyi(
    index: int,
    nodes: int,
    rng: random.Random,
    density_range: tuple[float, float],
) -> GraphInstance:
    density = rng.uniform(float(density_range[0]), float(density_range[1]))
    return erdos_renyi_graph(
        nodes, density, rng, name=f'erdos_renyi_{index:02d}'
    )


def _sample_holme_kim(
    index: int,
    nodes: int,
    rng: random.Random,
    density_range: tuple[float, float],
) -> GraphInstance:
    attach_count = max(2, round(nodes * HOLME_KIM_ATTACH_FRACTION))
    triad_probability = float(rng.uniform(*HOLME_KIM_TRIAD_RANGE))
    return holme_kim_graph(
        nodes,
        attach_count,
        triad_probability,
        rng,
        name=f'holme_kim_{index:02d}',
    )


def _sample_regular_expander(
    index: int,
    nodes: int,
    rng: random.Random,
    density_range: tuple[float, float],
) -> GraphInstance:
    degree = max(
        3, min(round(nodes * REGULAR_EXPANDER_DEGREE_FRACTION), nodes - 1)
    )
    if (nodes * degree) % 2 == 1:
        degree -= 1
    return regular_expander_graph(
        nodes, degree, rng, name=f'regular_expander_{index:02d}'
    )


def _sample_complete(
    index: int,
    nodes: int,
    rng: random.Random,
    density_range: tuple[float, float],
) -> GraphInstance:
    return complete_graph(nodes, name=f'complete_{index:02d}')


def _sample_paley(
    index: int,
    nodes: int,
    rng: random.Random,
    density_range: tuple[float, float],
) -> GraphInstance:
    return paley_graph(
        PALEY_PRIMES[index % len(PALEY_PRIMES)], name=f'paley_{index:02d}'
    )


Sampler = Callable[
    [int, int, random.Random, tuple[float, float]], GraphInstance
]

SAMPLERS: dict[str, Sampler] = {
    'erdos_renyi': _sample_erdos_renyi,
    'holme_kim': _sample_holme_kim,
    'regular_expander': _sample_regular_expander,
    'complete': _sample_complete,
    'paley': _sample_paley,
}
DISTRIBUTIONS = tuple(SAMPLERS)


def generate_distribution(
    distribution: str,
    rng: random.Random,
    node_range: tuple[int, int] = DEFAULT_NODE_RANGE,
    density_range: tuple[float, float] = DEFAULT_DENSITY_RANGE,
    count: int = 5,
) -> list[GraphInstance]:
    """Sample ``count`` instances of one named distribution.

    For every instance the node count is drawn uniformly from
    ``node_range`` first (even for ``paley``, which ignores it and
    cycles through :data:`PALEY_PRIMES`). Erdos-Renyi draws its density
    from ``density_range``; Holme-Kim attaches ``max(2, round(0.1 n))``
    nodes with a triad probability in ``[0.4, 0.8]``; the regular
    expander uses degree ``max(3, round(0.4 n))`` made even when needed.
    Instances are named ``<distribution>_<index:02d>``.

    Raises:
        ValueError: For an unknown distribution.
    """
    key = str(distribution).strip().lower()
    if key not in SAMPLERS:
        raise ValueError(f'unknown distribution: {distribution}')
    sampler = SAMPLERS[key]
    graphs: list[GraphInstance] = []
    for index in range(count):
        nodes = rng.randint(int(node_range[0]), int(node_range[1]))
        graphs.append(sampler(index, nodes, rng, density_range))
    return graphs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the dataset generator command line."""
    parser = argparse.ArgumentParser(
        description='Generate graph coloring instances as a JSON file.'
    )
    parser.add_argument(
        '--distribution',
        default='erdos_renyi',
        choices=DISTRIBUTIONS,
        help='distribution to sample (default: %(default)s)',
    )
    parser.add_argument(
        '--count',
        type=int,
        default=10,
        help='instances to generate (default: %(default)s)',
    )
    parser.add_argument(
        '--node-min',
        type=int,
        default=DEFAULT_NODE_RANGE[0],
        help='smallest node count, ignored by paley (default: %(default)s)',
    )
    parser.add_argument(
        '--node-max',
        type=int,
        default=DEFAULT_NODE_RANGE[1],
        help='largest node count, ignored by paley (default: %(default)s)',
    )
    parser.add_argument(
        '--density-min',
        type=float,
        default=DEFAULT_DENSITY_RANGE[0],
        help='smallest edge probability, erdos_renyi only '
        '(default: %(default)s)',
    )
    parser.add_argument(
        '--density-max',
        type=float,
        default=DEFAULT_DENSITY_RANGE[1],
        help='largest edge probability, erdos_renyi only '
        '(default: %(default)s)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random.Random seed (default: %(default)s)',
    )
    parser.add_argument(
        '--output', required=True, help='destination JSON file'
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Generate one distribution with a seeded generator and write it
    as JSON.
    """
    args = parse_args(argv)
    graphs = generate_distribution(
        args.distribution,
        random.Random(args.seed),
        node_range=(args.node_min, args.node_max),
        density_range=(args.density_min, args.density_max),
        count=args.count,
    )
    write_graphs(graphs, args.output)
    print(f'Wrote {len(graphs)} {args.distribution} graphs to {args.output}')


if __name__ == '__main__':
    main()
