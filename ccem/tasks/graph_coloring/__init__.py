"""Graph coloring with a composed per-edge convex factor.

A graph with ``N`` nodes is relaxed to one probability vector per node
over ``K`` colors. One convex factor ``E_theta([y_u, y_v])`` scores
every edge and the composed energy is the sum over all edges
(:mod:`.energy`). Graph instances, the greedy decoder and the JSON
format live in :mod:`.graphs`, the contrastive edge dataset in
:mod:`.data` and the random graph generators in :mod:`.generate_data`.
Entry points, run from the repository root:
``python scripts/train.py graph_coloring``,
``python scripts/evaluate.py graph_coloring`` and, for new instances,
``python -m ccem.tasks.graph_coloring.generate_data``.
"""
