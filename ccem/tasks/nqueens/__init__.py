"""N-Queens: convex factor energies for rows, columns and diagonals.

A board is relaxed to a doubly-stochastic matrix. Each row, column,
diagonal and anti-diagonal is a factor whose energy is a PICNN
conditioned on the line's descriptor (:mod:`ccem.tasks.nqueens.energy`);
the composed energy is their sum and is minimised with projected Adam
over the Birkhoff polytope (:mod:`ccem.projections`,
:mod:`ccem.solvers`). Entry points, run from the repository root:
``python scripts/train.py nqueens``,
``python scripts/evaluate.py nqueens`` and
``python -m ccem.tasks.nqueens.generate_data``.
"""
