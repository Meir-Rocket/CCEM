"""Random 3-SAT with one PICNN clause factor.

A formula over ``N`` variables is relaxed to one 2-simplex row per
variable (``[N, 2]``, class 1 = true). One convex clause factor,
conditioned on the three literal polarities, scores the ``[3, 2]``
block gathered at a clause's variables, and the composed energy is its
sum over the active clauses (:mod:`.energy`). Inference runs projected
gradient descent from random starts, rounds by ``argmax`` and refines
by coordinate descent on the learned energy; the ``.npz`` datasets live
in :mod:`.dataset` and the exact clause/formula metrics in
:mod:`.metrics`. Entry points, run from the repository root:
``python scripts/train.py sat3``, ``python scripts/evaluate.py sat3``
and, for new planted formulas,
``python -m ccem.tasks.sat3.generate_data``.
"""
