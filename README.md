# Convex Compositional Energy Minimization (CCEM)

Reference implementation of Convex Compositional Energy Minimization
(CCEM) from the NeurIPS 2026 paper
[*Convex Compositional Reasoning Models*](https://arxiv.org/abs/2605.23395).

Compositional energy-based models can generalize to larger combinatorial 
reasoning problems by reusing a learned factor energy across many local 
constraints. We show that a key bottleneck in compositional reasoning is 
not composition itself, but the non-convex geometry of the learned energy 
landscape. We introduce Convex Compositional Energy Minimization (CCEM), 
a framework that parameterizes each factor with an input-convex neural 
network and optimizes the composed energy over a tight convex relaxation 
of the feasible set. Because convexity is preserved under summation, 
the global relaxed objective remains convex, enabling deterministic 
projected first-order optimization. CCEM is trained in two stages: 
factor-level contrastive learning to shape local energy basins, 
followed by end-to-end refinement through an unrolled projected solver.

## Installation

Python 3.10 or newer. PyTorch, NumPy, PyYAML and tqdm are the only
dependencies; a GPU is optional (`--device cpu` works everywhere).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Training & Evaluation

Each trainer reads the `training:` block of its config (the settings of
the shipped checkpoints) and writes a `*_retrained.pt` file next to the
shipped checkpoint, which is never overwritten.

```bash
python scripts/train.py nqueens --board-size 8
python scripts/train.py graph_coloring
python scripts/train.py sat3
```

A trained model is evaluated by passing it explicitly:

```bash
python scripts/evaluate.py nqueens --board-size 8 --checkpoint checkpoints/nqueens/<...>.pt
python scripts/evaluate.py graph_coloring --checkpoint checkpoints/graph_coloring/<...>.pt
python scripts/evaluate.py sat3 --checkpoint checkpoints/sat3/<...>.pt
```

## Data

Everything the entry points read in `data/`:

* `data/nqueens/`: for N = 4 .. 10 the complete solution set
  `{N}_queens_all.txt`, which evaluation checks decoded boards against,
  and the single training board `{N}_queens_train.txt`.
* `data/sat3/`: planted-solution formulas as `.npz` splits, `train.npz`
  and `valid.npz` (4000 and 1000 formulas with 10-20 variables) and
  `test.npz` (100 formulas with 20 variables and 91 clauses).
* `data/graph_coloring/`: eight JSON instance files; every graph stores
  the colour budget (`chromatic_number`) that decoding is restricted to.
  Five graphs each in `erdos_renyi.json` (22-39 nodes), `holme_kim.json`
  (22-32), `regular_expander.json` (21-30), `complete.json` (8-12) and
  `paley.json` (5-37); `erdos_renyi_2.json` and `holme_kim_2.json` are
  larger variants with 84-99 nodes, and `regular_expander_2.json` holds
  100 graphs of 80-100 nodes with a certified chromatic number of 3.

New instances are generated with

```bash
python -m ccem.tasks.nqueens.generate_data --board-size 12 --output-dir data/nqueens
python -m ccem.tasks.sat3.generate_data --seed 1 --output-dir data/sat3_seed1
python -m ccem.tasks.graph_coloring.generate_data --distribution erdos_renyi --count 10 --output graphs.json
```


## Repository layout

```
ccem/
  icnn.py                 PositiveLinear, PICNN
  projections.py          simplex and Birkhoff projections, random feasible starts
  solvers.py              projected GD and projected Adam, SolverConfig
  utils.py                YAML and JSON helpers, seeding, --set overrides
  tasks/nqueens/          board.py, data.py, energy.py, train.py, evaluate.py, generate_data.py
  tasks/graph_coloring/   graphs.py, data.py, energy.py, train.py, evaluate.py, generate_data.py
  tasks/sat3/             dataset.py, metrics.py, energy.py, train.py, evaluate.py, generate_data.py
scripts/                  train.py, evaluate.py
configs/<task>/picnn.yaml model, training and solver settings of each task
checkpoints/<task>/       trained factors
data/<task>/              instances
```


## Citation

```bibtex
@inproceedings{roketlishvili2026ccem,
  title     = {Convex Compositional Reasoning Models},
  author    = {Roketlishvili, Meir and Semenov, Semyon and Bobrin, Maksim and Kovalchuk, Viktor and Baichorov, Albert and Shtanchaev, Abduragim and Karray, Fakhri and Dylov, Dmitry V. and Taka{\v{c}}, Martin and Asadulaev, Arip},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2605.23395},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2605.23395}
}
```

The code, data and checkpoints are released under the MIT License (see
`LICENSE`).
