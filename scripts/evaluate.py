"""Evaluate a trained CCEM factor energy on a task's test instances.

Run from the repository root::

    python scripts/evaluate.py <task> [task flags]

``<task>`` is ``nqueens``, ``sat3`` or ``graph_coloring``. The task
flags are forwarded unchanged to ``ccem.tasks.<task>.evaluate.main``;
``python scripts/evaluate.py <task> --help`` lists them. Each evaluator
loads the config's ``checkpoint`` (``--checkpoint`` overrides it), runs
the multi-start projected solver of the ``solver:`` block on the graphs
or formulas of ``test_path`` (for N-Queens on ``eval.num_samples``
random boards, checked against the ``test_path`` solution set), prints
a report and, with ``--output-json PATH``, writes one JSON summary.
"""

import sys
from collections.abc import Callable, Sequence
from typing import Optional

from ccem.tasks.graph_coloring import evaluate as graph_coloring_evaluate
from ccem.tasks.nqueens import evaluate as nqueens_evaluate
from ccem.tasks.sat3 import evaluate as sat3_evaluate

TaskMain = Callable[[Optional[Sequence[str]]], None]

TASKS: dict[str, TaskMain] = {
    'nqueens': nqueens_evaluate.main,
    'sat3': sat3_evaluate.main,
    'graph_coloring': graph_coloring_evaluate.main,
}

TASK_LIST = '\n'.join(f'  {name}' for name in TASKS)
USAGE = (
    'usage: python scripts/evaluate.py <task> [task flags]\n'
    '\n'
    'Evaluate the CCEM factor energy of one task. <task> is one of:\n'
    f'{TASK_LIST}\n'
    '\n'
    'python scripts/evaluate.py <task> --help lists the task flags.'
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Dispatch the arguments to the evaluator named by the first one.

    Args:
        argv: Command-line arguments without the program name; ``None``
            reads ``sys.argv[1:]``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ('-h', '--help'):
        print(USAGE)
        return
    if not args:
        sys.exit(f'{USAGE}\n\nerror: no task given')
    task, task_args = args[0], args[1:]
    if task not in TASKS:
        sys.exit(f'{USAGE}\n\nerror: unknown task {task!r}')
    # argparse names the task parser after sys.argv[0]; appending the
    # task makes its usage line read 'evaluate.py nqueens ...'.
    sys.argv[0] = f'{sys.argv[0]} {task}'
    TASKS[task](task_args)


if __name__ == '__main__':
    main()
