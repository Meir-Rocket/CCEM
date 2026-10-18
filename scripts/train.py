"""Train the CCEM factor energy of one task.

Run from the repository root::

    python scripts/train.py <task> [task flags]

``<task>`` is ``nqueens``, ``sat3`` or ``graph_coloring``. The task
flags are forwarded unchanged to ``ccem.tasks.<task>.train.main``;
``python scripts/train.py <task> --help`` lists them. Each trainer
reads the ``training:`` block of its YAML config and writes the trained
factor to the config's ``output_path`` (``--output`` overrides it), a
``*_retrained.pt`` file next to the shipped checkpoint.
"""

import sys
from collections.abc import Callable, Sequence
from typing import Optional

from ccem.tasks.graph_coloring import train as graph_coloring_train
from ccem.tasks.nqueens import train as nqueens_train
from ccem.tasks.sat3 import train as sat3_train

TaskMain = Callable[[Optional[Sequence[str]]], None]

TASKS: dict[str, TaskMain] = {
    'nqueens': nqueens_train.main,
    'sat3': sat3_train.main,
    'graph_coloring': graph_coloring_train.main,
}

TASK_LIST = '\n'.join(f'  {name}' for name in TASKS)
USAGE = (
    'usage: python scripts/train.py <task> [task flags]\n'
    '\n'
    'Train the CCEM factor energy of one task. <task> is one of:\n'
    f'{TASK_LIST}\n'
    '\n'
    'python scripts/train.py <task> --help lists the task flags.'
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Dispatch the arguments to the trainer named by the first one.

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
    # task makes its usage line read 'train.py nqueens ...'.
    sys.argv[0] = f'{sys.argv[0]} {task}'
    TASKS[task](task_args)


if __name__ == '__main__':
    main()
