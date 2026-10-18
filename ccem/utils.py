"""Small shared helpers: config loading, seeding, device selection and
IO.

Every train and evaluate entry point in :mod:`ccem.tasks` goes through
these helpers so that seeding, YAML handling, checkpoint loading and
``--set SECTION.KEY=VALUE`` overrides behave identically across tasks.
"""

import contextlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file whose top level is a mapping.

    Args:
        path: File to read.

    Returns:
        The parsed mapping; an empty file yields ``{}``.

    Raises:
        ValueError: If the top-level YAML node is not a mapping.
    """
    with open(path, encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError('YAML config must contain a mapping at the top level')
    return data


def save_json(data: Any, path: str | Path) -> None:
    """Write ``data`` as two-space-indented JSON, creating parent
    directories.

    Keys keep their insertion order (``sort_keys=False``) so emitted
    summaries read in the order the code built them.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write('\n')


def set_seed(seed: int) -> None:
    """Seed ``random``, NumPy, PyTorch and every CUDA device.

    The call sequence is part of the reproducibility contract: the
    entry points build the model and then draw their random starts
    from these streams, so the order of every call after seeding is
    part of the protocol.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_device(name: Optional[str]) -> torch.device:
    """Resolve a device name, falling back to CPU when CUDA is
    unavailable.

    Args:
        name: ``'cpu'``, ``'cuda'``, ``'cuda:1'``, ...; ``None`` means
            CPU.

    Returns:
        The requested device, or CPU (after a one-line notice) when a
        CUDA device was requested on a machine without CUDA.
    """
    requested = str(name or 'cpu')
    if requested.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA requested but not available; falling back to CPU.')
        return torch.device('cpu')
    return torch.device(requested)


def average_metrics(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average per-batch metric dictionaries key by key.

    The keys of the first item define the output; every item must
    contain them. An empty sequence yields ``{}``.
    """
    if not items:
        return {}
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def apply_overrides(
    cfg: dict[str, Any], overrides: Mapping[str, Optional[object]]
) -> None:
    """Set dotted keys such as ``'training.lr'`` in a nested config in
    place.

    ``None`` values are skipped so that unset CLI flags leave the YAML
    value untouched; missing intermediate sections are created.

    Raises:
        ValueError: If a key is empty or has an empty component.
        TypeError: If an intermediate entry exists but is not a mapping.
    """
    for dotted, value in overrides.items():
        if value is None:
            continue
        parts = dotted.split('.')
        if not all(parts):
            raise ValueError(f'invalid config key {dotted!r}')
        *sections, key = parts
        target = cfg
        for section in sections:
            child = target.setdefault(section, {})
            if not isinstance(child, dict):
                raise TypeError(
                    f'config entry {section!r} in {dotted!r} is not a mapping'
                )
            target = child
        target[key] = value


def format_paths(
    cfg: dict[str, Any], keys: Sequence[str], **fields: object
) -> None:
    """Fill ``{name}`` placeholders in the string entries ``keys`` of
    ``cfg`` in place.

    The N-Queens config stores its data and checkpoint paths as
    ``{board_size}`` templates so that ``--board-size N`` selects the
    files of that size; a path without braces, such as one given on the
    command line, passes through unchanged. Keys that are missing from
    ``cfg`` or whose values are not strings are skipped.

    Args:
        cfg: Config mapping, modified in place.
        keys: Entries of ``cfg`` to format.
        **fields: Placeholder values, e.g. ``board_size=8``.
    """
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, str):
            cfg[key] = value.format(**fields)


def parse_set_overrides(values: Optional[Sequence[str]]) -> dict[str, object]:
    """Turn ``--set SECTION.KEY=VALUE`` items into a dotted-key mapping.

    Values are parsed with ``yaml.safe_load`` so ``140`` becomes an
    ``int``, ``1.0e-4`` a ``float``, ``true`` a ``bool`` and ``relu`` a
    ``str``; a value YAML keeps as text but Python reads as a number
    (``1e-4``) becomes a ``float`` too.

    Raises:
        ValueError: If an item is not of the form ``KEY=VALUE`` with a
            non-empty key and value.
    """
    overrides: dict[str, object] = {}
    for item in values or ():
        key, separator, raw_value = item.partition('=')
        key = key.strip()
        if not separator or not key or not raw_value.strip():
            raise ValueError(f'expected SECTION.KEY=VALUE, got {item!r}')
        parsed = yaml.safe_load(raw_value)
        if isinstance(parsed, str):
            with contextlib.suppress(ValueError):
                parsed = float(parsed)
        overrides[key] = parsed
    return overrides


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint payload onto ``device`` with
    ``weights_only=True``.

    Raises:
        ValueError: If the file does not hold a payload dictionary.
    """
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f'checkpoint {path} does not contain a payload dict')
    return payload
