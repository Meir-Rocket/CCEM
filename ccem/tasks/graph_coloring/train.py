"""Train the graph coloring factor on random contrastive edge samples.

Run from the repository root::

    python scripts/train.py graph_coloring

The per-edge factor is trained with the InfoNCE objective of
:func:`~.energy.train_edge_energy_step` on the samples of
:class:`~.data.EdgeColorDataset` for ``training.local_epochs`` epochs
(Phase 1 of the paper: contrastive training of the local factor); the
epoch with the lowest validation loss is saved to ``output_path``. The
composed energy is never trained on whole graphs: the shipped
checkpoint was trained this way and inference composes the factor over
the edges of an instance.
"""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ccem.tasks.graph_coloring.data import (
    EdgeColorDataset,
    EdgeSample,
    collate_edge_samples,
)
from ccem.tasks.graph_coloring.energy import (
    EdgeColorEnergyModel,
    build_factor_model,
    train_edge_energy_step,
)
from ccem.utils import (
    apply_overrides,
    as_device,
    average_metrics,
    load_yaml,
    parse_set_overrides,
    set_seed,
)


def make_edge_dataset(
    num_colors: int, num_samples: int, training: Mapping[str, Any]
) -> EdgeColorDataset:
    """Build an :class:`~.data.EdgeColorDataset` from the
    ``training:`` block.
    """
    return EdgeColorDataset(
        num_colors=num_colors,
        num_samples=num_samples,
        negatives_per_positive=int(training.get('negatives_per_positive', 4)),
        same_color_fraction=float(training.get('same_color_fraction', 0.5)),
        perturbation_fraction=float(
            training.get('perturbation_fraction', 0.3)
        ),
        swap_fraction=float(training.get('swap_fraction', 0.2)),
    )


def local_epoch(
    loader: DataLoader[EdgeSample],
    model: EdgeColorEnergyModel,
    optimizer: Optional[AdamW],
    device: torch.device,
) -> dict[str, float]:
    """One pass over ``loader``; trains when ``optimizer`` is given.

    Returns:
        The batch metrics of :func:`~.energy.train_edge_energy_step`
        averaged over the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    metrics_all: list[dict[str, float]] = []
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = train_edge_energy_step(model, batch, device)
            if optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            metrics_all.append(metrics)
    return average_metrics(metrics_all)


def save_checkpoint(
    path: Path, cfg: Mapping[str, Any], model: EdgeColorEnergyModel
) -> None:
    """Write the ``{config, model_state, num_colors, model_type}``
    payload read by :func:`~.energy.load_factor_model`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'config': dict(cfg),
            'model_state': model.state_dict(),
            'num_colors': int(cfg['num_colors']),
            'model_type': 'picnn',
        },
        path,
    )


def train_local(
    cfg: Mapping[str, Any],
    model: EdgeColorEnergyModel,
    optimizer: AdamW,
    device: torch.device,
    output_path: Path,
) -> float:
    """``training.local_epochs`` InfoNCE epochs, saving the best
    validation epoch to ``output_path``.

    Returns:
        The best validation loss (``inf`` when no epoch ran).
    """
    training = cfg['training']
    num_colors = int(cfg['num_colors'])
    train_loader: DataLoader[EdgeSample] = DataLoader(
        make_edge_dataset(
            num_colors,
            int(training.get('local_samples_per_epoch', 8192)),
            training,
        ),
        batch_size=int(training['batch_size']),
        shuffle=True,
        collate_fn=collate_edge_samples,
    )
    valid_loader: DataLoader[EdgeSample] = DataLoader(
        make_edge_dataset(
            num_colors,
            int(training.get('local_valid_samples', 1024)),
            training,
        ),
        batch_size=int(training['batch_size']),
        shuffle=False,
        collate_fn=collate_edge_samples,
    )
    best_val = float('inf')
    for epoch in tqdm(range(int(training['local_epochs'])), desc='Phase 1'):
        train_metrics = local_epoch(train_loader, model, optimizer, device)
        valid_metrics = local_epoch(valid_loader, model, None, device)
        train_loss, train_margin = (
            train_metrics['loss'],
            train_metrics['margin'],
        )
        val_loss, val_margin = valid_metrics['loss'], valid_metrics['margin']
        tqdm.write(
            f'[local {epoch:04d}] train_loss={train_loss:.4f} '
            f'train_margin={train_margin:.4f} val_loss={val_loss:.4f} '
            f'val_margin={val_margin:.4f}'
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_path, cfg, model)
    return best_val


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI flags; every explicit flag overrides the config, ``--set``
    last.
    """
    parser = argparse.ArgumentParser(
        description='Train the CCEM graph coloring factor.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/graph_coloring/picnn.yaml',
        help='YAML config (default: %(default)s)',
    )
    parser.add_argument(
        '--num-colors',
        type=int,
        default=None,
        help='palette size K of the factor (num_colors)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='cpu or cuda (config device)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='RNG seed (config seed)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='checkpoint to write (config output_path)',
    )
    parser.add_argument(
        '--local-epochs',
        type=int,
        default=None,
        help='Phase-1 InfoNCE epochs (training.local_epochs)',
    )
    parser.add_argument(
        '--set',
        action='append',
        default=None,
        metavar='SECTION.KEY=VALUE',
        help='override any config entry (repeatable)',
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of ``python scripts/train.py graph_coloring``."""
    args = parse_args(argv)
    cfg = load_yaml(args.config)
    apply_overrides(
        cfg,
        {
            'num_colors': args.num_colors,
            'device': args.device,
            'seed': args.seed,
            'output_path': args.output,
            'training.local_epochs': args.local_epochs,
        },
    )
    apply_overrides(cfg, parse_set_overrides(args.set))
    for key in ('num_colors', 'output_path', 'training'):
        if key not in cfg:
            raise ValueError(
                f'{key} must be set in the config or on the command line'
            )
    set_seed(int(cfg.get('seed', 0)))
    device = as_device(cfg.get('device'))
    model = build_factor_model(cfg['model'], int(cfg['num_colors'])).to(
        device=device
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg['training']['lr']),
        weight_decay=float(cfg['training'].get('weight_decay', 0.0)),
    )
    output_path = Path(cfg['output_path'])
    best_val = train_local(cfg, model, optimizer, device, output_path)
    if best_val < float('inf'):
        print(
            f'Saved best checkpoint to: {output_path} '
            f'(val_loss={best_val:.4f})'
        )


if __name__ == '__main__':
    main()
