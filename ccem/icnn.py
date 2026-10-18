"""The partially input-convex network (Amos et al., 2017) that CCEM
uses as its factor energy.

One :class:`PICNN` scores a local constraint pattern ``y`` given its
context ``u`` and is convex in ``y``. The energy of a whole instance is
the sum of this factor over all of its constraints, so it is convex on
the relaxed feasible set and a projected first-order solver can find
its minimiser. The parameter names are baked into the shipped
checkpoints (listed in the class docstrings); renaming any of them
breaks loading with ``strict=True``.
"""

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as f
from torch import Tensor, nn


class PositiveLinear(nn.Module):
    """Linear layer whose effective weights are strictly positive.

    The weight is ``softplus(raw_weight)``, so it stays positive for any
    value of the free parameter. Non-negative weights on the convex path
    are what make the PICNN recurrence below convex in its input.
    Parameters: ``raw_weight`` and ``bias`` (checkpoint keys
    ``*.raw_weight``/``*.bias``).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw ``raw_weight`` around -4 (weights near 0.018) and a
        small bias.
        """
        nn.init.normal_(self.raw_weight, mean=-4.0, std=0.1)
        bound = 1.0 / math.sqrt(max(self.in_features, 1))
        nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> Tensor:
        """The positive effective weight ``softplus(raw_weight)``."""
        return f.softplus(self.raw_weight)

    def forward(self, x: Tensor) -> Tensor:
        return f.linear(x, self.weight, self.bias)


def build_activation(name: str) -> nn.Module:
    """Return the convex, non-decreasing activation called ``name``.

    ``'relu'`` and ``'softplus'`` are supported; both are convex and
    non-decreasing, which the convexity argument of :class:`PICNN`
    requires. The modules hold no parameters, so they never appear in a
    state dict.

    Raises:
        ValueError: For any other name.
    """
    key = str(name).lower()
    if key == 'relu':
        return nn.ReLU()
    if key == 'softplus':
        return nn.Softplus()
    raise ValueError(f'Unsupported activation: {name}')


class PICNN(nn.Module):
    """Partially input-convex network ``E(u, y)``, convex in ``y`` for
    fixed ``u``.

    Following Amos et al. (2017, Eq. 3) with ``u_0 = context`` and no
    ``z_0``::

        u_{k+1} = act(W_u^(k) u_k + b_u^(k))
        z_{k+1} = act( W_y^(k) (y   *          G_y^(k) u_{k+1})
                     + W_uz^(k) u_{k+1} + W_z^(k) (z_k *
                     softplus(G_z^(k) u_{k+1})) + b^(k) )
        E(u, y) = W_fy (y * G_fy u_K) + W_fz (z_K * softplus(G_fz u_K))
                  + W_fu u_K

    ``W_z^(k)`` and ``W_fz`` are :class:`PositiveLinear` layers and the
    gates on the ``z`` path pass through ``softplus``, so the
    coefficients that multiply ``z`` are non-negative; the gates on
    ``y`` are unconstrained because ``y`` enters affinely. With a
    convex, non-decreasing ``act`` every ``z_k`` is therefore convex in
    ``y`` and so is ``E``. The additions are performed in the order
    written above. Children (frozen by the shipped checkpoints):
    ``u_layers`` (``W_u``), ``y_layers`` (``W_y``), ``z_layers``
    (``W_z``), ``u_to_z_layers`` (``W_uz``), ``y_gate_layers``
    (``G_y``), ``z_gate_layers`` (``G_z``), ``final_y``, ``final_u``,
    ``final_y_gate``, ``final_z``, ``final_z_gate``.

    Args:
        context_dim: Size of the context vector ``u_0``.
        y_dim: Size of the convex input ``y``.
        hidden_dims: Widths of the ``K`` hidden layers (non-empty).
        activation: ``'relu'`` or ``'softplus'``, used on both paths.
    """

    def __init__(
        self,
        context_dim: int,
        y_dim: int,
        hidden_dims: Sequence[int],
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError('hidden_dims must be non-empty')
        self.activation = build_activation(activation)
        self.u_layers = nn.ModuleList()
        self.y_layers = nn.ModuleList()
        self.z_layers = nn.ModuleList()
        self.u_to_z_layers = nn.ModuleList()
        self.y_gate_layers = nn.ModuleList()
        self.z_gate_layers = nn.ModuleList()

        prev_u = int(context_dim)
        prev_z: Optional[int] = None
        for configured_width in hidden_dims:
            hidden_dim = int(configured_width)
            self.u_layers.append(nn.Linear(prev_u, hidden_dim))
            self.y_layers.append(nn.Linear(int(y_dim), hidden_dim))
            self.u_to_z_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.y_gate_layers.append(nn.Linear(hidden_dim, int(y_dim)))
            if prev_z is not None:
                self.z_layers.append(PositiveLinear(prev_z, hidden_dim))
                self.z_gate_layers.append(nn.Linear(hidden_dim, prev_z))
            prev_u = hidden_dim
            prev_z = hidden_dim

        last_hidden = int(hidden_dims[-1])
        self.final_y = nn.Linear(int(y_dim), 1)
        self.final_u = nn.Linear(last_hidden, 1)
        self.final_y_gate = nn.Linear(last_hidden, int(y_dim))
        self.final_z = PositiveLinear(last_hidden, 1)
        self.final_z_gate = nn.Linear(last_hidden, last_hidden)

    def forward(self, context: Tensor, y: Tensor) -> Tensor:
        u = context
        z: Optional[Tensor] = None
        for index, u_layer in enumerate(self.u_layers):
            u = self.activation(u_layer(u))
            y_term = self.y_layers[index](y * self.y_gate_layers[index](u))
            pre_activation = y_term + self.u_to_z_layers[index](u)
            if z is not None:
                z_term = self.z_layers[index - 1](
                    z * f.softplus(self.z_gate_layers[index - 1](u))
                )
                pre_activation = pre_activation + z_term
            z = self.activation(pre_activation)
        assert z is not None
        out: Tensor = self.final_y(y * self.final_y_gate(u))
        out = out + self.final_z(z * f.softplus(self.final_z_gate(u)))
        out = out + self.final_u(u)
        return out.squeeze(-1)
