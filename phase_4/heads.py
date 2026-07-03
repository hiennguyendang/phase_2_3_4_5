"""T-heads for M4, applied per-region on the last dim via one `make_head(in_dim, out_dim)` call.

Three interchangeable heads (ablation axis 0):
  mlp    — 1-hidden-layer GELU MLP (baseline)
  linear — single Linear (no hidden): a linear probe, the honest "is a head needed?" floor
  kan    — self-contained FastKAN (RBF-spline Kolmogorov-Arnold), the "T-KAN" claim itself

FastKAN is implemented inline (no pip dependency) so it runs on the frozen server image: each layer
is LayerNorm -> Gaussian-RBF spline over a fixed grid + a SiLU residual base (ZiyaoLi/fast-kan style).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = config.HEAD_HIDDEN,
                 dropout: float = config.HEAD_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearHead(nn.Module):
    """Linear probe: the floor. If mlp/kan don't beat this, the head is doing nothing."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RBFKANLayer(nn.Module):
    """One FastKAN layer: LayerNorm -> [Gaussian-RBF spline] + SiLU base residual.

    The learnable univariate functions are RBF splines over a FIXED grid of `num_grids` centers in
    [grid_min, grid_max]; LayerNorm keeps activations in that range. Cheap and differentiable — the
    fast approximation to a KAN B-spline layer.
    """

    def __init__(self, in_dim: int, out_dim: int, num_grids: int = config.KAN_GRIDS,
                 grid_min: float = -2.0, grid_max: float = 2.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)                        # [G]
        self.inv_h = (num_grids - 1) / (grid_max - grid_min)      # 1/spacing
        self.spline = nn.Linear(in_dim * num_grids, out_dim)      # combine RBF basis
        self.base = nn.Linear(in_dim, out_dim)                    # SiLU residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # [..., in_dim]
        xn = self.norm(x)
        basis = torch.exp(-((xn.unsqueeze(-1) - self.grid) * self.inv_h) ** 2)  # [...,in_dim,G]
        basis = basis.flatten(-2)                                               # [...,in_dim*G]
        return self.base(F.silu(xn)) + self.spline(basis)


class KANHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = config.HEAD_HIDDEN,
                 dropout: float = config.HEAD_DROPOUT, num_grids: int = config.KAN_GRIDS):
        super().__init__()
        self.l1 = RBFKANLayer(in_dim, hidden, num_grids)
        self.drop = nn.Dropout(dropout)
        self.l2 = RBFKANLayer(hidden, out_dim, num_grids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.drop(self.l1(x)))


_HEADS = {"mlp": MLPHead, "linear": LinearHead, "kan": KANHead}


def make_head(in_dim: int, out_dim: int, head_type: str = config.HEAD_TYPE,
              hidden: int = config.HEAD_HIDDEN, dropout: float = config.HEAD_DROPOUT) -> nn.Module:
    if head_type not in _HEADS:
        raise ValueError(f"unknown head_type: {head_type} (choose from {sorted(_HEADS)})")
    return _HEADS[head_type](in_dim, out_dim, hidden, dropout)
