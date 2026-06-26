"""T-head for M4. MLP now; swap FastKAN later WITHOUT touching model.py
(same `make_head(in_dim, out_dim)` interface, applied per-region on the last dim)."""

from __future__ import annotations

import torch
import torch.nn as nn

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


# Placeholder so swapping is a one-word config change later.
# class KANHead(nn.Module): ...  # FastKAN with the same (in_dim, out_dim) signature


def make_head(in_dim: int, out_dim: int, head_type: str = config.HEAD_TYPE,
              hidden: int = config.HEAD_HIDDEN, dropout: float = config.HEAD_DROPOUT) -> nn.Module:
    if head_type == "mlp":
        return MLPHead(in_dim, out_dim, hidden, dropout)
    if head_type == "kan":
        raise NotImplementedError("FastKAN head not wired yet — keep HEAD_TYPE='mlp' for now")
    raise ValueError(f"unknown head_type: {head_type}")
