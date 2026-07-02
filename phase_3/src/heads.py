"""Prediction heads. MLP now; swap FastKAN later WITHOUT touching model.py
(same `make_head(in_dim, out_dim)` interface, applied per-region on the last dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import constants as C


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None,
                 dropout: float | None = None):
        super().__init__()
        hidden = config.HEAD_HIDDEN if hidden is None else hidden       # resolve at build time,
        dropout = config.CONCEPT_DROPOUT if dropout is None else dropout  # not import time (ablations)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # operates on the last dim; region axis is just batched


# Placeholder so swapping is a one-word config change later.
# class KANHead(nn.Module): ...  # FastKAN with the same (in_dim, out_dim) signature


def make_head(in_dim: int, out_dim: int, head_type: str | None = None,
              hidden: int | None = None, dropout: float | None = None) -> nn.Module:
    head_type = config.HEAD_TYPE if head_type is None else head_type    # resolve at build time
    if head_type == "mlp":
        return MLPHead(in_dim, out_dim, hidden, dropout)
    if head_type == "kan":
        raise NotImplementedError("FastKAN head not wired yet — keep HEAD_TYPE='mlp' for now")
    raise ValueError(f"unknown head_type: {head_type}")


class ConceptDiseaseHead(nn.Module):
    """Faithful-CBM concept->disease head (mode B): logit_d = Σ_c softplus(W[d,c])·mask[d,c]·concept_c + b_d.

    Non-negative weights (softplus) + (optionally) a mask to each disease's mapped concepts
    (constants.CHEX_FROM_CONCEPTS) guarantee that raising a mapped concept can only RAISE its
    disease -> the concept-intervention test (spec 3.4) passes *by construction*. This is the
    principled "why"-faithful bottleneck; the free MLP head maximizes accuracy but entangles."""

    def __init__(self, num_concepts: int, num_chex: int, masked: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_chex, num_concepts) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_chex))
        m = torch.zeros(num_chex, num_concepts)
        if masked:
            for xi, cis in C.CHEX_FROM_CONCEPTS.items():
                for ci in cis:
                    m[xi, ci] = 1.0
        else:
            m[:] = 1.0
        self.register_buffer("cmask", m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x [...,num_concepts] concept activations
        w = F.softplus(self.weight) * self.cmask             # >=0, restricted to mapped concepts
        return x @ w.t() + self.bias


def make_disease_head(num_concepts: int, num_chex: int) -> nn.Module:
    """Concept->disease head for mode B, selected by config.DISEASE_HEAD."""
    dh = config.DISEASE_HEAD
    if dh == "mlp":
        return make_head(num_concepts, num_chex)
    if dh == "linear":
        return nn.Linear(num_concepts, num_chex)
    if dh == "nonneg":
        return ConceptDiseaseHead(num_concepts, num_chex, masked=False)
    if dh == "faithful":
        return ConceptDiseaseHead(num_concepts, num_chex, masked=True)
    raise ValueError(f"unknown DISEASE_HEAD: {dh}")
