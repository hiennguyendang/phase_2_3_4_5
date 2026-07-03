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


class _RadialBasis(nn.Module):
    """Gaussian RBF on a fixed grid (FastKAN, Li 2024): phi_g(x) = exp(-((x - c_g)/h)^2).

    Replaces the B-spline basis of the original KAN with cheap Gaussians -> same expressive
    'learnable activation on each edge' idea, but a plain matmul. x[...,D] -> [...,D,G]."""

    def __init__(self, num_grids: int, grid_min: float = -2.0, grid_max: float = 2.0):
        super().__init__()
        self.register_buffer("grid", torch.linspace(grid_min, grid_max, num_grids))
        self.denom = (grid_max - grid_min) / (num_grids - 1)  # RBF width = grid spacing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(((x.unsqueeze(-1) - self.grid) / self.denom) ** 2))


class _FastKANLayer(nn.Module):
    """One FastKAN layer: LayerNorm -> RBF spline (learned edges) + SiLU base residual.

    The LayerNorm keeps activations inside the [-2,2] grid so the basis stays informative."""

    def __init__(self, in_dim: int, out_dim: int, num_grids: int = 8, use_base: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.rbf = _RadialBasis(num_grids)
        self.spline = nn.Linear(in_dim * num_grids, out_dim, bias=False)
        nn.init.trunc_normal_(self.spline.weight, std=0.1 * (in_dim * num_grids) ** -0.5)
        self.use_base = use_base
        if use_base:
            self.base = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spline(self.rbf(self.ln(x)).flatten(-2))     # [...,D,G] -> [...,D*G] -> [...,out]
        if self.use_base:
            y = y + self.base(F.silu(x))
        return y


class KANHead(nn.Module):
    """FastKAN drop-in for MLPHead — SAME (in_dim, out_dim) interface, operates on the last dim
    (region axis is just batched). Two layers (in->hidden->out) so depth matches MLPHead;
    `num_grids` (config.KAN_GRIDS) is the spline resolution. Heavier than the MLP but the same
    call signature, so `HEAD_TYPE='kan'` (or `--head-type kan`) swaps EVERY MLP head — concept,
    mode-A direct-disease, global. The mode-B *faithful* concept->disease head is NOT touched
    (it's not built via make_head), so `--head-type kan --disease-head faithful` gives a KAN
    concept extractor feeding the same masked-non-negative bottleneck."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None,
                 dropout: float | None = None, num_grids: int | None = None):
        super().__init__()
        hidden = config.HEAD_HIDDEN if hidden is None else hidden        # resolve at build time
        dropout = config.CONCEPT_DROPOUT if dropout is None else dropout  # (matches MLPHead)
        g = getattr(config, "KAN_GRIDS", 8) if num_grids is None else num_grids
        self.net = nn.Sequential(
            _FastKANLayer(in_dim, hidden, g),
            nn.Dropout(dropout),
            _FastKANLayer(hidden, out_dim, g),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_head(in_dim: int, out_dim: int, head_type: str | None = None,
              hidden: int | None = None, dropout: float | None = None) -> nn.Module:
    head_type = config.HEAD_TYPE if head_type is None else head_type    # resolve at build time
    if head_type == "mlp":
        return MLPHead(in_dim, out_dim, hidden, dropout)
    if head_type == "kan":
        return KANHead(in_dim, out_dim, hidden, dropout)
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
