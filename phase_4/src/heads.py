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


class FaithfulTemporalConceptHead(nn.Module):
    """FTCB head: per-(region, disease) progression from DIRECTED concept deltas (spec Part A section 5).

    Given concept activations c_prior/c_current [B,29,69] and M3 disease logits logit_prior/current
    [B,29,14], with per-concept severity sign s_c and concept->disease mask M[d,c]:

        e_{r,c}   = s_c * (c_cur - c_prior)                      directed evidence (worse if >0)
        z_{r,d}   = sum_c softplus(W_dir[d,c]) * M[d,c] * e_{r,c}     direction score  (>0 -> worsened)
        m_{r,d}   = b_d + sum_c softplus(W_mag[d,c]) * M[d,c] * |e_{r,c}|
                       + softplus(w_logit_d) * |Δlogit_d|            change magnitude
        P(change) = sigmoid(m);  P(worse)=P(change)*sigmoid(z);  P(improve)=P(change)*(1-sigmoid(z))

    Faithful by construction (non-negative masked weights: a mapped concept can only push its own
    disease, and the direction comes from the signed concept delta, not an arbitrary weight). Time
    reversal is exact by construction: swapping prior<->current sends e -> -e, so z -> -z (worse<->improve)
    while |e| and m are unchanged (stable preserved) — no KL regularizer needed. Returns log-probs
    [B,29,14,3] in the shared order [stable, improved, worsened].
    """

    def __init__(self, severity_sign, concept_to_chex, num_chex: int):
        super().__init__()
        n_concept = len(severity_sign)
        mask = torch.zeros(num_chex, n_concept)
        for c, d in enumerate(concept_to_chex):
            if d is not None and d >= 0:
                mask[d, c] = 1.0
        self.register_buffer("mask", mask)                                   # [14,69]
        self.register_buffer("sign", torch.as_tensor(severity_sign, dtype=torch.float32))  # [69]
        self.W_dir = nn.Parameter(torch.zeros(num_chex, n_concept))
        self.W_mag = nn.Parameter(torch.zeros(num_chex, n_concept))
        self.w_logit = nn.Parameter(torch.zeros(num_chex))
        self.b_mag = nn.Parameter(torch.zeros(num_chex))

    def forward(self, c_prior, c_cur, logit_prior, logit_cur, return_contrib: bool = False):
        e = self.sign * (c_cur - c_prior)                                    # [B,29,69]
        w_dir = F.softplus(self.W_dir) * self.mask                           # [14,69] >=0, masked
        w_mag = F.softplus(self.W_mag) * self.mask
        z = torch.einsum("brc,dc->brd", e, w_dir)                            # [B,29,14]
        m = (self.b_mag + torch.einsum("brc,dc->brd", e.abs(), w_mag)
             + F.softplus(self.w_logit) * (logit_cur - logit_prior).abs())   # [B,29,14]
        log_change = F.logsigmoid(m)
        log_stable = F.logsigmoid(-m)
        pw = torch.sigmoid(z)                                                 # P(worsened | change)
        log_wor = log_change + torch.log(pw.clamp(min=1e-6))
        log_imp = log_change + torch.log((1.0 - pw).clamp(min=1e-6))
        out = torch.stack([log_stable, log_imp, log_wor], dim=-1)            # [B,29,14,3]
        if return_contrib:
            # exact per-concept direction contribution: contrib[b,r,d,c] = e * w_dir, sums to z
            contrib = e.unsqueeze(2) * w_dir.unsqueeze(0).unsqueeze(0)        # [B,29,14,69]
            return out, contrib
        return out


_HEADS = {"mlp": MLPHead, "linear": LinearHead, "kan": KANHead}


def make_head(in_dim: int, out_dim: int, head_type: str = config.HEAD_TYPE,
              hidden: int = config.HEAD_HIDDEN, dropout: float = config.HEAD_DROPOUT) -> nn.Module:
    if head_type not in _HEADS:
        raise ValueError(f"unknown head_type: {head_type} (choose from {sorted(_HEADS)})")
    return _HEADS[head_type](in_dim, out_dim, hidden, dropout)
