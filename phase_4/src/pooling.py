"""Temporal patch fusion + bbox-guided region pooling for the M4 `tempfuse` arch.

Self-contained (M4 never imports phase_3). Two pieces:
  CrossAttnFuse   — BioViL-T-style: current patches attend the prior grid (soft registration, no
                    explicit alignment) + self-attn + FFN. Fuses BEFORE pooling, so localised
                    change survives (late fusion of pooled vectors would wash it out).
  BBoxRegionPool  — one learnable query per anatomical region, attention-pooling the FUSED current
                    patches restricted to that region's bbox cells. alpha = M4's faithful
                    "where in the region did it change" (exposed for M5).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
import constants as C


def box_grid_disallow(boxes: torch.Tensor, gh: int, gw: int, cell: float) -> torch.Tensor:
    """boxes [B,29,4] in INPUT_RES px -> disallow mask [B,29,gh*gw] bool (True = cannot attend).
    A region with an empty/zero box (absent) is left attending the whole grid so MultiheadAttention
    does not NaN; its pooled vector is unused downstream (present_mask zeroes its loss). Mirrors the
    phase_3 pooling rasterisation exactly so M4 regions match M3 regions."""
    b, r, _ = boxes.shape
    f = boxes.float()
    x1 = (f[..., 0] / cell).floor().clamp(0, gw)
    y1 = (f[..., 1] / cell).floor().clamp(0, gh)
    x2 = (f[..., 2] / cell).ceil().clamp(0, gw)
    y2 = (f[..., 3] / cell).ceil().clamp(0, gh)
    cols = torch.arange(gw, device=boxes.device).view(1, 1, gw)
    rows = torch.arange(gh, device=boxes.device).view(1, 1, gh)
    col_in = (cols >= x1.unsqueeze(-1)) & (cols < x2.unsqueeze(-1))       # [B,R,gw]
    row_in = (rows >= y1.unsqueeze(-1)) & (rows < y2.unsqueeze(-1))       # [B,R,gh]
    allowed = (row_in.unsqueeze(-1) & col_in.unsqueeze(-2)).reshape(b, r, gh * gw)
    empty = allowed.sum(-1) == 0
    allowed = allowed | empty.unsqueeze(-1)
    return ~allowed


class CrossAttnFuse(nn.Module):
    """One temporal block: current <- prior cross-attention, then self-attention, then FFN (pre-LN)."""

    def __init__(self, dim: int, heads: int = config.FUSE_HEADS, dropout: float = config.HEAD_DROPOUT):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln_s = nn.LayerNorm(dim)
        self.self_ = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 2, dim))

    def forward(self, cur: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        q = self.ln_q(cur)
        cur = cur + self.cross(q, self.ln_kv(prior), self.ln_kv(prior), need_weights=False)[0]
        s = self.ln_s(cur)
        cur = cur + self.self_(s, s, s, need_weights=False)[0]
        return cur + self.ffn(self.ln_f(cur))


class BBoxRegionPool(nn.Module):
    """196 fused patches -> 29 region vectors via masked multi-head attention (returns alpha too)."""

    def __init__(self, dim: int, heads: int = config.POOL_HEADS):
        super().__init__()
        self.heads = heads
        self.region_queries = nn.Parameter(torch.randn(C.NUM_REGIONS, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, grid: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """grid [B,196,dim], boxes [B,29,4] -> (region_feats [B,29,dim], alpha [B,29,196])."""
        b = grid.shape[0]
        q = self.region_queries.unsqueeze(0).expand(b, -1, -1)               # [B,29,dim]
        attn_mask = None
        if config.MASK_BBOX:
            cell = config.INPUT_RES / config.GRID_W
            disallow = box_grid_disallow(boxes, config.GRID_H, config.GRID_W, cell)   # [B,29,196]
            attn_mask = disallow.repeat_interleave(self.heads, dim=0)        # (B*heads,29,196)
        pooled, alpha = self.attn(q, grid, grid, attn_mask=attn_mask,
                                  need_weights=True, average_attn_weights=True)
        return self.norm(pooled), alpha
