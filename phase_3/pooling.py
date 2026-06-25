"""Attention pooling: the 196x C BioViL-T grid -> 29x C region vectors.

Replaces ROI-pooling. Each of the 29 anatomical regions is a learnable QUERY that
cross-attends over the WHOLE 14x14 grid (so each region vector already carries global
context -> an explicit global node is usually unnecessary). Optionally appends
BioViL-T's own `projected_global_embedding` as a 30th "global" region (free recall net).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
import constants as C


class RegionAttentionPool(nn.Module):
    def __init__(self, feat_dim: int, n_heads: int = config.POOL_HEADS,
                 use_global: bool = config.USE_GLOBAL_TOKEN):
        super().__init__()
        self.use_global = use_global
        # one learnable query per anatomical region (conditioned only on identity)
        self.region_queries = nn.Parameter(torch.randn(C.NUM_REGIONS, feat_dim) * 0.02)
        self.attn = nn.MultiheadAttention(feat_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, grid: torch.Tensor, global_vec: torch.Tensor | None = None) -> torch.Tensor:
        """grid [B,196,C], global_vec [B,C] -> region_feats [B, 29(+1), C]."""
        b = grid.shape[0]
        q = self.region_queries.unsqueeze(0).expand(b, -1, -1)   # [B,29,C]
        pooled, _ = self.attn(q, grid, grid)                     # [B,29,C]
        pooled = self.norm(pooled)
        if self.use_global and global_vec is not None:
            pooled = torch.cat([pooled, global_vec.unsqueeze(1)], dim=1)  # [B,30,C]
        return pooled
