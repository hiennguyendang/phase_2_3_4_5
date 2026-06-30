"""Attention pooling: the 196x C BioViL-T grid -> 29x C region vectors.

Replaces ROI-pooling. Each of the 29 anatomical regions is a learnable QUERY. By default
(spec 3.1) the query is MASKED to the grid cells covered by that region's bbox, so:
  - alpha (the attention weights) is a *faithful within-region grounding signal* — "inside the
    lung, the model focused on these cells" — not a whole-image saliency soup;
  - a focal finding in a small box is not diluted by the rest of the image.
Set MASK_BBOX=False to fall back to attending the whole grid.

Returns (pooled, alpha): alpha[B,29,196] is exposed for M5 (spec 3.0 / 5 tier-2 "where").
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
import constants as C


def box_grid_disallow(boxes: torch.Tensor, gh: int, gw: int, cell: float) -> torch.Tensor:
    """boxes [B,29,4] in INPUT_RES px -> disallow mask [B,29, gh*gw] bool (True = cannot attend).

    A region with an empty/zero box (absent) is left attending the whole grid (all-allowed) so
    nn.MultiheadAttention does not produce NaN; its pooled vector is unused downstream anyway
    (present_mask zeroes its loss)."""
    b, r, _ = boxes.shape
    f = boxes.float()
    x1 = (f[..., 0] / cell).floor().clamp(0, gw)
    y1 = (f[..., 1] / cell).floor().clamp(0, gh)
    x2 = (f[..., 2] / cell).ceil().clamp(0, gw)
    y2 = (f[..., 3] / cell).ceil().clamp(0, gh)
    cols = torch.arange(gw, device=boxes.device).view(1, 1, gw)
    rows = torch.arange(gh, device=boxes.device).view(1, 1, gh)
    col_in = (cols >= x1.unsqueeze(-1)) & (cols < x2.unsqueeze(-1))   # [B,R,gw]
    row_in = (rows >= y1.unsqueeze(-1)) & (rows < y2.unsqueeze(-1))   # [B,R,gh]
    allowed = (row_in.unsqueeze(-1) & col_in.unsqueeze(-2)).reshape(b, r, gh * gw)
    empty = allowed.sum(-1) == 0                                       # absent / degenerate box
    allowed = allowed | empty.unsqueeze(-1)                            # -> attend everything
    return ~allowed                                                    # disallow = not allowed


class RegionAttentionPool(nn.Module):
    def __init__(self, feat_dim: int, n_heads: int = config.POOL_HEADS,
                 use_global: bool = config.USE_GLOBAL_TOKEN, mask_bbox: bool = config.MASK_BBOX):
        super().__init__()
        self.use_global = use_global
        self.mask_bbox = mask_bbox
        self.n_heads = n_heads
        # one learnable query per anatomical region (conditioned only on identity)
        self.region_queries = nn.Parameter(torch.randn(C.NUM_REGIONS, feat_dim) * 0.02)
        self.attn = nn.MultiheadAttention(feat_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, grid: torch.Tensor, global_vec: torch.Tensor | None = None,
                boxes: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """grid [B,196,C], global_vec [B,C], boxes [B,29,4]
        -> (region_feats [B,29(+1),C], alpha [B,29,196])."""
        b = grid.shape[0]
        q = self.region_queries.unsqueeze(0).expand(b, -1, -1)        # [B,29,C]

        attn_mask = None
        if self.mask_bbox and boxes is not None:
            cell = config.INPUT_RES / config.GRID_W
            disallow = box_grid_disallow(boxes, config.GRID_H, config.GRID_W, cell)  # [B,29,196]
            # nn.MultiheadAttention wants (B*heads, L, S) for a 3-D bool mask
            attn_mask = disallow.repeat_interleave(self.n_heads, dim=0)

        pooled, alpha = self.attn(q, grid, grid, attn_mask=attn_mask,
                                  need_weights=True, average_attn_weights=True)  # [B,29,C],[B,29,196]
        pooled = self.norm(pooled)
        if self.use_global and global_vec is not None:
            pooled = torch.cat([pooled, global_vec.unsqueeze(1)], dim=1)          # [B,30,C]
        return pooled, alpha
