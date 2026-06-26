"""M4 T-KAN: per-region temporal progression from frozen-M3 region tensors (spec 4.1-4.3).

Siamese-by-construction: the shared frozen branch already ran in phase_3/precompute_regions.py, so
here we only consume its cached outputs. Per region the head sees (spec 4.2):
    [feat_curr ; feat_prior ; feat_curr - feat_prior]   (3 * feat_dim)
  + [logit_curr ; logit_prior]                          (2 * 14)
keeping BOTH sides and the difference (no forced sign). Output: 29 x 14 x 3 progression logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
import constants as C
from heads import make_head


def region_in_dim(feat_dim: int) -> int:
    return 3 * feat_dim + 2 * C.NUM_CHEX


class TKAN(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.head = make_head(region_in_dim(feat_dim), C.NUM_CHEX * C.NUM_PROG)

    def forward(self, feat_curr, logit_curr, feat_prior, logit_prior) -> torch.Tensor:
        """all [B,29,*] -> progression logits [B,29,14,3]."""
        diff = feat_curr - feat_prior
        x = torch.cat([feat_curr, feat_prior, diff, logit_curr, logit_prior], dim=-1)  # [B,29,in]
        out = self.head(x)                                                              # [B,29,14*3]
        b, r, _ = out.shape
        return out.view(b, r, C.NUM_CHEX, C.NUM_PROG)


def build_model(feat_dim: int) -> TKAN:
    return TKAN(feat_dim)
