"""M4 T-KAN: per-region temporal progression from frozen-M3 region tensors (spec 4.1-4.3).

Siamese-by-construction: the shared frozen branch already ran in phase_3/precompute_regions.py, so
here we only consume its cached outputs. Per region the head sees a composition of (spec 4.2):
    feat_curr / feat_prior           (frozen-M3 region features, feat_dim each)
    logit_curr / logit_prior         (frozen-M3 disease logits, 14 each; soft — magnitude matters)
selected by `input_mode` (ablation axis; see config.INPUT_MODE). Output: 29 x 14 x 3 progression
logits regardless of input_mode (only the head's in_dim changes).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
import constants as C
from heads import make_head

# in_dim per input_mode, as a multiple of (feat_dim, NUM_CHEX)
_MODE_DIMS = {
    "full":   (3, 2),   # [c ; p ; c-p ; lc ; lp]
    "concat": (2, 2),   # [c ; p ; lc ; lp]         (no explicit difference)
    "diff":   (1, 1),   # [c-p ; lc-lp]             (pure Siamese difference)
    "logits": (0, 2),   # [lc ; lp]                 (M3 disease logits only)
    "feat":   (3, 0),   # [c ; p ; c-p]             (region features only)
}


def region_in_dim(feat_dim: int, input_mode: str = config.INPUT_MODE) -> int:
    if input_mode not in _MODE_DIMS:
        raise ValueError(f"unknown input_mode: {input_mode} (choose from {sorted(_MODE_DIMS)})")
    nf, nl = _MODE_DIMS[input_mode]
    return nf * feat_dim + nl * C.NUM_CHEX


class TKAN(nn.Module):
    def __init__(self, feat_dim: int, head_type: str = config.HEAD_TYPE,
                 input_mode: str = config.INPUT_MODE, hidden: int = config.HEAD_HIDDEN,
                 dropout: float = config.HEAD_DROPOUT):
        super().__init__()
        self.feat_dim = feat_dim
        self.input_mode = input_mode
        self.head = make_head(region_in_dim(feat_dim, input_mode), C.NUM_CHEX * C.NUM_PROG,
                              head_type, hidden, dropout)

    def _compose(self, fc, lc, fp, lp) -> torch.Tensor:
        m = self.input_mode
        if m == "full":
            return torch.cat([fc, fp, fc - fp, lc, lp], dim=-1)
        if m == "concat":
            return torch.cat([fc, fp, lc, lp], dim=-1)
        if m == "diff":
            return torch.cat([fc - fp, lc - lp], dim=-1)
        if m == "logits":
            return torch.cat([lc, lp], dim=-1)
        if m == "feat":
            return torch.cat([fc, fp, fc - fp], dim=-1)
        raise ValueError(f"unknown input_mode: {m}")

    def forward(self, feat_curr, logit_curr, feat_prior, logit_prior) -> torch.Tensor:
        """all [B,29,*] -> progression logits [B,29,14,3]."""
        x = self._compose(feat_curr, logit_curr, feat_prior, logit_prior)   # [B,29,in]
        out = self.head(x)                                                  # [B,29,14*3]
        b, r, _ = out.shape
        return out.view(b, r, C.NUM_CHEX, C.NUM_PROG)


def build_model(feat_dim: int, head_type: str = config.HEAD_TYPE,
                input_mode: str = config.INPUT_MODE, hidden: int = config.HEAD_HIDDEN,
                dropout: float = config.HEAD_DROPOUT) -> TKAN:
    return TKAN(feat_dim, head_type, input_mode, hidden, dropout)
