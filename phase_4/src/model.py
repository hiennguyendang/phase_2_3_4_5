"""M4 per-region temporal progression. Two architectures, both -> [B,29,14,3] and both taking a
batch dict so train/eval/infer call `model(batch)` uniformly:

  regiondiff (v1)  consume the frozen-M3 REGION cache; per region the head sees a composition of
      feat_curr/feat_prior (region features) + logit_curr/logit_prior (M3 disease logits), selected
      by `input_mode`. Cheap, but M3's static pooling can wash out localised change.

  tempfuse         read frozen M1 PATCH grids (196xC) for curr+prior; cross-attend current<-prior
      (BioViL-T-style soft registration) BEFORE pooling, then M4's OWN bbox-guided region pool ->
      per-region temporal feature -> head. Preserves localised change; pool alpha = faithful "where".

`head_mode` (flat | twostage) and the flat/twostage composition are shared by both archs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import constants as C
from heads import make_head
from pooling import BBoxRegionPool, CrossAttnFuse

# in_dim per input_mode, as a multiple of (feat_dim, NUM_CHEX)  [regiondiff only]
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


def _finish(out: torch.Tensor, head_mode: str) -> torch.Tensor:
    """[B,29,14*3] head output -> [B,29,14,3] progression logits (order: stable, improved, worsened).
    flat = raw 3-way logits; twostage = factorized P(change) x P(direction) returned as summing-to-1
    log-probs (so softmax/argmax unchanged and CE == BCE(change)+CE(direction) exactly)."""
    b, r, _ = out.shape
    out = out.view(b, r, C.NUM_CHEX, C.NUM_PROG)
    if head_mode == "flat":
        return out
    change = out[..., 0]                                       # [B,29,14] change gate
    log_no = F.logsigmoid(-change)                             # log P(stable)
    log_ch = F.logsigmoid(change)                             # log P(change)
    dir_lp = F.log_softmax(out[..., 1:3], dim=-1)             # [B,29,14,2] (improved, worsened)
    return torch.stack([log_no, log_ch + dir_lp[..., 0], log_ch + dir_lp[..., 1]], dim=-1)


class TKAN(nn.Module):
    """regiondiff arch — consumes the frozen-M3 region cache."""

    def __init__(self, feat_dim: int, head_type: str = config.HEAD_TYPE,
                 input_mode: str = config.INPUT_MODE, hidden: int = config.HEAD_HIDDEN,
                 dropout: float = config.HEAD_DROPOUT, head_mode: str = config.HEAD_MODE):
        super().__init__()
        if head_mode not in ("flat", "twostage"):
            raise ValueError(f"unknown head_mode: {head_mode}")
        self.arch = "regiondiff"
        self.feat_dim = feat_dim
        self.input_mode = input_mode
        self.head_mode = head_mode
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

    def forward(self, batch: dict) -> torch.Tensor:
        x = self._compose(batch["feat_curr"], batch["logit_curr"],
                          batch["feat_prior"], batch["logit_prior"])          # [B,29,in]
        return _finish(self.head(x), self.head_mode)


class TempFuseTKAN(nn.Module):
    """tempfuse arch — patch-level cross-attention fusion + M4's own bbox-guided region pool."""

    def __init__(self, feat_dim: int, head_type: str = config.HEAD_TYPE,
                 hidden: int = config.HEAD_HIDDEN, dropout: float = config.HEAD_DROPOUT,
                 head_mode: str = config.HEAD_MODE, fuse_blocks: int = config.FUSE_BLOCKS,
                 fuse_heads: int = config.FUSE_HEADS):
        super().__init__()
        if head_mode not in ("flat", "twostage"):
            raise ValueError(f"unknown head_mode: {head_mode}")
        self.arch = "tempfuse"
        self.feat_dim = feat_dim
        self.head_mode = head_mode
        self.fuse = nn.ModuleList([CrossAttnFuse(feat_dim, fuse_heads, dropout)
                                   for _ in range(fuse_blocks)])
        self.pool = BBoxRegionPool(feat_dim, config.POOL_HEADS)
        self.head = make_head(feat_dim, C.NUM_CHEX * C.NUM_PROG, head_type, hidden, dropout)

    def forward(self, batch: dict, return_alpha: bool = False):
        cur, prior = batch["patch_curr"], batch["patch_prior"]                # [B,196,dim]
        for blk in self.fuse:
            cur = blk(cur, prior)                                             # current<-prior fusion
        region, alpha = self.pool(cur, batch["box_curr"])                     # [B,29,dim],[B,29,196]
        out = _finish(self.head(region), self.head_mode)                      # [B,29,14,3]
        return (out, alpha) if return_alpha else out


def build_model(feat_dim: int, head_type: str = config.HEAD_TYPE,
                input_mode: str = config.INPUT_MODE, hidden: int = config.HEAD_HIDDEN,
                dropout: float = config.HEAD_DROPOUT, head_mode: str = config.HEAD_MODE,
                arch: str = config.ARCH, fuse_blocks: int = config.FUSE_BLOCKS):
    if arch == "regiondiff":
        return TKAN(feat_dim, head_type, input_mode, hidden, dropout, head_mode)
    if arch == "tempfuse":
        return TempFuseTKAN(feat_dim, head_type, hidden, dropout, head_mode, fuse_blocks)
    raise ValueError(f"unknown arch: {arch} (choose 'regiondiff' or 'tempfuse')")
