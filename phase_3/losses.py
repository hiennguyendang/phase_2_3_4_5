"""Masked-BCE losses (ignore the -100 "not mentioned" sentinel) for M3.

total = λc * concept + λr * region_chexpert + λi * image_chexpert
Concept/region terms are also gated by `present_mask` (no box -> no feature -> no loss).
Imbalance (spec 3.6, top priority) handled by RADAR-style log-scale pos_weight.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import config
import constants as C


def pos_weight_logscale(arr, num_classes: int, device) -> torch.Tensor:
    """RADAR-style per-class weight  α_i = log(1 + |D|/pos_i)  (|D| = pos+neg, ignoring -100).
    arr: int array whose last axis is `num_classes`. Classes with no positives -> weight 1."""
    a = np.asarray(arr).reshape(-1, num_classes)
    pos = (a == 1).sum(0).astype(np.float64)
    neg = (a == 0).sum(0).astype(np.float64)
    tot = pos + neg
    w = np.log1p(tot / np.clip(pos, 1.0, None))
    w[pos == 0] = 1.0
    return torch.tensor(w, dtype=torch.float32, device=device)


def masked_bce(logits: torch.Tensor, target: torch.Tensor,
               extra_mask: torch.Tensor | None = None,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """BCEWithLogits over elements where target != -100 (and extra_mask is true).
    Returns 0 if nothing is valid (keeps the batch alive)."""
    valid = target != C.UNKNOWN
    if extra_mask is not None:
        valid = valid & extra_mask.bool()
    if valid.sum() == 0:
        return logits.sum() * 0.0
    tgt = target.clamp_min(0).float()             # -100 entries are masked out anyway
    loss = F.binary_cross_entropy_with_logits(
        logits, tgt, reduction="none",
        pos_weight=pos_weight)                     # pos_weight broadcasts over the last dim
    return loss[valid].mean()


def compute_losses(out: dict, batch: dict, pos_weight: dict | None = None) -> tuple[torch.Tensor, dict]:
    pw = pos_weight or {}
    present = batch["present_mask"]                                  # [B,29]
    li = masked_bce(out["image_disease_logits"], batch["image_chexpert"],
                    pos_weight=pw.get("image"))
    lr = masked_bce(out["region_disease_logits"], batch["region_chexpert"],
                    extra_mask=present.unsqueeze(-1), pos_weight=pw.get("region"))
    lc = torch.zeros((), device=li.device)
    if out["concept_logits"] is not None:
        lc = masked_bce(out["concept_logits"], batch["region_concepts"],
                        extra_mask=present.unsqueeze(-1), pos_weight=pw.get("concept"))
    total = (config.LAMBDA_CONCEPT * lc
             + config.LAMBDA_REGION_CHEX * lr
             + config.LAMBDA_IMAGE_CHEX * li)
    return total, {"total": float(total), "concept": float(lc),
                   "region_chex": float(lr), "image_chex": float(li)}
