"""Masked-BCE losses (ignore the -100 "not mentioned" sentinel) for M3.

total = λc * concept + λr * region_chexpert + λi * image_chexpert
Concept/region terms are also gated by `present_mask` (no box -> no feature -> no loss).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import config
import constants as C


def masked_bce(logits: torch.Tensor, target: torch.Tensor,
               extra_mask: torch.Tensor | None = None) -> torch.Tensor:
    """BCEWithLogits over elements where target != -100 (and extra_mask is true).
    Returns 0 if nothing is valid (keeps the batch alive)."""
    valid = target != C.UNKNOWN
    if extra_mask is not None:
        valid = valid & extra_mask.bool()
    if valid.sum() == 0:
        return logits.sum() * 0.0
    tgt = target.clamp_min(0).float()             # -100 entries are masked out anyway
    loss = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return loss[valid].mean()


def compute_losses(out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    present = batch["present_mask"]                                  # [B,29]
    li = masked_bce(out["image_disease_logits"], batch["image_chexpert"])
    lr = masked_bce(out["region_disease_logits"], batch["region_chexpert"],
                    extra_mask=present.unsqueeze(-1))
    lc = torch.zeros((), device=li.device)
    if out["concept_logits"] is not None:
        lc = masked_bce(out["concept_logits"], batch["region_concepts"],
                        extra_mask=present.unsqueeze(-1))
    total = (config.LAMBDA_CONCEPT * lc
             + config.LAMBDA_REGION_CHEX * lr
             + config.LAMBDA_IMAGE_CHEX * li)
    return total, {"total": float(total), "concept": float(lc),
                   "region_chex": float(lr), "image_chex": float(li)}
