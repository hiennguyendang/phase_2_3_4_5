"""Masked, class-weighted cross-entropy for M4 progression (3 classes).

A (region, disease) cell is supervised only where progression != -100 AND the region is present
(in both current and prior, per dataset). "stable" dominates, so classes are inverse-freq weighted.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import config
import constants as C


def class_weight_from_counts(counts) -> torch.Tensor:
    """Mean-normalized inverse-frequency weight over the 3 progression classes (spec 4.4)."""
    counts = np.clip(np.asarray(counts, dtype=np.float64), 1.0, None)
    w = counts.sum() / (C.NUM_PROG * counts)
    return torch.tensor(w, dtype=torch.float32)


def class_weight(prog_arr) -> torch.Tensor:
    """Inverse-frequency weight straight from a label array (counts the 3 classes, ignores -100)."""
    a = np.asarray(prog_arr).reshape(-1)
    counts = np.array([(a == k).sum() for k in range(C.NUM_PROG)], dtype=np.float64)
    return class_weight_from_counts(counts)


def progression_loss(logits: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor,
                     weight: torch.Tensor | None = None, loss_type: str = "ce",
                     gamma: float = config.FOCAL_GAMMA) -> tuple[torch.Tensor, int]:
    """logits [B,29,14,3], target [B,29,14] in {0,1,2,-100}, region_mask [B,29].
    -> (mean loss over valid cells, n_valid). Returns 0 if nothing valid (keeps batch alive).

    loss_type "ce" = class-weighted cross-entropy (baseline); "focal" = the same weights times the
    focal (1-p_t)^gamma modulator (ablation axis 2: down-weight the easy dominant "stable" cells)."""
    b, r, d, k = logits.shape
    valid = (target != C.UNKNOWN) & region_mask.bool().unsqueeze(-1)     # [B,29,14]
    if valid.sum() == 0:
        return logits.sum() * 0.0, 0
    flat_logits = logits[valid]                      # [M,3]
    flat_target = target[valid]                      # [M]
    if weight is not None:
        weight = weight.to(flat_logits.device)
    if loss_type == "focal":
        logp = F.log_softmax(flat_logits, dim=-1)                        # [M,3]
        ce = F.nll_loss(logp, flat_target, weight=weight, reduction="none")   # [M] (weighted)
        pt = logp.gather(1, flat_target[:, None]).squeeze(1).exp()       # [M]
        loss = ((1.0 - pt) ** gamma * ce).mean()
    elif loss_type == "ce":
        loss = F.cross_entropy(flat_logits, flat_target, weight=weight)
    else:
        raise ValueError(f"unknown loss_type: {loss_type} (choose 'ce' or 'focal')")
    return loss, int(valid.sum())
