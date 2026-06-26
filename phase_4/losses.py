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
                     weight: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """logits [B,29,14,3], target [B,29,14] in {0,1,2,-100}, region_mask [B,29].
    -> (mean CE over valid cells, n_valid). Returns 0 if nothing valid (keeps batch alive)."""
    b, r, d, k = logits.shape
    valid = (target != C.UNKNOWN) & region_mask.bool().unsqueeze(-1)     # [B,29,14]
    if valid.sum() == 0:
        return logits.sum() * 0.0, 0
    flat_logits = logits[valid]                      # [M,3]
    flat_target = target[valid]                      # [M]
    if weight is not None:
        weight = weight.to(flat_logits.device)
    loss = F.cross_entropy(flat_logits, flat_target, weight=weight)
    return loss, int(valid.sum())
