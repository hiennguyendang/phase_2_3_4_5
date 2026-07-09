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
                     gamma: float = config.FOCAL_GAMMA,
                     label_smoothing: float = config.LABEL_SMOOTHING) -> tuple[torch.Tensor, int]:
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
        loss = F.cross_entropy(flat_logits, flat_target, weight=weight,
                               label_smoothing=label_smoothing)
    else:
        raise ValueError(f"unknown loss_type: {loss_type} (choose 'ce' or 'focal')")
    return loss, int(valid.sum())


def flip_consistency_loss(logits: torch.Tensor, flipped_logits: torch.Tensor,
                          region_mask: torch.Tensor,
                          flipped_region_mask: torch.Tensor | None = None,
                          temperature: float = 1.0) -> tuple[torch.Tensor, int]:
    """Symmetric KL for temporal flip consistency.

    A prediction for (current, prior) should match the prediction for (prior, current) after swapping
    improved<->worsened. Stable stays stable. Diseases listed in FLIP_EXCLUDE_DISEASES are skipped.
    """
    t = max(float(temperature), 1e-6)
    flip_idx = torch.tensor(C.FLIP_CLASS_MAP, dtype=torch.long, device=logits.device)
    aligned_flip = flipped_logits.index_select(-1, flip_idx)

    logp = F.log_softmax(logits / t, dim=-1)
    p = logp.exp()
    logq = F.log_softmax(aligned_flip / t, dim=-1)
    q = logq.exp()
    kl_pq = F.kl_div(logp, q, reduction="none").sum(dim=-1)
    kl_qp = F.kl_div(logq, p, reduction="none").sum(dim=-1)
    loss = 0.5 * (kl_pq + kl_qp) * (t * t)

    valid = region_mask.bool()
    if flipped_region_mask is not None:
        valid = valid & flipped_region_mask.bool()
    valid = valid.unsqueeze(-1).expand_as(loss)
    exclude = [C.CHEX_INDEX[n] for n in config.FLIP_EXCLUDE_DISEASES if n in C.CHEX_INDEX]
    if exclude:
        valid[:, :, exclude] = False
    if valid.sum() == 0:
        return logits.sum() * 0.0, 0
    return loss[valid].mean(), int(valid.sum())
