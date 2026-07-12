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
                     label_smoothing: float = config.LABEL_SMOOTHING,
                     opposite_penalty_weight: float = 0.0,
                     distance_penalty_weight: float = 0.0,
                     cdw_alpha: float = config.CDW_ALPHA,
                     cdw_weight: float = 0.0) -> tuple[torch.Tensor, int]:
    """logits [B,29,14,3], target [B,29,14] in {0,1,2,-100}, region_mask [B,29].
    -> (mean loss over valid cells, n_valid). Returns 0 if nothing valid (keeps batch alive).

    loss_type "ce" = class-weighted cross-entropy (baseline); "focal" = the same weights times the
    focal (1-p_t)^gamma modulator (ablation axis 2: down-weight the easy dominant "stable" cells);
    "cdw" = Class-Distance-Weighted CE (Polat et al. 2022/2024), a full CE replacement that penalizes
    each wrong class by its ordinal distance^alpha: L = -sum_{i!=c} log(1-p_i)*|i-c|^alpha. With
    improved<stable<worsened and alpha=5, an improved<->worsened error (dist 2) costs 2^5=32x an
    adjacent-to-stable error (dist 1). Honors the same class weight (on the true class) as ce/focal.

    Optional ordinal-safety regularizers (add-ons, usually left off with cdw which already encodes it):
      * opposite_penalty_weight discourages improved<->worsened reversals without pushing stable.
      * distance_penalty_weight uses improved < stable < worsened distance, so opposite-direction
        errors cost more than adjacent-to-stable errors.
    """
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
    elif loss_type == "cdw":
        # Class-Distance-Weighted CE. Ordinal axis improved < stable < worsened via `coords`; the
        # distance to the true class becomes an exponential penalty on every other class.
        probs_cdw = F.softmax(flat_logits, dim=-1)                       # [M,3]
        coords = torch.tensor([0.0, -1.0, 1.0], dtype=probs_cdw.dtype, device=probs_cdw.device)
        dist = (coords[flat_target, None] - coords[None, :]).abs()       # [M,3] ordinal distance
        w_dist = dist.pow(float(cdw_alpha))                             # |i-c|^alpha (0 at i==c)
        penalty = -(torch.log1p(-probs_cdw.clamp(max=1 - 1e-6)) * w_dist).sum(dim=-1)   # [M]
        if weight is not None:
            penalty = penalty * weight[flat_target]                     # same class weight as ce/focal
        loss = penalty.mean()
    else:
        raise ValueError(f"unknown loss_type: {loss_type} (choose 'ce', 'focal', or 'cdw')")
    probs = None
    if opposite_penalty_weight > 0:
        probs = F.softmax(flat_logits, dim=-1)
        opp_idx = torch.full_like(flat_target, -1)
        opp_idx = torch.where(flat_target == C.PROG_INDEX["improved"],
                              torch.full_like(opp_idx, C.PROG_INDEX["worsened"]), opp_idx)
        opp_idx = torch.where(flat_target == C.PROG_INDEX["worsened"],
                              torch.full_like(opp_idx, C.PROG_INDEX["improved"]), opp_idx)
        has_opp = opp_idx >= 0
        if has_opp.any():
            p_opp = probs[has_opp].gather(1, opp_idx[has_opp, None]).squeeze(1)
            loss = loss + float(opposite_penalty_weight) * (-torch.log1p(-p_opp.clamp(max=1 - 1e-6))).mean()
    if distance_penalty_weight > 0:
        if probs is None:
            probs = F.softmax(flat_logits, dim=-1)
        # Class ids are [stable, improved, worsened], but the ordinal axis is improved < stable < worsened.
        coords = torch.tensor([0.0, -1.0, 1.0], dtype=probs.dtype, device=probs.device)
        dist = (coords[flat_target, None] - coords[None, :]).abs()
        loss = loss + float(distance_penalty_weight) * (probs * dist).sum(dim=-1).mean()
    if cdw_weight > 0:
        # Hybrid CE + lambda*CDW-CE: keep CE's "get the right class" pull, ADD the exponential
        # distance penalty as a safety term. Sweeping cdw_weight traces the safety-sensitivity frontier
        # (pure --loss cdw collapses to stable; this add-on avoids that — see ORDAC 2025).
        if probs is None:
            probs = F.softmax(flat_logits, dim=-1)
        coords = torch.tensor([0.0, -1.0, 1.0], dtype=probs.dtype, device=probs.device)
        w_dist = (coords[flat_target, None] - coords[None, :]).abs().pow(float(cdw_alpha))
        cdw_pen = -(torch.log1p(-probs.clamp(max=1 - 1e-6)) * w_dist).sum(dim=-1).mean()
        loss = loss + float(cdw_weight) * cdw_pen
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
