"""Audit M4 temporal consistency.

Two deterministic checks:
  1. temporal-swap: f(current, prior) should match f(prior, current) after
     swapping improved <-> worsened while stable stays stable.
  2. identical-image null: f(current, current) should predict stable.

These are safety/provenance audits, not training losses.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_4/src

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import collate, move_batch
from eval import build_from_ckpt, dataset_from_ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M4 temporal-swap and identical-image consistency audit")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--region-cache", type=Path, default=config.DEFAULT_REGION_CACHE)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--m3-labels-dir", type=Path, default=config.DEFAULT_M3_LABELS_DIR)
    p.add_argument("--m4-labels-dir", type=Path, default=config.DEFAULT_M4_LABELS_DIR)
    p.add_argument("--pairs", type=Path, default=config.DEFAULT_PAIRS_PATH)
    p.add_argument("--split", default="test")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-batches", type=int, default=0,
                   help="limit batches for a quick smoke test; 0 = full split")
    return p.parse_args()


def _swap_pair(batch: dict) -> dict:
    out = dict(batch)
    for a, b in (
        ("feat_curr", "feat_prior"),
        ("logit_curr", "logit_prior"),
        ("patch_curr", "patch_prior"),
        ("box_curr", "box_prior"),
        ("concept_curr", "concept_prior"),
    ):
        if a in batch and b in batch:
            out[a], out[b] = batch[b], batch[a]
    if "region_mask_flip" in batch:
        out["region_mask"], out["region_mask_flip"] = batch["region_mask_flip"], batch["region_mask"]
    return out


def _identity_pair(batch: dict) -> dict:
    out = dict(batch)
    for current, prior in (
        ("feat_curr", "feat_prior"),
        ("logit_curr", "logit_prior"),
        ("patch_curr", "patch_prior"),
        ("box_curr", "box_prior"),
        ("concept_curr", "concept_prior"),
    ):
        if current in batch and prior in batch:
            out[prior] = batch[current]
    if "region_mask_flip" in batch:
        out["region_mask_flip"] = batch["region_mask"]
    return out


def _mean_or_nan(x: torch.Tensor) -> float:
    return float(x.float().mean().item()) if x.numel() else float("nan")


def _sum_or_zero(x: torch.Tensor) -> int:
    return int(x.sum().item()) if x.numel() else 0


@torch.no_grad()
def audit(model, loader, device, max_batches: int = 0) -> dict:
    model.eval()
    flip_idx = torch.tensor(C.FLIP_CLASS_MAP, dtype=torch.long, device=device)
    stable = C.PROG_INDEX["stable"]
    improved = C.PROG_INDEX["improved"]
    worsened = C.PROG_INDEX["worsened"]
    exclude = [C.CHEX_INDEX[n] for n in config.FLIP_EXCLUDE_DISEASES if n in C.CHEX_INDEX]

    totals = {
        "swap_valid": 0,
        "swap_match": 0,
        "swap_change_valid": 0,
        "swap_change_match": 0,
        "swap_target_change_valid": 0,
        "swap_target_change_correct": 0,
        "identity_all": 0,
        "identity_all_stable": 0,
        "identity_labeled": 0,
        "identity_labeled_stable": 0,
    }
    kl_sum, kl_n = 0.0, 0
    id_change_prob_sum, id_change_prob_n = 0.0, 0

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        b = move_batch(batch, device)
        logits = model(b)
        logits_swapped = model(move_batch(_swap_pair(batch), device))
        logits_identity = model(move_batch(_identity_pair(batch), device))

        probs = F.softmax(logits, dim=-1)
        probs_swapped = F.softmax(logits_swapped, dim=-1).index_select(-1, flip_idx)
        probs_identity = F.softmax(logits_identity, dim=-1)

        pred = probs.argmax(-1)
        pred_swapped = probs_swapped.argmax(-1)
        pred_identity = probs_identity.argmax(-1)

        target = b["progression"]
        valid = (target != C.UNKNOWN) & b["region_mask"].bool().unsqueeze(-1)
        if "region_mask_flip" in b:
            valid = valid & b["region_mask_flip"].bool().unsqueeze(-1)
        present_all = b["region_mask"].bool().unsqueeze(-1).expand_as(target).clone()
        if exclude:
            valid[:, :, exclude] = False
            present_all[:, :, exclude] = False

        match = (pred == pred_swapped) & valid
        totals["swap_valid"] += _sum_or_zero(valid)
        totals["swap_match"] += _sum_or_zero(match)

        pred_change = ((pred == improved) | (pred == worsened) |
                       (pred_swapped == improved) | (pred_swapped == worsened)) & valid
        totals["swap_change_valid"] += _sum_or_zero(pred_change)
        totals["swap_change_match"] += _sum_or_zero((pred == pred_swapped) & pred_change)

        target_change = ((target == improved) | (target == worsened)) & valid
        flipped_target = torch.where(target == improved, torch.full_like(target, worsened),
                                     torch.where(target == worsened, torch.full_like(target, improved), target))
        raw_swapped_pred = F.softmax(logits_swapped, dim=-1).argmax(-1)
        target_flip_correct = (pred == target) & (raw_swapped_pred == flipped_target) & target_change
        totals["swap_target_change_valid"] += _sum_or_zero(target_change)
        totals["swap_target_change_correct"] += _sum_or_zero(target_flip_correct)

        logp = torch.log(probs.clamp_min(1e-8))
        logq = torch.log(probs_swapped.clamp_min(1e-8))
        sym_kl = 0.5 * (
            F.kl_div(logp, probs_swapped, reduction="none").sum(-1) +
            F.kl_div(logq, probs, reduction="none").sum(-1)
        )
        if valid.any():
            kl_sum += float(sym_kl[valid].sum().item())
            kl_n += int(valid.sum().item())

        id_all = present_all
        id_labeled = valid
        totals["identity_all"] += _sum_or_zero(id_all)
        totals["identity_all_stable"] += _sum_or_zero((pred_identity == stable) & id_all)
        totals["identity_labeled"] += _sum_or_zero(id_labeled)
        totals["identity_labeled_stable"] += _sum_or_zero((pred_identity == stable) & id_labeled)
        if id_all.any():
            change_prob = 1.0 - probs_identity[..., stable]
            id_change_prob_sum += float(change_prob[id_all].sum().item())
            id_change_prob_n += int(id_all.sum().item())

    def ratio(num: str, den: str) -> float:
        return float(totals[num] / totals[den]) if totals[den] else float("nan")

    return {
        "swap_consistency": ratio("swap_match", "swap_valid"),
        "swap_change_consistency": ratio("swap_change_match", "swap_change_valid"),
        "swap_target_change_flip_acc": ratio("swap_target_change_correct", "swap_target_change_valid"),
        "swap_symmetric_kl": float(kl_sum / kl_n) if kl_n else float("nan"),
        "identity_stable_rate_all_present": ratio("identity_all_stable", "identity_all"),
        "identity_stable_rate_labeled": ratio("identity_labeled_stable", "identity_labeled"),
        "identity_change_rate_all_present": 1.0 - ratio("identity_all_stable", "identity_all"),
        "identity_mean_change_prob_all_present": float(id_change_prob_sum / id_change_prob_n) if id_change_prob_n else float("nan"),
        "counts": totals,
    }


def main() -> int:
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    m = build_from_ckpt(ck, args.device).eval()
    ds = dataset_from_ckpt(ck, args.m3_labels_dir, args.m4_labels_dir, args.pairs, args.split,
                           region_cache=args.region_cache, features_root=args.features_root)
    loader_kwargs = {"batch_size": args.batch, "collate_fn": collate, "num_workers": args.workers}
    if args.workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    res = audit(m, DataLoader(ds, **loader_kwargs), args.device, args.max_batches)
    res.update({
        "split": args.split,
        "n_pairs": len(ds),
        "ckpt": str(args.ckpt),
    })
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[diagnostics] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
