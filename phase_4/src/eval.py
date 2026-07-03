"""Metrics for M4: macro-F1 over the 3 progression classes (+ per-class), masked to valid cells.

accuracy ~= "stable" is a red flag (spec 4.4), so we report per-class F1 and a change-only macro
(improved/worsened) alongside the 3-class macro.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import collate, make_dataset, move_batch


def build_from_ckpt(ck: dict, device):
    """Rebuild the exact M4 model a ckpt was trained with (arch / head_type / input_mode / head_mode /
    hidden / dropout), defaulting to the shipping baseline for old ckpts that only stored feat_dim.
    Also restores the prior-present masking so eval scores the same cell set training supervised."""
    import model as M
    config.REQUIRE_PRIOR_PRESENT = ck.get("require_prior_present", config.REQUIRE_PRIOR_PRESENT)
    m = M.build_model(ck["feat_dim"], ck.get("head_type", "mlp"), ck.get("input_mode", "full"),
                      ck.get("hidden", config.HEAD_HIDDEN), ck.get("dropout", config.HEAD_DROPOUT),
                      ck.get("head_mode", "flat"), ck.get("arch", "regiondiff"),
                      ck.get("fuse_blocks", config.FUSE_BLOCKS))
    m.load_state_dict(ck["model"])
    return m.to(device)


def dataset_from_ckpt(ck: dict, m3_labels_dir, m4_labels_dir, pairs, split, *,
                      region_cache, features_root):
    """Build the split dataset MATCHING a ckpt (same arch / same_view / box_source as training)."""
    return make_dataset(ck.get("arch", "regiondiff"), m3_labels_dir, m4_labels_dir, pairs, split,
                        region_cache=region_cache, features_root=features_root,
                        same_view_only=ck.get("same_view", False),
                        box_source=ck.get("box_source", config.BOX_SOURCE))


def multiclass_f1(pred: np.ndarray, tgt: np.ndarray) -> tuple[float, dict, float]:
    """pred/tgt are 1-D class indices (0/1/2). -> (macro-F1, per-class F1, change-only macro-F1)."""
    per = {}
    for k in range(C.NUM_PROG):
        tp = int(((pred == k) & (tgt == k)).sum())
        fp = int(((pred == k) & (tgt != k)).sum())
        fn = int(((pred != k) & (tgt == k)).sum())
        denom = 2 * tp + fp + fn
        per[C.PROG_NAMES[k]] = (2.0 * tp / denom) if denom > 0 else float("nan")
    macro = float(np.nanmean(list(per.values())))
    change = float(np.nanmean([per[C.PROG_NAMES[1]], per[C.PROG_NAMES[2]]]))  # improved+worsened
    return macro, per, change


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, tgts = [], []
    for b in loader:
        logits = model(move_batch(b, device))                                      # [B,29,14,3]
        target = b["progression"]                                                  # [B,29,14]
        valid = (target != C.UNKNOWN) & b["region_mask"].bool().unsqueeze(-1)
        pred = logits.argmax(-1).cpu()
        preds.append(pred[valid].numpy())
        tgts.append(target[valid].numpy())
    pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    tgt = np.concatenate(tgts) if tgts else np.array([], dtype=np.int64)
    if pred.size == 0:
        return {"prog_f1_macro": float("nan"), "per_class": {}, "change_f1_macro": float("nan"), "n_valid": 0}
    macro, per, change = multiclass_f1(pred, tgt)
    return {"prog_f1_macro": macro, "per_class": per, "change_f1_macro": change, "n_valid": int(pred.size)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an M4 checkpoint")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--region-cache", type=Path, default=config.DEFAULT_REGION_CACHE)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--m3-labels-dir", type=Path, default=config.DEFAULT_M3_LABELS_DIR)
    p.add_argument("--m4-labels-dir", type=Path, default=config.DEFAULT_M4_LABELS_DIR)
    p.add_argument("--pairs", type=Path, default=config.DEFAULT_PAIRS_PATH)
    p.add_argument("--split", default="test")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    m = build_from_ckpt(ck, args.device)                    # sets config.REQUIRE_PRIOR_PRESENT first
    ds = dataset_from_ckpt(ck, args.m3_labels_dir, args.m4_labels_dir, args.pairs, args.split,
                           region_cache=args.region_cache, features_root=args.features_root)
    loader = DataLoader(ds, batch_size=args.batch, collate_fn=collate)
    res = evaluate(m, loader, args.device)
    print(f"[{args.split}] prog macro-F1 = {res['prog_f1_macro']:.4f}  "
          f"change-only F1 = {res['change_f1_macro']:.4f}  (n={res['n_valid']:,})")
    for k, v in res["per_class"].items():
        print(f"  {k:<10} {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
