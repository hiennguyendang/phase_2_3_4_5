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
from dataset import M4Dataset, collate


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
        logits = model(b["feat_curr"].to(device), b["logit_curr"].to(device),
                       b["feat_prior"].to(device), b["logit_prior"].to(device))   # [B,29,14,3]
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
    p.add_argument("--m3-labels-dir", type=Path, default=config.DEFAULT_M3_LABELS_DIR)
    p.add_argument("--m4-labels-dir", type=Path, default=config.DEFAULT_M4_LABELS_DIR)
    p.add_argument("--pairs", type=Path, default=config.DEFAULT_PAIRS_PATH)
    p.add_argument("--split", default="test")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    import model as M
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    ds = M4Dataset(args.region_cache, args.m3_labels_dir, args.m4_labels_dir, args.pairs, args.split)
    loader = DataLoader(ds, batch_size=args.batch, collate_fn=collate)
    m = M.build_model(ck["feat_dim"]).to(args.device)
    m.load_state_dict(ck["model"])
    res = evaluate(m, loader, args.device)
    print(f"[{args.split}] prog macro-F1 = {res['prog_f1_macro']:.4f}  "
          f"change-only F1 = {res['change_f1_macro']:.4f}  (n={res['n_valid']:,})")
    for k, v in res["per_class"].items():
        print(f"  {k:<10} {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
