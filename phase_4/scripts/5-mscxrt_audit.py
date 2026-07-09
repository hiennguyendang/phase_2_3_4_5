"""Evaluate an M4 checkpoint on MS-CXR-T image-level temporal labels.

MS-CXR-T labels five findings at image-pair level, while VERA M4 predicts per-region progression.
This audit reports several transparent region-to-image aggregations instead of pretending the
mapping is unique.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_4/src

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import collate, move_batch
from eval import build_from_ckpt, multiclass_f1
from mscxrt import FINDINGS, MSCXRTDataset, aggregate_mscxrt_probs, read_mscxrt_rows


def _confusion(pred: np.ndarray, tgt: np.ndarray) -> list[list[int]]:
    mat = np.zeros((C.NUM_PROG, C.NUM_PROG), dtype=np.int64)
    for t, p in zip(tgt.astype(np.int64), pred.astype(np.int64)):
        if 0 <= t < C.NUM_PROG and 0 <= p < C.NUM_PROG:
            mat[t, p] += 1
    return mat.tolist()


def _metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    valid = tgt != C.UNKNOWN
    flat_pred = pred[valid]
    flat_tgt = tgt[valid]
    if flat_tgt.size == 0:
        return {"prog_f1_macro": float("nan"), "change_f1_macro": float("nan"),
                "per_class": {}, "per_finding": {}, "n_valid": 0}
    macro, per, change = multiclass_f1(flat_pred, flat_tgt)
    out = {
        "prog_f1_macro": macro,
        "change_f1_macro": change,
        "per_class": per,
        "n_valid": int(flat_tgt.size),
        "confusion": {"labels": C.PROG_NAMES, "matrix_true_by_pred": _confusion(flat_pred, flat_tgt)},
        "per_finding": {},
    }
    for j, finding in enumerate(FINDINGS):
        m = valid[:, j]
        if not m.any():
            continue
        f_macro, f_per, f_change = multiclass_f1(pred[m, j], tgt[m, j])
        out["per_finding"][finding] = {
            "n": int(m.sum()),
            "macro_f1": f_macro,
            "change_f1": f_change,
            "per_class": f_per,
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MS-CXR-T audit for M4 checkpoints")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--csv", type=Path,
                   default=Path("data/MS_CXR_T_temporal_image_classification_v1.0.0.csv"))
    p.add_argument("--region-cache", type=Path, default=config.DEFAULT_REGION_CACHE)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--m3-labels-dir", type=Path, default=config.DEFAULT_M3_LABELS_DIR)
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"],
                   help="subject-hash split for adapter development; use all for external audit")
    p.add_argument("--agg", default="all", choices=["all", "mean", "max", "lse"],
                   help="region-to-image aggregation")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    model = build_from_ckpt(ck, args.device).eval()
    arch = ck.get("arch", "regiondiff")
    tf_input = ck.get("tempfuse_input_mode", "feat")
    ds = MSCXRTDataset(args.csv, arch=arch, m3_labels_dir=args.m3_labels_dir,
                       region_cache=args.region_cache, features_root=args.features_root,
                       split=args.split, box_source=ck.get("box_source", config.BOX_SOURCE),
                       tempfuse_input_mode=tf_input)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)
    aggs = ["mean", "max", "lse"] if args.agg == "all" else [args.agg]
    pred_by_agg = {a: [] for a in aggs}
    targets = []
    for b in loader:
        logits = model(move_batch(b, args.device))
        for agg in aggs:
            probs = aggregate_mscxrt_probs(logits.cpu(), b["region_mask"], agg)
            pred_by_agg[agg].append(probs.argmax(-1).numpy())
        targets.append(b["target_mscxrt"].numpy())

    tgt = np.concatenate(targets) if targets else np.zeros((0, len(FINDINGS)), dtype=np.int64)
    raw_rows = read_mscxrt_rows(args.csv)
    result = {
        "ckpt": str(args.ckpt),
        "csv": str(args.csv),
        "split": args.split,
        "arch": arch,
        "tempfuse_input_mode": tf_input,
        "coverage": {
            "csv_rows": len(raw_rows),
            "used_pairs": len(ds),
            "skipped": ds.skipped,
            "class_counts_used": {C.PROG_NAMES[i]: int(c) for i, c in enumerate(ds.class_counts())},
        },
        "aggregations": {},
    }
    for agg in aggs:
        pred = np.concatenate(pred_by_agg[agg]) if pred_by_agg[agg] else np.zeros_like(tgt)
        result["aggregations"][agg] = _metrics(pred, tgt)

    print(f"[MS-CXR-T] rows={len(raw_rows):,} used={len(ds):,} skipped={ds.skipped}")
    for agg, res in result["aggregations"].items():
        print(f"  agg={agg:<4} macro-F1={res['prog_f1_macro']:.4f} "
              f"change-F1={res['change_f1_macro']:.4f} n={res['n_valid']:,}")
        for name, val in res["per_class"].items():
            print(f"    {name:<10} {val:.4f}")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[MS-CXR-T] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
