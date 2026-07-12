"""Metrics for M4: macro-F1 over the 3 progression classes (+ per-class), masked to valid cells.

accuracy ~= "stable" is a red flag (spec 4.4), so we report per-class F1 and a change-only macro
(improved/worsened) alongside the 3-class macro.
"""

from __future__ import annotations

import argparse
import json
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
                      ck.get("fuse_blocks", config.FUSE_BLOCKS),
                      ck.get("tempfuse_input_mode", "feat"))
    m.load_state_dict(ck["model"])
    return m.to(device)


def dataset_from_ckpt(ck: dict, m3_labels_dir, m4_labels_dir, pairs, split, *,
                      region_cache, features_root):
    """Build the split dataset MATCHING a ckpt (same arch / same_view / box_source as training)."""
    return make_dataset(ck.get("arch", "regiondiff"), m3_labels_dir, m4_labels_dir, pairs, split,
                        region_cache=region_cache, features_root=features_root,
                        same_view_only=ck.get("same_view", False),
                        box_source=ck.get("box_source", config.BOX_SOURCE),
                        tempfuse_input_mode=ck.get("tempfuse_input_mode", "feat"))


def _safe_nanmean(values) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def multiclass_f1(pred: np.ndarray, tgt: np.ndarray) -> tuple[float, dict, float]:
    """pred/tgt are 1-D class indices (0/1/2). -> (macro-F1, per-class F1, change-only macro-F1)."""
    per = {}
    for k in range(C.NUM_PROG):
        tp = int(((pred == k) & (tgt == k)).sum())
        fp = int(((pred == k) & (tgt != k)).sum())
        fn = int(((pred != k) & (tgt == k)).sum())
        denom = 2 * tp + fp + fn
        per[C.PROG_NAMES[k]] = (2.0 * tp / denom) if denom > 0 else float("nan")
    macro = _safe_nanmean(list(per.values()))
    change = _safe_nanmean([per[C.PROG_NAMES[1]], per[C.PROG_NAMES[2]]])  # improved+worsened
    return macro, per, change


def _confusion(pred: np.ndarray, tgt: np.ndarray) -> list[list[int]]:
    mat = np.zeros((C.NUM_PROG, C.NUM_PROG), dtype=np.int64)
    for t, p in zip(tgt.astype(np.int64), pred.astype(np.int64)):
        if 0 <= t < C.NUM_PROG and 0 <= p < C.NUM_PROG:
            mat[t, p] += 1
    return mat.tolist()


# Ordinal rank on the clinical axis improved < stable < worsened.
# Class ids are stable=0, improved=1, worsened=2, so rank-by-id = [1, 0, 2].
_PROG_RANK = np.array([1.0, 0.0, 2.0], dtype=np.float64)


def ordinal_safety(pred: np.ndarray, tgt: np.ndarray) -> dict:
    """Ordinal-safety diagnostics (report alongside F1, spec: adjacent-safe vs opposite-catastrophic).

    - qwk: Quadratic Weighted Kappa on the ordered classes (penalizes improved<->worsened 4x adjacent).
    - opposite_rate: P(pred opposite pole | true is a change class) — the catastrophic-error rate, target ~0.
    - mean_ordinal_distance: mean |rank(pred) - rank(tgt)| on improved<stable<worsened.
    - stable_pred_rate: fraction predicted stable (~majority baseline = collapse red flag).
    """
    if pred.size == 0:
        return {k: float("nan") for k in
                ("qwk", "opposite_rate", "mean_ordinal_distance", "stable_pred_rate")}
    rp, rt = _PROG_RANK[pred], _PROG_RANK[tgt]
    mod = float(np.abs(rp - rt).mean())
    stable_pred_rate = float((pred == C.PROG_INDEX["stable"]).mean())
    imp, wor = C.PROG_INDEX["improved"], C.PROG_INDEX["worsened"]
    n_change = int(((tgt == imp) | (tgt == wor)).sum())
    if n_change > 0:
        opp = int(((tgt == imp) & (pred == wor)).sum() + ((tgt == wor) & (pred == imp)).sum())
        opposite_rate = opp / n_change
    else:
        opposite_rate = float("nan")
    k = C.NUM_PROG
    obs = np.zeros((k, k), dtype=np.float64)
    for t, p in zip(tgt.astype(np.int64), pred.astype(np.int64)):
        obs[t, p] += 1
    w = (_PROG_RANK[:, None] - _PROG_RANK[None, :]) ** 2 / (k - 1) ** 2
    row, col, n = obs.sum(1), obs.sum(0), obs.sum()
    exp = np.outer(row, col) / max(n, 1.0)
    denom = float((w * exp).sum())
    qwk = 1.0 - float((w * obs).sum()) / denom if denom > 0 else float("nan")
    return {"qwk": qwk, "opposite_rate": opposite_rate,
            "mean_ordinal_distance": mod, "stable_pred_rate": stable_pred_rate}


def _slice_f1(pred: np.ndarray, tgt: np.ndarray, keys: np.ndarray, names: list[str]) -> dict:
    """Macro/per-class F1 for each key value, e.g. per disease or per region."""
    out = {}
    for i, name in enumerate(names):
        m = keys == i
        if not m.any():
            continue
        macro, per, change = multiclass_f1(pred[m], tgt[m])
        out[name] = {
            "n": int(m.sum()),
            "macro_f1": macro,
            "change_f1": change,
            "per_class": per,
        }
    return out


@torch.no_grad()
def evaluate(model, loader, device, *, diagnostics: bool = False) -> dict:
    model.eval()
    preds, tgts = [], []
    regs, diseases, same_views = [], [], []
    for b in loader:
        logits = model(move_batch(b, device))                                      # [B,29,14,3]
        target = b["progression"]                                                  # [B,29,14]
        valid = (target != C.UNKNOWN) & b["region_mask"].bool().unsqueeze(-1)
        pred = logits.argmax(-1).cpu()
        preds.append(pred[valid].numpy())
        tgts.append(target[valid].numpy())
        if diagnostics:
            rr, dd = torch.meshgrid(torch.arange(C.NUM_REGIONS), torch.arange(C.NUM_CHEX), indexing="ij")
            rr = rr.unsqueeze(0).expand_as(target)
            dd = dd.unsqueeze(0).expand_as(target)
            sv = b.get("same_view")
            if sv is not None:
                sv = sv.view(-1, 1, 1).expand_as(target)
            regs.append(rr[valid].numpy())
            diseases.append(dd[valid].numpy())
            if sv is not None:
                same_views.append(sv[valid].numpy().astype(np.int64))
    pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    tgt = np.concatenate(tgts) if tgts else np.array([], dtype=np.int64)
    if pred.size == 0:
        return {"prog_f1_macro": float("nan"), "per_class": {}, "change_f1_macro": float("nan"),
                "acc": float("nan"), "n_valid": 0}
    macro, per, change = multiclass_f1(pred, tgt)
    acc = float((pred == tgt).mean())
    res = {"prog_f1_macro": macro, "per_class": per, "change_f1_macro": change, "n_valid": int(pred.size),
           "acc": acc, "ordinal_safety": ordinal_safety(pred, tgt)}
    if diagnostics:
        reg = np.concatenate(regs) if regs else np.array([], dtype=np.int64)
        dis = np.concatenate(diseases) if diseases else np.array([], dtype=np.int64)
        res["confusion"] = {
            "labels": C.PROG_NAMES,
            "matrix_true_by_pred": _confusion(pred, tgt),
        }
        res["per_disease"] = _slice_f1(pred, tgt, dis, C.CHEX_NAMES)
        res["per_region"] = _slice_f1(pred, tgt, reg, C.REGION_NAMES)
        if same_views:
            sv = np.concatenate(same_views)
            res["per_view_pair"] = _slice_f1(pred, tgt, sv, ["cross_view", "same_view"])
    return res


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
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers for eval. Use >0 on server to parallelize feature loading.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--diagnostics-json", type=Path, default=None,
                   help="write confusion matrix plus per-disease/per-region F1 diagnostics")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    m = build_from_ckpt(ck, args.device)                    # sets config.REQUIRE_PRIOR_PRESENT first
    ds = dataset_from_ckpt(ck, args.m3_labels_dir, args.m4_labels_dir, args.pairs, args.split,
                           region_cache=args.region_cache, features_root=args.features_root)
    loader_kwargs = {
        "batch_size": args.batch,
        "collate_fn": collate,
        "num_workers": args.workers,
    }
    if args.workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })
    loader = DataLoader(ds, **loader_kwargs)
    res = evaluate(m, loader, args.device, diagnostics=args.diagnostics_json is not None)
    print(f"[{args.split}] acc = {res['acc']:.4f}  prog macro-F1 = {res['prog_f1_macro']:.4f}  "
          f"change-only F1 = {res['change_f1_macro']:.4f}  (n={res['n_valid']:,})")
    for k, v in res["per_class"].items():
        print(f"  {k:<10} {v:.4f}")
    if args.diagnostics_json is not None:
        args.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[diagnostics] wrote {args.diagnostics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
