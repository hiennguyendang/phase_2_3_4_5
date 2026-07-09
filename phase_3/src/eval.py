"""Metrics for M3: macro-F1 (headline, spec 3.6) + AUC, for image / region / concept.

Both are computed dependency-free (ignore the -100 sentinel), so no sklearn needed.
F1 is the metric that drives checkpoint selection; AUC is reported alongside.
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
from dataset import M3Dataset, collate


def auc_binary(scores: np.ndarray, targets: np.ndarray) -> float:
    pos, neg = targets == 1, targets == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _auc_table(prob: np.ndarray, tgt: np.ndarray, names: list[str]) -> tuple[float, dict]:
    aucs = {}
    for c in range(prob.shape[-1]):
        m = (tgt[..., c] == 0) | (tgt[..., c] == 1)
        aucs[names[c]] = auc_binary(prob[..., c][m], tgt[..., c][m]) if m.any() else float("nan")
    macro = float(np.nanmean(list(aucs.values())))
    return macro, aucs


def _f1_table(prob: np.ndarray, tgt: np.ndarray, names: list[str],
              thr: float = 0.5) -> tuple[float, dict]:
    """Binary F1 per class at a fixed threshold (ignores -100). Macro = mean over classes."""
    f1s = {}
    for c in range(prob.shape[-1]):
        m = (tgt[..., c] == 0) | (tgt[..., c] == 1)
        if not m.any():
            f1s[names[c]] = float("nan"); continue
        p = (prob[..., c][m] >= thr).astype(np.int64)
        t = tgt[..., c][m].astype(np.int64)
        tp = int(((p == 1) & (t == 1)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        denom = 2 * tp + fp + fn
        f1s[names[c]] = (2.0 * tp / denom) if denom > 0 else float("nan")
    macro = float(np.nanmean(list(f1s.values())))
    return macro, f1s


def _binary_counts(prob: np.ndarray, tgt: np.ndarray, thr: float) -> tuple[int, int, int]:
    p = (prob >= thr).astype(np.int64)
    t = tgt.astype(np.int64)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    return tp, fp, fn


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else float("nan")


def _reliability_bins(prob: np.ndarray, tgt: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error bins for binary confidence=max(p,1-p), correctness at threshold .5."""
    if prob.size == 0:
        return float("nan"), []
    conf = np.maximum(prob, 1.0 - prob)
    correct = ((prob >= 0.5).astype(np.int64) == tgt.astype(np.int64)).astype(np.float64)
    ece = 0.0
    bins = []
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if m.any():
            weight = float(m.mean())
            confidence = float(conf[m].mean())
            accuracy = float(correct[m].mean())
            ece += weight * abs(confidence - accuracy)
            bins.append({
                "lo": float(lo),
                "hi": float(hi),
                "n": int(m.sum()),
                "confidence": confidence,
                "accuracy": accuracy,
                "gap": abs(confidence - accuracy),
            })
        else:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "confidence": float("nan"), "accuracy": float("nan"), "gap": float("nan")})
    return ece, bins


def _diagnostic_table(prob: np.ndarray, tgt: np.ndarray, names: list[str],
                      thresholds: np.ndarray | None = None) -> dict:
    """Per-label AUC/F1/ECE plus best threshold over a small fixed sweep."""
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 19)
    out = {}
    for c, name in enumerate(names):
        m = (tgt[..., c] == 0) | (tgt[..., c] == 1)
        if not m.any():
            continue
        pc = prob[..., c][m]
        tc = tgt[..., c][m].astype(np.int64)
        tp, fp, fn = _binary_counts(pc, tc, 0.5)
        ece, bins = _reliability_bins(pc, tc)
        best_thr, best_f1 = 0.5, _f1_from_counts(tp, fp, fn)
        for thr in thresholds:
            btp, bfp, bfn = _binary_counts(pc, tc, float(thr))
            bf1 = _f1_from_counts(btp, bfp, bfn)
            if np.isnan(best_f1) or bf1 > best_f1:
                best_thr, best_f1 = float(thr), bf1
        out[name] = {
            "n": int(tc.size),
            "pos": int((tc == 1).sum()),
            "neg": int((tc == 0).sum()),
            "f1_at_0_5": _f1_from_counts(tp, fp, fn),
            "auc": auc_binary(pc, tc),
            "ece": ece,
            "reliability_bins": bins,
            "best_threshold": best_thr,
            "best_f1": best_f1,
        }
    return out


@torch.no_grad()
def evaluate(model, loader, device, *, diagnostics: bool = False,
             pred_dump: Path | None = None) -> dict:
    model.eval()
    img_p, img_t = [], []
    rd_p, rd_t, rd_m = [], [], []
    cc_p, cc_t, cc_m = [], [], []
    for b in loader:
        out = model(b["grid"].to(device), b["global"].to(device),
                    b["present_mask"].to(device), b["boxes"].to(device))
        img_p.append(torch.sigmoid(out["image_disease_logits"]).cpu().numpy())
        img_t.append(b["image_chexpert"].numpy())
        rd_p.append(torch.sigmoid(out["region_disease_logits"]).cpu().numpy())
        rd_t.append(b["region_chexpert"].numpy())
        rd_m.append(b["present_mask"].numpy())
        if out["concept_logits"] is not None:
            cc_p.append(torch.sigmoid(out["concept_logits"]).cpu().numpy())
            cc_t.append(b["region_concepts"].numpy())
            cc_m.append(b["present_mask"].numpy())

    res = {}
    P, T = np.concatenate(img_p), np.concatenate(img_t)
    res["image_auc_macro"], res["image_per_class"] = _auc_table(P, T, C.CHEX_NAMES)
    res["image_f1_macro"], res["image_f1_per_class"] = _f1_table(P, T, C.CHEX_NAMES)
    if diagnostics:
        res["image_diagnostics"] = _diagnostic_table(P, T, C.CHEX_NAMES)

    rp, rt, rm = np.concatenate(rd_p), np.concatenate(rd_t), np.concatenate(rd_m).astype(bool)
    rp, rt = rp[rm], rt[rm]                      # [n_present, 14]
    res["region_auc_macro"], _ = _auc_table(rp, rt, C.CHEX_NAMES)
    res["region_f1_macro"], _ = _f1_table(rp, rt, C.CHEX_NAMES)
    if diagnostics:
        res["region_diagnostics"] = _diagnostic_table(rp, rt, C.CHEX_NAMES)

    if cc_p:
        cp, ct, cm = np.concatenate(cc_p), np.concatenate(cc_t), np.concatenate(cc_m).astype(bool)
        cp, ct = cp[cm], ct[cm]                  # [n_present, 69]
        res["concept_auc_macro"], _ = _auc_table(cp, ct, C.CONCEPT_NAMES)
        res["concept_f1_macro"], _ = _f1_table(cp, ct, C.CONCEPT_NAMES)
        if diagnostics:
            res["concept_diagnostics"] = _diagnostic_table(cp, ct, C.CONCEPT_NAMES)
    if pred_dump is not None:
        pred_dump.parent.mkdir(parents=True, exist_ok=True)
        dump = {
            "image_prob": P.astype(np.float32),
            "image_target": T.astype(np.int8),
            "region_prob": rp.astype(np.float32),
            "region_target": rt.astype(np.int8),
        }
        if cc_p:
            dump["concept_prob"] = cp.astype(np.float32)
            dump["concept_target"] = ct.astype(np.int8)
        np.savez_compressed(pred_dump, **dump)
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an M3 checkpoint")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, default=config.DEFAULT_LABELS_DIR)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--box-source", choices=["detector", "gt"], default=config.BOX_SOURCE,
                   help="bbox source: detector (default) or gt (oracle ablation)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--diagnostics-json", type=Path, default=None,
                   help="write per-class/per-concept AUC, F1, ECE, and threshold sweep diagnostics")
    p.add_argument("--pred-dump", type=Path, default=None,
                   help="write compressed probabilities/targets NPZ for bootstrap or custom audit")
    return p.parse_args()


def main() -> int:
    import model as M
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    config.apply(ck.get("cfg", {}))                     # rebuild the exact trained architecture
    ds = M3Dataset(args.labels_dir, args.features_root, args.split, box_source=args.box_source)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)
    config.USE_GLOBAL_TOKEN = ck.get("use_global", config.USE_GLOBAL_TOKEN)
    m = M.build_model(ck["feat_dim"], ck["mode"]).to(args.device)
    m.load_state_dict(ck["model"])
    res = evaluate(m, loader, args.device, diagnostics=args.diagnostics_json is not None,
                   pred_dump=args.pred_dump)
    print(f"[{args.split}] image  F1 macro = {res['image_f1_macro']:.4f}  AUC macro = {res['image_auc_macro']:.4f}")
    print(f"          region F1 {res['region_f1_macro']:.4f}  AUC {res['region_auc_macro']:.4f}"
          + (f"  | concept F1 {res.get('concept_f1_macro', float('nan')):.4f}"
             f"  AUC {res.get('concept_auc_macro', float('nan')):.4f}" if "concept_auc_macro" in res else ""))
    print(f"  {'class':<26} {'F1':>7} {'AUC':>7}")
    for c in res["image_per_class"]:
        print(f"  {c:<26} {res['image_f1_per_class'][c]:>7.4f} {res['image_per_class'][c]:>7.4f}")
    if args.diagnostics_json is not None:
        args.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[diagnostics] wrote {args.diagnostics_json}")
    if args.pred_dump is not None:
        print(f"[pred-dump] wrote {args.pred_dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
