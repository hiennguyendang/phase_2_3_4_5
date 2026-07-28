"""Metrics for M3: macro-F1 (headline, spec 3.6) + AUC, for image / region / concept.

Both are computed dependency-free (ignore the -100 sentinel), so no sklearn needed.
F1 is the metric that drives checkpoint selection; AUC is reported alongside.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import M3Dataset, collate


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _patient_id(image_id: str) -> str:
    """Extract the stable MIMIC patient key without requiring metadata joins."""
    for part in str(image_id).split("_"):
        if part.startswith("p") and part[1:].isdigit():
            return part
    return str(image_id)


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


def _supported_metrics(prob: np.ndarray, tgt: np.ndarray, names: list[str],
                       min_pos: int, min_neg: int, thr: float = 0.5) -> dict:
    """Per-class metrics with an explicit support gate for region breakdowns."""
    per_class = {}
    supported_f1, supported_auc = [], []
    total_labeled = total_pos = total_neg = 0
    for c, name in enumerate(names):
        valid = (tgt[..., c] == 0) | (tgt[..., c] == 1)
        pc = prob[..., c][valid]
        tc = tgt[..., c][valid].astype(np.int64)
        n_pos = int((tc == 1).sum())
        n_neg = int((tc == 0).sum())
        n = n_pos + n_neg
        total_labeled += n; total_pos += n_pos; total_neg += n_neg
        supported = n_pos >= min_pos and n_neg >= min_neg
        if supported:
            tp, fp, fn = _binary_counts(pc, tc, thr)
            f1 = _f1_from_counts(tp, fp, fn)
            auc = auc_binary(pc, tc)
            supported_f1.append(f1); supported_auc.append(auc)
        else:
            f1 = auc = float("nan")
        per_class[name] = {
            "n": n, "pos": n_pos, "neg": n_neg, "supported": supported,
            "f1_at_0_5": f1, "auc": auc,
        }
    return {
        "macro_f1": float(np.nanmean(supported_f1)) if supported_f1 else float("nan"),
        "macro_auc": float(np.nanmean(supported_auc)) if supported_auc else float("nan"),
        "n_supported_classes": len(supported_f1),
        "n_classes": len(names),
        "n_labeled": total_labeled,
        "n_pos": total_pos,
        "n_neg": total_neg,
        "per_class": per_class,
    }


def _coverage_summary(detected: np.ndarray, gt_present: np.ndarray) -> dict:
    detected = detected.astype(bool); gt_present = gt_present.astype(bool)
    tp = int((detected & gt_present).sum())
    fp = int((detected & ~gt_present).sum())
    fn = int((~detected & gt_present).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall > 0 else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_detected": int(detected.sum()), "n_gt_present": int(gt_present.sum()),
        "precision": precision, "recall": recall, "f1": f1,
    }


def _regional_breakdown(prob: np.ndarray, tgt: np.ndarray,
                        detected: np.ndarray, gt_present: np.ndarray,
                        class_names: list[str], min_pos: int, min_neg: int) -> dict:
    """Metrics by named anatomical slot under conditional and detector-aware protocols."""
    per_region = {}
    conditional_f1, end_to_end_f1 = [], []
    conditional_auc, end_to_end_auc = [], []
    for ri, region_name in enumerate(C.REGION_NAMES):
        det = detected[:, ri].astype(bool)
        gt = gt_present[:, ri].astype(bool)
        conditional = _supported_metrics(prob[:, ri, :][det], tgt[:, ri, :][det],
                                         class_names, min_pos, min_neg)
        # A missed GT region cannot emit a positive regional finding. Setting its
        # probabilities to zero turns any labeled positive into a false negative.
        e2e_prob = prob[:, ri, :][gt].copy()
        if e2e_prob.size:
            e2e_prob[~det[gt]] = 0.0
        end_to_end = _supported_metrics(e2e_prob, tgt[:, ri, :][gt],
                                       class_names, min_pos, min_neg)
        cov = _coverage_summary(det[:, None], gt[:, None])
        per_region[region_name] = {
            "region_index": ri,
            "coverage": cov,
            "conditional_on_detected": conditional,
            "end_to_end": end_to_end,
        }
        conditional_f1.append(conditional["macro_f1"])
        conditional_auc.append(conditional["macro_auc"])
        end_to_end_f1.append(end_to_end["macro_f1"])
        end_to_end_auc.append(end_to_end["macro_auc"])

    def mean_valid(values):
        values = np.asarray(values, dtype=np.float64)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    return {
        "support_policy": {"min_positive": min_pos, "min_negative": min_neg,
                           "threshold": 0.5},
        "coverage": _coverage_summary(detected, gt_present),
        "macro_over_regions": {
            "conditional_f1": mean_valid(conditional_f1),
            "conditional_auc": mean_valid(conditional_auc),
            "end_to_end_f1": mean_valid(end_to_end_f1),
            "end_to_end_auc": mean_valid(end_to_end_auc),
        },
        "per_region": per_region,
    }


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


def _absent_counts(prob: np.ndarray, tgt: np.ndarray, thr: float) -> tuple[int, int, int]:
    """Counts for the absent class, where p <= thr is an explicit negative prediction."""
    pred_absent = prob <= thr
    true_absent = tgt == 0
    tp = int((pred_absent & true_absent).sum())
    fp = int((pred_absent & ~true_absent).sum())
    fn = int((~pred_absent & true_absent).sum())
    return tp, fp, fn


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
                      thresholds: np.ndarray | None = None,
                      target_present_precision: float = 0.90,
                      target_present_specificity: float = 0.90,
                      min_present_support: int = 30,
                      min_present_negatives: int = 30,
                      target_absent_npv: float = 0.95,
                      min_absent_support: int = 30) -> dict:
    """Fit F1-optimal and precision-constrained present thresholds plus absent thresholds."""
    if not 0.0 < target_present_precision <= 1.0:
        raise ValueError("target_present_precision must be in (0, 1]")
    if not 0.0 < target_present_specificity <= 1.0:
        raise ValueError("target_present_specificity must be in (0, 1]")
    if min_present_support < 1:
        raise ValueError("min_present_support must be positive")
    if min_present_negatives < 1:
        raise ValueError("min_present_negatives must be positive")
    if not 0.0 < target_absent_npv <= 1.0:
        raise ValueError("target_absent_npv must be in (0, 1]")
    if min_absent_support < 1:
        raise ValueError("min_absent_support must be positive")
    thresholds = thresholds if thresholds is not None else np.linspace(0.01, 0.99, 99)
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
        present_tp, present_fp, present_fn = _binary_counts(pc, tc, best_thr)
        present_support = present_tp + present_fp
        present_precision = (present_tp / present_support) if present_support else float("nan")
        present_recall = (present_tp / (present_tp + present_fn)) \
            if present_tp + present_fn else float("nan")

        precision_present_thr = None
        precision_present_f1 = float("nan")
        precision_present_precision = float("nan")
        precision_present_recall = float("nan")
        precision_present_specificity = float("nan")
        precision_present_support = 0
        neg_total = int((tc == 0).sum())
        if neg_total >= min_present_negatives:
            # Lowest qualifying threshold maximizes retained validation coverage.
            for thr in thresholds:
                ptp, pfp, pfn = _binary_counts(pc, tc, float(thr))
                support = ptp + pfp
                precision = ptp / support if support else float("nan")
                specificity = (neg_total - pfp) / neg_total
                if (support >= min_present_support
                        and precision >= target_present_precision
                        and specificity >= target_present_specificity):
                    precision_present_thr = float(thr)
                    precision_present_f1 = _f1_from_counts(ptp, pfp, pfn)
                    precision_present_precision = precision
                    precision_present_recall = ptp / (ptp + pfn) if ptp + pfn else float("nan")
                    precision_present_specificity = specificity
                    precision_present_support = support
                    break

        absent_candidates = [float(x) for x in thresholds if float(x) < best_thr]
        if not absent_candidates:
            absent_candidates = [max(0.0, best_thr / 2.0)]
        best_abs_thr = None
        best_abs_f1 = float("nan")
        best_abs_counts = (0, 0, 0)
        best_abs_npv = float("nan")
        for thr in absent_candidates:
            atp, afp, afn = _absent_counts(pc, tc, thr)
            support = atp + afp
            npv = atp / support if support else float("nan")
            af1 = _f1_from_counts(atp, afp, afn)
            if support >= min_absent_support and npv >= target_absent_npv:
                # The largest valid threshold has the greatest absent coverage.
                best_abs_thr, best_abs_f1, best_abs_npv = thr, af1, npv
                best_abs_counts = (atp, afp, afn)
        atp, afp, _ = best_abs_counts
        absent_support = atp + afp
        present_rate = float((pc >= best_thr).mean())
        absent_rate = float((pc <= best_abs_thr).mean()) if best_abs_thr is not None else 0.0
        unknown_rate = float(((pc > best_abs_thr) & (pc < best_thr)).mean()) \
            if best_abs_thr is not None else float((pc < best_thr).mean())
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
            "present_threshold": best_thr,
            "present_f1": best_f1,
            "present_precision": present_precision,
            "present_recall": present_recall,
            "present_support": present_support,
            "precision_present_threshold": precision_present_thr,
            "precision_present_f1": precision_present_f1,
            "precision_present_precision": precision_present_precision,
            "precision_present_recall": precision_present_recall,
            "precision_present_specificity": precision_present_specificity,
            "precision_present_support": precision_present_support,
            "target_present_precision": target_present_precision,
            "target_present_specificity": target_present_specificity,
            "min_present_support": min_present_support,
            "min_present_negatives": min_present_negatives,
            "precision_present_threshold_available": precision_present_thr is not None,
            "absent_threshold": best_abs_thr,
            "absent_f1": best_abs_f1,
            "absent_npv": best_abs_npv,
            "absent_precision": best_abs_npv,
            "absent_support": absent_support,
            "target_absent_npv": target_absent_npv,
            "min_absent_support": min_absent_support,
            "absent_threshold_available": best_abs_thr is not None,
            "present_rate": present_rate,
            "absent_rate": absent_rate,
            "unknown_rate": unknown_rate,
        }
    return out


@torch.no_grad()
def evaluate(model, loader, device, *, diagnostics: bool = False,
             pred_dump: Path | None = None,
             pred_metadata: dict[str, str] | None = None,
             target_absent_npv: float = 0.95,
             min_absent_support: int = 30,
             min_region_pos: int = 30,
             min_region_neg: int = 30) -> dict:
    model.eval()
    img_p, img_t = [], []
    rd_p, rd_t, rd_m = [], [], []
    cc_p, cc_t, cc_m = [], [], []
    gt_m = []
    image_ids: list[str] = []
    for b in loader:
        out = model(b["grid"].to(device), b["global"].to(device),
                    b["present_mask"].to(device), b["boxes"].to(device))
        img_p.append(torch.sigmoid(out["image_disease_logits"]).cpu().numpy())
        img_t.append(b["image_chexpert"].numpy())
        image_ids.extend(str(x) for x in b["image_id"])
        gt_m.append(b.get("gt_present_mask", b["present_mask"]).numpy())
        if out.get("region_disease_logits") is not None:
            rd_p.append(torch.sigmoid(out["region_disease_logits"]).cpu().numpy())
            rd_t.append(b["region_chexpert"].numpy())
            rd_m.append(b["present_mask"].numpy())
        if out.get("concept_logits") is not None:
            cc_p.append(torch.sigmoid(out["concept_logits"]).cpu().numpy())
            cc_t.append(b["region_concepts"].numpy())
            cc_m.append(b["present_mask"].numpy())

    res = {}
    P, T = np.concatenate(img_p), np.concatenate(img_t)
    res["image_auc_macro"], res["image_per_class"] = _auc_table(P, T, C.CHEX_NAMES)
    res["image_f1_macro"], res["image_f1_per_class"] = _f1_table(P, T, C.CHEX_NAMES)
    if diagnostics:
        res["threshold_policy"] = {
            "present": "maximize_validation_f1",
            "absent": "maximize_coverage_subject_to_validation_npv",
            "target_absent_npv": target_absent_npv,
            "min_absent_support": min_absent_support,
        }
        res["image_diagnostics"] = _diagnostic_table(
            P, T, C.CHEX_NAMES, target_absent_npv=target_absent_npv,
            min_absent_support=min_absent_support)

    if rd_p:
        rp_full, rt_full = np.concatenate(rd_p), np.concatenate(rd_t)
        rm = np.concatenate(rd_m).astype(bool)
        gm = np.concatenate(gt_m).astype(bool)
        rp, rt = rp_full[rm], rt_full[rm]            # [n_detected_regions, 14]
        res["region_auc_macro"], _ = _auc_table(rp, rt, C.CHEX_NAMES)
        res["region_f1_macro"], _ = _f1_table(rp, rt, C.CHEX_NAMES)
        if diagnostics:
            res["region_diagnostics"] = _diagnostic_table(
                rp, rt, C.CHEX_NAMES, target_absent_npv=target_absent_npv,
                min_absent_support=min_absent_support)
            res["regional_disease_breakdown"] = _regional_breakdown(
                rp_full, rt_full, rm, gm, C.CHEX_NAMES, min_region_pos, min_region_neg)
    else:
        rp = rt = None
        res["region_auc_macro"] = float("nan")
        res["region_f1_macro"] = float("nan")

    if cc_p:
        cp_full, ct_full = np.concatenate(cc_p), np.concatenate(cc_t)
        cm = np.concatenate(cc_m).astype(bool)
        gm = np.concatenate(gt_m).astype(bool)
        cp, ct = cp_full[cm], ct_full[cm]        # [n_detected_regions, 69]
        res["concept_auc_macro"], _ = _auc_table(cp, ct, C.CONCEPT_NAMES)
        res["concept_f1_macro"], _ = _f1_table(cp, ct, C.CONCEPT_NAMES)
        if diagnostics:
            res["concept_diagnostics"] = _diagnostic_table(
                cp, ct, C.CONCEPT_NAMES, target_absent_npv=target_absent_npv,
                min_absent_support=min_absent_support)
            res["regional_concept_breakdown"] = _regional_breakdown(
                cp_full, ct_full, cm, gm, C.CONCEPT_NAMES, min_region_pos, min_region_neg)
    else:
        cp = ct = None
        res["concept_auc_macro"] = float("nan")
        res["concept_f1_macro"] = float("nan")
    if pred_dump is not None:
        pred_dump.parent.mkdir(parents=True, exist_ok=True)
        image_id_array = np.asarray(image_ids, dtype=str)
        patient_id_array = np.asarray([_patient_id(x) for x in image_ids], dtype=str)
        dump = {
            "image_prob": P.astype(np.float32),
            "image_target": T.astype(np.int8),
            "image_id": image_id_array,
            "patient_id": patient_id_array,
            "region_prob": rp.astype(np.float32) if rp is not None else np.empty((0, len(C.CHEX_NAMES)), dtype=np.float32),
            "region_target": rt.astype(np.int8) if rt is not None else np.empty((0, len(C.CHEX_NAMES)), dtype=np.int8),
        }
        if rd_p:
            region_idx = np.broadcast_to(np.arange(C.NUM_REGIONS), rm.shape)[rm]
            image_idx = np.broadcast_to(np.arange(rm.shape[0])[:, None], rm.shape)[rm]
            dump["region_index"] = region_idx.astype(np.int8)
            dump["region_image_index"] = image_idx.astype(np.int32)
            dump["region_patient_id"] = patient_id_array[image_idx]
        if cp is not None:
            dump["concept_prob"] = cp.astype(np.float32)
            dump["concept_target"] = ct.astype(np.int8)
            concept_idx = np.broadcast_to(np.arange(C.NUM_REGIONS), cm.shape)[cm]
            concept_image_idx = np.broadcast_to(np.arange(cm.shape[0])[:, None], cm.shape)[cm]
            dump["concept_region_index"] = concept_idx.astype(np.int8)
            dump["concept_image_index"] = concept_image_idx.astype(np.int32)
            dump["concept_patient_id"] = patient_id_array[concept_image_idx]
        for key, value in (pred_metadata or {}).items():
            dump[f"meta_{key}"] = np.asarray(str(value))
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
    p.add_argument("--target-absent-npv", type=float, default=0.95,
                   help="minimum validation NPV required for an explicit absent call")
    p.add_argument("--min-absent-support", type=int, default=30,
                   help="minimum validation absent calls required to enable an absent threshold")
    p.add_argument("--min-region-pos", type=int, default=30,
                   help="minimum positives per (region,class) cell included in regional macro metrics")
    p.add_argument("--min-region-neg", type=int, default=30,
                   help="minimum negatives per (region,class) cell included in regional macro metrics")
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
                   pred_dump=args.pred_dump,
                   pred_metadata={
                       "schema_version": "2",
                       "split": args.split,
                       "box_source": args.box_source,
                       "checkpoint_sha256": _sha256(args.ckpt),
                       "manifest_sha256": _sha256(args.labels_dir / "manifest.jsonl"),
                   },
                   target_absent_npv=args.target_absent_npv,
                   min_absent_support=args.min_absent_support,
                   min_region_pos=args.min_region_pos, min_region_neg=args.min_region_neg)
    print(f"[{args.split}] image  F1 macro = {res['image_f1_macro']:.4f}  AUC macro = {res['image_auc_macro']:.4f}")
    region_txt = "region F1 N/A  AUC N/A" if math.isnan(res.get("region_f1_macro", float("nan"))) \
        else f"region F1 {res['region_f1_macro']:.4f}  AUC {res['region_auc_macro']:.4f}"
    concept_f1 = res.get("concept_f1_macro", float("nan"))
    concept_auc = res.get("concept_auc_macro", float("nan"))
    concept_txt = "" if math.isnan(concept_f1) else f"  | concept F1 {concept_f1:.4f}  AUC {concept_auc:.4f}"
    print(f"          {region_txt}{concept_txt}")
    if args.diagnostics_json is not None and "regional_disease_breakdown" in res:
        rb = res["regional_disease_breakdown"]
        cov = rb["coverage"]
        macro = rb["macro_over_regions"]
        print(f"          detector coverage P/R/F1 = {cov['precision']:.4f}/{cov['recall']:.4f}/{cov['f1']:.4f}")
        print(f"          disease by-region macro-F1: conditional={macro['conditional_f1']:.4f} "
              f"end-to-end={macro['end_to_end_f1']:.4f}")
    if args.diagnostics_json is not None and "regional_concept_breakdown" in res:
        macro = res["regional_concept_breakdown"]["macro_over_regions"]
        print(f"          concept by-region macro-F1: conditional={macro['conditional_f1']:.4f} "
              f"end-to-end={macro['end_to_end_f1']:.4f}")
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
