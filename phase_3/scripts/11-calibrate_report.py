"""Fit report-facing M3 disease thresholds and regional concept gates.

Input is the full, non-truncated validation NPZ written by ``5-eval.py``.  The
script never loads model weights and never touches the test split.  It emits
strict pair-specific JSON artifacts plus flat CSV audit tables containing every
supported and unsupported pair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import constants as C  # noqa: E402


def auc_binary(scores: np.ndarray, targets: np.ndarray) -> float:
    positive, negative = targets == 1, targets == 0
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        # Mann-Whitney AUC requires average ranks for tied scores.
        sorted_ranks[start:stop] = (start + 1 + stop) / 2.0
        start = stop
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = sorted_ranks
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def meta(npz, key: str, default: str = "") -> str:
    name = f"meta_{key}"
    return str(npz[name].item()) if name in npz.files else default


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    if total <= 0:
        return {"lo": None, "hi": None, "width": None}
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    return {"lo": lo, "hi": hi, "width": hi - lo}


def confusion(prob: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    pred = prob >= threshold
    positive = target == 1
    tp = int((pred & positive).sum())
    fp = int((pred & ~positive).sum())
    fn = int((~pred & positive).sum())
    tn = int((~pred & ~positive).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def confusion_for_absent(prob: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    """2x2 table whose negative call exactly matches the report rule p <= threshold."""
    pred_positive = prob > threshold
    positive = target == 1
    return {
        "tp": int((pred_positive & positive).sum()),
        "fp": int((pred_positive & ~positive).sum()),
        "tn": int((~pred_positive & ~positive).sum()),
        "fn": int((~pred_positive & positive).sum()),
    }


def ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def rates(counts: dict) -> dict:
    tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
    present_calls = tp + fp
    absent_calls = tn + fn
    positives = tp + fn
    negatives = tn + fp
    total = positives + negatives
    values = {
        "ppv": ratio(tp, present_calls),
        "npv": ratio(tn, absent_calls),
        "sensitivity": ratio(tp, positives),
        "specificity": ratio(tn, negatives),
        "present_rate": ratio(present_calls, total),
        "absent_rate": ratio(absent_calls, total),
    }
    intervals = {
        "ppv": wilson(tp, present_calls),
        "npv": wilson(tn, absent_calls),
        "sensitivity": wilson(tp, positives),
        "specificity": wilson(tn, negatives),
        "present_rate": wilson(present_calls, total),
        "absent_rate": wilson(absent_calls, total),
    }
    return {
        **counts,
        "n": total,
        "positive_labels": positives,
        "negative_labels": negatives,
        "present_calls": present_calls,
        "absent_calls": absent_calls,
        **values,
        "wilson95": intervals,
    }


def f1(counts: dict) -> float:
    den = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 2 * counts["tp"] / den if den else float("nan")


def unique_call_patients(prob: np.ndarray, patients: np.ndarray,
                         threshold: float, present: bool) -> int:
    mask = prob >= threshold if present else prob <= threshold
    return int(np.unique(patients[mask]).size)


def ci_width_ok(stats: dict, names: tuple[str, ...], maximum: float | None) -> bool:
    if maximum is None:
        return True
    for name in names:
        width = stats["wilson95"][name]["width"]
        if width is None or width > maximum:
            return False
    return True


def cluster_bootstrap(prob: np.ndarray, target: np.ndarray, patients: np.ndarray,
                      threshold: float, n_bootstrap: int, seed: int,
                      positive_inclusive: bool = True) -> dict | None:
    """Patient-cluster bootstrap intervals at one frozen operating point."""
    if n_bootstrap <= 0:
        return None
    patient_names, inverse = np.unique(patients, return_inverse=True)
    n_patients = len(patient_names)
    if n_patients < 2:
        return None
    pred = prob >= threshold if positive_inclusive else prob > threshold
    positive = target == 1
    cell_masks = (pred & positive, pred & ~positive, ~pred & ~positive, ~pred & positive)
    per_patient = np.stack([
        np.bincount(inverse, weights=mask.astype(np.int16), minlength=n_patients)
        for mask in cell_masks
    ], axis=1)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, 4), dtype=np.float64)
    for i in range(n_bootstrap):
        sampled = rng.integers(0, n_patients, size=n_patients)
        draws[i] = per_patient[sampled].sum(axis=0)
    tp, fp, tn, fn = (draws[:, i] for i in range(4))

    def interval(num, den):
        vals = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            return {"lo": None, "hi": None, "width": None}
        lo, hi = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
        return {"lo": lo, "hi": hi, "width": hi - lo}

    return {
        "n_resamples": n_bootstrap,
        "n_unique_patients": n_patients,
        "ppv": interval(tp, tp + fp),
        "npv": interval(tn, tn + fn),
        "sensitivity": interval(tp, tp + fn),
        "specificity": interval(tn, tn + fp),
    }


def prepare(prob: np.ndarray, target: np.ndarray,
            patients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (target == 0) | (target == 1)
    return (np.asarray(prob[valid], dtype=np.float64),
            np.asarray(target[valid], dtype=np.int8),
            np.asarray(patients[valid], dtype=str))


def threshold_count_grid(prob: np.ndarray, target: np.ndarray,
                         thresholds: np.ndarray) -> list[dict]:
    """All threshold confusion tables in O(n log n + number_of_thresholds)."""
    if not len(prob):
        return [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for _ in thresholds]
    order = np.argsort(prob, kind="mergesort")
    sorted_prob = prob[order]
    sorted_positive = (target[order] == 1).astype(np.int64)
    cumulative_positive = np.concatenate(([0], np.cumsum(sorted_positive)))
    cumulative_negative = np.arange(len(prob) + 1) - cumulative_positive
    total_positive = int(cumulative_positive[-1])
    total_negative = len(prob) - total_positive
    cut = np.searchsorted(sorted_prob, thresholds, side="left")
    return [
        {
            "tp": total_positive - int(cumulative_positive[i]),
            "fp": total_negative - int(cumulative_negative[i]),
            "tn": int(cumulative_negative[i]),
            "fn": int(cumulative_positive[i]),
        }
        for i in cut
    ]


def best_f1_threshold(thresholds: np.ndarray,
                      count_grid: list[dict]) -> tuple[float | None, float | None]:
    best_threshold, best_value = None, float("nan")
    for threshold, counts in zip(thresholds, count_grid):
        value = f1(counts)
        if best_threshold is None or (finite(value) and (not finite(best_value) or value > best_value)):
            best_threshold, best_value = float(threshold), value
    return best_threshold, best_value


def fit_disease(prob: np.ndarray, target: np.ndarray, patients: np.ndarray, args,
                seed_offset: int = 0) -> dict:
    prob, target, patients = prepare(prob, target, patients)
    pos, neg = int((target == 1).sum()), int((target == 0).sum())
    unique_patients = int(np.unique(patients).size)
    thresholds = np.linspace(0.01, 0.99, 99)
    count_grid = threshold_count_grid(prob, target, thresholds)
    best_threshold, best_value = best_f1_threshold(thresholds, count_grid) \
        if len(prob) else (None, None)

    present_threshold = None
    present_stats = None
    present_unique_patients = 0
    if neg >= args.min_negatives and unique_patients >= args.min_unique_patients:
        for threshold, counts in zip(thresholds, count_grid):
            threshold = float(threshold)
            stats = rates(counts)
            call_patients = unique_call_patients(prob, patients, threshold, True)
            if (stats["present_calls"] >= args.min_calls
                    and call_patients >= args.min_unique_patients
                    and stats["ppv"] >= args.target_ppv
                    and stats["specificity"] >= args.target_specificity
                    and ci_width_ok(stats, ("ppv", "specificity"), args.max_ci_width)):
                present_threshold = threshold
                present_stats = stats
                present_unique_patients = call_patients
                break

    absent_threshold = None
    absent_stats = None
    absent_unique_patients = 0
    if present_threshold is not None:
        for threshold, _ in zip(thresholds, count_grid):
            threshold = float(threshold)
            if threshold >= present_threshold:
                break
            stats = rates(confusion_for_absent(prob, target, threshold))
            call_patients = unique_call_patients(prob, patients, threshold, False)
            if (stats["absent_calls"] >= args.min_calls
                    and call_patients >= args.min_unique_patients
                    and stats["npv"] >= args.target_npv
                    and ci_width_ok(stats, ("npv",), args.max_ci_width)):
                absent_threshold = threshold
                absent_stats = stats
                absent_unique_patients = call_patients

    if present_stats is not None:
        present_stats["cluster_bootstrap95"] = cluster_bootstrap(
            prob, target, patients, present_threshold, args.bootstrap,
            args.seed + seed_offset * 2)
    if absent_stats is not None:
        absent_stats["cluster_bootstrap95"] = cluster_bootstrap(
            prob, target, patients, absent_threshold, args.bootstrap,
            args.seed + seed_offset * 2 + 1, positive_inclusive=False)

    present_rate = float((prob >= present_threshold).mean()) if present_threshold is not None else 0.0
    absent_rate = float((prob <= absent_threshold).mean()) if absent_threshold is not None else 0.0
    return safe({
        "n": len(prob), "positive_labels": pos, "negative_labels": neg,
        "unique_patients": unique_patients,
        "best_f1_threshold": best_threshold, "best_f1": best_value,
        "present_threshold": present_threshold,
        "present_supported": present_threshold is not None,
        "present_call_unique_patients": present_unique_patients,
        "present_stats": present_stats,
        "absent_threshold": absent_threshold,
        "absent_supported": absent_threshold is not None,
        "absent_call_unique_patients": absent_unique_patients,
        "absent_stats": absent_stats,
        "present_rate": present_rate, "absent_rate": absent_rate,
        "unknown_rate": max(0.0, 1.0 - present_rate - absent_rate),
        "supported": present_threshold is not None or absent_threshold is not None,
    })


def fit_concept(prob: np.ndarray, target: np.ndarray, patients: np.ndarray, args,
                seed_offset: int = 0) -> dict:
    prob, target, patients = prepare(prob, target, patients)
    pos, neg = int((target == 1).sum()), int((target == 0).sum())
    unique_patients = int(np.unique(patients).size)
    thresholds = np.linspace(0.01, 0.99, 99)
    count_grid = threshold_count_grid(prob, target, thresholds)
    best_threshold, best_value = best_f1_threshold(thresholds, count_grid) \
        if len(prob) else (None, None)
    auc = auc_binary(prob, target) if pos and neg else float("nan")

    candidate = None
    candidate_stats = None
    call_unique_patients = 0
    if (pos >= args.min_concept_positives and neg >= args.min_negatives
            and unique_patients >= args.min_unique_patients):
        for threshold, counts in zip(thresholds, count_grid):
            threshold = float(threshold)
            stats = rates(counts)
            call_patients = unique_call_patients(prob, patients, threshold, True)
            if (stats["present_calls"] >= args.min_calls
                    and call_patients >= args.min_unique_patients
                    and stats["ppv"] >= args.target_concept_ppv
                    and stats["specificity"] >= args.target_concept_specificity
                    and ci_width_ok(stats, ("ppv", "specificity"), args.max_ci_width)):
                candidate, candidate_stats = threshold, stats
                call_unique_patients = call_patients
                break

    allowed = bool(candidate is not None and finite(best_value)
                   and best_value >= args.min_concept_f1
                   and finite(auc) and auc >= args.min_concept_auc)
    if allowed and candidate_stats is not None:
        candidate_stats["cluster_bootstrap95"] = cluster_bootstrap(
            prob, target, patients, candidate, args.bootstrap, args.seed + seed_offset)
    reason = "pass" if allowed else "below pair support/PPV/specificity/F1/AUC gate"
    return safe({
        "n": len(prob), "positive_labels": pos, "negative_labels": neg,
        "unique_patients": unique_patients,
        "best_f1_threshold": best_threshold, "best_f1": best_value, "auc": auc,
        "candidate_present_threshold": candidate,
        "present_threshold": candidate if allowed else None,
        "present_stats": candidate_stats,
        "present_call_unique_patients": call_unique_patients,
        "allowed_for_why": allowed,
        "supported": allowed,
        "reason": reason,
    })


def audit_row(scope: str, region: str, label: str, item: dict) -> dict:
    present = item.get("present_stats") or {}
    absent = item.get("absent_stats") or {}
    return {
        "scope": scope, "region": region, "label": label,
        "supported": item.get("supported", False),
        "n": item.get("n"), "positive_labels": item.get("positive_labels"),
        "negative_labels": item.get("negative_labels"),
        "unique_patients": item.get("unique_patients"),
        "best_f1_threshold": item.get("best_f1_threshold"), "best_f1": item.get("best_f1"),
        "auc": item.get("auc"),
        "present_threshold": item.get("present_threshold"),
        "present_calls": present.get("present_calls"), "ppv": present.get("ppv"),
        "ppv_wilson_lo": ((present.get("wilson95") or {}).get("ppv") or {}).get("lo"),
        "specificity": present.get("specificity"),
        "specificity_wilson_lo": ((present.get("wilson95") or {}).get("specificity") or {}).get("lo"),
        "absent_threshold": item.get("absent_threshold"),
        "absent_calls": absent.get("absent_calls"), "npv": absent.get("npv"),
        "npv_wilson_lo": ((absent.get("wilson95") or {}).get("npv") or {}).get("lo"),
        "reason": item.get("reason", ""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["scope", "region", "label", "supported"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit strict M3 report thresholds from validation NPZ")
    p.add_argument("--pred-dump", type=Path, required=True)
    p.add_argument("--thresholds-json", type=Path, required=True)
    p.add_argument("--concept-gate-json", type=Path, required=True)
    p.add_argument("--disease-audit-csv", type=Path, required=True)
    p.add_argument("--concept-audit-csv", type=Path, required=True)
    p.add_argument("--target-ppv", type=float, default=0.90)
    p.add_argument("--target-specificity", type=float, default=0.90)
    p.add_argument("--target-npv", type=float, default=0.95)
    p.add_argument("--target-concept-ppv", type=float, default=0.90)
    p.add_argument("--target-concept-specificity", type=float, default=0.90)
    p.add_argument("--min-calls", type=int, default=30)
    p.add_argument("--min-negatives", type=int, default=30)
    p.add_argument("--min-unique-patients", type=int, default=30)
    p.add_argument("--min-concept-positives", type=int, default=10)
    p.add_argument("--min-concept-f1", type=float, default=0.55)
    p.add_argument("--min-concept-auc", type=float, default=0.75)
    p.add_argument("--max-ci-width", type=float, default=None,
                   help="optional preregistered maximum Wilson 95%% interval width")
    p.add_argument("--bootstrap", type=int, default=200,
                   help="patient-cluster bootstrap resamples for supported operating points; 0 disables")
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--allow-non-validation", action="store_true",
                   help="testing only; final calibration must use meta_split=val")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_calls < 1 or args.min_negatives < 1 or args.min_unique_patients < 1:
        raise SystemExit("[ERROR] minimum support values must be positive")
    with np.load(args.pred_dump, allow_pickle=False) as z:
        split = meta(z, "split")
        if split.lower() != "val" and not args.allow_non_validation:
            raise SystemExit(f"[ERROR] report thresholds require a validation dump, found split={split!r}")
        required = {
            "image_prob", "image_target", "patient_id", "region_prob", "region_target",
            "region_index", "region_patient_id", "concept_prob", "concept_target",
            "concept_region_index", "concept_patient_id",
        }
        missing = sorted(required - set(z.files))
        if missing:
            raise SystemExit(f"[ERROR] prediction dump schema is stale; missing {missing}; rerun 5-eval.py")
        arrays = {name: np.asarray(z[name]) for name in required}
        provenance = {
            "prediction_dump": str(args.pred_dump),
            "prediction_dump_sha256": sha256(args.pred_dump),
            "prediction_schema_version": meta(z, "schema_version"),
            "split": split,
            "box_source": meta(z, "box_source"),
            "checkpoint_sha256": meta(z, "checkpoint_sha256"),
            "manifest_sha256": meta(z, "manifest_sha256"),
        }

    policy = {
        "threshold_grid": "0.01..0.99 step 0.01",
        "present": "lowest threshold satisfying PPV, specificity, call, and patient support",
        "absent": "largest threshold below present satisfying NPV, call, and patient support",
        "target_ppv": args.target_ppv, "target_specificity": args.target_specificity,
        "target_npv": args.target_npv, "min_calls": args.min_calls,
        "min_negatives": args.min_negatives, "min_unique_patients": args.min_unique_patients,
        "max_wilson95_width": args.max_ci_width,
        "cluster_bootstrap_resamples": args.bootstrap,
        "unknown_labels": "excluded (-100)", "unsupported_pair": "abstain; no pooled fallback",
    }
    thresholds = {
        "schema_version": 2, "source": str(args.pred_dump), "provenance": provenance,
        "policy": policy, "thresholds": {"image": {}, "region_by_name": {}},
    }
    disease_rows: list[dict] = []
    image_prob, image_target = arrays["image_prob"], arrays["image_target"]
    image_patients = arrays["patient_id"].astype(str)
    for di, disease in enumerate(C.CHEX_NAMES):
        item = fit_disease(image_prob[:, di], image_target[:, di], image_patients, args, di)
        thresholds["thresholds"]["image"][disease] = item
        disease_rows.append(audit_row("image", "", disease, item))

    region_prob, region_target = arrays["region_prob"], arrays["region_target"]
    region_indices = arrays["region_index"]
    region_patients = arrays["region_patient_id"].astype(str)
    pair_counter = 100
    for ri, region in enumerate(C.REGION_NAMES):
        mask = region_indices == ri
        thresholds["thresholds"]["region_by_name"][region] = {}
        for di, disease in enumerate(C.CHEX_NAMES):
            item = fit_disease(region_prob[mask, di], region_target[mask, di],
                               region_patients[mask], args, pair_counter)
            pair_counter += 1
            thresholds["thresholds"]["region_by_name"][region][disease] = item
            disease_rows.append(audit_row("region", region, disease, item))
        print(f"[disease calibration] {ri + 1:02d}/{C.NUM_REGIONS} {region}")

    concept_policy = {
        "scope": "pair-specific (region, concept) present-display gate",
        "target_ppv": args.target_concept_ppv,
        "target_specificity": args.target_concept_specificity,
        "min_calls": args.min_calls, "min_negatives": args.min_negatives,
        "min_unique_patients": args.min_unique_patients,
        "min_positive_labels": args.min_concept_positives,
        "min_best_f1": args.min_concept_f1, "min_auc": args.min_concept_auc,
        "max_wilson95_width": args.max_ci_width,
        "unsupported_pair": "omit; no pooled fallback",
        "disease_forward": "continuous concepts; this gate is display-only",
    }
    gate = {
        "schema_version": 2, "source": str(args.pred_dump), "provenance": provenance,
        "policy": concept_policy, "region_by_name": {}, "allow": [], "deny": [],
    }
    concept_rows: list[dict] = []
    concept_prob, concept_target = arrays["concept_prob"], arrays["concept_target"]
    concept_indices = arrays["concept_region_index"]
    concept_patients = arrays["concept_patient_id"].astype(str)
    concept_counter = 10000
    for ri, region in enumerate(C.REGION_NAMES):
        mask = concept_indices == ri
        gate["region_by_name"][region] = {}
        for ci, concept in enumerate(C.CONCEPT_NAMES):
            item = fit_concept(concept_prob[mask, ci], concept_target[mask, ci],
                               concept_patients[mask], args, concept_counter)
            concept_counter += 1
            gate["region_by_name"][region][concept] = item
            listed = {"region": region, "concept": concept, **item}
            gate["allow" if item["allowed_for_why"] else "deny"].append(listed)
            concept_rows.append(audit_row("concept", region, concept, item))
        print(f"[concept calibration] {ri + 1:02d}/{C.NUM_REGIONS} {region}")

    thresholds["summary"] = {
        "image_present_supported": sum(x["present_supported"] for x in thresholds["thresholds"]["image"].values()),
        "region_present_supported": sum(
            x["present_supported"] for region in thresholds["thresholds"]["region_by_name"].values()
            for x in region.values()),
        "region_absent_supported": sum(
            x["absent_supported"] for region in thresholds["thresholds"]["region_by_name"].values()
            for x in region.values()),
    }
    gate["summary"] = {"allowed": len(gate["allow"]), "denied": len(gate["deny"])}
    args.thresholds_json.parent.mkdir(parents=True, exist_ok=True)
    args.concept_gate_json.parent.mkdir(parents=True, exist_ok=True)
    args.thresholds_json.write_text(json.dumps(safe(thresholds), indent=2, ensure_ascii=False), encoding="utf-8")
    args.concept_gate_json.write_text(json.dumps(safe(gate), indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.disease_audit_csv, disease_rows)
    write_csv(args.concept_audit_csv, concept_rows)
    print(f"[DONE] thresholds -> {args.thresholds_json}")
    print(f"[DONE] concept gate -> {args.concept_gate_json} "
          f"(allow={len(gate['allow'])}, deny={len(gate['deny'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
