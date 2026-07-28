"""Export M3 disease thresholds and concept explanation gates from diagnostics JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export thresholds/gates from M3 diagnostics")
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--thresholds-json", type=Path, required=True)
    p.add_argument("--concept-gate-json", type=Path, required=True)
    p.add_argument("--present-policy", choices=["f1", "precision"], default="f1",
                   help="F1 for benchmark decisions; precision for conservative report display")
    p.add_argument("--min-concept-pos", type=int, default=10)
    p.add_argument("--min-concept-f1", type=float, default=0.55)
    p.add_argument("--min-concept-auc", type=float, default=0.75)
    p.add_argument("--allow-single-class-concepts", action="store_true",
                   help="allow concepts whose AUC is NaN because only one target class appears")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    policy = data.get("threshold_policy", {})
    if policy.get("absent") != "maximize_coverage_subject_to_validation_npv":
        raise SystemExit(
            "[ERROR] diagnostics do not contain the finalized NPV threshold policy; "
            "rerun phase_3/scripts/5-eval.py on validation before exporting")

    thresholds = {
        "source": str(args.diagnostics),
        "policy": {**policy, "selected_present": args.present_policy},
        "thresholds": {"image": {}, "region": {}, "region_by_name": {}},
    }
    for level, diagnostic_key in (("image", "image_diagnostics"),
                                  ("region", "region_diagnostics")):
        for label, rec in data.get(diagnostic_key, {}).items():
            if args.present_policy == "precision":
                present = rec.get("precision_present_threshold")
            else:
                present = rec.get("present_threshold", rec.get("best_threshold", 0.5))
            present_available = _finite(present)
            absent = rec.get("absent_threshold")
            thresholds["thresholds"][level][label] = {
                "present_threshold": float(present) if present_available else None,
                "present_threshold_available": present_available,
                "present_policy": args.present_policy,
                "absent_threshold": float(absent) if _finite(absent) else None,
                "present_f1": rec.get("present_f1", rec.get("best_f1")),
                "absent_f1": rec.get("absent_f1"),
                "absent_npv": rec.get("absent_npv", rec.get("absent_precision")),
                "absent_precision": rec.get("absent_precision"),
                "absent_support": rec.get("absent_support"),
                "target_absent_npv": rec.get("target_absent_npv"),
                "min_absent_support": rec.get("min_absent_support"),
                "absent_threshold_available": rec.get("absent_threshold_available", absent is not None),
                "present_rate": rec.get("present_rate"),
                "precision_present_f1": rec.get("precision_present_f1"),
                "precision_present_precision": rec.get("precision_present_precision"),
                "precision_present_recall": rec.get("precision_present_recall"),
                "precision_present_specificity": rec.get("precision_present_specificity"),
                "precision_present_support": rec.get("precision_present_support"),
                "target_present_precision": rec.get("target_present_precision"),
                "target_present_specificity": rec.get("target_present_specificity"),
                "absent_rate": rec.get("absent_rate"),
                "unknown_rate": rec.get("unknown_rate"),
                "f1_at_0_5": rec.get("f1_at_0_5"),
                "auc": rec.get("auc"),
                "pos": rec.get("pos"),
                "neg": rec.get("neg"),
            }

    for region, records in data.get("region_pair_diagnostics", {}).items():
        thresholds["thresholds"]["region_by_name"][region] = {}
        for label, rec in records.items():
            present = (rec.get("precision_present_threshold")
                       if args.present_policy == "precision"
                       else rec.get("present_threshold", rec.get("best_threshold")))
            absent = rec.get("absent_threshold")
            present_available = _finite(present)
            thresholds["thresholds"]["region_by_name"][region][label] = {
                "present_threshold": float(present) if present_available else None,
                "present_threshold_available": present_available,
                "present_policy": args.present_policy,
                "absent_threshold": float(absent) if _finite(absent) else None,
                "absent_threshold_available": _finite(absent),
                "present_precision": (rec.get("precision_present_precision")
                                      if args.present_policy == "precision"
                                      else rec.get("present_precision")),
                "present_recall": (rec.get("precision_present_recall")
                                   if args.present_policy == "precision"
                                   else rec.get("present_recall")),
                "present_specificity": (rec.get("precision_present_specificity")
                                        if args.present_policy == "precision" else None),
                "present_support": (rec.get("precision_present_support")
                                    if args.present_policy == "precision"
                                    else rec.get("present_support")),
                "absent_npv": rec.get("absent_npv"),
                "absent_support": rec.get("absent_support"),
                "pos": rec.get("pos"),
                "neg": rec.get("neg"),
            }

    gate = {
        "source": str(args.diagnostics),
        "policy": {
            "min_pos": args.min_concept_pos,
            "min_best_f1": args.min_concept_f1,
            "min_auc": args.min_concept_auc,
            "allow_single_class_concepts": args.allow_single_class_concepts,
        },
        "allow": [],
        "deny": [],
    }
    for label, rec in data.get("concept_diagnostics", {}).items():
        pos = int(rec.get("pos", 0) or 0)
        auc = rec.get("auc")
        best_f1 = rec.get("best_f1")
        has_auc = _finite(auc)
        pass_auc = (has_auc and float(auc) >= args.min_concept_auc) or (
            args.allow_single_class_concepts and not has_auc)
        explanation_threshold = (rec.get("precision_present_threshold")
                                 if args.present_policy == "precision"
                                 else rec.get("best_threshold"))
        ok = (pos >= args.min_concept_pos and _finite(best_f1)
              and float(best_f1) >= args.min_concept_f1 and pass_auc
              and _finite(explanation_threshold))
        item = {
            "concept": label,
            "allowed_for_why": bool(ok),
            "pos": pos,
            "neg": rec.get("neg"),
            "best_threshold": rec.get("best_threshold"),
            "precision_present_threshold": rec.get("precision_present_threshold"),
            "explanation_threshold": explanation_threshold,
            "best_f1": best_f1,
            "f1_at_0_5": rec.get("f1_at_0_5"),
            "auc": auc,
            "ece": rec.get("ece"),
            "reason": "pass" if ok else "below support/F1/AUC gate",
        }
        gate["allow" if ok else "deny"].append(item)

    args.thresholds_json.parent.mkdir(parents=True, exist_ok=True)
    args.concept_gate_json.parent.mkdir(parents=True, exist_ok=True)
    args.thresholds_json.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
    args.concept_gate_json.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[thresholds] wrote {args.thresholds_json} "
          f"(image={len(thresholds['thresholds']['image'])}, "
          f"region={len(thresholds['thresholds']['region'])})")
    print(f"[concept-gate] wrote {args.concept_gate_json} "
          f"allow={len(gate['allow'])} deny={len(gate['deny'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
