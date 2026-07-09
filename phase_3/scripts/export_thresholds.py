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
    p.add_argument("--min-concept-pos", type=int, default=10)
    p.add_argument("--min-concept-f1", type=float, default=0.55)
    p.add_argument("--min-concept-auc", type=float, default=0.75)
    p.add_argument("--allow-single-class-concepts", action="store_true",
                   help="allow concepts whose AUC is NaN because only one target class appears")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.diagnostics.read_text(encoding="utf-8"))

    thresholds = {
        "source": str(args.diagnostics),
        "level": "image_diagnostics",
        "thresholds": {},
    }
    for label, rec in data.get("image_diagnostics", {}).items():
        thr = rec.get("best_threshold", 0.5)
        thresholds["thresholds"][label] = {
            "threshold": float(thr) if _finite(thr) else 0.5,
            "f1_at_0_5": rec.get("f1_at_0_5"),
            "best_f1": rec.get("best_f1"),
            "auc": rec.get("auc"),
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
        ok = pos >= args.min_concept_pos and _finite(best_f1) and float(best_f1) >= args.min_concept_f1 and pass_auc
        item = {
            "concept": label,
            "allowed_for_why": bool(ok),
            "pos": pos,
            "neg": rec.get("neg"),
            "best_threshold": rec.get("best_threshold"),
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
    print(f"[thresholds] wrote {args.thresholds_json} ({len(thresholds['thresholds'])} diseases)")
    print(f"[concept-gate] wrote {args.concept_gate_json} "
          f"allow={len(gate['allow'])} deny={len(gate['deny'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
