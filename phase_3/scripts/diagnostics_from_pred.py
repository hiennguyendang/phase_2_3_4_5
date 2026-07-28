"""Build M3 calibration diagnostics from an existing prediction JSONL.

This avoids a second model pass when inference has already dumped all image and
detector-region probabilities. Thresholds are still fitted only against the
requested validation labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import constants as C  # noqa: E402
from eval import _auc_table, _diagnostic_table, _f1_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit M3 image/region diagnostics from prediction JSONL")
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target-present-precision", type=float, default=0.90)
    p.add_argument("--target-present-specificity", type=float, default=0.90)
    p.add_argument("--min-present-support", type=int, default=30)
    p.add_argument("--min-present-negatives", type=int, default=30)
    p.add_argument("--target-absent-npv", type=float, default=0.95)
    p.add_argument("--min-absent-support", type=int, default=30)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    labels_dir = args.labels_dir
    image_y = np.load(labels_dir / "image_chexpert.npy", mmap_mode="r")
    region_y = np.load(labels_dir / "region_chexpert.npy", mmap_mode="r")
    concept_y = np.load(labels_dir / "region_concepts.npy", mmap_mode="r")
    manifest = []
    with (labels_dir / "manifest.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row.get("ok", True) and str(row.get("split", "")).lower() == args.split:
                manifest.append((row["image_id"], i))
    index = dict(manifest)

    image_p, image_t = [], []
    region_p, region_t = [], []
    concept_p, concept_t = [], []
    pair_p = {name: [] for name in C.REGION_NAMES}
    pair_t = {name: [] for name in C.REGION_NAMES}
    with args.pred.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            iid = rec.get("image_id")
            if iid not in index:
                continue
            row_index = index[iid]
            image_p.append([float((rec.get("image_disease") or {}).get(name, 0.0))
                            for name in C.CHEX_NAMES])
            image_t.append(np.asarray(image_y[row_index], dtype=np.int16))
            for region, entry in (rec.get("regions") or {}).items():
                if region not in C.REGION_NAMES:
                    continue
                r = C.REGION_NAMES.index(region)
                region_p.append([float((entry.get("disease") or {}).get(name, 0.0))
                                 for name in C.CHEX_NAMES])
                region_t.append(np.asarray(region_y[row_index, r], dtype=np.int16))
                pair_p[region].append(region_p[-1])
                pair_t[region].append(region_t[-1])
                concept_p.append([float((entry.get("concepts") or {}).get(name, 0.0))
                                  for name in C.CONCEPT_NAMES])
                concept_t.append(np.asarray(concept_y[row_index, r], dtype=np.int16))

    if not image_p:
        raise SystemExit("[ERROR] no prediction/label overlap")
    ip, it = np.asarray(image_p), np.asarray(image_t)
    rp, rt = np.asarray(region_p), np.asarray(region_t)
    cp, ct = np.asarray(concept_p), np.asarray(concept_t)
    result = {
        "source": str(args.pred),
        "split": args.split,
        "n_images": int(len(ip)),
        "n_region_rows": int(len(rp)),
        "threshold_policy": {
            "present": "maximize_validation_f1",
            "alternate_present": "maximize_coverage_subject_to_validation_precision",
            "target_present_precision": args.target_present_precision,
            "target_present_specificity": args.target_present_specificity,
            "min_present_support": args.min_present_support,
            "min_present_negatives": args.min_present_negatives,
            "absent": "maximize_coverage_subject_to_validation_npv",
            "target_absent_npv": args.target_absent_npv,
            "min_absent_support": args.min_absent_support,
        },
        "image_auc_macro": _auc_table(ip, it, C.CHEX_NAMES)[0],
        "image_f1_macro": _f1_table(ip, it, C.CHEX_NAMES)[0],
        "image_diagnostics": _diagnostic_table(
            ip, it, C.CHEX_NAMES, target_present_precision=args.target_present_precision,
            target_present_specificity=args.target_present_specificity,
            min_present_support=args.min_present_support,
            min_present_negatives=args.min_present_negatives,
            target_absent_npv=args.target_absent_npv,
            min_absent_support=args.min_absent_support),
        "region_auc_macro": _auc_table(rp, rt, C.CHEX_NAMES)[0] if len(rp) else float("nan"),
        "region_f1_macro": _f1_table(rp, rt, C.CHEX_NAMES)[0] if len(rp) else float("nan"),
        "region_diagnostics": _diagnostic_table(
            rp, rt, C.CHEX_NAMES, target_present_precision=args.target_present_precision,
            target_present_specificity=args.target_present_specificity,
            min_present_support=args.min_present_support,
            min_present_negatives=args.min_present_negatives,
            target_absent_npv=args.target_absent_npv,
            min_absent_support=args.min_absent_support) if len(rp) else {},
        "region_pair_diagnostics": {
            region: _diagnostic_table(
                np.asarray(pair_p[region]), np.asarray(pair_t[region]), C.CHEX_NAMES,
                target_present_precision=args.target_present_precision,
                target_present_specificity=args.target_present_specificity,
                min_present_support=args.min_present_support,
                min_present_negatives=args.min_present_negatives,
                target_absent_npv=args.target_absent_npv,
                min_absent_support=args.min_absent_support)
            for region in C.REGION_NAMES if pair_p[region]
        },
        "concept_auc_macro": _auc_table(cp, ct, C.CONCEPT_NAMES)[0] if len(cp) else float("nan"),
        "concept_f1_macro": _f1_table(cp, ct, C.CONCEPT_NAMES)[0] if len(cp) else float("nan"),
        "concept_diagnostics": _diagnostic_table(
            cp, ct, C.CONCEPT_NAMES, target_present_precision=args.target_present_precision,
            target_present_specificity=args.target_present_specificity,
            min_present_support=args.min_present_support,
            min_present_negatives=args.min_present_negatives,
            target_absent_npv=args.target_absent_npv,
            min_absent_support=args.min_absent_support) if len(cp) else {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] {len(ip):,} images, {len(rp):,} region rows -> {args.out}")
    for name in C.CHEX_NAMES:
        img = result["image_diagnostics"].get(name, {})
        print(f"  {name:<26} present={img.get('present_threshold')} absent={img.get('absent_threshold')} "
              f"NPV={img.get('absent_npv')} unknown={img.get('unknown_rate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
