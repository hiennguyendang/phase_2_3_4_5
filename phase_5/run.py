"""M5 CLI: join M3 + M4 predictions -> faithful reports + verification report.

    python phase_5/run.py --m3-pred data/m3_pred.jsonl --m4-pred data/m4_pred.jsonl \
        --out data/m5_reports.jsonl

Each output line contains the three-part VERA report: classification,
progression (when a prior exists), and optional ground_truth reference blocks.
The constrained paraphraser is OFF by default (template only). If a finding's realized text fails
verify (out-of-table / coverage / temporal), it falls back to the template — by construction the
template cannot go out of table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from assemble import assemble_image, realize_interval_template, realize_template
from ground_truth import attach_ground_truth
from paraphrase import paraphrase
from verify import verify


def _load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path or not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "image_id" in r:
                out[r["image_id"]] = r
    return out


def _load_temperature(path) -> dict | None:
    p = Path(path) if path else None
    if p and p.exists():
        t = json.loads(p.read_text(encoding="utf-8"))
        return t.get("per_class", t) if isinstance(t, dict) else None
    return None


def _load_temporal_calibration(path) -> tuple[dict | None, dict | None]:
    p = Path(path) if path else None
    if not p or not p.exists():
        return None, None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None, None
    return data.get("per_class"), data.get("confidence_gates")


def _load_thresholds(path) -> dict | None:
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("thresholds", data) if isinstance(data, dict) else {}
    return raw if isinstance(raw, dict) else None


def _load_concept_gate(path) -> dict | None:
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    allowed = data.get("allow", []) if isinstance(data, dict) else []
    gate = {}
    for item in allowed:
        threshold = item.get("explanation_threshold", item.get("best_threshold"))
        if item.get("allowed_for_why") and threshold is not None:
            gate[str(item["concept"])] = float(threshold)
    return gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M5 assemble faithful reports")
    p.add_argument("--m3-pred", type=Path, default=config.DEFAULT_M3_PRED)
    p.add_argument("--m4-pred", type=Path, default=config.DEFAULT_M4_PRED, help="optional; omit => no temporal")
    p.add_argument("--out", type=Path, default=config.DEFAULT_OUT)
    p.add_argument("--temperature", type=Path, default=config.TEMPERATURE_PATH,
                   help="per-class temperature json from calibrate.py (optional)")
    p.add_argument("--temporal-temperature", type=Path, default=config.TEMPORAL_TEMPERATURE_PATH,
                   help="per-disease temporal temperature json (optional)")
    p.add_argument("--thresholds", type=Path, default=config.DISEASE_THRESHOLDS_PATH,
                   help="validation-selected image/region absent+present thresholds (optional)")
    p.add_argument("--concept-gate", type=Path, default=None,
                   help="validation-selected concept allow-list and thresholds")
    p.add_argument("--allow-fixed-threshold-fallback", action="store_true",
                   help="allow provisional 0.10/0.50 thresholds for smoke demos only")
    p.add_argument("--include-absent", action="store_true",
                   help="deprecated compatibility flag; present/absent is now the report default")
    p.add_argument("--realize", default=config.REALIZE, choices=["template", "paraphrase"])
    p.add_argument("--stats-json", type=Path, default=None, help="optional machine-readable verify stats")
    p.add_argument("--ground-truth-metadata", type=Path, default=None,
                   help="optional MIMIC metadata JSONL; appends GT tables and one metadata report text")
    p.add_argument("--ground-truth-m3-labels-dir", type=Path,
                   default=config.REPO_ROOT / "data" / "m3_labels")
    p.add_argument("--ground-truth-m4-labels-dir", type=Path,
                   default=config.REPO_ROOT / "data" / "m4_labels")
    return p.parse_args()


def run(m3_map: dict, m4_map: dict, realize: str = "template", backend=None,
        temps: dict | None = None, thresholds: dict | None = None,
        include_absent: bool = False, temporal_temps: dict | None = None,
        temporal_gates: dict | None = None,
        concept_gate: dict | None = None) -> tuple[list[dict], dict]:
    reports, stats = [], {"n": 0, "normal": 0, "with_prior": 0, "out_of_table": 0,
                          "coverage_miss": 0, "temporal_halluc": 0, "paraphrase_fallback": 0}
    for iid, m3rec in m3_map.items():
        m4rec = m4_map.get(iid)
        prior_m3rec = m3_map.get((m4rec or {}).get("prior_image_id")) if m4rec else None
        report = assemble_image(m3rec, m4rec, temps, prior_m3rec,
                                thresholds=thresholds, temporal_temps=temporal_temps,
                                temporal_gates=temporal_gates,
                                concept_gate=concept_gate,
                                include_absent=include_absent)
        text = realize_template(report)
        interval_text = realize_interval_template(report)
        if realize == "paraphrase":
            cand = paraphrase(report, text, backend)
            if verify(report, cand)["ok"]:
                text = cand
            else:
                stats["paraphrase_fallback"] += 1            # keep the faithful template
        v = verify(report, text)
        report["current_text"] = text
        report["interval_text"] = interval_text
        report["classification"]["text"] = text
        if report.get("progression") is not None:
            report["progression"]["text"] = interval_text
        report["text"], report["verify"] = text, v
        reports.append(report)
        stats["n"] += 1
        stats["normal"] += int(report["normal"])
        stats["with_prior"] += int(report["has_prior"])
        stats["out_of_table"] += int(bool(v["out_of_table"]))
        stats["coverage_miss"] += int(bool(v["coverage_miss"]))
        stats["temporal_halluc"] += int(v["temporal_halluc"])
    return reports, stats


def main() -> int:
    args = parse_args()
    m3_map = _load_jsonl(args.m3_pred)
    m4_map = _load_jsonl(args.m4_pred)
    if not m3_map:
        raise SystemExit(f"[ERROR] no M3 predictions at {args.m3_pred}")
    temps = _load_temperature(args.temperature)
    temporal_temps, temporal_gates = _load_temporal_calibration(args.temporal_temperature)
    thresholds = _load_thresholds(args.thresholds)
    concept_gate = _load_concept_gate(args.concept_gate)
    print(f"M3 rows: {len(m3_map):,} | M4 rows: {len(m4_map):,} | "
          f"temperature: {'per-class' if temps else 'identity (T=1)'} | "
          f"temporal temperature: {'per-disease' if temporal_temps else 'raw'} | "
          f"temporal gates: {'validation-fitted' if temporal_gates else 'fixed fallback'} | "
          f"thresholds: {'dual image/region' if thresholds else 'none'} | "
          f"concept evidence: {len(concept_gate) if concept_gate is not None else 'ungated'} | "
          f"ground truth: {'metadata labels' if args.ground_truth_metadata else 'none'} | "
          "visible states: present/absent (unknown omitted)")
    if thresholds is None and not args.allow_fixed_threshold_fallback:
        raise SystemExit(
            "[ERROR] validation-fitted dual thresholds are required. Run M3 validation "
            "diagnostics and export_thresholds.py, or pass "
            "--allow-fixed-threshold-fallback for a non-paper smoke demo.")
    if thresholds is None:
        print("[WARNING] no validation threshold artifact: using fixed 0.10/0.50 "
              "provisional fallback; do not use this report as paper evidence")

    reports, stats = run(m3_map, m4_map, args.realize, temps=temps,
                         thresholds=thresholds, include_absent=args.include_absent,
                         temporal_temps=temporal_temps, temporal_gates=temporal_gates,
                         concept_gate=concept_gate)
    if args.ground_truth_metadata:
        reports = [attach_ground_truth(
            report,
            metadata_path=args.ground_truth_metadata,
            m3_labels_dir=args.ground_truth_m3_labels_dir,
            m4_labels_dir=args.ground_truth_m4_labels_dir,
        ) for report in reports]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = max(stats["n"], 1)
    print(f"[DONE] {stats['n']:,} reports -> {args.out}")
    print(f"  normal {stats['normal']:,} | with prior {stats['with_prior']:,}")
    print(f"  out-of-table {stats['out_of_table']}/{n}  coverage-miss {stats['coverage_miss']}/{n}  "
          f"temporal-halluc {stats['temporal_halluc']}/{n}  paraphrase-fallback {stats['paraphrase_fallback']}")
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  stats-json {args.stats_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
