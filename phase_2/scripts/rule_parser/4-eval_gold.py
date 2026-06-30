"""Evaluate the rule parser against the GOLD subset — the held-out, human-verified scene graphs.

gold_ids.txt lists the 784 gold image_ids (excluded from train/val/test by build_sft_dataset). This
scores parse_report directly against those scene graphs (sourcing the report from the metadata), with
the SAME metrics as eval_rule_parser. This is the apples-to-apples comparison to ImaGenome's reported
silver-vs-gold F1 0.939 (Table 3): their NLP pipeline vs gold, here OUR parser vs gold.

    python scripts/eval_gold.py --scene-root <chest-imagenome>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

_PH2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PH2 / "src"))

import config
from sg_eval_lib import pos_cells, prf, prog_cells, unc_cells
from rule_parser import parse_report
from scene_to_yolo import dicom_id_from_image_id, iter_jsonl
from sg_lib import assemble_objects_from_scene, available_regions
from sg_schema import flat_from_scene_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the rule parser vs the gold scene graphs")
    p.add_argument("--scene-root", type=Path, default=config.DEFAULT_SCENE_ROOT)
    p.add_argument("--metadata", type=Path, default=config.DEFAULT_METADATA)
    p.add_argument("--gold-ids", type=Path, default=_PH2 / "gold_ids.txt")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "sg_eval_gold.json")
    p.add_argument("--top-findings", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gold_ids = {ln.strip() for ln in args.gold_ids.read_text(encoding="utf-8").splitlines() if ln.strip()}
    reports = {}
    for row in iter_jsonl(args.metadata):
        iid = str(row.get("image_id", "")).strip()
        if iid in gold_ids:
            reports[iid] = str(row.get("report", "")).strip()
    print(f"gold ids: {len(gold_ids):,} | with report in metadata: {len(reports):,}")

    rows = []
    miss = 0
    for iid in gold_ids:
        report = reports.get(iid, "")
        if not report:
            miss += 1
            continue
        path = args.scene_root / f"{dicom_id_from_image_id(iid)}_SceneGraph.json"
        if not path.exists():
            miss += 1
            continue
        try:
            scene = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            miss += 1
            continue
        regions = available_regions(assemble_objects_from_scene(scene))
        if not regions:
            miss += 1
            continue
        gold = {r: v for r, v in flat_from_scene_graph(scene).items() if r in regions}
        rows.append((report, regions, gold))
    print(f"eval samples: {len(rows):,}  (skipped {miss})")

    P_tp = P_fp = P_fn = F_tp = F_fp = F_fn = U_tp = U_fp = U_fn = 0
    per_find = defaultdict(lambda: [0, 0, 0])
    prog_total = prog_correct = prog_covered = prog_gold = 0

    for report, regions, gold in rows:
        pred = parse_report(report, regions)
        g_pos, p_pos = pos_cells(gold), pos_cells(pred)
        P_tp += len(g_pos & p_pos); P_fp += len(p_pos - g_pos); P_fn += len(g_pos - p_pos)
        for (_r, f) in (g_pos & p_pos):
            per_find[f][0] += 1
        for (_r, f) in (p_pos - g_pos):
            per_find[f][1] += 1
        for (_r, f) in (g_pos - p_pos):
            per_find[f][2] += 1
        g_find = {f for _r, f in g_pos}; p_find = {f for _r, f in p_pos}
        F_tp += len(g_find & p_find); F_fp += len(p_find - g_find); F_fn += len(g_find - p_find)
        g_unc, p_unc = unc_cells(gold), unc_cells(pred)
        U_tp += len(g_unc & p_unc); U_fp += len(p_unc - g_unc); U_fn += len(g_unc - p_unc)
        g_prog, p_prog = prog_cells(gold), prog_cells(pred)
        prog_gold += len(g_prog)
        for cell, gp in g_prog.items():
            if cell in p_prog:
                prog_covered += 1; prog_total += 1
                prog_correct += int(p_prog[cell] == gp)

    macro = [prf(tp, fp, fn)["f1"] for tp, fp, fn in per_find.values() if tp + fn > 0]
    report_out = {
        "engine": "rule_parser", "eval_set": "gold", "n_samples": len(rows),
        "presence_with_region": prf(P_tp, P_fp, P_fn),
        "presence_finding_only": prf(F_tp, F_fp, F_fn),
        "uncertain_with_region": prf(U_tp, U_fp, U_fn),
        "presence_macro_f1": round(sum(macro) / len(macro), 4) if macro else 0.0,
        "progression": {"coverage": round(prog_covered / max(1, prog_gold), 4),
                        "accuracy_on_covered": round(prog_correct / max(1, prog_total), 4)},
        "per_finding_top": [],
    }
    top = sorted(per_find.items(), key=lambda kv: -(kv[1][0] + kv[1][2]))[: args.top_findings]
    for finding, (tp, fp, fn) in top:
        report_out["per_finding_top"].append({"finding": finding, **prf(tp, fp, fn)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report_out, ensure_ascii=False, indent=2), encoding="utf-8")

    pr = report_out["presence_with_region"]
    print("\n=== EVAL vs GOLD (rule parser) ===")
    print(f"presence (region): P {pr['precision']:.3f}  R {pr['recall']:.3f}  F1 {pr['f1']:.3f}")
    print(f"presence (macro) : {report_out['presence_macro_f1']:.3f}")
    print(f"finding-only F1  : {report_out['presence_finding_only']['f1']:.3f}")
    uc = report_out["uncertain_with_region"]
    print(f"uncertain (hedge): F1 {uc['f1']:.3f}")
    pg = report_out["progression"]
    print(f"progression      : acc {pg['accuracy_on_covered']:.3f} (coverage {pg['coverage']:.3f})")
    print("top findings (P/R/F1):")
    for d in report_out["per_finding_top"]:
        print(f"  {d['finding']:<40} {d['precision']:.2f}/{d['recall']:.2f}/{d['f1']:.2f}")
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
