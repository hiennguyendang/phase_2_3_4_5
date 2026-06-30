"""Evaluate the NON-LLM rule_parser against the silver targets — LOCAL, CPU, no GPU, seconds.

Reads an SFT split (build_sft_dataset.py output), recovers (report, regions) from each user
message, runs rule_parser.parse_report, and scores it with the SAME metrics as eval_sg_llm.py
(presence P/R/F1 with/without region, uncertain, 3-class progression, per-finding). Lets you
compare the rule parser head-to-head with the LLM zero-shot / finetuned numbers.

    python eval_rule_parser.py --val phase_2/_work/sg_sft/val.jsonl --limit 0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

# runnable from phase_2/scripts/rule_parser/; add phase_2/src/ (library) to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config
from rule_parser import parse_report
from sg_eval_lib import pos_cells, prf, prog_cells, unc_cells, report_and_regions
from sg_schema import parse_flat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the rule-based report parser vs silver")
    p.add_argument("--val", type=Path, default=config.WORK_ROOT / "sg_sft" / "val.jsonl")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "sg_eval_rule.json")
    p.add_argument("--limit", type=int, default=0, help="eval first N samples (0 = all)")
    p.add_argument("--top-findings", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.val.exists():
        raise SystemExit(f"[ERROR] val file not found: {args.val} (run build_sft_dataset.py)")

    rows = []
    for line in args.val.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        msgs = json.loads(line)["messages"]
        usr = next(m["content"] for m in msgs if m["role"] == "user")
        gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        report, regions = report_and_regions(usr)
        rows.append((report, regions, parse_flat(gold)))
        if args.limit and len(rows) >= args.limit:
            break
    print(f"eval samples: {len(rows):,}")

    P_tp = P_fp = P_fn = F_tp = F_fp = F_fn = U_tp = U_fp = U_fn = 0
    per_find = defaultdict(lambda: [0, 0, 0])
    prog_total = prog_correct = prog_covered = prog_gold = 0
    prog_conf = Counter()

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
                prog_conf[(gp, p_prog[cell])] += 1

    macro = [prf(tp, fp, fn)["f1"] for tp, fp, fn in per_find.values() if tp + fn > 0]
    report = {
        "engine": "rule_parser", "n_samples": len(rows),
        "presence_with_region": prf(P_tp, P_fp, P_fn),
        "presence_finding_only": prf(F_tp, F_fp, F_fn),
        "uncertain_with_region": prf(U_tp, U_fp, U_fn),
        "localization_gap_f1": round(prf(F_tp, F_fp, F_fn)["f1"] - prf(P_tp, P_fp, P_fn)["f1"], 4),
        "presence_macro_f1": round(sum(macro) / len(macro), 4) if macro else 0.0,
        "progression": {
            "gold_cued_cells": prog_gold,
            "coverage": round(prog_covered / max(1, prog_gold), 4),
            "accuracy_on_covered": round(prog_correct / max(1, prog_total), 4),
            "confusion": {f"{g}->{p}": n for (g, p), n in sorted(prog_conf.items())},
        },
        "per_finding_top": [],
    }
    top = sorted(per_find.items(), key=lambda kv: -(kv[1][0] + kv[1][2]))[: args.top_findings]
    for finding, (tp, fp, fn) in top:
        report["per_finding_top"].append({"finding": finding, **prf(tp, fp, fn)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== EVAL (rule parser, no LLM) ===")
    pr = report["presence_with_region"]
    print(f"presence (region): P {pr['precision']:.3f}  R {pr['recall']:.3f}  F1 {pr['f1']:.3f}")
    print(f"presence (macro) : {report['presence_macro_f1']:.3f}")
    uc = report["uncertain_with_region"]
    print(f"uncertain (hedge): P {uc['precision']:.3f}  R {uc['recall']:.3f}  F1 {uc['f1']:.3f}")
    print(f"localiz. gap F1  : {report['localization_gap_f1']:.3f}  "
          f"(finding-only F1 {report['presence_finding_only']['f1']:.3f})")
    pg = report["progression"]
    print(f"progression      : acc {pg['accuracy_on_covered']:.3f} on {prog_total} covered "
          f"(coverage {pg['coverage']:.3f} of {prog_gold} cued)")
    print("top findings (finding: P/R/F1):")
    for d in report["per_finding_top"]:
        print(f"  {d['finding']:<40} {d['precision']:.2f}/{d['recall']:.2f}/{d['f1']:.2f}"
              f"  (tp{d['tp']} fp{d['fp']} fn{d['fn']})")
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
