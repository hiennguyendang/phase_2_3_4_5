"""[DIAG — local, CPU] Decompose rule_parser errors on val to find the next lever.

For every (region, finding) cell:
  * FP (predicted yes, gold not-yes) is split into
      - LOCALIZE : the finding IS gold-positive somewhere ELSE in the same image (we over-sprayed
                   the footprint into a region silver didn't tag) -> a PRECISION/region problem.
      - DETECT   : the finding is NOT gold-positive anywhere in the image (a wrong trigger / negation
                   miss) -> a real false detection.
  * FN (gold yes, not predicted yes) is split into
      - LOCALIZE : we DID predict the finding, just in other regions (footprint missed this region).
      - MISS     : we never fired the finding at all (lexicon/negation gap).

Prints global splits + per-finding tables + the worst region over-spray pairs, so we can see whether
to attack region footprints (localization) or triggers/negation (detection).

    python diagnose_rule_parser.py --val _work/sg_sft/val.jsonl --limit 0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config
from rule_parser import parse_report
from sg_eval_lib import report_and_regions
from sg_schema import parse_flat


def pos(flat):
    return {(r, f["finding"]) for r, fs in flat.items() for f in fs
            if f.get("presence", "yes") == "yes"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=Path, default=config.WORK_ROOT / "sg_sft" / "val.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top", type=int, default=25)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for line in args.val.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        usr = next(m["content"] for m in msgs if m["role"] == "user")
        gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        report, regions = report_and_regions(usr)
        rows.append((report, regions, parse_flat(gold)))
        if args.limit and len(rows) >= args.limit:
            break
    print(f"eval samples: {len(rows):,}")

    fp_loc = fp_det = fn_loc = fn_miss = tp = 0
    per = defaultdict(lambda: Counter())          # finding -> {tp,fp_loc,fp_det,fn_loc,fn_miss}
    overspray = Counter()                          # (finding, region) -> # FP_localize
    miss_region = Counter()                        # (finding, region) -> # FN_miss
    det_fp = Counter()                             # finding -> # FP_detect (wrong trigger)

    for report, regions, gold in rows:
        pred = parse_report(report, regions)
        g, p = pos(gold), pos(pred)
        g_find = {f for _r, f in g}
        p_find = {f for _r, f in p}
        for cell in (p & g):
            tp += 1; per[cell[1]]["tp"] += 1
        for (r, f) in (p - g):                      # false positive
            if f in g_find:
                fp_loc += 1; per[f]["fp_loc"] += 1; overspray[(f, r)] += 1
            else:
                fp_det += 1; per[f]["fp_det"] += 1; det_fp[f] += 1
        for (r, f) in (g - p):                      # false negative
            if f in p_find:
                fn_loc += 1; per[f]["fn_loc"] += 1
            else:
                fn_miss += 1; per[f]["fn_miss"] += 1; miss_region[(f, r)] += 1

    tot_fp = fp_loc + fp_det
    tot_fn = fn_loc + fn_miss
    print("\n=== GLOBAL ERROR DECOMPOSITION ===")
    print(f"TP            : {tp:,}")
    print(f"FP total      : {tot_fp:,}")
    print(f"  localize    : {fp_loc:,}  ({fp_loc/max(1,tot_fp):.0%})  over-sprayed footprint")
    print(f"  detect      : {fp_det:,}  ({fp_det/max(1,tot_fp):.0%})  wrong trigger / negation miss")
    print(f"FN total      : {tot_fn:,}")
    print(f"  localize    : {fn_loc:,}  ({fn_loc/max(1,tot_fn):.0%})  finding fired elsewhere")
    print(f"  miss        : {fn_miss:,}  ({fn_miss/max(1,tot_fn):.0%})  never fired")
    # if every FP_localize and FN_localize were fixed (perfect regions), what's the F1 ceiling?
    p_perfreg = tp + fp_det                         # only detection FPs remain
    g_count = tp + tot_fn
    prec_c = tp / max(1, tp + fp_det)
    rec_c = (tp + fn_loc) / max(1, g_count)         # localize FNs become TP if regions perfect
    print(f"\nregion-perfect CEILING (fix all localize errors): "
          f"P~{prec_c:.3f} R~{rec_c:.3f} -> F1~{2*prec_c*rec_c/max(1e-9,prec_c+rec_c):.3f}")

    print(f"\n=== PER-FINDING (sorted by total error) — tp / fpLoc fpDet / fnLoc fnMiss ===")
    order = sorted(per.items(), key=lambda kv: -(kv[1]["fp_loc"]+kv[1]["fp_det"]
                                                  + kv[1]["fn_loc"]+kv[1]["fn_miss"]))
    for f, c in order[: args.top]:
        print(f"  {f:<40} tp{c['tp']:>6} | fpL{c['fp_loc']:>6} fpD{c['fp_det']:>6} "
              f"| fnL{c['fn_loc']:>6} fnM{c['fn_miss']:>6}")

    print(f"\n=== WORST FOOTPRINT OVER-SPRAY (finding, region): #FP where finding is elsewhere-gold ===")
    for (f, r), n in overspray.most_common(args.top):
        print(f"  {n:>6}  {f:<34} -> {r}")

    print(f"\n=== WORST MISSED REGIONS (finding, region): #FN where finding never fired ===")
    for (f, r), n in miss_region.most_common(args.top):
        print(f"  {n:>6}  {f:<34} @ {r}")

    print(f"\n=== WORST FALSE DETECTIONS (finding): #FP where finding NOT in image at all ===")
    for f, n in det_fp.most_common(args.top):
        print(f"  {n:>6}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
