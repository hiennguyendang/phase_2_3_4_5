"""[PREP — local, no GPU] Mine the rule_parser's two priors from the silver SFT targets.

1. rule_region_priors.json   per finding: P(region occupied | finding positive) — silver's region
   footprint. rule_parser uses regions above DEFAULT_THRESH as the finding's default region set.
2. rule_concept_parents.json per finding: parent concepts A with P(A | finding) >= 0.97 and A more
   frequent (the ImaGenome ontology parent-child, e.g. consolidation -> lung opacity). rule_parser
   propagates a child's presence to its parents in the same region (a big recall win).

    python mine_region_priors.py --train phase_2/_work/sg_sft/train.jsonl

Re-run whenever build_sft_dataset.py is rebuilt. Both JSONs are bundled in phase_2/ so they travel
with the code on Kaggle (like m3_concept_space.json).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

_PH2 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_PH2), str(_PH2 / "src")]

import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine rule_parser region priors + concept parents")
    p.add_argument("--train", type=Path, default=config.WORK_ROOT / "sg_sft" / "train.jsonl")
    src = Path(__file__).resolve().parents[2] / "src"   # bundled data lives beside rule_parser
    p.add_argument("--priors-out", type=Path, default=src / "rule_region_priors.json")
    p.add_argument("--parents-out", type=Path, default=src / "rule_concept_parents.json")
    p.add_argument("--parent-thresh", type=float, default=0.97)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.train.exists():
        raise SystemExit(f"[ERROR] {args.train} not found (run build_sft_dataset.py).")

    reg_in_img: dict[str, Counter] = defaultdict(Counter)   # finding -> region -> #images
    fimg = Counter()                                         # finding -> #positive images
    cell = Counter()                                         # finding -> #(image,region) cells present
    co = defaultdict(Counter)                                # child -> co-present finding -> #cells
    n = 0
    for line in args.train.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            asst = next(m["content"] for m in json.loads(line)["messages"] if m["role"] == "assistant")
            flat = json.loads(asst)
        except (json.JSONDecodeError, StopIteration, KeyError):
            continue
        per: dict[str, set] = defaultdict(set)
        for region, fs in flat.items():
            present = [f["finding"] for f in fs if f.get("presence", "yes") == "yes"]
            for b in present:
                per[b].add(region)
                cell[b] += 1
                for a in present:
                    if a != b:
                        co[b][a] += 1
        for finding, regs in per.items():
            fimg[finding] += 1
            for r in regs:
                reg_in_img[finding][r] += 1
        n += 1
        if n % 40000 == 0:
            print(f"  ...{n:,}")

    priors = {f: {r: round(cnt / fimg[f], 3) for r, cnt in reg_in_img[f].items()} for f in fimg}
    parents = {}
    for b in cell:
        ps = [a for a, c in co[b].items() if c / cell[b] >= args.parent_thresh and cell[a] > cell[b]]
        if ps:
            parents[b] = ps

    args.priors_out.write_text(json.dumps(priors, ensure_ascii=False, indent=0), encoding="utf-8")
    args.parents_out.write_text(json.dumps(parents, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"\n[DONE] {n:,} images | {len(priors)} findings priors -> {args.priors_out.name}")
    print(f"        {len(parents)} findings with parents -> {args.parents_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
