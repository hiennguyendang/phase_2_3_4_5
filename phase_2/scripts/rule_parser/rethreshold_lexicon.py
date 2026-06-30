"""[PREP — local] Re-threshold the mined lexicon from the rich sidecar WITHOUT re-reading scenes.

mine_finding_lexicon.py writes _work/rule_finding_triggers_rich.json with, per concept,
[phrase, mention_count, precision, pos_rate] for every kept trigger at the LOOSEST setting. This
script applies tighter cutoffs to that pool and rewrites rule_finding_triggers.json, so the
precision/recall trade can be tuned in seconds instead of a 5-minute mine.

    python rethreshold_lexicon.py --min-prec 0.9 --min-pos-rate 0.35 --min-count 40 --cap 25
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ph2 = Path(__file__).resolve().parents[2]          # phase_2/
    p = argparse.ArgumentParser(description="Re-threshold mined lexicon from the rich sidecar")
    p.add_argument("--rich", type=Path, default=ph2 / "_work" / "rule_finding_triggers_rich.json")
    p.add_argument("--out", type=Path, default=ph2 / "src" / "rule_finding_triggers.json")
    p.add_argument("--min-prec", type=float, default=0.85)
    p.add_argument("--min-pos-rate", type=float, default=0.25)
    p.add_argument("--min-count", type=int, default=25)
    p.add_argument("--cap", type=int, default=40)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rich = json.loads(args.rich.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for c, rows in rich.items():
        kept = [phrase for phrase, men, prec, pos in rows
                if men >= args.min_count and prec >= args.min_prec and pos >= args.min_pos_rate]
        if kept:
            out[c] = kept[: args.cap]
    out = {c: out[c] for c in sorted(out, key=lambda c: -len(out[c]))}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(v) for v in out.values())
    print(f"min_prec={args.min_prec} min_pos={args.min_pos_rate} min_count={args.min_count} "
          f"cap={args.cap}  -> {n} triggers / {len(out)} concepts -> {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
