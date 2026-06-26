"""M5 CLI: join M3 + M4 predictions -> faithful reports + verification report.

    python phase_5/run.py --m3-pred data/m3_pred.jsonl --m4-pred data/m4_pred.jsonl \
        --out data/m5_reports.jsonl

Each output line: {image_id, prior_image_id, has_prior, normal, findings[<provenance>], text, verify}.
The constrained paraphraser is OFF by default (template only). If a finding's realized text fails
verify (out-of-table / coverage / temporal), it falls back to the template — by construction the
template cannot go out of table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from assemble import assemble_image, realize_template
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M5 assemble faithful reports")
    p.add_argument("--m3-pred", type=Path, default=config.DEFAULT_M3_PRED)
    p.add_argument("--m4-pred", type=Path, default=config.DEFAULT_M4_PRED, help="optional; omit => no temporal")
    p.add_argument("--out", type=Path, default=config.DEFAULT_OUT)
    p.add_argument("--temperature", type=Path, default=config.TEMPERATURE_PATH,
                   help="per-class temperature json from calibrate.py (optional)")
    p.add_argument("--realize", default=config.REALIZE, choices=["template", "paraphrase"])
    return p.parse_args()


def run(m3_map: dict, m4_map: dict, realize: str = "template", backend=None,
        temps: dict | None = None) -> tuple[list[dict], dict]:
    reports, stats = [], {"n": 0, "normal": 0, "with_prior": 0, "out_of_table": 0,
                          "coverage_miss": 0, "temporal_halluc": 0, "paraphrase_fallback": 0}
    for iid, m3rec in m3_map.items():
        m4rec = m4_map.get(iid)
        report = assemble_image(m3rec, m4rec, temps)
        text = realize_template(report)
        if realize == "paraphrase":
            cand = paraphrase(report, text, backend)
            if verify(report, cand)["ok"]:
                text = cand
            else:
                stats["paraphrase_fallback"] += 1            # keep the faithful template
        v = verify(report, text)
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
    print(f"M3 rows: {len(m3_map):,} | M4 rows: {len(m4_map):,} | "
          f"temperature: {'per-class' if temps else 'identity (T=1)'}")

    reports, stats = run(m3_map, m4_map, args.realize, temps=temps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = max(stats["n"], 1)
    print(f"[DONE] {stats['n']:,} reports -> {args.out}")
    print(f"  normal {stats['normal']:,} | with prior {stats['with_prior']:,}")
    print(f"  out-of-table {stats['out_of_table']}/{n}  coverage-miss {stats['coverage_miss']}/{n}  "
          f"temporal-halluc {stats['temporal_halluc']}/{n}  paraphrase-fallback {stats['paraphrase_fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
