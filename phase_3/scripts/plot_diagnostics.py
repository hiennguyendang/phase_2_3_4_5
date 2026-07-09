"""Render lightweight reliability SVGs and CSV summaries from M3 diagnostics JSON.

No matplotlib dependency: writes simple SVG reliability diagrams plus CSV tables for
image_diagnostics / region_diagnostics / concept_diagnostics.

    python phase_3/scripts/plot_diagnostics.py --diagnostics artifacts/diagnostics/m3.json \
        --out-dir artifacts/diagnostics/m3_plots --top-ece 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _num(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else x


def write_summary(name: str, table: dict, out_dir: Path, top_ece: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    fields = ["label", "n", "pos", "neg", "f1_at_0_5", "auc", "ece", "best_threshold", "best_f1"]
    rows = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, rec in sorted(table.items()):
            row = {"label": label, **{k: _num(rec.get(k)) for k in fields[1:]}}
            w.writerow(row)
            rows.append((label, rec))
    rows.sort(key=lambda kv: (-1 if math.isnan(kv[1].get("ece", float("nan"))) else kv[1].get("ece", -1)),
              reverse=True)
    chosen = [label for label, rec in rows[:top_ece] if rec.get("reliability_bins")]
    return chosen


def render_svg(label: str, rec: dict, path: Path) -> None:
    bins = rec.get("reliability_bins") or []
    w, h, pad = 520, 420, 54
    plot = h - 2 * pad

    def xy(conf, acc):
        x = pad + (conf - 0.5) / 0.5 * plot
        y = h - pad - acc * plot
        return x, y

    bars = []
    for b in bins:
        if not b.get("n"):
            continue
        lo, hi = b["lo"], b["hi"]
        x0, _ = xy(lo, 0)
        x1, y = xy(hi, b["accuracy"])
        _, y0 = xy(hi, 0)
        bars.append(
            f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(1, x1-x0-2):.1f}" '
            f'height="{max(0, y0-y):.1f}" fill="#7aa6c2" opacity="0.75"/>'
        )
    title = label.replace("&", "&amp;")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{pad}" y="28" font-family="sans-serif" font-size="16">{title}</text>
  <text x="{pad}" y="48" font-family="sans-serif" font-size="12">ECE={rec.get("ece", float("nan")):.4f} · F1@0.5={rec.get("f1_at_0_5", float("nan")):.4f} · best τ={rec.get("best_threshold", float("nan")):.2f}</text>
  <line x1="{pad}" y1="{h-pad}" x2="{h-pad}" y2="{h-pad}" stroke="black"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="black"/>
  <line x1="{pad}" y1="{h-pad}" x2="{h-pad}" y2="{pad}" stroke="#555" stroke-dasharray="5 5"/>
  {''.join(bars)}
  <text x="{pad}" y="{h-16}" font-family="sans-serif" font-size="12">confidence</text>
  <text x="8" y="{pad-10}" font-family="sans-serif" font-size="12">accuracy</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot M3 diagnostics JSON")
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-ece", type=int, default=8, help="render SVGs for the highest-ECE labels per table")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("image_diagnostics", "region_diagnostics", "concept_diagnostics"):
        table = data.get(name)
        if not table:
            continue
        labels = write_summary(name, table, args.out_dir, args.top_ece)
        sub = args.out_dir / name
        sub.mkdir(exist_ok=True)
        for label in labels:
            safe = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
            render_svg(label, table[label], sub / f"{safe}.svg")
    print(f"[DONE] diagnostics plots/tables -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
