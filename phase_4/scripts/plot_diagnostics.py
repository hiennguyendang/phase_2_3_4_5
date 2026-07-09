"""Render lightweight M4 audit plots from diagnostics JSON and train logs.

No matplotlib dependency. It scans existing artifacts by default and writes SVG/CSV files for:
  - M4 confusion matrices and per-class F1
  - per-disease macro/change F1 tables
  - MS-CXR-T aggregation comparison
  - train-log loss / val-F1 curves
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path


PROG = ["stable", "improved", "worsened"]
COLORS = ["#6f8fb3", "#69a67a", "#c9826b", "#8c79b5", "#b7a65d"]


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s).strip("_")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _svg_wrap(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f"{body}\n</svg>\n"
    )


def _txt(x, y, text, size=12, anchor="start", weight="normal", fill="#1f2933"):
    text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{text}</text>')


def _bar_svg(title: str, rows: list[tuple[str, list[float]]], labels: list[str], path: Path) -> None:
    width = 760
    height = max(260, 90 + 34 * len(rows))
    left, top = 230, 56
    plot_w = 450
    body = [_txt(24, 30, title, 18, weight="bold")]
    for j, lab in enumerate(labels):
        body.append(f'<rect x="{left + j*110}" y="42" width="14" height="14" fill="{COLORS[j % len(COLORS)]}"/>')
        body.append(_txt(left + j*110 + 20, 54, lab, 12))
    for i, (name, vals) in enumerate(rows):
        y = top + i * 34
        body.append(_txt(24, y + 16, name, 12))
        for j, v in enumerate(vals):
            v = 0.0 if v is None or math.isnan(v) else max(0.0, min(1.0, float(v)))
            x = left + j * 110
            body.append(f'<rect x="{x}" y="{y}" width="92" height="18" fill="#edf2f7"/>')
            body.append(f'<rect x="{x}" y="{y}" width="{92*v:.1f}" height="18" fill="{COLORS[j % len(COLORS)]}"/>')
            body.append(_txt(x + 46, y + 14, f"{v:.2f}", 10, anchor="middle", fill="#111827"))
    path.write_text(_svg_wrap(width, height, "\n".join(body)), encoding="utf-8")


def _confusion_svg(title: str, labels: list[str], matrix: list[list[int]], path: Path) -> None:
    cell = 88
    left, top = 160, 78
    max_v = max(max(row) for row in matrix) if matrix else 1
    body = [_txt(24, 32, title, 18, weight="bold"),
            _txt(left + 1.5 * cell, 58, "predicted", 13, anchor="middle"),
            _txt(34, top + 1.6 * cell, "true", 13)]
    for j, lab in enumerate(labels):
        body.append(_txt(left + j * cell + cell / 2, top - 16, lab, 12, anchor="middle"))
    for i, lab in enumerate(labels):
        body.append(_txt(left - 16, top + i * cell + cell / 2 + 4, lab, 12, anchor="end"))
        row_sum = sum(matrix[i]) or 1
        for j, val in enumerate(matrix[i]):
            shade = int(245 - 170 * (val / max_v))
            fill = f"rgb({shade},{shade + 8},{255})"
            x, y = left + j * cell, top + i * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" fill="{fill}" stroke="#d0d7de"/>')
            body.append(_txt(x + cell / 2, y + cell / 2 - 4, f"{val:,}", 13, anchor="middle", weight="bold"))
            body.append(_txt(x + cell / 2, y + cell / 2 + 15, f"{100*val/row_sum:.1f}%", 11, anchor="middle"))
    path.write_text(_svg_wrap(520, 390, "\n".join(body)), encoding="utf-8")


def plot_m4_diagnostics(paths: list[Path], out_dir: Path) -> None:
    rows = []
    diag_dir = out_dir / "m4_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        data = _read_json(p)
        name = _safe_name(p.stem.replace(".diagnostics", ""))
        rows.append({
            "run": name,
            "macro_f1": data.get("prog_f1_macro"),
            "change_f1": data.get("change_f1_macro"),
            **{k: data.get("per_class", {}).get(k) for k in PROG},
            "n_valid": data.get("n_valid"),
        })
        if "confusion" in data:
            _confusion_svg(f"{name} confusion", data["confusion"]["labels"],
                           data["confusion"]["matrix_true_by_pred"],
                           diag_dir / f"{name}.confusion.svg")
        per = data.get("per_class", {})
        _bar_svg(f"{name} per-class F1", [(name, [per.get(k) for k in PROG])], PROG,
                 diag_dir / f"{name}.per_class.svg")
        disease = data.get("per_disease") or {}
        if disease:
            csv_path = diag_dir / f"{name}.per_disease.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["disease", "n", "macro_f1", "change_f1",
                                                  "stable", "improved", "worsened"])
                w.writeheader()
                for dname, rec in sorted(disease.items()):
                    row = {"disease": dname, "n": rec.get("n"),
                           "macro_f1": rec.get("macro_f1"), "change_f1": rec.get("change_f1")}
                    row.update({k: rec.get("per_class", {}).get(k) for k in PROG})
                    w.writerow(row)
            ranked = sorted(disease.items(), key=lambda kv: (kv[1].get("macro_f1") or -1))
            chosen = ranked[:8] + ranked[-8:]
            _bar_svg(f"{name} disease slices", [(k, [v.get("macro_f1"), v.get("change_f1")])
                                                for k, v in chosen],
                     ["macro", "change"], diag_dir / f"{name}.per_disease.svg")
    if rows:
        with open(out_dir / "m4_run_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        _bar_svg("M4 diagnostic run comparison",
                 [(r["run"], [r.get("macro_f1"), r.get("change_f1"),
                              r.get("stable"), r.get("improved"), r.get("worsened")])
                  for r in rows],
                 ["macro", "change", "stable", "improved", "worsened"],
                 out_dir / "m4_run_comparison.svg")


def plot_mscxrt(paths: list[Path], out_dir: Path) -> None:
    rows = []
    ms_dir = out_dir / "mscxrt"
    ms_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        data = _read_json(p)
        run = _safe_name(p.stem.replace(".diagnostics", "").replace(".mscxrt", ""))
        for agg, rec in data.get("aggregations", {}).items():
            row = {
                "run": run,
                "aggregation": agg,
                "macro_f1": rec.get("prog_f1_macro"),
                "change_f1": rec.get("change_f1_macro"),
                "stable": rec.get("per_class", {}).get("stable"),
                "improved": rec.get("per_class", {}).get("improved"),
                "worsened": rec.get("per_class", {}).get("worsened"),
                "used_pairs": data.get("coverage", {}).get("used_pairs"),
            }
            rows.append(row)
    if rows:
        with open(ms_dir / "mscxrt_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        _bar_svg("MS-CXR-T external audit",
                 [(f"{r['run']}:{r['aggregation']}",
                   [r["macro_f1"], r["change_f1"], r["stable"], r["improved"], r["worsened"]])
                  for r in rows],
                 ["macro", "change", "stable", "improved", "worsened"],
                 ms_dir / "mscxrt_comparison.svg")


_EPOCH_RE = re.compile(
    r"epoch\s+(\d+)/(\d+)\s+\|\s+loss\s+([0-9.]+)\s+\|\s+val prog-F1\s+([0-9.]+)\s+"
    r"change-F1\s+([0-9.]+).*stable:([0-9.]+),\s+improved:([0-9.]+),\s+worsened:([0-9.]+)"
)


def _parse_train_log(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _EPOCH_RE.search(line)
        if m:
            rows.append({
                "epoch": int(m.group(1)),
                "epochs": int(m.group(2)),
                "loss": float(m.group(3)),
                "macro_f1": float(m.group(4)),
                "change_f1": float(m.group(5)),
                "stable": float(m.group(6)),
                "improved": float(m.group(7)),
                "worsened": float(m.group(8)),
            })
    return rows


def _line_svg(title: str, rows: list[dict], keys: list[str], path: Path) -> None:
    if not rows:
        return
    width, height = 760, 360
    left, right, top, bottom = 58, 24, 42, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    xvals = [r["epoch"] for r in rows]
    xmin, xmax = min(xvals), max(xvals)
    vals = [r[k] for r in rows for k in keys if k in r]
    ymin, ymax = min(vals), max(vals)
    if abs(ymax - ymin) < 1e-9:
        ymax += 1.0
    def xy(e, v):
        x = left + (e - xmin) / max(1, xmax - xmin) * plot_w
        y = top + (ymax - v) / (ymax - ymin) * plot_h
        return x, y
    body = [_txt(24, 26, title, 18, weight="bold"),
            f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111"/>']
    for j, key in enumerate(keys):
        pts = [xy(r["epoch"], r[key]) for r in rows if key in r]
        color = COLORS[j % len(COLORS)]
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="' +
                    " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
        body.append(f'<rect x="{left + j*120}" y="{height-28}" width="12" height="12" fill="{color}"/>')
        body.append(_txt(left + j*120 + 18, height - 18, key, 12))
    body.append(_txt(left, height - 8, f"epoch {xmin}-{xmax}", 11))
    body.append(_txt(8, top + 12, f"{ymax:.3f}", 10))
    body.append(_txt(8, top + plot_h, f"{ymin:.3f}", 10))
    path.write_text(_svg_wrap(width, height, "\n".join(body)), encoding="utf-8")


def plot_train_logs(paths: list[Path], out_dir: Path) -> None:
    log_dir = out_dir / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for p in paths:
        rows = _parse_train_log(p)
        if not rows:
            continue
        run = _safe_name(p.name.replace(".train.log", ""))
        with open(log_dir / f"{run}.curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        _line_svg(f"{run} loss", rows, ["loss"], log_dir / f"{run}.loss.svg")
        _line_svg(f"{run} validation F1", rows, ["macro_f1", "change_f1", "stable", "improved", "worsened"],
                  log_dir / f"{run}.val_f1.svg")
        best = max(rows, key=lambda r: r["change_f1"])
        summary.append({"run": run, "best_epoch": best["epoch"], "best_change_f1": best["change_f1"],
                        "best_macro_f1": best["macro_f1"], "last_loss": rows[-1]["loss"]})
    if summary:
        with open(log_dir / "train_log_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        _bar_svg("M4 train-log best validation scores",
                 [(r["run"], [r["best_macro_f1"], r["best_change_f1"]]) for r in summary],
                 ["macro", "change"], log_dir / "train_log_comparison.svg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot M4 diagnostics/logs")
    p.add_argument("--diagnostics", nargs="*", type=Path, default=None)
    p.add_argument("--mscxrt", nargs="*", type=Path, default=None)
    p.add_argument("--train-logs", nargs="*", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/phase4"))
    p.add_argument("--scan-defaults", action="store_true",
                   help="scan artifacts/diagnostics, LOGS, and logs for likely M4 files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.scan_defaults or (args.diagnostics is None and args.mscxrt is None and args.train_logs is None):
        diagnostics = [Path(p) for p in glob.glob("artifacts/diagnostics/m4*.diagnostics.json")
                       if "mscxrt" not in Path(p).name]
        mscxrt = [Path(p) for p in glob.glob("artifacts/diagnostics/*mscxrt*.json")]
        logs = [Path(p) for p in glob.glob("LOGS/m4*.train.log") + glob.glob("logs/**/*.train.log", recursive=True)]
    else:
        diagnostics = args.diagnostics or []
        mscxrt = args.mscxrt or []
        logs = args.train_logs or []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics:
        plot_m4_diagnostics(diagnostics, args.out_dir)
    if mscxrt:
        plot_mscxrt(mscxrt, args.out_dir)
    if logs:
        plot_train_logs(logs, args.out_dir)
    print(f"[DONE] M4 plots/tables -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
