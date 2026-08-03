#!/usr/bin/env python3
"""Collect M2/M3/M4 progress and completed metrics into CSV, Markdown and JSON."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path


M3_NAMES = [
    "m3v2_vera_graph_lse_det", "m3v2_no_concept_det", "m3v2_concept_mlp_det",
    "m3v2_graph_global_fusion_det", "m3v2_global_only_det",
    "m3v2_graph_attention_det", "m3v2_graph_mean_det", "m3v2_graph_max_det",
    "m3v2_vera_graph_lse_gt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--m2-root", type=Path, required=True)
    parser.add_argument("--m3-diagdir", type=Path, required=True)
    parser.add_argument("--m4-diagdir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--m3-epochs", type=int, required=True)
    parser.add_argument("--m4-epochs", type=int, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}


def number(value: object) -> str:
    try:
        parsed = float(value)
        return "" if not math.isfinite(parsed) else f"{parsed:.6f}"
    except (TypeError, ValueError):
        return ""


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    targets = {"M3": args.m3_epochs, "M4": args.m4_epochs}
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    rows: list[dict] = []

    m2_success = load_json(args.m2_root / "M2.SUCCESS.json")
    rows.append({
        "stage": "M2", "run": "yolo29_full_inference",
        "status": "complete" if m2_success else "pending", "epochs": "", "target_epochs": "",
        "box_source": "detector", "val_primary": "", "test_primary": "", "val_aux": "", "test_aux": "",
        "best_checkpoint": str(args.m2_root / "predictions.jsonl") if m2_success else "", "updated_at": now,
    })

    run_dirs = {path.name: path for path in args.runs_root.glob("*") if path.is_dir()} if args.runs_root.exists() else {}
    names = M3_NAMES + sorted(name for name in run_dirs if name.startswith("m4v2_"))
    for name in names:
        stage = "M3" if name.startswith("m3v2_") else "M4"
        run = run_dirs.get(name, args.runs_root / name)
        last = load_checkpoint(run / "last.pt")
        best_path = run / "best.pt"
        best = load_checkpoint(best_path)
        epochs = int(last.get("epoch", -1)) + 1 if last else 0
        target = targets[stage]
        status = "complete" if epochs >= target and best_path.is_file() else ("running" if last or run.exists() else "pending")
        diagdir = args.m3_diagdir if stage == "M3" else args.m4_diagdir
        val, test = load_json(diagdir / f"{name}.val.json"), load_json(diagdir / f"{name}.test.json")
        if stage == "M3":
            val_primary, test_primary = val.get("image_auc_macro", best.get("val_auc")), test.get("image_auc_macro")
            val_aux, test_aux = val.get("image_f1_macro", best.get("val_f1")), test.get("image_f1_macro")
        else:
            val_primary, test_primary = val.get("change_f1_macro", best.get("val_change_f1")), test.get("change_f1_macro")
            val_aux, test_aux = val.get("prog_f1_macro", best.get("val_f1")), test.get("prog_f1_macro")
        rows.append({
            "stage": stage, "run": name, "status": status, "epochs": epochs, "target_epochs": target,
            "box_source": best.get("box_source", last.get("box_source", "")),
            "val_primary": number(val_primary), "test_primary": number(test_primary),
            "val_aux": number(val_aux), "test_aux": number(test_aux),
            "best_checkpoint": str(best_path) if best_path.is_file() else "", "updated_at": now,
        })

    fields = ["stage", "run", "status", "epochs", "target_epochs", "box_source", "val_primary",
              "test_primary", "val_aux", "test_aux", "best_checkpoint", "updated_at"]
    csv_tmp = args.results_dir / "runs.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_tmp, args.results_dir / "runs.csv")

    markdown = [
        "# VERA server result collector", "", f"Updated: `{now}`", "",
        "For M3, primary = image macro-AUC and auxiliary = image macro-F1. For M4, primary = change-only macro-F1 and auxiliary = progression macro-F1.", "",
        "| Stage | Run | Status | Epoch | Box | Val primary | Test primary | Val aux | Test aux |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        epoch = f"{row['epochs']}/{row['target_epochs']}" if row["target_epochs"] else "-"
        markdown.append(
            f"| {row['stage']} | `{row['run']}` | {row['status']} | {epoch} | {row['box_source']} | "
            f"{row['val_primary'] or '-'} | {row['test_primary'] or '-'} | "
            f"{row['val_aux'] or '-'} | {row['test_aux'] or '-'} |"
        )
    atomic_text(args.results_dir / "runs.md", "\n".join(markdown) + "\n")

    summary = {
        "updated_at": now,
        "counts": {status: sum(row["status"] == status for row in rows) for status in ("complete", "running", "pending")},
        "rows": rows,
    }
    atomic_text(args.results_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(f"[collector] {now}: {summary['counts']} -> {args.results_dir / 'runs.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
