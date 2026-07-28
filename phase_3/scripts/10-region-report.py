"""Render a human-readable audit of M3 regional disease/concept predictions.

Input is the diagnostics JSON produced by `5-eval.py --diagnostics-json ...`.
Outputs Markdown, summary/detail CSV files, and conditional/end-to-end F1 heatmaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = {
    "disease": "regional_disease_breakdown",
    "concept": "regional_concept_breakdown",
}


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{x:.{digits}f}" if math.isfinite(x) else "--"


def _class_names(breakdown: dict) -> list[str]:
    regions = breakdown.get("per_region", {})
    if not regions:
        return []
    first = next(iter(regions.values()))
    return list(first["conditional_on_detected"]["per_class"])


def write_summary_csv(task: str, breakdown: dict, path: Path) -> None:
    fields = [
        "task", "region_index", "region", "detected", "gt_present",
        "detector_precision", "detector_recall", "detector_f1",
        "conditional_macro_f1", "conditional_macro_auc", "conditional_supported_classes",
        "end_to_end_macro_f1", "end_to_end_macro_auc", "end_to_end_supported_classes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for region, rec in breakdown["per_region"].items():
            cov = rec["coverage"]
            cond = rec["conditional_on_detected"]
            e2e = rec["end_to_end"]
            writer.writerow({
                "task": task, "region_index": rec["region_index"], "region": region,
                "detected": cov["n_detected"], "gt_present": cov["n_gt_present"],
                "detector_precision": cov["precision"], "detector_recall": cov["recall"],
                "detector_f1": cov["f1"], "conditional_macro_f1": cond["macro_f1"],
                "conditional_macro_auc": cond["macro_auc"],
                "conditional_supported_classes": cond["n_supported_classes"],
                "end_to_end_macro_f1": e2e["macro_f1"],
                "end_to_end_macro_auc": e2e["macro_auc"],
                "end_to_end_supported_classes": e2e["n_supported_classes"],
            })


def write_detail_csv(task: str, breakdown: dict, path: Path) -> None:
    fields = [
        "task", "region_index", "region", "label", "protocol", "supported",
        "n", "positive", "negative", "f1_at_0_5", "auc",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for region, region_rec in breakdown["per_region"].items():
            for protocol_key, protocol_name in (
                ("conditional_on_detected", "conditional"), ("end_to_end", "end_to_end")
            ):
                for label, rec in region_rec[protocol_key]["per_class"].items():
                    writer.writerow({
                        "task": task, "region_index": region_rec["region_index"],
                        "region": region, "label": label, "protocol": protocol_name,
                        "supported": rec["supported"], "n": rec["n"],
                        "positive": rec["pos"], "negative": rec["neg"],
                        "f1_at_0_5": rec["f1_at_0_5"], "auc": rec["auc"],
                    })


def render_heatmap(task: str, breakdown: dict, protocol: str, path: Path) -> None:
    regions = list(breakdown["per_region"])
    classes = _class_names(breakdown)
    matrix = np.full((len(regions), len(classes)), np.nan, dtype=np.float64)
    for ri, region in enumerate(regions):
        table = breakdown["per_region"][region][protocol]["per_class"]
        for ci, label in enumerate(classes):
            rec = table[label]
            if rec.get("supported"):
                value = rec.get("f1_at_0_5")
                if value is not None and math.isfinite(float(value)):
                    matrix[ri, ci] = float(value)

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#d9d9d9")
    width = max(14.0, len(classes) * (0.30 if task == "concept" else 0.62))
    fig, ax = plt.subplots(figsize=(width, 11.5))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(regions)), labels=regions, fontsize=8)
    title_protocol = "conditional on detected regions" if protocol == "conditional_on_detected" \
        else "end-to-end (missed GT regions forced absent)"
    ax.set_title(f"M3 regional {task} F1@0.5: {title_protocol}")
    ax.set_xlabel(f"{task} class")
    ax.set_ylabel("anatomical region")
    fig.colorbar(image, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_report(data: dict, available: dict[str, dict], source: Path) -> str:
    lines = [
        "# M3 Regional Prediction Audit",
        "",
        f"Source diagnostics: `{source}`",
        "",
        "## How to read this report",
        "",
        "- **Cell-pooled** metrics flatten every detected anatomical slot before computing a class macro average.",
        "- **Conditional** metrics preserve the 29 region identities but evaluate only slots found by the detector.",
        "- **End-to-end** metrics evaluate GT-present slots and force a missed detector slot to probability zero; a missed positive therefore becomes a false negative.",
        "- A region/class cell is supported only when it reaches the configured minimum number of explicit positive and negative labels. Unsupported cells are gray in the heatmaps.",
        "- Targets equal to `-100` remain unknown and never become negatives.",
        "",
        "## Global summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Image disease macro-F1 | {_number(data.get('image_f1_macro'))} |",
        f"| Image disease macro-AUC | {_number(data.get('image_auc_macro'))} |",
        f"| Cell-pooled regional disease macro-F1 | {_number(data.get('region_f1_macro'))} |",
        f"| Cell-pooled regional disease macro-AUC | {_number(data.get('region_auc_macro'))} |",
        f"| Cell-pooled regional concept macro-F1 | {_number(data.get('concept_f1_macro'))} |",
        f"| Cell-pooled regional concept macro-AUC | {_number(data.get('concept_auc_macro'))} |",
    ]
    first = next(iter(available.values()), None)
    if first:
        cov = first["coverage"]
        lines.extend([
            f"| Detector slot precision | {_number(cov.get('precision'))} |",
            f"| Detector slot recall | {_number(cov.get('recall'))} |",
            f"| Detector slot F1 | {_number(cov.get('f1'))} |",
        ])

    for task, breakdown in available.items():
        policy = breakdown["support_policy"]
        macro = breakdown["macro_over_regions"]
        lines.extend([
            "",
            f"## Regional {task.capitalize()} Summary",
            "",
            f"Support gate: at least {policy['min_positive']} positives and "
            f"{policy['min_negative']} negatives per `(region, {task})` cell.",
            "",
            f"Macro over 29 regions: conditional F1 `{_number(macro['conditional_f1'])}`, "
            f"end-to-end F1 `{_number(macro['end_to_end_f1'])}`.",
            "",
            "| Region | Det./GT | Recall | Cond. F1 | Supported | E2E F1 | Supported |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for region, rec in breakdown["per_region"].items():
            cov = rec["coverage"]
            cond = rec["conditional_on_detected"]
            e2e = rec["end_to_end"]
            lines.append(
                f"| {region} | {cov['n_detected']}/{cov['n_gt_present']} | "
                f"{_number(cov['recall'], 3)} | {_number(cond['macro_f1'])} | "
                f"{cond['n_supported_classes']}/{cond['n_classes']} | "
                f"{_number(e2e['macro_f1'])} | {e2e['n_supported_classes']}/{e2e['n_classes']} |"
            )
        lines.extend([
            "",
            f"- Conditional heatmap: `{task}_f1_conditional.png`",
            f"- End-to-end heatmap: `{task}_f1_end_to_end.png`",
            f"- Per-region summary: `{task}_region_summary.csv`",
            f"- Per-cell metrics and support: `{task}_region_class_detail.csv`",
        ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render M3 regional diagnostics")
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    available = {task: data[key] for task, key in TASKS.items() if key in data}
    if not available:
        raise SystemExit("[ERROR] diagnostics has no regional breakdown; rerun 5-eval.py with the updated evaluator")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task, breakdown in available.items():
        write_summary_csv(task, breakdown, args.out_dir / f"{task}_region_summary.csv")
        write_detail_csv(task, breakdown, args.out_dir / f"{task}_region_class_detail.csv")
        render_heatmap(task, breakdown, "conditional_on_detected",
                       args.out_dir / f"{task}_f1_conditional.png")
        render_heatmap(task, breakdown, "end_to_end",
                       args.out_dir / f"{task}_f1_end_to_end.png")
    report = args.out_dir / "regional_audit.md"
    report.write_text(markdown_report(data, available, args.diagnostics), encoding="utf-8")
    print(f"[DONE] regional audit -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
