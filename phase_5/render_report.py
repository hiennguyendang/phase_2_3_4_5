"""Render VERA JSONL reports as self-contained, paper-like HTML previews.

The renderer does not change predictions. It embeds the source CXRs, overlays
the boxes referenced by the visible rows, and keeps the complete tables in
expandable sections for audit.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from collections import defaultdict
from pathlib import Path


CANDIDATE_LABELS = {
    "MIMIC_p10019385_s55200571_ec57652d-415ec4ea-c342e27a-53cce3cc-f83ef7a9": "A",
    "MIMIC_p10013643_s53841005_6e58e826-fc08e4ee-a9ff0245-34c6f4cd-cd0aedc8": "B",
    "MIMIC_p10013569_s57874790_f1057fa6-32b37b7c-913c3d01-dcd710bc-b269dde8": "C",
}

COLORS = ["#d1495b", "#00798c", "#edae49", "#30638e", "#6a994e", "#7b2cbf"]
REGION_NAMES = [
    "abdomen", "aortic arch", "cardiac silhouette", "carina", "cavoatrial junction",
    "left apical zone", "left clavicle", "left costophrenic angle", "left hemidiaphragm",
    "left hilar structures", "left lower lung zone", "left lung", "left mid lung zone",
    "left upper lung zone", "mediastinum", "right apical zone", "right atrium",
    "right clavicle", "right costophrenic angle", "right hemidiaphragm",
    "right hilar structures", "right lower lung zone", "right lung", "right mid lung zone",
    "right upper lung zone", "spine", "svc", "trachea", "upper mediastinum",
]
REGION_PALETTE = [
    "#e63946", "#f77f00", "#2a9d8f", "#457b9d", "#8e44ad", "#d1495b", "#118ab2",
    "#06a77d", "#c44536", "#5c4d7d", "#ef476f", "#1d4e89", "#6a994e", "#bc6c25",
    "#7b2cbf", "#00798c", "#f9844a", "#3a86ff", "#d62828", "#588157", "#8338ec",
    "#277da1", "#f3722c", "#43aa8b", "#577590", "#9b5de5", "#00b4d8", "#f9c74f",
    "#90be6d",
]
REGION_COLORS = dict(zip(REGION_NAMES, REGION_PALETTE))


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def image_uri(path: str | None) -> str:
    if not path:
        return ""
    source = Path(path)
    if not source.exists():
        return ""
    mime = mimetypes.guess_type(source.name)[0] or "image/jpeg"
    payload = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def evidence_text(evidence: list[dict] | None) -> str:
    if not evidence:
        return "-"
    return ", ".join(esc(item.get("concept")) for item in evidence)


def region_label(region: str | None) -> str:
    if not region:
        return "-"
    color = REGION_COLORS.get(str(region), "#34495e")
    return f"<span class='region-label' style='--region-color:{color}'>{esc(region)}</span>"


def region_text(regions: list[dict] | None) -> str:
    names = [str(item.get("region")) for item in (regions or []) if item.get("region")]
    return ", ".join(region_label(name) for name in names) if names else "-"


def representative_classification(rows: list[dict], per_disease: int = 2) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        disease = str(row.get("disease", ""))
        if disease not in grouped:
            order.append(disease)
        grouped[disease].append(row)
    selected = []
    for disease in order:
        ranked = sorted(grouped[disease], key=lambda row: float(row.get("confidence") or 0), reverse=True)
        selected.extend(ranked[:per_disease])
    return selected


def classification_table(rows: list[dict], table_id: str) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr data-disease='{disease}' data-state='{state}'>"
            "<td>{disease}</td><td>{region}</td><td><span class='state {state}'>{state}</span></td>"
            "<td>{evidence}</td><td class='number'>{confidence:.3f}</td></tr>".format(
                disease=esc(row.get("disease")), region=region_label(row.get("region")),
                state=esc(row.get("state")), evidence=evidence_text(row.get("evidence")),
                confidence=float(row.get("confidence") or 0),
            )
        )
    return (
        f"<table class='classification-table' id='{esc(table_id)}'><thead><tr><th>Disease</th><th>Region</th>"
        "<th>State</th><th>Evidence</th><th>Confidence</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def progression_table(rows: list[dict], table_id: str) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr data-disease='{disease}' data-change='{change}'>"
            "<td>{disease}</td><td><span class='state {change}'>{change}</span></td>"
            "<td>{regions}</td><td>{lead}</td><td class='number'>{confidence:.3f}</td></tr>".format(
                disease=esc(row.get("disease")), change=esc(row.get("change")),
                regions=region_text(row.get("regions")), lead=region_label(row.get("lead_region")),
                confidence=float(row.get("confidence") or 0),
            )
        )
    return (
        f"<table class='progression-table' id='{esc(table_id)}'><thead><tr><th>Disease</th><th>Change</th>"
        "<th>Region(s)</th><th>Lead region</th><th>Confidence</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def overlay_figure(image: dict | None, title: str, allowed_regions: set[str] | None = None) -> str:
    image = image or {}
    uri = image_uri(image.get("path"))
    boxes = image.get("boxes") or []
    if allowed_regions is not None:
        boxes = [box for box in boxes if box.get("region") in allowed_regions]
    overlays = []
    seen = set()
    for index, box in enumerate(boxes):
        coords = box.get("bbox")
        region = str(box.get("region") or "region")
        if not coords or len(coords) != 4 or (region, tuple(coords)) in seen:
            continue
        seen.add((region, tuple(coords)))
        x1, y1, x2, y2 = [float(value) for value in coords]
        color = REGION_COLORS.get(region, COLORS[index % len(COLORS)])
        overlays.append(
            f"<rect x='{x1}' y='{y1}' width='{max(1, x2-x1)}' height='{max(1, y2-y1)}' "
            f"stroke='{color}'/><text x='{x1+3}' y='{max(12, y1+13)}' fill='{color}'>{esc(region)}</text>"
        )
    missing = "<div class='missing'>Image unavailable</div>" if not uri else ""
    return (
        "<figure><div class='cxr-frame'>"
        f"<img src='{uri}' alt='{esc(title)}'/>{missing}"
        f"<svg viewBox='0 0 448 448' aria-hidden='true'>{''.join(overlays)}</svg>"
        f"</div><figcaption>{esc(title)}</figcaption></figure>"
    )


def section_header(kicker: str, title: str, note: str = "") -> str:
    return (
        "<header class='section-head'><div>"
        f"<div class='kicker'>{esc(kicker)}</div><h2>{esc(title)}</h2></div>"
        f"<p>{esc(note)}</p></header>"
    )


def render_report(report: dict) -> str:
    candidate = CANDIDATE_LABELS.get(report.get("image_id"), "")
    classification = report.get("classification") or {}
    class_all = classification.get("table") or []
    class_visible = representative_classification(class_all)
    class_regions = {str(row.get("region")) for row in class_visible if row.get("state") == "present"}

    progression = report.get("progression") or {}
    prog_rows = progression.get("table") or []
    prog_regions = {str(row.get("lead_region")) for row in prog_rows if row.get("lead_region")}

    gt = report.get("ground_truth") or {}
    gt_class = (gt.get("classification") or {}).get("table") or []
    gt_prog = (gt.get("progression") or {}).get("table") or []
    gt_class_regions = {str(row.get("region")) for row in gt_class if row.get("state") == "present"}
    gt_prog_regions = {str(row.get("lead_region")) for row in gt_prog if row.get("lead_region")}

    policy = report.get("threshold_policy", "unknown")
    provisional = policy == "fixed_provisional_fallback"
    warning = (
        "Provisional visualization: fixed smoke-test thresholds; rerender after validation calibration."
        if provisional else "Validation-fitted per-disease dual thresholds."
    )

    images = progression.get("images") or {}
    gt_images = (gt.get("progression") or {}).get("images") or {}
    title = f"Candidate {candidate} · VERA structured report" if candidate else "VERA structured report"
    all_class_details = ""
    if len(class_visible) < len(class_all):
        all_class_details = (
            f"<details><summary>Inspect all {len(class_all)} classification rows</summary>"
            f"<div class='table-scroll'>{classification_table(class_all, 'classification-all')}</div></details>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head>
<body><main>
  <header class="report-head"><div><div class="brand">VERA</div><h1>{esc(title)}</h1></div>
    <div class="status {'provisional' if provisional else ''}">{esc(warning)}</div></header>

  <section>
    {section_header('01 · Current study', 'Classification', f'Showing {len(class_visible)} representative rows; {len(class_all)} retained in the source report.')}
    <div class="content-grid single">
      <div class="images">{overlay_figure(classification.get('image'), 'Current CXR', class_regions)}</div>
      <div class="readout">{classification_table(class_visible, 'classification-visible')}
        <h3>Findings</h3><p class="narrative">{esc(classification.get('text') or report.get('current_text'))}</p>
        {all_class_details}</div>
    </div>
  </section>

  <section>
    {section_header('02 · Longitudinal comparison', 'Progression', 'Stable is an explicit model class; lead boxes are shown only for non-stable changes.')}
    <div class="content-grid pair">
      <div class="images two">{overlay_figure(images.get('prior'), 'Prior CXR', prog_regions)}{overlay_figure(images.get('current'), 'Current CXR', prog_regions)}</div>
      <div class="readout">{progression_table(prog_rows, 'progression-visible')}
        <h3>Interval changes</h3><p class="narrative">{esc(progression.get('text') or report.get('interval_text'))}</p></div>
    </div>
  </section>

  <section class="ground-truth">
    {section_header('03 · Reference', 'Ground truth', 'Reference labels and boxes are shown separately; the prose is copied once from metadata.')}
    <div class="content-grid single">
      <div class="images">{overlay_figure((gt.get('classification') or {{}}).get('image'), 'GT current CXR', gt_class_regions)}</div>
      <div class="readout">{classification_table(gt_class, 'gt-classification')}</div>
    </div>
    <div class="content-grid pair gt-progression">
      <div class="images two">{overlay_figure(gt_images.get('prior'), 'GT prior CXR', gt_prog_regions)}{overlay_figure(gt_images.get('current'), 'GT current CXR', gt_prog_regions)}</div>
      <div class="readout">{progression_table(gt_prog, 'gt-progression')}</div>
    </div>
    <div class="reference-text"><h3>Reference report</h3><p>{esc(gt.get('text'))}</p></div>
  </section>
</main></body></html>"""


CSS = r"""
:root { --ink:#17212b; --muted:#5b6773; --line:#9ba9b7; --soft:#eef3f7; --header:#c8d8ea;
  --present:#9f2d3f; --absent:#176b87; --stable:#5c6670; --improved:#247b5a; --worsened:#a63d40; }
* { box-sizing:border-box; }
body { margin:0; background:#f4f6f8; color:var(--ink); font-family:Arial,Helvetica,sans-serif; font-size:14px; letter-spacing:0; }
main { width:min(1500px,calc(100% - 32px)); margin:18px auto 48px; background:#fff; border:1px solid #c8d0d8; }
.report-head { min-height:92px; padding:18px 24px; display:flex; align-items:center; justify-content:space-between; gap:24px; border-bottom:3px solid #25394d; }
.brand { font-size:13px; font-weight:800; letter-spacing:0; color:#30638e; } h1 { margin:5px 0 0; font-size:25px; }
.status { max-width:440px; padding:9px 12px; border-left:4px solid #247b5a; background:#edf7f2; font-size:12px; line-height:1.4; }
.status.provisional { border-color:#c27c0e; background:#fff7e5; }
section { padding:20px 24px 26px; border-bottom:1px solid #cbd3da; }
.section-head { display:flex; align-items:end; justify-content:space-between; gap:24px; margin-bottom:16px; }
.section-head h2 { margin:3px 0 0; font-size:21px; } .section-head p { max-width:620px; margin:0; color:var(--muted); text-align:right; }
.kicker { color:#47647f; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0; }
.content-grid,.content-grid.pair { display:block; }
.images { display:grid; gap:12px; margin:0 auto 18px; max-width:480px; }
.images.two { grid-template-columns:repeat(2,minmax(0,1fr)); max-width:760px; }
figure { margin:0; min-width:0; } .cxr-frame { position:relative; width:100%; aspect-ratio:1; background:#071017; overflow:hidden; border:1px solid #4c5c69; }
.cxr-frame img { display:block; width:100%; height:100%; object-fit:contain; }
.cxr-frame svg { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
.cxr-frame rect { fill:none; stroke-width:2; vector-effect:non-scaling-stroke; }
.cxr-frame text { font-size:10px; font-weight:700; paint-order:stroke; stroke:#071017; stroke-width:2px; }
.missing { position:absolute; inset:0; display:grid; place-items:center; color:#fff; }
figcaption { padding-top:7px; text-align:center; font-weight:700; }
.readout { min-width:0; } table { width:100%; border-collapse:collapse; table-layout:fixed; }
th,td { border:1px solid #62717f; padding:9px 10px; vertical-align:top; line-height:1.35; overflow-wrap:anywhere; }
th { background:var(--header); text-align:center; font-weight:800; }
.classification-table th:nth-child(1),.classification-table td:nth-child(1) { width:21%; }
.classification-table th:nth-child(2),.classification-table td:nth-child(2) { width:20%; }
.classification-table th:nth-child(3),.classification-table td:nth-child(3) { width:11%; }
.classification-table th:nth-child(4),.classification-table td:nth-child(4) { width:38%; }
.classification-table th:last-child,.classification-table td:last-child { width:10%; }
.progression-table th:nth-child(1),.progression-table td:nth-child(1) { width:18%; }
.progression-table th:nth-child(2),.progression-table td:nth-child(2) { width:12%; }
.progression-table th:nth-child(3),.progression-table td:nth-child(3) { width:38%; }
.progression-table th:nth-child(4),.progression-table td:nth-child(4) { width:22%; }
.progression-table th:last-child,.progression-table td:last-child { width:10%; }
.number { text-align:right; font-variant-numeric:tabular-nums; font-weight:700; }
.state { display:inline-block; font-weight:800; text-transform:capitalize; }
.state.present,.state.worsened,.state.new { color:var(--worsened); }
.state.absent,.state.improved,.state.resolved { color:var(--improved); }
.state.stable { color:var(--stable); }
.region-label { color:var(--region-color); font-weight:800; }
h3 { margin:15px 0 6px; font-size:16px; } .narrative,.reference-text p { margin:0; font-weight:600; line-height:1.55; }
details { margin-top:14px; border-top:1px solid #cbd3da; padding-top:10px; }
summary { cursor:pointer; color:#30638e; font-weight:700; } .table-scroll { max-height:520px; overflow:auto; margin-top:10px; }
.ground-truth { background:#fafbfc; } .gt-progression { margin-top:22px; padding-top:22px; border-top:1px solid #cbd3da; }
.reference-text { margin-top:20px; padding:14px 16px; border-left:4px solid #47647f; background:var(--soft); }
tr.match { background:#eaf6ef; } tr.mismatch { background:#fff0f0; } tr.partial { background:#fff8df; }
@media (max-width:950px) { .section-head,.report-head { align-items:flex-start; flex-direction:column; } .section-head p { text-align:left; } }
@media print { body { background:#fff; } main { width:100%; margin:0; border:0; } section { break-inside:avoid; } details { display:none; } }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render VERA report JSONL as self-contained HTML")
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-id", action="append", default=[], help="optional current image ID filter")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wanted = set(args.image_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with args.reports.open(encoding="utf-8") as source:
        for line in source:
            report = json.loads(line)
            image_id = report.get("image_id")
            if not report.get("has_prior") or (wanted and image_id not in wanted):
                continue
            candidate = CANDIDATE_LABELS.get(image_id, "report")
            patient = image_id.split("_")[1] if image_id and "_" in image_id else "case"
            target = args.out_dir / f"{candidate}_{patient}_report.html"
            target.write_text(render_report(report), encoding="utf-8")
            written.append(target)
    if not written:
        raise SystemExit("[ERROR] no matching longitudinal reports")
    for target in written:
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
