"""Build report-side ground-truth tables from the local MIMIC label artifacts.

This is an audit view for qualitative inspection. It does not alter model
predictions and never turns uncertain (-100) labels into present or absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import constants as C


_PROG = {0: "stable", 1: "improved", 2: "worsened"}


def _metadata(path: Path, wanted: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row.get("image_id")
            if iid in wanted:
                found[iid] = row
                if len(found) == len(wanted):
                    break
    return found


def _manifest_rows(labels_dir: Path, wanted: set[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    with open(labels_dir / "manifest.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            iid = json.loads(line).get("image_id")
            if iid in wanted:
                found[iid] = i
                if len(found) == len(wanted):
                    break
    return found


def _bbox(boxes: np.ndarray, present: np.ndarray, region_idx: int) -> list[int] | None:
    if int(present[region_idx]) != 1:
        return None
    box = [int(x) for x in boxes[region_idx].tolist()]
    return box if box[2] > box[0] and box[3] > box[1] else None


def _concept_evidence(concept_labels: np.ndarray, region_idx: int, disease: str) -> list[dict]:
    allowed = C.CONCEPTS_BY_DISEASE.get(disease, set())
    return [
        {"concept": name, "confidence": 1.0}
        for concept_idx, name in enumerate(C.CONCEPT_NAMES)
        if name in allowed and int(concept_labels[region_idx, concept_idx]) == 1
    ]


def _state_rows(image_labels: np.ndarray, region_labels: np.ndarray,
                concept_labels: np.ndarray, boxes: np.ndarray,
                present: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for d, disease in enumerate(C.CHEX_NAMES):
        if disease == C.NO_FINDING:
            continue
        value = int(image_labels[d])
        if value not in (0, 1):
            continue
        state = "present" if value == 1 else "absent"
        for region_idx, cell in enumerate(region_labels[:, d]):
            if int(cell) != value or int(present[region_idx]) != 1:
                continue
            rows.append({
                "disease": disease,
                "region": C.REGION_NAMES[region_idx],
                "state": state,
                "evidence": (_concept_evidence(concept_labels, region_idx, disease)
                             if state == "present" else []),
                "confidence": 1.0,
                "bbox": _bbox(boxes, present, region_idx),
                "source": "ground_truth_label",
            })
    return rows


def _progression_rows(progression: np.ndarray, current_image: np.ndarray,
                      prior_image: np.ndarray | None, current_regions: np.ndarray,
                      prior_regions: np.ndarray | None, current_boxes: np.ndarray,
                      prior_boxes: np.ndarray | None, current_present: np.ndarray,
                      prior_present: np.ndarray | None) -> list[dict]:
    rows: list[dict] = []
    for d, disease in enumerate(C.CHEX_NAMES):
        if disease == C.NO_FINDING:
            continue
        cur_state = int(current_image[d])
        prior_state = int(prior_image[d]) if prior_image is not None else -100
        groups: list[tuple[str, list[int]]] = []
        if cur_state == 1 and prior_state == 0:
            groups.append(("new", [r for r, cell in enumerate(current_regions[:, d]) if int(cell) == 1]))
        elif cur_state == 0 and prior_state == 1 and prior_regions is not None:
            groups.append(("resolved", [r for r, cell in enumerate(prior_regions[:, d]) if int(cell) == 1]))
        else:
            for cls, name in _PROG.items():
                region_ids = [r for r, cell in enumerate(progression[:, d]) if int(cell) == cls]
                if region_ids:
                    groups.append((name, region_ids))

        for change, region_ids in groups:
            if not region_ids:
                continue
            regions = []
            for region_idx in region_ids:
                regions.append({
                    "region": C.REGION_NAMES[region_idx],
                    "current_bbox": _bbox(current_boxes, current_present, region_idx),
                    "prior_bbox": (_bbox(prior_boxes, prior_present, region_idx)
                                   if prior_boxes is not None and prior_present is not None else None),
                })
            lead_region = C.REGION_NAMES[region_ids[0]] if change in {"new", "resolved"} and len(region_ids) == 1 else None
            rows.append({
                "disease": disease,
                "change": change,
                "regions": regions,
                "lead_region": lead_region,
                "lead_bbox": ({
                    "current": regions[0]["current_bbox"],
                    "prior": regions[0]["prior_bbox"],
                } if lead_region else None),
                "confidence": 1.0,
                "lead_region_source": ("single_presence_crossing" if lead_region
                                       else "not_annotated"),
                "source": "ground_truth_progression_label",
            })
    return rows


def _image_path(meta: dict, image_id: str | None) -> str | None:
    if not image_id:
        return None
    raw_path = meta.get(image_id, {}).get("image_path")
    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return str(candidate)
    patient = image_id.split("_s", 1)[0].removeprefix("MIMIC_p")
    local = C.REPO_ROOT / "data" / "mimic-cxr-448" / f"p{patient[:2]}" / f"p{patient}" / f"{image_id}.jpg"
    return str(local) if local.exists() else None


def attach_ground_truth(report: dict, *, metadata_path: Path,
                        m3_labels_dir: Path, m4_labels_dir: Path) -> dict:
    """Attach one current-image GT table, one interval GT table, and metadata text."""
    current_id = report.get("image_id")
    prior_id = report.get("prior_image_id")
    wanted = {x for x in (current_id, prior_id) if x}
    meta = _metadata(Path(metadata_path), wanted)
    m3_row_idx = _manifest_rows(Path(m3_labels_dir), wanted)
    m4_row_idx = _manifest_rows(Path(m4_labels_dir), {current_id})
    image_labels = np.load(Path(m3_labels_dir) / "image_chexpert.npy", mmap_mode="r")
    region_labels = np.load(Path(m3_labels_dir) / "region_chexpert.npy", mmap_mode="r")
    concept_labels = np.load(Path(m3_labels_dir) / "region_concepts.npy", mmap_mode="r")
    boxes = np.load(Path(m3_labels_dir) / "boxes.npy", mmap_mode="r")
    present = np.load(Path(m3_labels_dir) / "present_mask.npy", mmap_mode="r")
    progression = np.load(Path(m4_labels_dir) / "progression.npy", mmap_mode="r")

    current_idx = m3_row_idx.get(current_id)
    if current_idx is None:
        return {**report, "ground_truth": {"available": False}}

    prior_idx = m3_row_idx.get(prior_id)
    current_table = _state_rows(
        image_labels[current_idx], region_labels[current_idx], concept_labels[current_idx],
        boxes[current_idx], present[current_idx],
    )
    progression_idx = m4_row_idx.get(current_id)
    interval_table = []
    if prior_id and progression_idx is not None:
        interval_table = _progression_rows(
            progression[progression_idx], image_labels[current_idx],
            image_labels[prior_idx] if prior_idx is not None else None,
            region_labels[current_idx], region_labels[prior_idx] if prior_idx is not None else None,
            boxes[current_idx], boxes[prior_idx] if prior_idx is not None else None,
            present[current_idx], present[prior_idx] if prior_idx is not None else None,
        )

    current_path = _image_path(meta, current_id)
    prior_path = _image_path(meta, prior_id)
    gt_classification_box_map: dict[str, dict] = {}
    for row in current_table:
        if row["state"] != "present" or not row.get("bbox"):
            continue
        box = gt_classification_box_map.setdefault(row["region"], {
            "region": row["region"], "bbox": row["bbox"], "diseases": [],
        })
        box["diseases"].append(row["disease"])
    gt_classification_boxes = list(gt_classification_box_map.values())
    gt_progression_current_boxes = [{
        "disease": row["disease"], "change": row["change"],
        "region": region["region"], "bbox": region["current_bbox"],
    } for row in interval_table for region in row["regions"] if region.get("current_bbox")]
    gt_progression_prior_boxes = [{
        "disease": row["disease"], "change": row["change"],
        "region": region["region"], "bbox": region["prior_bbox"],
    } for row in interval_table for region in row["regions"] if region.get("prior_bbox")]

    gt = {
        "available": True,
        "image_id": current_id,
        "prior_image_id": prior_id,
        "classification": {
            "image": {"image_id": current_id, "path": current_path, "boxes": gt_classification_boxes},
            "table": current_table,
        },
        "progression": {
            "images": {
                "prior": {"image_id": prior_id, "path": prior_path, "boxes": gt_progression_prior_boxes},
                "current": {"image_id": current_id, "path": current_path,
                            "boxes": gt_progression_current_boxes},
            },
            "table": interval_table,
        } if prior_id else None,
        "current_table": current_table,
        "interval_table": interval_table,
        "text": meta.get(current_id, {}).get("report"),
        "image_path": current_path,
        "source": "MIMIC metadata + data/m3_labels + data/m4_labels",
        "uncertain_label_policy": "-100 labels are omitted",
        "lead_region_policy": "null unless a new/resolved GT crossing contains exactly one region",
    }
    report["classification"]["image"]["path"] = current_path
    if report.get("progression") is not None:
        report["progression"]["images"]["current"]["path"] = current_path
        report["progression"]["images"]["prior"]["path"] = prior_path
    return {**report, "ground_truth": gt}
