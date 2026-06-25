"""Core conversion: ImaGenome *_SceneGraph.json  ->  YOLO label lines.

A scene graph holds `objects[]`, each with a `bbox_name` and resized-space box
`x1,y1,x2,y2` (the same space as the resized image, per the project convention).
We keep only the 29 canonical regions, drop the (0,0,0,0) sentinel and degenerate
boxes, clip to image bounds, and emit normalized YOLO lines:

    <class_id> <cx> <cy> <w> <h>     # all in [0,1]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from constants import CLASS_TO_ID, canonical_name

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SCENE_SUFFIX = "_SceneGraph.json"


def dicom_id_from_image_id(image_id: str) -> str:
    """MIMIC image_id = MIMIC_<patient>_<study>_<dicom>; dicom is the last field
    (it contains hyphens, never underscores, so split on the first 3 '_')."""
    parts = image_id.split("_", 3)
    return parts[3] if len(parts) == 4 else ""


def dicom_id_from_scene_filename(filename: str) -> str:
    return filename[: -len(SCENE_SUFFIX)] if filename.endswith(SCENE_SUFFIX) else ""


@dataclass
class ConvertStats:
    objects_total: int = 0
    kept: int = 0
    dropped_not_canonical: int = 0
    dropped_sentinel: int = 0
    dropped_degenerate: int = 0
    dropped_out_of_bounds: int = 0
    clipped: int = 0

    def add(self, other: "ConvertStats") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))


def scene_to_yolo_lines(
    scene: dict[str, Any], img_w: int, img_h: int, bounds_tol: float = 0.02
) -> tuple[list[str], ConvertStats]:
    """Convert one scene-graph dict to YOLO lines for an `img_w x img_h` image.

    bounds_tol: a box whose center falls outside the image by more than this
    fraction is dropped (signals an image/bbox scale mismatch); boxes slightly
    past the edge are clipped instead.
    """
    stats = ConvertStats()
    lines: list[str] = []
    if img_w <= 0 or img_h <= 0:
        return lines, stats

    margin_x = img_w * bounds_tol
    margin_y = img_h * bounds_tol

    for obj in scene.get("objects", []) or []:
        stats.objects_total += 1
        name = canonical_name(str(obj.get("bbox_name", "")))
        if name is None:
            stats.dropped_not_canonical += 1
            continue

        try:
            x1, y1, x2, y2 = (
                float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            stats.dropped_degenerate += 1
            continue

        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            stats.dropped_sentinel += 1
            continue

        # normalize ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # reject boxes that sit well outside the image (scale mismatch guard)
        if (
            x2 < -margin_x or y2 < -margin_y
            or x1 > img_w + margin_x or y1 > img_h + margin_y
        ):
            stats.dropped_out_of_bounds += 1
            continue

        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(float(img_w), x2), min(float(img_h), y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            stats.clipped += 1

        bw, bh = cx2 - cx1, cy2 - cy1
        if bw <= 1.0 or bh <= 1.0:
            stats.dropped_degenerate += 1
            continue

        cx = (cx1 + cx2) / 2.0 / img_w
        cy = (cy1 + cy2) / 2.0 / img_h
        nw = bw / img_w
        nh = bh / img_h
        lines.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        stats.kept += 1

    return lines, stats


def index_images(root: Path) -> dict[str, Path]:
    """Map image filename stem -> path, for all images under `root`."""
    index: dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMAGE_SUFFIXES and stem not in index:
                index[stem] = Path(dirpath) / fn
    return index


def index_scene_graphs(root: Path) -> dict[str, Path]:
    """Map dicom_id -> *_SceneGraph.json path, for all scene graphs under `root`."""
    index: dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(SCENE_SUFFIX):
                dicom = dicom_id_from_scene_filename(fn)
                if dicom and dicom not in index:
                    index[dicom] = Path(dirpath) / fn
    return index


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    import json

    with open(path, "r", encoding="utf-8-sig") as stream:  # utf-8-sig tolerates a BOM
        for raw in stream:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row
