"""image_id helpers + CXR file-path resolution for phase_1.

image_id convention (repo-wide): "MIMIC_p<patient>_s<study>_<dicom>", where <dicom> is the
last field (it contains hyphens, never underscores). The 448 jpgs mirror MIMIC-CXR-JPG's
sharded layout:  <root>/p<patient[:2]>/p<patient>/<image_id>.jpg
"""

from __future__ import annotations

import os
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def dicom_from_image_id(image_id: str) -> str:
    """'MIMIC_p..._s..._<dicom>' -> <dicom> (split on the first 3 '_'; dicom keeps its hyphens)."""
    parts = image_id.split("_", 3)
    return parts[3] if len(parts) == 4 else ""


def patient_folder(image_id: str) -> str:
    """-> 'p<patient>' (the second underscore field), e.g. MIMIC_p10000032_s.. -> 'p10000032'."""
    parts = image_id.split("_")
    return parts[1] if len(parts) > 1 and parts[1].startswith("p") else ""


def guess_image_path(root: Path, image_id: str) -> Path | None:
    """Direct guess at <root>/p<pid[:2]>/p<pid>/<image_id>.{jpg,...} without walking the tree.

    Returns the first existing candidate, else None (caller can fall back to build_image_index)."""
    pf = patient_folder(image_id)
    if not pf:
        return None
    base = Path(root) / pf[:3] / pf
    for suf in IMAGE_SUFFIXES:
        cand = base / f"{image_id}{suf}"
        if cand.exists():
            return cand
    return None


def build_image_index(root: Path) -> dict[str, Path]:
    """Fallback: one os.walk over `root` mapping image stem (== image_id) -> path.

    Used when the sharded layout differs from the guess (e.g. a flat Kaggle mount)."""
    index: dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMAGE_SUFFIXES:
                index.setdefault(stem, Path(dirpath) / fn)
    return index
