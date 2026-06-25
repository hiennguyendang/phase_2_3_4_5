"""29 anatomical regions — single source of truth for the detector.

The class id of a region = its index in CLASS_NAMES.  CLASS_NAMES is the 29
canonical bbox names sorted ALPHABETICALLY (matches the "alphabetical is the
single source of truth" rule in docs/phase2_progress.md, and keeps the detector
class ids aligned with phase_3 REGION_NAMES / dataset.yaml downstream).

Names MUST be exactly the `bbox_name` strings used in the ImaGenome
*_SceneGraph.json (e.g. "svc", not "svc (superior vena cava)") so conversion can
match on bbox_name directly.
"""

from __future__ import annotations

# 29 canonical regions, listed in the anatomical order of docs/29_bboxes.md.
# (Order here is only for readability — class ids come from the sorted list below.)
CANONICAL_REGIONS: tuple[str, ...] = (
    "right lung",
    "left lung",
    "mediastinum",
    "right apical zone",
    "left apical zone",
    "right upper lung zone",
    "left upper lung zone",
    "right mid lung zone",
    "left mid lung zone",
    "right lower lung zone",
    "left lower lung zone",
    "right hilar structures",
    "left hilar structures",
    "right costophrenic angle",
    "left costophrenic angle",
    "upper mediastinum",
    "cardiac silhouette",
    "trachea",
    "right hemidiaphragm",
    "left hemidiaphragm",
    "right clavicle",
    "left clavicle",
    "spine",
    "right atrium",
    "cavoatrial junction",
    "svc",
    "carina",
    "aortic arch",
    "abdomen",
)

assert len(CANONICAL_REGIONS) == 29, "expected exactly 29 canonical regions"
assert len(set(CANONICAL_REGIONS)) == 29, "duplicate region name"

# Class id = index in this alphabetically-sorted list (deterministic & stable).
CLASS_NAMES: list[str] = sorted(CANONICAL_REGIONS)
CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS: dict[int, str] = {idx: name for name, idx in CLASS_TO_ID.items()}
NUM_CLASSES: int = len(CLASS_NAMES)

# Some scene graphs carry alias spellings; map them onto a canonical name.
# (Extend if you find more variants while converting.)
NAME_ALIASES: dict[str, str] = {
    "svc (superior vena cava)": "svc",
}


def canonical_name(bbox_name: str) -> str | None:
    """Return the canonical region name for a raw scene-graph bbox_name, or None
    if it is not one of the 29 regions we keep."""
    if not bbox_name:
        return None
    name = bbox_name.strip().lower()
    name = NAME_ALIASES.get(name, name)
    return name if name in CLASS_TO_ID else None
