"""Single source of truth for phase_3 (M3 C-KAN) label spaces.

- 29 anatomical REGIONS  (reused from phase_2 — the detector's classes)
- 69 CONCEPTS            (from data/m3_concept_space.json: 43 finding + 10 disease
                          + 12 tubesandlines + 4 device)
- 14 CHEXPERT diseases   (data/mimic_label_map.json order; No Finding at idx 8)

Plus the maps that let phase_3 turn a scene graph's `category|polarity|label`
strings into per-region concept targets, and aggregate 69 concepts -> 14 CheXpert.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent


def _find(name: str) -> Path:
    """Resolve a data file across likely locations (self-contained on Kaggle)."""
    for cand in (REPO_ROOT / "data" / name, _HERE / name, Path("data") / name, Path(name)):
        if cand.exists():
            return cand
    return REPO_ROOT / "data" / name  # default (will raise a clear error if missing)


CONCEPT_SPACE_PATH = _find("m3_concept_space.json")
LABEL_MAP_PATH = _find("mimic_label_map.json")

# ---- 29 regions (must match phase_2/constants.CLASS_NAMES; hardcoded so phase_3 is
# self-contained — no cross-folder import needed on Kaggle) --------------------
REGION_NAMES: list[str] = [
    "abdomen", "aortic arch", "cardiac silhouette", "carina", "cavoatrial junction",
    "left apical zone", "left clavicle", "left costophrenic angle", "left hemidiaphragm",
    "left hilar structures", "left lower lung zone", "left lung", "left mid lung zone",
    "left upper lung zone", "mediastinum", "right apical zone", "right atrium",
    "right clavicle", "right costophrenic angle", "right hemidiaphragm",
    "right hilar structures", "right lower lung zone", "right lung", "right mid lung zone",
    "right upper lung zone", "spine", "svc", "trachea", "upper mediastinum",
]
REGION_INDEX: dict[str, int] = {r: i for i, r in enumerate(REGION_NAMES)}
NUM_REGIONS = len(REGION_NAMES)  # 29


# ---- 14 CheXpert -------------------------------------------------------------
_label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
CHEX_NAMES: list[str] = [_label_map[str(i)] for i in range(len(_label_map))]
CHEX_INDEX: dict[str, int] = {c: i for i, c in enumerate(CHEX_NAMES)}
NUM_CHEX = len(CHEX_NAMES)  # 14
NO_FINDING_IDX = CHEX_INDEX["No Finding"]  # 8


# ---- 69 concepts -------------------------------------------------------------
_cs = json.loads(CONCEPT_SPACE_PATH.read_text(encoding="utf-8"))
_concepts = _cs["concepts"]                       # ordered list, idx 0..68
CONCEPT_NAMES: list[str] = [c["name"] for c in _concepts]
CONCEPT_CATEGORY: list[str] = [c["category"] for c in _concepts]
NUM_CONCEPTS = len(_concepts)  # 69

# match scene-graph "category|pol|label" -> concept idx  (key = (category, label))
CONCEPT_BY_CATLABEL: dict[tuple[str, str], int] = {
    (c["category"], c["name"]): c["idx"] for c in _concepts
}

# concept idx -> CheXpert idx (or -1 if the concept has no CheXpert slot)
CONCEPT_TO_CHEX: list[int] = [
    CHEX_INDEX[c["chexpert"]] if c.get("chexpert") else -1 for c in _concepts
]
# CheXpert idx -> list of concept idxs that feed it (for deriving per-region CheXpert)
CHEX_FROM_CONCEPTS: dict[int, list[int]] = {i: [] for i in range(NUM_CHEX)}
for ci, xi in enumerate(CONCEPT_TO_CHEX):
    if xi >= 0:
        CHEX_FROM_CONCEPTS[xi].append(ci)

# scene-graph attribute categories that ARE concepts (the rest are cues, ignored here)
CONCEPT_CATEGORIES = ("anatomicalfinding", "disease", "tubesandlines", "device")

UNKNOWN = -100  # masked-BCE sentinel (not-mentioned), matches the repo-wide convention


if __name__ == "__main__":  # quick sanity dump
    print(f"regions  : {NUM_REGIONS}")
    print(f"concepts : {NUM_CONCEPTS}  (mapped to CheXpert: {sum(x >= 0 for x in CONCEPT_TO_CHEX)})")
    print(f"chexpert : {NUM_CHEX}")
    for xi in range(NUM_CHEX):
        names = [CONCEPT_NAMES[ci] for ci in CHEX_FROM_CONCEPTS[xi]]
        print(f"  [{xi:2}] {CHEX_NAMES[xi]:<26} <- {len(names)} concepts")
