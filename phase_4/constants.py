"""Single source of truth for phase_4 (M4 T-KAN) label spaces. Self-contained on Kaggle.

Reuses the 29 regions / 14 CheXpert / 69 concept maps (same bundled JSONs as phase_3) to turn a
scene graph's per-region `comparison_cues` into per-(region, disease) progression targets, plus the
3 progression classes themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent


def _find(name: str) -> Path:
    for cand in (REPO_ROOT / "data" / name, _HERE / name, _HERE.parent / "phase_3" / name,
                 Path("data") / name, Path(name)):
        if cand.exists():
            return cand
    return REPO_ROOT / "data" / name


CONCEPT_SPACE_PATH = _find("m3_concept_space.json")
LABEL_MAP_PATH = _find("mimic_label_map.json")

# ---- 29 regions (must match phase_2/phase_3) ---------------------------------
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

# ---- 69 concepts -> CheXpert map ---------------------------------------------
_cs = json.loads(CONCEPT_SPACE_PATH.read_text(encoding="utf-8"))
_concepts = _cs["concepts"]
NUM_CONCEPTS = len(_concepts)  # 69
CONCEPT_BY_CATLABEL: dict[tuple[str, str], int] = {(c["category"], c["name"]): c["idx"] for c in _concepts}
CONCEPT_TO_CHEX: list[int] = [CHEX_INDEX[c["chexpert"]] if c.get("chexpert") else -1 for c in _concepts]
CONCEPT_CATEGORIES = ("anatomicalfinding", "disease", "tubesandlines", "device")

# ---- 3 progression classes (spec 4.3) ----------------------------------------
# ImaGenome `comparison_cues` carry exactly these 3 values (verified on the corpus).
PROG_NAMES: list[str] = ["stable", "improved", "worsened"]
NUM_PROG = len(PROG_NAMES)  # 3
PROG_INDEX: dict[str, int] = {p: i for i, p in enumerate(PROG_NAMES)}
# scene-graph cue label -> progression class
CUE_TO_PROG: dict[str, int] = {"no change": 0, "improved": 1, "worsened": 2}
COMPARISON_CATEGORY = "comparison"
# conflict resolution when one (region,disease) gets several cues: the more salient change wins
PROG_PRIORITY = {0: 0, 1: 1, 2: 2}  # worsened(2) > improved(1) > stable(0)

# time-flip augmentation: swapping (prior,current) flips the progression label.
# stable stays stable; improved <-> worsened. Index by class -> flipped class.
FLIP_CLASS_MAP = [0, 2, 1]

UNKNOWN = -100  # masked cells (no comparison cue for that region/disease)


if __name__ == "__main__":
    print(f"regions {NUM_REGIONS} | chex {NUM_CHEX} | concepts {NUM_CONCEPTS} | prog {PROG_NAMES}")
