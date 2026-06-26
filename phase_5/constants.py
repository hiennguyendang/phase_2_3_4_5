"""Label spaces + surface vocab for phase_5 (M5 assembler). Self-contained on Kaggle.

M5 reads M3/M4 prediction JSON (string keys = CheXpert / region / progression names) and never
needs the numeric label vectors, so it only loads the CheXpert order + a small realization vocab.
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


_label_map = json.loads(_find("mimic_label_map.json").read_text(encoding="utf-8"))
CHEX_NAMES: list[str] = [_label_map[str(i)] for i in range(len(_label_map))]
CHEX_INDEX: dict[str, int] = {c: i for i, c in enumerate(CHEX_NAMES)}
NO_FINDING = "No Finding"

# 29 anatomical regions (for the coverage map; same order as phase_2/3/4)
REGION_NAMES: list[str] = [
    "abdomen", "aortic arch", "cardiac silhouette", "carina", "cavoatrial junction",
    "left apical zone", "left clavicle", "left costophrenic angle", "left hemidiaphragm",
    "left hilar structures", "left lower lung zone", "left lung", "left mid lung zone",
    "left upper lung zone", "mediastinum", "right apical zone", "right atrium",
    "right clavicle", "right costophrenic angle", "right hemidiaphragm",
    "right hilar structures", "right lower lung zone", "right lung", "right mid lung zone",
    "right upper lung zone", "spine", "svc", "trachea", "upper mediastinum",
]

PROG_NAMES = ["stable", "improved", "worsened"]
# how a progression class is realized in prose (stable is normally NOT spoken unless asserted change)
PROG_PHRASE = {"improved": "improved", "worsened": "worsened", "stable": "unchanged"}

# disease -> a readable noun phrase for templates (fallback = the CheXpert name lowercased)
DISEASE_PHRASE = {
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinum",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "lung opacity",
    "Lung Lesion": "a lung lesion",
    "Edema": "pulmonary edema",
    "Consolidation": "consolidation",
    "Pneumonia": "pneumonia",
    "Atelectasis": "atelectasis",
    "Pneumothorax": "pneumothorax",
    "Pleural Effusion": "pleural effusion",
    "Pleural Other": "another pleural abnormality",
    "Fracture": "a fracture",
    "Support Devices": "support devices",
}
