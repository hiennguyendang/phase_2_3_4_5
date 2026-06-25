"""Paths + hyperparameters for phase_3 (M3 C-KAN). CLI flags override everything.

Kaggle-friendly: inputs under /kaggle/input, outputs under /kaggle/working.
The data-prep scripts (labels.py, pairing.py) run fine off-Kaggle (pure data).
"""

from __future__ import annotations

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- inputs ------------------------------------------------------------------
# unified metadata (split + image-level 14-CheXpert `labels` + which images have a scene graph)
DEFAULT_METADATA = REPO_ROOT / "data" / "mimic_metadata_final.jsonl"
# ImaGenome scene graphs (rescaled to 448) — source of per-region concept labels
DEFAULT_SCENE_ROOT = Path(
    r"C:\Users\Dang Hien\Downloads\chest-imagenome"
) if not ON_KAGGLE else Path("/kaggle/input/mimic-scene-graph")
# mimic-cxr metadata csv (StudyDate/Time/ViewPosition) — for prior<->current pairing
DEFAULT_CXR_META_CSV = REPO_ROOT / "data" / "mimic-cxr-2.0.0-metadata.csv"

# precomputed BioViL-T feature grids (196x512 per image). *** format TBD by user ***
# expected: one file per image keyed by image_id; loader lives in features.py
DEFAULT_FEATURES_ROOT = Path("/kaggle/input/mimic-biovilt-features") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "features")

# ---- outputs (prep artifacts) ------------------------------------------------
WORK_ROOT = Path("/kaggle/working") if ON_KAGGLE else (REPO_ROOT / "phase_3" / "_work")
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "m3_labels"      # region_concepts.npy, ... + manifest
DEFAULT_PAIRS_PATH = REPO_ROOT / "data" / "m3_pairs.jsonl"  # prior<->current
DEFAULT_RUNS_DIR = WORK_ROOT / "m3_runs"

# ---- feature grid geometry ---------------------------------------------------
GRID_H = 14
GRID_W = 14
GRID_TOKENS = GRID_H * GRID_W   # 196
FEAT_DIM = 512                  # BioViL-T channel dim

# ---- model -------------------------------------------------------------------
# pooling: attention-pool 196x512 -> 29x512 (each region query attends the full grid).
POOL_HEADS = 4                  # multi-head attention pooling
USE_GLOBAL_TOKEN = False        # add a 30th "global" query (recall safety net). Default off
                                # because full-grid attention already carries global context.

# head direction: "C" feat->14 | "A" feat->69->14 (pure CBM) | "B" feat+69->14 (hybrid)
HEAD_MODE = "B"
HEAD_TYPE = "mlp"               # "mlp" now; "kan" (FastKAN) later — same interface
HEAD_HIDDEN = 512
CONCEPT_DROPOUT = 0.1
FEATURE_LEAK_DROPOUT = 0.3      # (B only) drop on the raw-feature path into disease head,
                                # so disease leans on concepts -> more faithful, "leaky CBM"

# how to get the image-level 14 from the 29 region predictions
REGION_AGG = "attention"        # "attention" | "max" | "mean"

# ---- training ----------------------------------------------------------------
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40
BATCH = 32
LAMBDA_CONCEPT = 1.0            # weight on per-region concept loss
LAMBDA_REGION_CHEX = 0.5        # weight on per-region CheXpert loss
LAMBDA_IMAGE_CHEX = 1.0         # weight on image-level CheXpert loss
USE_GT_BOXES = True             # train on scene-graph GT boxes; detector boxes at inference

# train only on rows whose `dataset` is in this set (have per-region concept labels)
CONCEPT_SUPERVISED_DATASETS = ("mimic",)


def env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default
