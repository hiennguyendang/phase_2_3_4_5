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
INPUT_RES = 448                 # BioViL-T center-crop side; boxes (labels.py) live in this space.
                                # cell = INPUT_RES / GRID_W = 32 px  -> maps a bbox to grid cells.

# ---- model -------------------------------------------------------------------
# pooling: attention-pool 196x512 -> 29x512.  Each region query attends the grid, but
# MASKED to its own bbox cells (spec 3.1) -> alpha is a faithful within-region grounding
# signal and small focal findings survive. Set MASK_BBOX=False to attend the full grid.
POOL_HEADS = 4                  # multi-head attention pooling
MASK_BBOX = True                # restrict each region query to its bbox grid cells (spec 3.1)
USE_GLOBAL_TOKEN = False        # add a 30th "global" query as an extra region. Default off;
                                # relational findings are handled by the GlobalHead+gate below.

# neck: Linear feat->NECK_DIM + LayerNorm + GELU (spec 3.2). DISABLED by choice — we keep the
# full 512-d region feature (richer signal). region_feat[512] is then the contract shared with M4
# (its input/region = 512*3 + 14*2 = 1564, still light). Set NECK_DIM=128 to re-enable the neck.
NECK_DIM = None

# global branch (spec 3.5): GAP/global vector -> GlobalHead(14), fused with the region path at
# image-logit via a learned per-disease gate  g=σ(gate(global)); fused = g*global + (1-g)*local.
# Catches relational findings (cardiomegaly, diffuse edema, low lung volumes) not in one box.
USE_GLOBAL_HEAD = True

# head direction (spec 3.3 letters — A=safe fallback, C=most dangerous for faithfulness):
#   "A" Direct:   region_feat -> 14                       (faithful "where", accuracy ceiling)
#   "B" CBM:      region_feat -> 69 concept -> 14         (pure bottleneck, the only "why"-faithful path)
#   "C" Hybrid:   region_feat -> 69; [feat(+leak) ⊕ 69] -> 14   (accuracy, but CBM-leakage risk)
HEAD_MODE = "A"
HEAD_TYPE = "mlp"               # "mlp" now; "kan" (FastKAN) later — same interface
HEAD_HIDDEN = 512
CONCEPT_DROPOUT = 0.1
FEATURE_LEAK_DROPOUT = 0.3      # (C/Hybrid only) dropout on the raw-feature path into the disease
                                # head, so disease leans on concepts -> "leaky CBM"

# how to get the image-level 14 from the 29 region predictions
REGION_AGG = "attention"        # "attention" | "max" | "mean"

# ---- imbalance (spec 3.6, top priority) --------------------------------------
USE_POS_WEIGHT = True           # RADAR-style log-scale pos_weight  α_i = log(1 + |D|/pos_i)

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
