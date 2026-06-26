"""Paths + hyperparameters for phase_4 (M4 T-KAN). CLI flags override everything.

M4 is staged on top of a FROZEN M3: its inputs are the cached region features + disease logits
(`phase_3/precompute_regions.py`), so this module never imports phase_3 or runs the backbone.
"""

from __future__ import annotations

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- inputs ------------------------------------------------------------------
# frozen-M3 region cache: <image_id>.npy float16 [29, feat_dim + 14] (feat ‖ disease logits)
DEFAULT_REGION_CACHE = Path("/kaggle/input/m3-region-cache") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "m3_region_cache")
# per-region present mask lives in the m3 label arrays (present_mask.npy + manifest.jsonl)
DEFAULT_M3_LABELS_DIR = Path("/kaggle/input/m3-labels") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "m3_labels")
# M4 progression targets (this module's labels.py output)
DEFAULT_M4_LABELS_DIR = Path("/kaggle/input/m4-labels") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "m4_labels")
# prior<->current pairs (phase_3/pairing.py) — bundled alongside the m4 labels on Kaggle
DEFAULT_PAIRS_PATH = Path("/kaggle/input/m4-labels/m3_pairs.jsonl") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "m3_pairs.jsonl")
# scene graphs (prep only, for labels.py)
DEFAULT_SCENE_ROOT = Path(r"C:\Users\Dang Hien\Downloads\chest-imagenome") if not ON_KAGGLE \
    else Path("/kaggle/input/mimic-scene-graph")
DEFAULT_METADATA = REPO_ROOT / "data" / "mimic_metadata_final.jsonl"

# ---- outputs -----------------------------------------------------------------
WORK_ROOT = Path("/kaggle/working") if ON_KAGGLE else (REPO_ROOT / "phase_4" / "_work")
DEFAULT_RUNS_DIR = WORK_ROOT / "m4_runs"

# ---- model -------------------------------------------------------------------
NUM_CHEX = 14
# T-head input/region = [feat_curr ; feat_prior ; feat_curr-feat_prior] + [logit_curr ; logit_prior]
#                     = 3*feat_dim + 2*14   (feat_dim auto-detected from the cache; 512 -> 1564)
HEAD_TYPE = "mlp"               # "mlp" now; "kan" (FastKAN) later — same interface
HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.1

REQUIRE_PRIOR_PRESENT = True    # a region is supervised only if present in BOTH curr and prior

# ---- training ----------------------------------------------------------------
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40
BATCH = 64
USE_CLASS_WEIGHT = True         # "stable" dominates -> inverse-frequency weighting (spec 4.4)

# ---- time-flip augmentation (TRAIN ONLY) -------------------------------------
# Doubles train pairs by flipping (prior,current)->(current,prior) with labels improved<->worsened.
# Forces M4 to learn "flip the input -> flip the output" instead of cheating on which slot is current,
# and balances improved/worsened against the dominant "stable". Only valid for symmetric labels:
# diseases without a clean antonym (device placement/removal) are excluded -> their flipped cells are
# masked (-100), never given a wrong label. Inspect label symmetry before widening FLIP_EXCLUDE.
AUGMENT_TIME_FLIP = True
FLIP_EXCLUDE_DISEASES = ("Support Devices",)   # not antisymmetric under improved<->worsened


def env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default
