"""Paths + hyperparameters for phase_4 (M4 T-KAN). CLI flags override everything.

M4 is staged on top of a FROZEN M3: its inputs are the cached region features + disease logits
(`phase_3/precompute_regions.py`), so this module never imports phase_3 or runs the backbone.
"""

from __future__ import annotations

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
REPO_ROOT = Path(__file__).resolve().parents[2]   # src/ -> phase_4/ -> repo root

# ---- inputs ------------------------------------------------------------------
# frozen-M3 region cache: <image_id>.npy float16 [29, feat_dim + 14] (feat ‖ disease logits)
DEFAULT_REGION_CACHE = Path("/kaggle/input/m3-region-cache") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "m3_region_cache")
# frozen-M3 concept cache (ftcb arch): <image_id>.npy float16 [29, 69] sigmoid concept activations
DEFAULT_CONCEPT_CACHE = Path("/kaggle/input/m3-concept-cache") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "m4_concept_cache")
# per-region present mask lives in the m3 label arrays (present_mask.npy + manifest.jsonl)
DEFAULT_M3_LABELS_DIR = Path("/kaggle/input/m3-labels") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "m3_labels")
# frozen M1 BioViL-T patch grids ([1+196, C] per image) — the tempfuse arch reads these DIRECTLY
# (no bridge/region-cache needed: M4 pools the patches itself). Boxes come from the m3 label arrays.
DEFAULT_FEATURES_ROOT = Path("/kaggle/input/m1-features") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "features" / "frozen")
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

# ---- architecture ------------------------------------------------------------
# How M4 turns the two studies into per-region temporal features:
#   regiondiff  consume the frozen-M3 REGION cache; head sees [c;p;c-p;lc;lp]        (v1 default)
#   tempfuse    read the frozen M1 PATCH grids (196xC) for curr+prior, cross-attend current<-prior
#               (BioViL-T-style soft registration), then M4's OWN bbox-guided attention pool ->
#               region temporal feats -> head. Fixes "region-pool washes out localised change"
#               and yields M4's own faithful "where progressed" (the pool alpha). Reads
#               data/features/frozen directly; boxes from the m3 label arrays.
ARCH = "regiondiff"

# tempfuse geometry (must match how boxes were rasterised in phase_2/3: 448 frame, 14x14 grid)
GRID_H = 14
GRID_W = 14
GRID_TOKENS = GRID_H * GRID_W    # 196
INPUT_RES = 448                  # boxes live in this pixel space; cell = 448/14 = 32 px
POOL_HEADS = 4                   # heads for the bbox-guided region attention pool
MASK_BBOX = True                 # restrict each region query to its bbox cells (faithful "where")
FUSE_BLOCKS = 1                  # cross-attn(current<-prior)+self+FFN blocks (keep shallow: overfits fast)
FUSE_HEADS = 4
BOX_SOURCE = "detector"          # "detector" (boxes_det.npy) | "gt" (boxes.npy, oracle only)
TEMPFUSE_INPUT_MODE = "feat"     # "feat" = v3; "feat_logits" adds M3 curr/prior/delta logits to the head

# ---- model -------------------------------------------------------------------
NUM_CHEX = 14
# T-head input/region = [feat_curr ; feat_prior ; feat_curr-feat_prior] + [logit_curr ; logit_prior]
#                     = 3*feat_dim + 2*14   (feat_dim auto-detected from the cache; 512 -> 1564)
HEAD_TYPE = "mlp"               # "mlp" (baseline) | "kan" (FastKAN) | "linear" — same make_head interface
HEAD_HIDDEN = 512
HEAD_DROPOUT = 0.1
KAN_GRIDS = 8                   # FastKAN: #RBF centers per input dim (ablation head only)

# (c) prediction structure — how the 3 classes are produced from the head output.
#   flat     one 3-way softmax over {stable, improved, worsened}          (baseline)
#   twostage factorized: P(change) [stable vs change] x P(dir|change) [improved vs worsened].
#            Separates the EASY sub-task (did it change?) from the HARD one (which direction?),
#            and targets the weak `improved` class. Same head width (14x3), reinterpreted; the
#            composed log-probs are returned as "logits" so eval/infer/loss stay unchanged —
#            plain cross-entropy on them decomposes EXACTLY into BCE(change)+CE(direction).
HEAD_MODE = "flat"

# ---- input composition (spec 4.2) --------------------------------------------
# What the per-region head sees. Ablation axis: where does the progression signal live?
#   full   [feat_curr ; feat_prior ; feat_curr-feat_prior ; logit_curr ; logit_prior]  (baseline)
#   concat [feat_curr ; feat_prior ; logit_curr ; logit_prior]        (no explicit difference)
#   diff   [feat_curr-feat_prior ; logit_curr-logit_prior]            (pure Siamese difference)
#   logits [logit_curr ; logit_prior]                                 (M3 disease logits ONLY)
#   feat   [feat_curr ; feat_prior ; feat_curr-feat_prior]            (region features ONLY, no logits)
INPUT_MODE = "full"

# ---- loss --------------------------------------------------------------------
LOSS_TYPE = "ce"                # "ce" (baseline) | "focal" | "cdw" — all honor USE_CLASS_WEIGHT
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.0           # CE only; useful for noisy silver comparison_cues
CDW_ALPHA = 5.0                 # "cdw" only: |i-c|^alpha ordinal distance penalty (Polat 2022/24)

REQUIRE_PRIOR_PRESENT = True    # a region is supervised only if present in BOTH curr and prior

# (b) prior view alignment. Cross-view pairs (PA prior vs AP current, lateral) make feat_curr-feat_prior
# noise -> the Siamese fails silently. `m3_pairs.jsonl` carries a `same_view` flag; when True we keep
# ONLY same-ViewPosition pairs. Default False = old behaviour (all pairs). Eval/infer inherit the
# training value from the ckpt so the test set matches.
SAME_VIEW_ONLY = False

# ---- training ----------------------------------------------------------------
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40
BATCH = 64
USE_CLASS_WEIGHT = True         # "stable" dominates -> inverse-frequency weighting (spec 4.4)
# (a) checkpoint selection. Val curves peak by epoch ~1-10 then overfit, and best.pt historically
# tracked macro-F1 while the HEADLINE is change-only F1. Pick which val metric selects best.pt, and
# optionally early-stop after PATIENCE evals with no improvement (0 = off, old behaviour).
SELECT_METRIC = "macro"         # "macro" (old) | "change"
PATIENCE = 0

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
