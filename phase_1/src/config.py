"""Paths + constants for phase_1 (M1 — FROZEN BioViL-T feature extraction).

M1 writes one  <image_id>.pt  per CXR — a torch.save'd [197, C] float16 tensor:
  row 0      = projected_global_embedding   (BioViL-T's own global vector)
  rows 1..196 = projected_patch_embeddings [C,14,14] flattened to [196,C], index = y*14+x
This is the EXACT cache contract phase_3/src/features.py loads. Do NOT drift from it.
CLI flags override everything here.

Kaggle-friendly: inputs under /kaggle/input, outputs staged under /kaggle/working and
flushed to Google Drive (rclone) in batches so a dead session can resume.
"""

from __future__ import annotations

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
REPO_ROOT = Path(__file__).resolve().parents[2]   # src/ -> phase_1/ -> repo root

# ---- feature-grid geometry (MUST match phase_3/config.py) --------------------
INPUT_RES = 448                 # BioViL-T input side; the m3 boxes (labels.py) live in this
                                # 448x448 frame, so we feed images at exactly this size.
GRID_H = 14
GRID_W = 14
GRID_TOKENS = GRID_H * GRID_W    # 196
FEAT_DIM = 512                   # BioViL-T joint_feature_size — AUTO-DETECTED at runtime
                                 # (see biovilt.py); kept here only as the expected default.
CELL = INPUT_RES / GRID_W        # 32 px per grid cell (same as pooling.py)

# ---- model -------------------------------------------------------------------
# BioViL-T image encoder, loaded FROZEN (health_multimodal downloads the weights).
BIOVILT_ENCODER = "biovil_t"

# Which BioViL-T output feeds the cache. The existing collaborator reference is the 512-d
# PRE-projection backbone features (img_embedding + patch_embeddings), NOT the 128-d projected
# head (projected_global/patch_embeddings). Confirmed by the reference-reproduce in 3-verify.
#   "backbone"  : img_embedding [B,512] + patch_embeddings [B,512,14,14]   (DEFAULT, matches ref)
#   "projected" : projected_global_embedding [B,128] + projected_patch_embeddings [B,128,14,14]
FEATURE_SOURCE = "backbone"      # "backbone" | "projected"

# Image preprocessing geometry. The mimic-cxr-448 jpgs are already a straight STRETCH to
# 448x448 (the frame the m3 boxes were rescaled into), so the default feeds them as-is with
# NO geometric resize/crop -> the box<->grid mapping stays exact. "resize_crop" reproduces
# BioViL-T's *default* Resize(512)+CenterCrop(448) instead; only use it if the existing
# feature cache was built that way (the reference-compare in 3-verify_features.py decides).
TRANSFORM_MODE = "stretch448"    # "stretch448" | "resize_crop"

# ---- inputs ------------------------------------------------------------------
# pre-resized 448x448 CXR jpgs, laid out  <root>/p<pid[:2]>/p<pid>/<image_id>.jpg
DEFAULT_IMAGES_ROOT = Path("/kaggle/input/datasets/nguynnghin/mimic-cxr-448") if ON_KAGGLE \
    else (REPO_ROOT / "data" / "mimic-cxr-448")
# image_id universe = manifest images  ∪  prior_image_id in the pairs file (M4 needs priors)
DEFAULT_MANIFEST = REPO_ROOT / "data" / "m3_labels" / "manifest.jsonl"
DEFAULT_PAIRS = REPO_ROOT / "data" / "m4_labels" / "m3_pairs.jsonl"

# ground-truth reference output (one real collaborator .pt) — 3-verify reproduces it and
# asserts cosine≈1, proving model variant + preprocessing + flatten order end-to-end. Shipped
# inside phase_1/ (the repo-root docs/ copy is gitignored, so it isn't on the Kaggle clone).
_REFERENCE_ID = "MIMIC_p10000032_s50414267_02aa804e-bde0afdd-112c0b34-7bc16630-4e384014"
DEFAULT_REFERENCE = next(
    (p for p in (Path(__file__).resolve().parents[1] / "reference" / f"{_REFERENCE_ID}.pt",
                 REPO_ROOT / "docs" / f"{_REFERENCE_ID}.pt") if p.exists()),
    Path(__file__).resolve().parents[1] / "reference" / f"{_REFERENCE_ID}.pt",
)

# ---- outputs -----------------------------------------------------------------
WORK_ROOT = Path("/kaggle/working") if ON_KAGGLE else (REPO_ROOT / "phase_1" / "_work")
DEFAULT_WORKLIST = WORK_ROOT / "worklist.jsonl"
DEFAULT_FEATURES_OUT = WORK_ROOT / "features"   # local staging; flushed to Drive in batches

# ---- extraction loop ---------------------------------------------------------
BATCH = 32                # images per forward pass
FLUSH_EVERY = 1000        # after this many NEW .pt: rclone copy to Drive, then delete local
NUM_WORKERS = 2           # image-loading dataloader workers


def env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default
