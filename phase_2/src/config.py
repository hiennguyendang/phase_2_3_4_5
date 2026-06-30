"""Paths + hyperparameters for the phase_2 detector.

Everything here can be overridden on the CLI of each script.  The defaults are
Kaggle-friendly:  inputs read from /kaggle/input (read-only), all outputs written
under /kaggle/working (the only writable, persisted-to-output dir).

EDIT `KAGGLE_*` below to match YOUR dataset slugs, or pass --images-root /
--scene-root / --metadata on the CLI.
"""

from __future__ import annotations

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()

# ----------------------------------------------------------------------------
# Inputs (read-only on Kaggle).  Override per-run with CLI flags if these guesses
# are wrong; build_yolo_dataset.py also AUTO-DETECTS by scanning /kaggle/input.
# ----------------------------------------------------------------------------
# Folder that (recursively) contains the resized images  <image_id>.jpg
KAGGLE_IMAGES_ROOT = Path("/kaggle/input/mimic-processed")
# Folder that (recursively) contains the  <dicom_id>_SceneGraph.json  files
KAGGLE_SCENE_ROOT = Path("/kaggle/input/mimic-scene-graph")
# mimic_metadata_final.jsonl  (gives split + which images have a scene graph)
KAGGLE_METADATA = Path("/kaggle/input/mimic-metadata/mimic_metadata_final.jsonl")

# Local (Windows) fallbacks so the scripts are runnable off-Kaggle for testing.
_REPO = Path(__file__).resolve().parents[2]   # src/ -> phase_2/ -> repo root
LOCAL_METADATA = _REPO / "data" / "mimic_metadata_final.jsonl"

DEFAULT_IMAGES_ROOT = KAGGLE_IMAGES_ROOT if ON_KAGGLE else (_REPO / "data" / "images")
DEFAULT_SCENE_ROOT = KAGGLE_SCENE_ROOT if ON_KAGGLE else (_REPO / "data" / "scene_graph")
DEFAULT_METADATA = KAGGLE_METADATA if ON_KAGGLE else LOCAL_METADATA

# ----------------------------------------------------------------------------
# Outputs (writable).
# ----------------------------------------------------------------------------
WORK_ROOT = Path("/kaggle/working") if ON_KAGGLE else (_REPO / "phase_2" / "_work")
DEFAULT_DATASET_DIR = WORK_ROOT / "yolo_ds"        # built YOLO dataset (images+labels)
DEFAULT_RUNS_DIR = WORK_ROOT / "runs"              # ultralytics project dir

# ----------------------------------------------------------------------------
# Split mapping: metadata `split` value  ->  YOLO split folder.
# MIMIC uses train / valid / test / gold.  gold = human-verified ImaGenome set,
# routed to `test` as a clean held-out eval set.
# ----------------------------------------------------------------------------
SPLIT_MAP: dict[str, str] = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "test": "test",
    "gold": "test",
}

# ----------------------------------------------------------------------------
# Detector hyperparameters.
# Reference 4090 values from docs/phase2_progress.md (imgsz=1024, batch~12).
# Kaggle GPUs (P100 16GB / T4) are smaller + sessions cap at ~9-12h, so the
# defaults below are lighter.  Bump imgsz/batch if your GPU allows.
# ----------------------------------------------------------------------------
MODEL_WEIGHTS = "yolov8l.pt"      # base checkpoint to fine-tune
IMGSZ = 448                       # images are center-cropped to 448; matches source res
                                  # (bump to 640 to upsample for small boxes if GPU allows)
BATCH = -1                        # -1 = ultralytics auto-batch; or a fixed int
EPOCHS = 100
PATIENCE = 15
SAVE_PERIOD = 5                   # checkpoint every N epochs (multi-session resume)
# Anatomy-safe augmentation: NO mosaic/mixup (they tear apart anatomy), tiny rotate.
AUG = dict(mosaic=0.0, mixup=0.0, degrees=2.0, perspective=0.0005)


# ----------------------------------------------------------------------------
# LLM branch — report -> flat per-region findings extractor (see sg_schema.py).
# 3B (not 7B): the task is closed-vocab extraction after SFT on ~200k ImaGenome
# pairs, so a 3B matches 7B here while running far cheaper at launch. Bump to
# Qwen2.5-7B-Instruct only if eval_sg_llm.py shows a real per-finding-F1 gap.
# ----------------------------------------------------------------------------
SG_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def env_path(name: str, default: Path) -> Path:
    """Allow overriding any default via environment variable."""
    val = os.environ.get(name)
    return Path(val) if val else default
