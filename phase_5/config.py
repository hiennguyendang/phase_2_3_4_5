"""Thresholds + paths for phase_5 (M5 assembler). All overridable on the CLI.

M5 is mostly deterministic rules — these thresholds are the only "knobs", and tier-3 calibration
(temperature scaling) feeds them. Defaults are placeholders to be set from a calibration split.
"""

from __future__ import annotations

from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- inputs (M3 / M4 prediction JSONL from their infer.py) -------------------
DEFAULT_M3_PRED = REPO_ROOT / "data" / "m3_pred.jsonl"
DEFAULT_M4_PRED = REPO_ROOT / "data" / "m4_pred.jsonl"      # optional; absent => no temporal language
DEFAULT_OUT = (Path("/kaggle/working") if ON_KAGGLE else REPO_ROOT / "phase_5" / "_work") / "m5_reports.jsonl"

# ---- tier 3: calibration + abstention thresholds (spec 5.3) ------------------
# calibrated prob bands:  >=ASSERT assert | >=UNCERTAIN hedge | >=ABSTAIN abstain ("cannot exclude") | omit
TAU_ASSERT = 0.50
TAU_UNCERTAIN = 0.20   # hedge floor ("there may be ...")
TAU_ABSTAIN = 0.10     # abstain floor ("X cannot be excluded"); below -> omit (silent)
TAU_PROG = 0.50        # min M4 class prob to SPEAK a progression (else fall back to "stable"/silent)
TEMPERATURE = 1.0      # global temperature (1.0 = identity); per-class from TEMPERATURE_PATH if present
TEMPERATURE_PATH = REPO_ROOT / "data" / "m5_temperature.json"  # written by calibrate.py (optional)

# ---- tier 2: grounding -------------------------------------------------------
TAU_REGION = 0.50      # min per-region disease prob to name a location

# ---- tier 5: realization -----------------------------------------------------
REALIZE = "template"   # "template" (faithful, default) | "paraphrase" (constrained LLM, pluggable)
