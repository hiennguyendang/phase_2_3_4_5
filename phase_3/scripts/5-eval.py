"""Run order 5 — evaluate an M3 checkpoint: macro-F1 (headline, spec 3.6) + AUC.

Thin CLI wrapper; the metrics + evaluate() live in src/eval.py (also imported by 4-train.py and
6-faithfulness.py, so they must stay a clean-named library module — hence this separate entry).

    python phase_3/scripts/5-eval.py --ckpt <run>/m3_A/best.pt --split test
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_3/src

from eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
