"""Run order 3 — evaluate an M4 checkpoint: macro-F1 + change-only F1 (headline; spec 4.4).

Thin CLI wrapper; the metrics + evaluate() + build_from_ckpt() live in src/eval.py (also imported
by 2-train.py and 4-infer.py, so they must stay a clean-named library module — hence this entry).

    python phase_4/scripts/3-eval.py --ckpt <run>/m4_kan/best.pt --split test
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_4/src

from eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
