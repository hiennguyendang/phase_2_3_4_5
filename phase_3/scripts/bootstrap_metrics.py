"""Bootstrap confidence intervals from an M3 eval prediction dump.

The dump is produced by:
  python phase_3/scripts/5-eval.py ... --pred-dump artifacts/predictions/m3.test.npz
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_3/src

import argparse
import json
from pathlib import Path

import numpy as np

import constants as C
from eval import _auc_table, _f1_table


def _ci(vals: np.ndarray) -> dict:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {
        "mean": float(vals.mean()),
        "lo": float(np.percentile(vals, 2.5)),
        "hi": float(np.percentile(vals, 97.5)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap M3 headline metrics")
    p.add_argument("--pred-dump", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--level", choices=["image", "region", "concept"], default="image")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    z = np.load(args.pred_dump)
    prob = z[f"{args.level}_prob"]
    tgt = z[f"{args.level}_target"]
    names = C.CHEX_NAMES if args.level in ("image", "region") else C.CONCEPT_NAMES
    rng = np.random.default_rng(args.seed)
    m = prob.shape[0]
    f1s = np.zeros(args.n, dtype=np.float64)
    aucs = np.zeros(args.n, dtype=np.float64)
    for i in range(args.n):
        idx = rng.integers(0, m, size=m)
        f1s[i], _ = _f1_table(prob[idx], tgt[idx], names)
        aucs[i], _ = _auc_table(prob[idx], tgt[idx], names)
    full_f1, full_per_f1 = _f1_table(prob, tgt, names)
    full_auc, full_per_auc = _auc_table(prob, tgt, names)
    out = {
        "pred_dump": str(args.pred_dump),
        "level": args.level,
        "n_rows": int(m),
        "n_bootstrap": int(args.n),
        "metrics": {
            "f1_macro": {"point": full_f1, **_ci(f1s)},
            "auc_macro": {"point": full_auc, **_ci(aucs)},
        },
        "per_class_point": {
            name: {"f1": full_per_f1[name], "auc": full_per_auc[name]} for name in names
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bootstrap] {args.level} F1 {full_f1:.4f} "
          f"CI [{out['metrics']['f1_macro']['lo']:.4f}, {out['metrics']['f1_macro']['hi']:.4f}]")
    print(f"[bootstrap] {args.level} AUC {full_auc:.4f} "
          f"CI [{out['metrics']['auc_macro']['lo']:.4f}, {out['metrics']['auc_macro']['hi']:.4f}]")
    print(f"[bootstrap] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
