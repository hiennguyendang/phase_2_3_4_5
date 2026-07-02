"""Plot a training run's metrics.jsonl -> curves.png, and print a convergence summary.

Headless-safe (Agg backend). Works for one run or several (compare val_image_f1).

    python phase_3/scripts/plot_metrics.py /home/jovyan/runs/m3_B
    python phase_3/scripts/plot_metrics.py /home/jovyan/runs/*        # compare many runs
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(run: str):
    p = Path(run)
    if p.is_dir():
        p = p / "metrics.jsonl"
    if not p.exists():
        return None, p
    return [json.loads(l) for l in open(p, encoding="utf-8")], p


def _summary(name: str, rows: list[dict]) -> None:
    ep = [r["epoch"] for r in rows]
    valid = [(r["epoch"], r["val_image_f1"]) for r in rows if r.get("val_image_f1") is not None]
    if not valid:
        print(f"{name}: no val metrics (val split empty at train time)"); return
    be, bv = max(valid, key=lambda x: x[1])
    since = ep[-1] - be
    last = [v for _, v in valid[-10:]]
    sd = statistics.pstdev(last) if len(last) > 1 else 0.0
    conv = "CONVERGED" if (since >= 8 and sd < 0.01) else "still improving / not settled"
    print(f"{name:16} best img_F1={bv:.4f} @ep{be:<3} | since_best={since:<3} | last10_std={sd:.4f} -> {conv}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot metrics.jsonl + convergence summary")
    ap.add_argument("runs", nargs="+", help="run dir(s) or metrics.jsonl path(s)")
    ap.add_argument("--out", default=None, help="output PNG (default: <run>/curves.png, or compare.png for many)")
    args = ap.parse_args()

    loaded = [(Path(r).name if Path(r).is_dir() else Path(r).parent.name, *_load(r)) for r in args.runs]
    loaded = [(n, rows, p) for (n, rows, p) in loaded if rows]
    if not loaded:
        raise SystemExit("[ERROR] no readable metrics.jsonl in the given paths")

    print("== convergence ==")
    for n, rows, _ in loaded:
        _summary(n, rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(loaded) == 1:  # single run: loss breakdown + val metrics
        n, rows, p = loaded[0]
        ep = [r["epoch"] for r in rows]
        col = lambda k: [r.get(k) for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for k in ("train_loss", "train_concept", "train_region_chex", "train_image_chex"):
            if any(v is not None for v in col(k)):
                ax[0].plot(ep, col(k), label=k, alpha=1.0 if k == "train_loss" else 0.5)
        ax[0].set_title(f"{n} — loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        for k, lb in [("val_image_f1", "img_F1"), ("val_region_f1", "region_F1"),
                      ("val_concept_f1", "concept_F1"), ("val_image_auc", "img_AUC")]:
            if any(v is not None for v in col(k)):
                ax[1].plot(ep, col(k), label=lb)
        ax[1].set_title(f"{n} — val"); ax[1].set_xlabel("epoch"); ax[1].legend()
        out = Path(args.out) if args.out else p.with_name("curves.png")
    else:  # many runs: compare val_image_f1
        fig, axx = plt.subplots(figsize=(9, 5))
        for n, rows, _ in loaded:
            ep = [r["epoch"] for r in rows]
            f1 = [r.get("val_image_f1") for r in rows]
            if any(v is not None for v in f1):
                axx.plot(ep, f1, label=n)
        axx.set_title("val image macro-F1"); axx.set_xlabel("epoch"); axx.legend()
        out = Path(args.out) if args.out else Path("compare.png")

    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
