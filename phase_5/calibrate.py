"""[Tier 3 prep] Fit per-class temperature scaling for M5 calibration (spec 5.3).

Reads M3 predictions (image-level probs) + the true image-level CheXpert labels (m3_labels), fits a
per-disease temperature T that minimizes BCE on a calibration split, and writes m5_temperature.json
for run.py to consume. Dependency-free (grid search), CPU-only. Needs M3 predictions to exist, so it
runs AFTER M3 inference — the code is ready now.

    python phase_5/calibrate.py --m3-pred data/m3_pred_val.jsonl --m3-labels-dir data/m3_labels \
        --split val --out data/m5_temperature.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config
import constants as C

UNKNOWN = -100


def _bce(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.any():
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """T minimizing BCE of sigmoid(logit(p)/T) vs y. Grid + refine, no scipy."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    grid = np.linspace(0.3, 6.0, 200)
    best_t, best_l = 1.0, float("inf")
    for t in grid:
        q = 1.0 / (1.0 + np.exp(-logit / t))
        l = _bce(q, y)
        if l < best_l:
            best_l, best_t = l, t
    fine = np.linspace(max(0.2, best_t - 0.1), best_t + 0.1, 80)
    for t in fine:
        q = 1.0 / (1.0 + np.exp(-logit / t))
        l = _bce(q, y)
        if l < best_l:
            best_l, best_t = l, t
    return float(best_t)


def load_labels(m3_labels_dir: Path, split: str) -> dict[str, np.ndarray]:
    labels = np.load(Path(m3_labels_dir) / "image_chexpert.npy", mmap_mode="r")
    out: dict[str, np.ndarray] = {}
    with open(Path(m3_labels_dir) / "manifest.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            if m.get("ok", True) and str(m.get("split", "")).lower() == split:
                out[m["image_id"]] = np.asarray(labels[i])
    return out


def load_preds(m3_pred: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(m3_pred, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["image_id"]] = r.get("image_disease") or {}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit per-class temperature for M5")
    p.add_argument("--m3-pred", type=Path, required=True, help="M3 infer JSONL on the calibration split")
    p.add_argument("--m3-labels-dir", type=Path, default=config.REPO_ROOT / "data" / "m3_labels")
    p.add_argument("--split", default="val")
    p.add_argument("--out", type=Path, default=config.TEMPERATURE_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    labels = load_labels(args.m3_labels_dir, args.split)
    preds = load_preds(args.m3_pred)
    ids = [i for i in preds if i in labels]
    if not ids:
        raise SystemExit("[ERROR] no overlap between predictions and labels for this split")
    print(f"calibrating on {len(ids):,} images (split={args.split})")

    per_class, report = {}, {}
    for d, name in enumerate(C.CHEX_NAMES):
        p = np.array([preds[i].get(name, 0.0) for i in ids], dtype=np.float64)
        y = np.array([labels[i][d] for i in ids], dtype=np.float64)
        m = (y == 0) | (y == 1)
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            per_class[name] = 1.0
            report[name] = {"T": 1.0, "n": int(m.sum()), "note": "insufficient/one-class -> T=1"}
            continue
        pp, yy = p[m], y[m]
        t = fit_temperature(pp, yy)
        q = 1.0 / (1.0 + np.exp(-np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6))) / t))
        per_class[name] = round(t, 4)
        report[name] = {"T": round(t, 4), "n": int(m.sum()),
                        "ece_before": round(_ece(pp, yy), 4), "ece_after": round(_ece(q, yy), 4)}

    out = {"per_class": per_class, "split": args.split, "n": len(ids), "report": report}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[DONE] temperatures -> {args.out}")
    for name in C.CHEX_NAMES:
        r = report[name]
        extra = (f"ECE {r.get('ece_before','-')}->{r.get('ece_after','-')}" if "ece_before" in r else r.get("note", ""))
        print(f"  {name:<26} T={per_class[name]:<6} {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
