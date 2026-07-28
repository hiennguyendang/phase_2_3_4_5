"""Fit per-disease temperature scaling for M4 disease-level readouts.

The M4 JSON contains the LSE disease-level three-class probabilities. Validation
targets are derived from the valid regional progression labels by majority vote,
which matches the label granularity available to this local audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import constants as C  # noqa: E402


def fit_temperature(probs: np.ndarray, target: np.ndarray) -> float:
    probs = np.clip(probs, 1e-8, 1.0)
    logits = np.log(probs)
    grid = np.linspace(0.3, 6.0, 200)
    best_t, best_loss = 1.0, float("inf")
    for t in grid:
        z = logits / t
        z -= z.max(axis=1, keepdims=True)
        logsum = np.log(np.exp(z).sum(axis=1))
        loss = float((-z[np.arange(len(target)), target] + logsum).mean())
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


def calibrate_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-8, 1.0)) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_confidence_gates(probs: np.ndarray, target: np.ndarray,
                         target_precision: float, min_support: int) -> dict[str, dict]:
    pred = probs.argmax(axis=1)
    out = {}
    for cls, label in enumerate(C.PROG_NAMES):
        mask = pred == cls
        conf = probs[mask, cls]
        correct = target[mask] == cls
        selected = int(mask.sum())
        gate = None
        achieved = float("nan")
        support = 0
        for threshold in np.linspace(0.34, 0.99, 66):
            keep = conf >= threshold
            n = int(keep.sum())
            precision = float(correct[keep].mean()) if n else float("nan")
            if n >= min_support and precision >= target_precision:
                gate = float(threshold)
                achieved = precision
                support = n
                break
        out[label] = {
            "threshold": gate,
            "precision": achieved,
            "support": support,
            "selected_before_gate": selected,
            "target_precision": target_precision,
            "min_support": min_support,
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit M4 temporal temperature scaling")
    p.add_argument("--m4-pred", type=Path, required=True)
    p.add_argument("--m4-labels-dir", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target-precision", type=float, default=0.90)
    p.add_argument("--min-gate-support", type=int, default=30)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    progression = np.load(args.m4_labels_dir / "progression.npy", mmap_mode="r")
    masks = np.load(args.m4_labels_dir / "present_mask.npy", mmap_mode="r") \
        if (args.m4_labels_dir / "present_mask.npy").exists() else None
    index = {}
    with (args.m4_labels_dir / "manifest.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row.get("ok", True) and str(row.get("split", "")).lower() == args.split:
                index[row["image_id"]] = i

    values = {name: [] for name in C.CHEX_NAMES}
    targets = {name: [] for name in C.CHEX_NAMES}
    for line in args.m4_pred.open(encoding="utf-8"):
        rec = json.loads(line)
        iid = rec.get("image_id")
        if iid not in index:
            continue
        i = index[iid]
        for d, name in enumerate(C.CHEX_NAMES):
            item = (rec.get("diseases") or {}).get(name)
            if not item or not item.get("probs"):
                continue
            target = progression[i, :, d]
            valid = target != C.UNKNOWN
            if masks is not None:
                valid &= masks[i].astype(bool)
            target = target[valid].astype(int)
            if target.size == 0:
                continue
            counts = np.bincount(target, minlength=C.NUM_PROG)
            values[name].append(item["probs"])
            targets[name].append(int(counts.argmax()))

    temps, gates, report = {}, {}, {}
    for name in C.CHEX_NAMES:
        if name in {"Support Devices", "No Finding"}:
            temps[name] = 1.0
            gates[name] = {label: {"threshold": None, "note": "excluded"}
                           for label in C.PROG_NAMES}
            report[name] = {"T": 1.0, "n": 0, "note": "excluded from disease progression"}
            continue
        if len(values[name]) < 50 or len(set(targets[name])) < 2:
            temps[name] = 1.0
            gates[name] = {label: {"threshold": None, "note": "insufficient/one-class"}
                           for label in C.PROG_NAMES}
            report[name] = {"T": 1.0, "n": len(values[name]), "note": "insufficient/one-class"}
            continue
        p = np.asarray(values[name], dtype=np.float64)
        y = np.asarray(targets[name], dtype=np.int64)
        temps[name] = round(fit_temperature(p, y), 4)
        calibrated = calibrate_probs(p, temps[name])
        gates[name] = fit_confidence_gates(
            calibrated, y, args.target_precision, args.min_gate_support)
        report[name] = {"T": temps[name], "n": int(len(y)), "gates": gates[name]}

    out = {"per_class": temps, "confidence_gates": gates, "split": args.split,
           "policy": "regional-majority-validation-target",
           "gate_policy": {
               "scope": "per-disease-selected-change-class",
               "target_precision": args.target_precision,
               "min_support": args.min_gate_support,
           },
           "report": report}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[DONE] temporal temperatures -> {args.out}")
    for name in C.CHEX_NAMES:
        print(f"  {name:<26} T={temps[name]} n={report[name]['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
