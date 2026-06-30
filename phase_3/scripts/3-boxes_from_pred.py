"""[PREP — CPU, no GPU] Turn YOLO detector predictions into phase_3 detector boxes.

Reads the detector output (infer_yolo.py: predictions.jsonl OR a dir of per-image .json) and
the phase_3 label manifest, and writes — ALIGNED to the manifest row order so they drop in
next to the GT arrays:

    boxes_det.npy        [N, 29, 4]  detector box per region in INPUT_RES (448) px, 0 if undetected
    present_mask_det.npy [N, 29]     1 where the detector produced a box

phase_3 then trains/infers on these instead of the silver GT boxes (config.BOX_SOURCE="detector",
the B1-faithful default: M3 sees the SAME box source at train and launch). Keep the GT arrays
(boxes.npy/present_mask.npy) for the gold-vs-detector oracle ablation.

Coordinates: a detector box is stored as its NORMALIZED corners x N (x1n*448 ...) so it lands on
the 448x448 canvas pooling.py assumes (cell = 448/14), regardless of the source image's real size.

    python phase_3/boxes_from_pred.py \
        --pred /kaggle/working/pred/predictions.jsonl \
        --manifest data/m3_labels/manifest.jsonl --out-dir data/m3_labels
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_3/src

import argparse
import json
from pathlib import Path

import numpy as np

import config
import constants as C


def iter_predictions(pred: Path):
    """Yield prediction records from a predictions.jsonl OR a dir of per-image .json."""
    if pred.is_dir():
        for pf in sorted(pred.glob("*.json")):
            if pf.name == "predictions.jsonl":
                continue
            try:
                yield json.loads(pf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    else:
        with open(pred, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build phase_3 detector boxes from YOLO predictions")
    p.add_argument("--pred", type=Path, required=True,
                   help="predictions.jsonl file OR a dir of per-image .json (infer_yolo.py output)")
    p.add_argument("--manifest", type=Path, default=config.DEFAULT_LABELS_DIR / "manifest.jsonl")
    p.add_argument("--out-dir", type=Path, default=config.DEFAULT_LABELS_DIR)
    p.add_argument("--input-res", type=int, default=config.INPUT_RES)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"[ERROR] manifest not found: {args.manifest} (run phase_3/labels.py first)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res = float(args.input_res)

    manifest = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    row_of: dict[str, int] = {m["image_id"]: i for i, m in enumerate(manifest)}
    n = len(manifest)
    print(f"manifest rows: {n:,}")

    boxes = np.zeros((n, C.NUM_REGIONS, 4), dtype=np.int16)
    present = np.zeros((n, C.NUM_REGIONS), dtype=np.uint8)

    seen = matched = n_boxes = 0
    for rec in iter_predictions(args.pred):
        seen += 1
        i = row_of.get(str(rec.get("image_id", "")))
        if i is None:                       # a prediction for an image not in the manifest
            continue
        matched += 1
        for o in rec.get("objects", []) or []:
            ri = C.REGION_INDEX.get(o.get("bbox_name"))
            if ri is None:
                continue
            try:                            # prefer normalized corners (canvas-independent)
                x1 = float(o["x1n"]) * res; y1 = float(o["y1n"]) * res
                x2 = float(o["x2n"]) * res; y2 = float(o["y2n"]) * res
            except (KeyError, TypeError, ValueError):
                continue
            x1, x2 = sorted((max(0.0, min(res, x1)), max(0.0, min(res, x2))))
            y1, y2 = sorted((max(0.0, min(res, y1)), max(0.0, min(res, y2))))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            boxes[i, ri] = (round(x1), round(y1), round(x2), round(y2))
            present[i, ri] = 1
            n_boxes += 1
        if seen % 20000 == 0:
            print(f"  ...{seen:,} predictions read")

    np.save(args.out_dir / "boxes_det.npy", boxes)
    np.save(args.out_dir / "present_mask_det.npy", present)

    covered = int((present.sum(axis=1) > 0).sum())
    print("\n=== DONE ===")
    print(f"predictions read     : {seen:,}  (matched to manifest: {matched:,})")
    print(f"manifest rows w/ box : {covered:,}/{n:,}  ({100*covered/max(1,n):.1f}%)")
    print(f"avg regions/image    : {present.sum()/max(1,covered):.1f}  (total boxes {n_boxes:,})")
    print(f"boxes_det.npy + present_mask_det.npy -> {args.out_dir}")
    if matched < n:
        print(f"[note] {n - matched:,} manifest images had NO prediction -> their boxes are 0 "
              "(masked like an absent region). Re-run infer on the missing images if unexpected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
