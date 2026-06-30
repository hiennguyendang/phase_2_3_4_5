"""Evaluate the trained detector (mAP) on a split of the built dataset.

    python eval_yolo.py --weights <runs>/detect/det29/weights/best.pt --split test
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
from pathlib import Path

import config
from constants import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval YOLO detector (mAP)")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=config.DEFAULT_DATASET_DIR / "dataset.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=config.IMGSZ)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data), split=args.split, imgsz=args.imgsz,
        batch=args.batch, device=args.device, verbose=True,
    )
    print("\n=== OVERALL ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall   : {metrics.box.mr:.4f}")

    print("\n=== PER CLASS (mAP50-95) ===")
    try:
        for i, ap in zip(metrics.box.ap_class_index, metrics.box.maps[metrics.box.ap_class_index]):
            print(f"  {CLASS_NAMES[int(i)]:<26} {ap:.4f}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
