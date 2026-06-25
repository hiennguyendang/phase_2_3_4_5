"""Step 2 — run the trained detector on images and dump 29-region boxes.

Produces one JSON per image (plus an optional combined .jsonl) holding, for each
detected region, its pixel + normalized box and confidence. Each of the 29
regions appears at most once (highest-confidence detection kept) — matching the
"one box per anatomical region" assumption used downstream by phase_3 ROI-pool.

    python infer_yolo.py --weights <runs>/detect/det29/weights/best.pt \
        --source /kaggle/input/some-images --out /kaggle/working/pred --jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from constants import ID_TO_CLASS

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer 29-region boxes with trained YOLO")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True, help="image file or folder")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "pred")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=config.IMGSZ)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--jsonl", action="store_true", help="also write a combined predictions.jsonl")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    images = list_images(args.source)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"[ERROR] no images under {args.source}")
    print(f"Running on {len(images):,} images")

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    combined = None
    if args.jsonl:
        combined = open(args.out / "predictions.jsonl", "w", encoding="utf-8")

    n = 0
    for start in range(0, len(images), args.batch):
        batch = images[start: start + args.batch]
        results = model.predict(
            source=[str(p) for p in batch], conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, verbose=False,
        )
        for img_path, res in zip(batch, results):
            h, w = int(res.orig_shape[0]), int(res.orig_shape[1])
            best: dict[int, dict] = {}
            for box in res.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                if cls not in best or conf > best[cls]["conf"]:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    best[cls] = {
                        "bbox_name": ID_TO_CLASS[cls],
                        "class_id": cls,
                        "conf": round(conf, 4),
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "x1n": round(x1 / w, 6), "y1n": round(y1 / h, 6),
                        "x2n": round(x2 / w, 6), "y2n": round(y2 / h, 6),
                    }
            record = {
                "image_id": img_path.stem,
                "width": w, "height": h,
                "objects": [best[k] for k in sorted(best)],
            }
            (args.out / f"{img_path.stem}.json").write_text(
                json.dumps(record), encoding="utf-8")
            if combined is not None:
                combined.write(json.dumps(record) + "\n")
            n += 1
        if n % 500 == 0 or start + args.batch >= len(images):
            print(f"  ...{n:,}/{len(images):,}")

    if combined is not None:
        combined.close()
    print(f"\nDONE. {n:,} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
