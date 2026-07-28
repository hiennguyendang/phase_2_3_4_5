"""Step 2 — run the trained detector on images and dump 29-region boxes.

Produces one JSON per image (plus an optional combined .jsonl) holding, for each
detected region, its pixel + normalized box and confidence. Each of the 29
regions appears at most once (highest-confidence detection kept) — matching the
"one box per anatomical region" assumption used downstream by phase_3 ROI-pool.

    python infer_yolo.py --weights <runs>/detect/det29/weights/best.pt \
        --source /kaggle/input/some-images --out /kaggle/working/pred --jsonl
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

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
    p.add_argument("--manifest", type=Path, default=None,
                   help="optional manifest.jsonl; resolves only its image_id rows without rglob")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "pred")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=config.IMGSZ)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--jsonl", action="store_true", help="also write a combined predictions.jsonl")
    p.add_argument("--no-per-image", action="store_true",
                   help="write ONLY the combined predictions.jsonl, skip per-image .json "
                        "(use for the full ~220k MIMIC run to avoid a flood of tiny files; "
                        "implies --jsonl)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0,
                   help="zero-based shard index for deterministic multi-GPU inference")
    p.add_argument("--num-shards", type=int, default=1,
                   help="number of deterministic shards; use 1 for the normal single-process run")
    return p.parse_args()


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def list_manifest_images(source: Path, manifest: Path, shard_index: int,
                         num_shards: int, limit: int | None) -> list[Path]:
    """Resolve manifest image IDs directly under pXX/pXXXXXXXX, avoiding a full tree scan."""
    image_ids = set()
    with manifest.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                image_ids.add(str(json.loads(line)["image_id"]))
    selected = sorted(image_ids)[shard_index::num_shards]
    if limit is not None:
        selected = selected[:limit]
    images, missing = [], []
    for image_id in selected:
        patient = image_id.split("_", 2)[1]  # p12345678
        parent = source / patient[:3] / patient
        path = next((parent / f"{image_id}{suffix}" for suffix in IMAGE_SUFFIXES
                     if (parent / f"{image_id}{suffix}").exists()), None)
        if path is None:
            missing.append(image_id)
        else:
            images.append(path)
    if missing:
        raise SystemExit(f"[ERROR] {len(missing):,} manifest images missing under {source}; first={missing[:5]}")
    return images


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("[ERROR] require 0 <= shard-index < num-shards and num-shards >= 1")
    if args.manifest is not None:
        if not args.manifest.exists():
            raise SystemExit(f"[ERROR] manifest not found: {args.manifest}")
        images = list_manifest_images(args.source, args.manifest, args.shard_index,
                                      args.num_shards, args.limit)
    else:
        images = list_images(args.source)
        if args.num_shards > 1:
            images = images[args.shard_index::args.num_shards]
        if args.limit is not None:
            images = images[: args.limit]
    if not images:
        raise SystemExit(f"[ERROR] no images under {args.source}")
    print(f"Running on {len(images):,} images (shard {args.shard_index + 1}/{args.num_shards})")

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    write_per_image = not args.no_per_image
    combined = None
    if args.jsonl or args.no_per_image:
        combined = open(args.out / "predictions.jsonl", "w", encoding="utf-8")

    n = 0
    last_report = 0
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
            if write_per_image:
                (args.out / f"{img_path.stem}.json").write_text(
                    json.dumps(record), encoding="utf-8")
            if combined is not None:
                combined.write(json.dumps(record) + "\n")
            n += 1
        if n - last_report >= 500 or start + args.batch >= len(images):
            print(f"  ...{n:,}/{len(images):,}")
            last_report = n

    if combined is not None:
        combined.close()
    print(f"\nDONE. {n:,} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
