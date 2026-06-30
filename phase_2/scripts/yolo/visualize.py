"""Sanity check — draw the converted GT boxes on a few images.

Run this BEFORE a long training run to confirm images and bboxes line up (the
whole pipeline assumes they were resized/scaled together).

    python visualize.py                       # draws from the built dataset
    python visualize.py --split val --n 12
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

import config
from constants import CLASS_NAMES

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize converted YOLO GT boxes")
    p.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET_DIR)
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "viz")
    return p.parse_args()


def draw(image_path: Path, label_path: Path, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    drw = ImageDraw.Draw(img)
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            drw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=2)
            drw.text((x1 + 2, max(0, y1 - 10)), CLASS_NAMES[cid], fill=(255, 220, 0))
    img.save(out_path)


def main() -> int:
    args = parse_args()
    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    if not img_dir.exists():
        raise SystemExit(f"[ERROR] no images at {img_dir} (build the dataset first)")

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: args.n]
    args.out.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        draw(p, lbl_dir / f"{p.stem}.txt", args.out / f"viz_{p.stem}.jpg")
    print(f"Wrote {len(imgs)} visualizations -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
