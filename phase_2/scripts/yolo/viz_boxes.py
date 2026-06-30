"""Visualize 29-region boxes: draw GT boxes and YOLO-predicted boxes, one colour per
region (same colour in GT and pred so you can compare the same region across the two).

Runs LOCALLY (GT drawing is pure PIL — no torch). Predicted boxes need `ultralytics`
+ a trained best.pt; omit --weights to draw GT only.

    # side-by-side GT|pred for 10 val images from a built yolo_ds
    python viz_boxes.py --yolo-ds /path/to/yolo_ds --split val \
        --weights runs/det29/weights/best.pt --n 10 --out viz

    # or point at raw dirs
    python viz_boxes.py --images-root <imgs> --labels-dir <yolo_labels/val> --n 10 --out viz
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import colorsys
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from constants import CLASS_NAMES, NUM_CLASSES

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def class_colors() -> list[tuple[int, int, int]]:
    """29 visually-distinct RGB colours, deterministic (evenly spaced hues)."""
    cols = []
    for i in range(NUM_CLASSES):
        h = (i * 0.61803398875) % 1.0  # golden-ratio hop -> well-separated hues
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


def _font(size: int = 12):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_gt(label_path: Path):
    """YOLO label -> list of (cls, cx,cy,w,h) normalized."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            out.append((int(float(p[0])), *map(float, p[1:])))
    return out


def draw_boxes(img: Image.Image, boxes, colors, font, title: str | None = None):
    """boxes: list of (cls, x1,y1,x2,y2) in PIXELS + optional conf as last item."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for b in boxes:
        cls, x1, y1, x2, y2 = b[0], b[1], b[2], b[3], b[4]
        conf = b[5] if len(b) > 5 else None
        col = colors[cls % NUM_CLASSES]
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        lab = f"{name} {conf:.2f}" if conf is not None else name
        tw = d.textlength(lab, font=font)
        d.rectangle([x1, y1, x1 + tw + 4, y1 + 14], fill=col)
        d.text((x1 + 2, y1 + 1), lab, fill=(0, 0, 0), font=font)
    if title:
        d.rectangle([0, 0, d.textlength(title, font=font) + 6, 16], fill=(0, 0, 0))
        d.text((3, 1), title, fill=(255, 255, 255), font=font)
    return im


def gt_to_pixels(gt, W, H):
    out = []
    for cls, cx, cy, w, h in gt:
        out.append((cls, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def predict_pixels(model, img_path, imgsz, device, conf):
    r = model.predict(str(img_path), imgsz=imgsz, device=device, conf=conf,
                      verbose=False, max_det=60)[0]
    out = []
    if r.boxes is not None and len(r.boxes):
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        for c, cc, xy in zip(cls, cf, xyxy):
            out.append((int(c), float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3]), float(cc)))
    return out


def side_by_side(a: Image.Image, b: Image.Image, gap: int = 8) -> Image.Image:
    W = a.width + gap + b.width
    H = max(a.height, b.height)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    return canvas


def make_legend(colors, font, out: Path):
    rows = NUM_CLASSES
    rh, w = 16, 220
    im = Image.new("RGB", (w, rows * rh + 4), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i, name in enumerate(CLASS_NAMES):
        y = i * rh + 2
        d.rectangle([2, y + 2, 14, y + 12], fill=colors[i])
        d.text((20, y), f"{i}:{name}", fill=(0, 0, 0), font=font)
    im.save(out / "_legend.png")


def resolve_labels_dir(labels_dir: Path, split: str) -> Path:
    """Accept a dir that holds *.txt directly, OR a parent that nests them under
    {split} / labels/{split} / labels/labels/{split} (build_yolo_dataset + Kaggle wrap)."""
    if any(labels_dir.glob("*.txt")):
        return labels_dir
    for cand in (labels_dir / split, labels_dir / "labels" / split,
                 labels_dir / "labels" / "labels" / split):
        if cand.is_dir() and any(cand.glob("*.txt")):
            return cand
    return labels_dir  # let the caller error with the original path


def find_images(root: Path):
    idx = {}
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMAGE_EXTS:
                idx.setdefault(stem, Path(dp) / fn)
    return idx


def parse_args():
    p = argparse.ArgumentParser(description="Draw GT vs YOLO boxes, one colour per region")
    p.add_argument("--yolo-ds", type=Path, default=None, help="dir with images/ labels/ (sets roots)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--images-root", type=Path, default=None)
    p.add_argument("--labels-dir", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None, help="best.pt (omit -> GT only)")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=448)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0")
    p.add_argument("--out", type=Path, default=Path("viz"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.yolo_ds:
        args.images_root = args.images_root or args.yolo_ds / "images" / args.split
        args.labels_dir = args.labels_dir or args.yolo_ds / "labels" / args.split
    if not args.images_root or not args.labels_dir:
        raise SystemExit("[ERROR] give --yolo-ds OR (--images-root AND --labels-dir)")
    args.labels_dir = resolve_labels_dir(Path(args.labels_dir), args.split)
    args.out.mkdir(parents=True, exist_ok=True)

    colors = class_colors()
    font = _font(12)
    make_legend(colors, font, args.out)

    label_files = sorted(Path(args.labels_dir).glob("*.txt"))
    if not label_files:
        raise SystemExit(f"[ERROR] no .txt labels under {args.labels_dir}")
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(label_files)
    label_files = label_files[: args.n]

    img_index = find_images(args.images_root)
    print(f"labels: {args.labels_dir} | images: {args.images_root} | picked {len(label_files)}")

    model = None
    if args.weights:
        from ultralytics import YOLO
        model = YOLO(str(args.weights))
        print(f"weights: {args.weights}")
    else:
        print("no --weights -> drawing GT only")

    done = 0
    for lf in label_files:
        stem = lf.stem
        ip = img_index.get(stem)
        if ip is None:
            print(f"  [skip] no image for {stem}")
            continue
        img = Image.open(ip)
        W, H = img.size
        gt_px = gt_to_pixels(read_gt(lf), W, H)
        gt_im = draw_boxes(img, gt_px, colors, font, title="GT")
        gt_im.save(args.out / f"{stem}_gt.jpg")
        if model is not None:
            pred_px = predict_pixels(model, ip, args.imgsz, args.device, args.conf)
            pred_im = draw_boxes(img, pred_px, colors, font, title=f"YOLO conf>={args.conf}")
            pred_im.save(args.out / f"{stem}_pred.jpg")
            side_by_side(gt_im, pred_im).save(args.out / f"{stem}_cmp.jpg")
        done += 1
        print(f"  [{done}] {stem}  GT boxes={len(gt_px)}"
              + (f"  pred boxes={len(pred_px)}" if model is not None else ""))

    print(f"\n[DONE] {done} images -> {args.out}  (legend: _legend.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
