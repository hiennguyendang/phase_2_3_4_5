"""Assemble a ready-to-train YOLO dataset from a PREBUILT labels tree (run on Kaggle).

The slow part of `build_yolo_dataset.py` (open every image for W/H, convert boxes) is done ONCE
LOCALLY with `--labels-only`, then uploaded as a small dataset. This script just rebuilds the
`images/{split}/` symlinks (matching label stems to the mounted images) + writes dataset.yaml.
Fast: a single os.walk over the images, no PIL, no box math.

    python link_yolo_images.py --labels-dir /kaggle/input/yolo-labels \
        --images-root /kaggle/input/mimic-cropped448 --out /kaggle/working/yolo_ds
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import os
from pathlib import Path

from constants import CLASS_NAMES, NUM_CLASSES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")

# Gold (human-verified) image_ids are held out of the model ENTIRELY (no train/val
# /test). build_yolo_dataset routed them into labels/test/, so we skip them here at
# link time — no rebuild needed. Default list ships next to this script.
DEFAULT_EXCLUDE = Path(__file__).resolve().parent / "gold_ids.txt"


def load_exclude_ids(path: Path | None) -> set[str]:
    """Read one image_id (== label/image stem) per line; '#' comments + blanks ignored."""
    if path is None:
        path = DEFAULT_EXCLUDE if DEFAULT_EXCLUDE.exists() else None
    if path is None:
        return set()
    if not path.exists():
        raise SystemExit(f"[ERROR] --exclude-ids file not found: {path}")
    ids = {
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    }
    print(f"exclude ids : {len(ids):,} held-out stems (from {path})")
    return ids


def index_images(root: Path) -> dict[str, Path]:
    """stem (== image_id) -> image path. One walk, no PIL."""
    idx: dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMAGE_EXTS:
                idx.setdefault(stem, Path(dirpath) / fn)
    return idx


def find_labels_root(labels_dir: Path) -> Path:
    """Accept either <dir>/labels/{split} or <dir>/{split} (Kaggle may wrap one folder deep)."""
    for cand in (labels_dir / "labels", labels_dir):
        if any((cand / s).is_dir() for s in SPLITS):
            return cand
    # one level down (Kaggle wrapping)
    for child in labels_dir.iterdir() if labels_dir.is_dir() else []:
        for cand in (child / "labels", child):
            if any((cand / s).is_dir() for s in SPLITS):
                return cand
    raise SystemExit(f"[ERROR] no labels/{{train,val,test}} found under {labels_dir}")


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil
        shutil.copyfile(src, dst)


def write_dataset_yaml(out: Path) -> None:
    lines = [f"path: {out.as_posix()}", "train: images/train", "val: images/val", "test: images/test",
             f"nc: {NUM_CLASSES}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    (out / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Link images for a prebuilt YOLO labels tree")
    p.add_argument("--labels-dir", type=Path, required=True, help="uploaded build_yolo_dataset --labels-only output")
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/yolo_ds"))
    p.add_argument("--exclude-ids", type=Path, default=None,
                   help="file of image_id stems to hold out of ALL splits (one per line). "
                        f"Defaults to {DEFAULT_EXCLUDE.name} next to this script if present "
                        "(the 784 gold image_ids → kept for a final-only eval, never trained).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.images_root.exists():
        raise SystemExit(f"[ERROR] images-root not found: {args.images_root}")
    labels_root = find_labels_root(args.labels_dir)
    exclude = load_exclude_ids(args.exclude_ids)
    print(f"labels root : {labels_root}")
    print(f"images root : {args.images_root}")
    print("Indexing images (one walk, no PIL) ...")
    img_index = index_images(args.images_root)
    print(f"  images indexed: {len(img_index):,}")

    per_split, missing = {s: 0 for s in SPLITS}, []
    excluded = {s: 0 for s in SPLITS}
    for split in SPLITS:
        sdir = labels_root / split
        if not sdir.is_dir():
            continue
        for txt in sdir.glob("*.txt"):
            stem = txt.stem
            if stem in exclude:
                excluded[split] += 1
                continue
            img = img_index.get(stem)
            if img is None:
                if len(missing) < 10:
                    missing.append(stem)
                continue
            link(img, args.out / "images" / split / f"{stem}{img.suffix}")
            link(txt, args.out / "labels" / split / f"{stem}.txt")
            per_split[split] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    write_dataset_yaml(args.out)
    print(f"\n[DONE] linked per split: {per_split}")
    if exclude:
        print(f"[HELD OUT] gold/excluded skipped per split: {excluded} "
              f"({sum(excluded.values()):,} total — never linked into images/labels)")
    if missing:
        print(f"[WARN] {len(missing)}+ label stems had no image (e.g. {missing[:3]}). "
              f"Check --images-root matches the dataset the labels were built from.")
    print(f"dataset.yaml -> {args.out / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
