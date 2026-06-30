"""Step 0 — build a YOLO detection dataset from scene graphs + metadata.

Reads mimic_metadata_final.jsonl (for split + the set of images that have a
scene graph), matches each image to its *_SceneGraph.json by dicom_id, converts
the 29-region boxes to YOLO labels, and lays out a ready-to-train dataset:

    <out>/
      images/{train,val,test}/<image_id>.jpg   # symlink (or hardlink/copy)
      labels/{train,val,test}/<image_id>.txt
      dataset.yaml

Run on Kaggle (paths auto-detected under /kaggle/input when possible):

    python build_yolo_dataset.py
    python build_yolo_dataset.py --limit 2000           # quick smoke
    python build_yolo_dataset.py --images-root /kaggle/input/foo --scene-root ...
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import os
from pathlib import Path

from PIL import Image

import config
from constants import CLASS_NAMES, NUM_CLASSES
from scene_to_yolo import (
    ConvertStats,
    dicom_id_from_image_id,
    index_images,
    index_scene_graphs,
    iter_jsonl,
    scene_to_yolo_lines,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build YOLO dataset from MIMIC scene graphs")
    p.add_argument("--metadata", type=Path, default=config.DEFAULT_METADATA)
    p.add_argument("--images-root", type=Path, default=config.DEFAULT_IMAGES_ROOT)
    p.add_argument("--scene-root", type=Path, default=config.DEFAULT_SCENE_ROOT)
    p.add_argument("--out", type=Path, default=config.DEFAULT_DATASET_DIR)
    p.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    p.add_argument("--limit", type=int, default=None, help="process only first N rows (debug)")
    p.add_argument("--keep-empty", action="store_true",
                   help="also write images whose scene graph yields 0 boxes")
    p.add_argument("--labels-only", action="store_true",
                   help="write only labels/ + dataset.yaml, NOT the image links. Build this LOCALLY "
                        "and upload it; on Kaggle run link_yolo_images.py to add the image symlinks.")
    p.add_argument("--fixed-size", type=int, default=None,
                   help="assume every image is this square size (e.g. 448 for center-cropped) instead "
                        "of opening each image for W/H. With --labels-only this needs NO local images "
                        "at all — just scene graphs + metadata.")
    return p.parse_args()


def autodetect(root: Path, kind: str) -> Path:
    """If `root` is missing and we're on Kaggle, scan /kaggle/input for a folder
    that contains the right kind of files ('image' or 'scene')."""
    if root.exists():
        return root
    base = Path("/kaggle/input")
    if not base.exists():
        return root
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(child):
            for fn in files:
                if kind == "scene" and fn.endswith("_SceneGraph.json"):
                    print(f"[autodetect] scene-root -> {child}")
                    return child
                if kind == "image" and os.path.splitext(fn)[1].lower() in (".jpg", ".jpeg", ".png"):
                    print(f"[autodetect] images-root -> {child}")
                    return child
    return root


def link_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "symlink":
            os.symlink(src, dst)
        elif mode == "hardlink":
            os.link(src, dst)
        else:
            import shutil

            shutil.copyfile(src, dst)
    except OSError:
        import shutil

        shutil.copyfile(src, dst)


def write_dataset_yaml(out: Path) -> None:
    lines = [
        f"path: {out.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {NUM_CLASSES}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    (out / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    # with --labels-only --fixed-size we never touch images -> don't require/index them
    need_images = not (args.labels_only and args.fixed_size)
    images_root = autodetect(args.images_root, "image") if need_images else args.images_root
    scene_root = autodetect(args.scene_root, "scene")

    print(f"metadata    : {args.metadata}")
    print(f"images-root : {images_root if need_images else '(skipped: fixed-size labels-only)'}")
    print(f"scene-root  : {scene_root}")
    print(f"out         : {args.out}")
    checks = [("metadata", args.metadata), ("scene-root", scene_root)]
    if need_images:
        checks.append(("images-root", images_root))
    for label, path in checks:
        if not path.exists():
            raise SystemExit(f"[ERROR] {label} not found: {path}")

    print("Indexing scene graphs (one-time walk)...")
    img_index = index_images(images_root) if need_images else {}
    scene_index = index_scene_graphs(scene_root)
    if need_images:
        print(f"  images indexed       : {len(img_index):,}")
    print(f"  scene graphs indexed : {len(scene_index):,}")

    import json

    per_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    total_stats = ConvertStats()
    seen = no_image = no_scene = no_box = unmapped_split = 0

    for n, row in enumerate(iter_jsonl(args.metadata)):
        if args.limit is not None and seen >= args.limit:
            break
        if str(row.get("dataset", "")).lower() not in ("mimic", ""):
            continue
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue
        seen += 1

        split = config.SPLIT_MAP.get(str(row.get("split", "")).strip().lower())
        if split is None:
            unmapped_split += 1
            continue

        img_path = img_index.get(image_id) if need_images else None
        if need_images and img_path is None:
            no_image += 1
            continue

        dicom = dicom_id_from_image_id(image_id)
        scene_path = scene_index.get(dicom)
        if scene_path is None:
            # fall back to the (possibly stale) basename in the metadata row
            sp = str(row.get("scene_path", "")).strip()
            if sp:
                scene_path = scene_index.get(
                    Path(sp).name[: -len("_SceneGraph.json")]
                )
        if scene_path is None:
            no_scene += 1
            continue

        if args.fixed_size:
            img_w = img_h = args.fixed_size
        else:
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                no_image += 1
                continue

        try:
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            no_scene += 1
            continue

        lines, stats = scene_to_yolo_lines(scene, img_w, img_h)
        total_stats.add(stats)
        if not lines and not args.keep_empty:
            no_box += 1
            continue

        label_path = args.out / "labels" / split / f"{image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not args.labels_only:
            link_file(img_path, args.out / "images" / split / f"{image_id}{img_path.suffix}",
                      args.link_mode)
        per_split[split] += 1

        if seen % 5000 == 0:
            print(f"  ...{seen:,} rows seen, written {sum(per_split.values()):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    write_dataset_yaml(args.out)

    print("\n=== DONE ===")
    print(f"written per split   : {per_split}")
    print(f"rows seen (mimic)   : {seen:,}")
    print(f"skipped no image    : {no_image:,}")
    print(f"skipped no scene    : {no_scene:,}")
    print(f"skipped 0 boxes     : {no_box:,}")
    print(f"skipped bad split   : {unmapped_split:,}")
    print(f"boxes kept          : {total_stats.kept:,}")
    print(f"  not-canonical     : {total_stats.dropped_not_canonical:,}")
    print(f"  sentinel (0,0,0,0): {total_stats.dropped_sentinel:,}")
    print(f"  degenerate        : {total_stats.dropped_degenerate:,}")
    print(f"  out-of-bounds     : {total_stats.dropped_out_of_bounds:,}  (clipped {total_stats.clipped:,})")
    print(f"dataset.yaml        : {args.out / 'dataset.yaml'}")
    if args.labels_only:
        print("\n[labels-only] no image links written. Upload this folder as a Kaggle dataset, then on\n"
              "Kaggle run:  python link_yolo_images.py --labels-dir <ds> --images-root <imgs> --out yolo_ds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
