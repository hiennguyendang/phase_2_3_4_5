"""Step 3 — verify the M1 feature cache against the phase_3 contract (run before trusting it).

Checks (each prints PASS/FAIL):
  1. STRUCTURE  every .pt is [197, C] float16, one consistent C, no NaN/Inf.
  2. NAMING     every stem is a valid manifest image_id or a pairs prior_image_id.
  3. COVERAGE   how many manifest image_ids have a feature file (and how many priors).
  4. REFERENCE  reproduce a known image's features with the live encoder and assert cosine≈1
                vs docs/<id>.pt — this proves model variant + preprocessing + flatten order
                end-to-end (needs health_multimodal + the source jpg; skipped if unavailable).
  5. ALIGNMENT  overlay a region's bbox on the 14x14 grid (same cell math as pooling.py) and
                save a PNG so a human confirms the cells fall on the right anatomy.

    python 3-verify_features.py --features-root /kaggle/working/features \
        --labels-dir data/m3_labels --images-root data/mimic-cxr-448 --reference docs/<id>.pt
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_1/src

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
import constants as K


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def _load_pt(path: Path) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        t = next((v for v in t.values() if torch.is_tensor(v)), None)
    return t


# ---- 1+2: structure + naming -------------------------------------------------
def check_structure_naming(root: Path, valid_ids: set[str], sample: int) -> int:
    files = list(root.rglob("*.pt"))
    if not files:
        print(f"[1/2 FAIL] no .pt under {root}")
        return 0
    if sample and len(files) > sample:
        idx = np.linspace(0, len(files) - 1, sample).astype(int)
        check = [files[i] for i in idx]
        print(f"structure: sampling {sample}/{len(files):,} files")
    else:
        check = files

    dims, bad_shape, bad_dtype, bad_nan, bad_name = set(), 0, 0, 0, 0
    name_examples: list[str] = []
    for p in check:
        t = _load_pt(p)
        if t is None or t.ndim != 2 or t.shape[0] != config.GRID_TOKENS + 1:
            bad_shape += 1
            continue
        dims.add(int(t.shape[1]))
        if t.dtype != torch.float16:
            bad_dtype += 1
        if not torch.isfinite(t.to(torch.float32)).all():
            bad_nan += 1
        if valid_ids and p.stem not in valid_ids:
            bad_name += 1
            if len(name_examples) < 5:
                name_examples.append(p.stem)

    ok1 = bad_shape == 0 and bad_dtype == 0 and bad_nan == 0 and len(dims) == 1
    print(f"[1 {'PASS' if ok1 else 'FAIL'}] structure: {len(check):,} checked | "
          f"C={sorted(dims)} | bad_shape={bad_shape} bad_dtype={bad_dtype} nonfinite={bad_nan}")
    if dims and sorted(dims)[0] != config.FEAT_DIM:
        print(f"        note: C={sorted(dims)} (config.FEAT_DIM={config.FEAT_DIM}); "
              f"phase_3 auto-detects but the whole cache must share ONE C.")
    ok2 = (not valid_ids) or bad_name == 0
    print(f"[2 {'PASS' if ok2 else 'FAIL'}] naming: {bad_name} stems not in manifest+priors"
          + (f"  e.g. {name_examples}" if bad_name else ""))
    return len(dims) == 1 and sorted(dims)[0] if ok1 else 0


# ---- 3: coverage -------------------------------------------------------------
def check_coverage(root: Path, manifest_ids: set[str], prior_ids: set[str]) -> None:
    have = {p.stem for p in root.rglob("*.pt")}
    cov_m = len(manifest_ids & have)
    cov_p = len(prior_ids & have)
    miss = sorted(manifest_ids - have)
    print(f"[3] coverage: manifest {cov_m:,}/{len(manifest_ids):,} "
          f"({100 * cov_m / max(1, len(manifest_ids)):.1f}%) | priors {cov_p:,}/{len(prior_ids):,}")
    if miss:
        print(f"    missing {len(miss):,} manifest features, e.g. {miss[:3]}")


# ---- 4: reference reproduce --------------------------------------------------
def check_reference(reference: Path, images_root: Path, device: str) -> None:
    if not reference.exists():
        print(f"[4 SKIP] reference not found: {reference}")
        return
    ref = _load_pt(reference).to(torch.float32)
    image_id = reference.stem
    src = K.guess_image_path(images_root, image_id) or K.build_image_index(images_root).get(image_id)
    if src is None:
        print(f"[4 SKIP] source jpg for {image_id} not under {images_root}")
        return
    try:
        import biovilt
        model = biovilt.load_encoder(device)
    except Exception as e:  # noqa: BLE001
        print(f"[4 SKIP] could not load BioViL-T ({e}); run on Kaggle with health_multimodal")
        return
    img = biovilt.load_image(src).unsqueeze(0)
    got = biovilt.encode_batch(model, img, device)[0].to(torch.float32)
    if got.shape != ref.shape:
        print(f"[4 FAIL] shape {tuple(got.shape)} != reference {tuple(ref.shape)}")
        return
    cos = torch.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    row_cos = torch.cosine_similarity(got, ref, dim=1)            # per-row [197]
    worst = row_cos.min().item()
    ok = cos > 0.99 and worst > 0.97
    print(f"[4 {'PASS' if ok else 'FAIL'}] reference reproduce: flat cos={cos:.5f}, "
          f"min row cos={worst:.5f}  (transform={config.TRANSFORM_MODE})")
    if not ok:
        print("    -> features DIFFER from the reference. Try TRANSFORM_MODE='resize_crop' "
              "(or check the encoder variant); the cache must match the existing collaborator set.")


# ---- 5: alignment overlay ----------------------------------------------------
def box_to_cells(box, cell: float, gh: int, gw: int):
    """bbox (x1,y1,x2,y2) in 448px -> (cols, rows) ranges of covered cells (pooling.py math)."""
    x1, y1, x2, y2 = box
    c1, r1 = int(np.floor(x1 / cell)), int(np.floor(y1 / cell))
    c2, r2 = int(np.ceil(x2 / cell)), int(np.ceil(y2 / cell))
    return (max(0, c1), min(gw, c2)), (max(0, r1), min(gh, r2))


def _alignment(labels_dir: Path, images_root: Path, out_png: Path, region_idx: int,
               region_name: str) -> None:
    boxes = np.load(labels_dir / "boxes.npy")                    # [N,29,4] int16, 448 space
    manifest = list(_iter_jsonl(labels_dir / "manifest.jsonl"))
    # first image whose chosen region has a non-empty box
    pick = None
    for i, row in enumerate(manifest):
        b = boxes[i, region_idx]
        if (b != 0).any() and b[2] > b[0] and b[3] > b[1]:
            pick = (i, row["image_id"], b)
            break
    if pick is None:
        print(f"[5 SKIP] no non-empty '{region_name}' box found")
        return
    i, image_id, box = pick
    src = K.guess_image_path(images_root, image_id) or K.build_image_index(images_root).get(image_id)
    (c1, c2), (r1, r2) = box_to_cells(box, config.CELL, config.GRID_H, config.GRID_W)
    print(f"[5] alignment: '{region_name}' on {image_id}")
    print(f"    box(px)={tuple(int(v) for v in box)} -> cols[{c1}:{c2}] rows[{r1}:{r2}] "
          f"(token idx y*14+x; cell={config.CELL:.0f}px)")
    if src is None:
        print(f"    [SKIP png] source jpg not found under {images_root}")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image

        img = Image.open(src).convert("L").resize((config.INPUT_RES, config.INPUT_RES))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray")
        for k in range(config.GRID_W + 1):                       # 14x14 grid
            ax.axvline(k * config.CELL, color="cyan", lw=0.4, alpha=0.5)
            ax.axhline(k * config.CELL, color="cyan", lw=0.4, alpha=0.5)
        for rr in range(r1, r2):                                 # shade covered cells
            for cc in range(c1, c2):
                ax.add_patch(Rectangle((cc * config.CELL, rr * config.CELL), config.CELL,
                                       config.CELL, color="lime", alpha=0.30))
        ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                               edgecolor="red", facecolor="none", lw=2))
        ax.set_title(f"{region_name}\n{image_id[:32]}…", fontsize=8)
        ax.axis("off")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"    [PASS] overlay saved -> {out_png}  (confirm green cells sit on the {region_name})")
    except Exception as e:  # noqa: BLE001
        print(f"    [SKIP png] could not render ({e})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify the M1 feature cache")
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_OUT)
    p.add_argument("--labels-dir", type=Path, default=config.REPO_ROOT / "data" / "m3_labels")
    p.add_argument("--manifest", type=Path, default=config.DEFAULT_MANIFEST)
    p.add_argument("--pairs", type=Path, default=config.DEFAULT_PAIRS)
    p.add_argument("--images-root", type=Path, default=config.DEFAULT_IMAGES_ROOT)
    p.add_argument("--reference", type=Path, default=config.DEFAULT_REFERENCE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample", type=int, default=2000, help="max .pt to deep-check (0 = all)")
    p.add_argument("--region", default="right lung", help="anatomical region for the overlay")
    p.add_argument("--align-png", type=Path, default=config.WORK_ROOT / "alignment_check.png")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_ids = {str(r.get("image_id", "")) for r in _iter_jsonl(args.manifest)} \
        if args.manifest.exists() else set()
    prior_ids = {str(r.get("prior_image_id", "")) for r in _iter_jsonl(args.pairs)} \
        if args.pairs.exists() else set()
    prior_ids.discard("")
    valid = manifest_ids | prior_ids
    print(f"manifest ids {len(manifest_ids):,} | prior ids {len(prior_ids):,}\n")

    check_structure_naming(args.features_root, valid, args.sample)
    check_coverage(args.features_root, manifest_ids, prior_ids)
    check_reference(args.reference, args.images_root, args.device)

    # region index from phase_3's canonical region list (29 detector classes)
    region_names = [
        "abdomen", "aortic arch", "cardiac silhouette", "carina", "cavoatrial junction",
        "left apical zone", "left clavicle", "left costophrenic angle", "left hemidiaphragm",
        "left hilar structures", "left lower lung zone", "left lung", "left mid lung zone",
        "left upper lung zone", "mediastinum", "right apical zone", "right atrium",
        "right clavicle", "right costophrenic angle", "right hemidiaphragm",
        "right hilar structures", "right lower lung zone", "right lung", "right mid lung zone",
        "right upper lung zone", "spine", "svc", "trachea", "upper mediastinum",
    ]
    ridx = region_names.index(args.region) if args.region in region_names else 22  # right lung
    _alignment(args.labels_dir, args.images_root, args.align_png, ridx, region_names[ridx])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
