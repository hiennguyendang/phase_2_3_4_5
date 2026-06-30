"""[PREP — run local, no GPU] Build per-region M3 targets from scene graphs.

For every MIMIC image that has a scene graph, produce:
  region_concepts [29, 69]  per-region concept target   (1 yes / 0 no / -100 unknown)
  region_chexpert [29, 14]  per-region CheXpert, DERIVED from concepts via the map
  boxes           [29, 4]   GT bbox per region (448-crop space; 0 if region absent)
  present_mask    [29]      1 if the region has a usable box
and per image:
  image_chexpert  [14]      image-level CheXpert, straight from metadata `labels`

Saved as stacked arrays + manifest.jsonl (row i <-> image_id), so phase_3 training
just memory-maps them. This is the biggest "prepare before Kaggle" step — pure data.

    python phase_3/labels.py --scene-root <dir> --metadata data/mimic_metadata_final.jsonl \
                             --out-dir data/m3_labels
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_3/src

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import constants as C

# canonical hedge detector (repo root) — shared with phase_2 + phase_4 so "uncertain" never diverges.
# Search ancestor dirs (robust to the code being copied/relocated on Kaggle).
for _cand in Path(__file__).resolve().parents:
    if (_cand / "hedge.py").exists():
        sys.path.insert(0, str(_cand))
        break
from hedge import is_hedged  # noqa: E402


def _phrase_is_uncertain(entry: dict, p_idx: int) -> bool:
    """A phrase group is uncertain if its source sentence is hedged (silver) OR it carries an
    `uncertainty_cues` entry (LLM-assembled pseudo scene graph, where phrases are empty)."""
    phrases = entry.get("phrases", []) or []
    if p_idx < len(phrases) and is_hedged(phrases[p_idx]):
        return True
    unc = entry.get("uncertainty_cues", []) or []
    return bool(p_idx < len(unc) and unc[p_idx])

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

_SCENE_SUFFIX = "_SceneGraph.json"


# ---- small scene-graph helpers (kept inline to avoid phase_2 import collisions) ----
def dicom_id_from_image_id(image_id: str) -> str:
    parts = image_id.split("_", 3)
    return parts[3] if len(parts) == 4 else ""


def index_scene_graphs(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(_SCENE_SUFFIX):
                dicom = fn[: -len(_SCENE_SUFFIX)]
                index.setdefault(dicom, Path(dirpath) / fn)
    return index


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8-sig") as stream:
        for raw in stream:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def parse_triplet(s: str):
    parts = str(s).split("|", 2)
    if len(parts) != 3:
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()  # cat, pol, label


def region_concepts_from_scene(scene: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (region_concepts[29,69] int8, boxes[29,4] int16, present_mask[29] uint8)."""
    rc = np.full((C.NUM_REGIONS, C.NUM_CONCEPTS), C.UNKNOWN, dtype=np.int8)
    boxes = np.zeros((C.NUM_REGIONS, 4), dtype=np.int16)
    mask = np.zeros(C.NUM_REGIONS, dtype=np.uint8)

    # boxes from objects[]
    for obj in scene.get("objects", []) or []:
        name = obj.get("bbox_name")
        ri = C.REGION_INDEX.get(name)
        if ri is None:
            continue
        try:
            x1, y1, x2, y2 = int(obj["x1"]), int(obj["y1"]), int(obj["x2"]), int(obj["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if (x1, y1, x2, y2) == (0, 0, 0, 0) or x2 <= x1 or y2 <= y1:
            continue  # sentinel / degenerate
        boxes[ri] = (x1, y1, x2, y2)
        mask[ri] = 1

    # concept polarity from attributes[] (yes wins over no; absent stays -100).
    # HEDGED mentions are SKIPPED -> a hedged-only concept stays -100 (masked-BCE ignores it),
    # so "possible pneumonia" never trains the image classifier as a confident finding. A certain
    # mention in another phrase still sets 1/0 normally.
    for entry in scene.get("attributes", []) or []:
        ri = C.REGION_INDEX.get(entry.get("bbox_name"))
        if ri is None:
            continue
        for p_idx, per_phrase in enumerate(entry.get("attributes", []) or []):
            if _phrase_is_uncertain(entry, p_idx):
                continue
            for s in (per_phrase or []):
                t = parse_triplet(s)
                if t is None:
                    continue
                cat, pol, label = t
                if cat not in C.CONCEPT_CATEGORIES:
                    continue
                ci = C.CONCEPT_BY_CATLABEL.get((cat, label))
                if ci is None:
                    continue
                if pol == "yes":
                    rc[ri, ci] = 1
                elif pol == "no" and rc[ri, ci] == C.UNKNOWN:
                    rc[ri, ci] = 0
    return rc, boxes, mask


def derive_region_chexpert(rc: np.ndarray) -> np.ndarray:
    """[29,69] -> [29,14]: a CheXpert is 1 if any feeding concept is 1, else 0 if any is 0,
    else -100. (No Finding has no feeding concepts -> stays -100 at region level.)"""
    out = np.full((C.NUM_REGIONS, C.NUM_CHEX), C.UNKNOWN, dtype=np.int8)
    for xi, cis in C.CHEX_FROM_CONCEPTS.items():
        if not cis:
            continue
        sub = rc[:, cis]                       # [29, k]
        has_pos = (sub == 1).any(axis=1)
        has_neg = (sub == 0).any(axis=1)
        col = np.full(C.NUM_REGIONS, C.UNKNOWN, dtype=np.int8)
        col[has_neg] = 0
        col[has_pos] = 1                       # positive overrides
        out[:, xi] = col
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-region M3 concept/CheXpert labels")
    p.add_argument("--metadata", type=Path, default=C.REPO_ROOT / "data" / "mimic_metadata_final.jsonl")
    p.add_argument("--scene-root", type=Path,
                   default=Path(r"C:\Users\Dang Hien\Downloads\chest-imagenome"))
    p.add_argument("--out-dir", type=Path, default=C.REPO_ROOT / "data" / "m3_labels")
    p.add_argument("--dataset", default="mimic", help="only rows with this `dataset` value")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.scene_root.is_dir():
        raise SystemExit(f"[ERROR] scene-root not found: {args.scene_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Indexing scene graphs ...")
    scene_index = index_scene_graphs(args.scene_root)
    print(f"  {len(scene_index):,} scene graphs")

    # pass 1: collect eligible rows (mimic + has scene graph)
    eligible = []
    for row in iter_jsonl(args.metadata):
        if str(row.get("dataset", "")).lower() != args.dataset:
            continue
        iid = str(row.get("image_id", "")).strip()
        dicom = dicom_id_from_image_id(iid)
        sp = scene_index.get(dicom)
        if not iid or sp is None:
            continue
        eligible.append((iid, dicom, str(row.get("split", "")), row.get("labels"), sp))
        if args.limit and len(eligible) >= args.limit:
            break
    n = len(eligible)
    print(f"  {n:,} eligible images (mimic + scene graph)")
    if n == 0:
        raise SystemExit("[ERROR] nothing eligible")

    # preallocate
    region_concepts = np.full((n, C.NUM_REGIONS, C.NUM_CONCEPTS), C.UNKNOWN, dtype=np.int8)
    region_chexpert = np.full((n, C.NUM_REGIONS, C.NUM_CHEX), C.UNKNOWN, dtype=np.int8)
    boxes = np.zeros((n, C.NUM_REGIONS, 4), dtype=np.int16)
    present = np.zeros((n, C.NUM_REGIONS), dtype=np.uint8)
    image_chex = np.full((n, C.NUM_CHEX), C.UNKNOWN, dtype=np.int8)

    manifest = []
    bad = 0
    for i, (iid, dicom, split, labels, sp) in enumerate(tqdm(eligible, desc="labels", unit="img")):
        try:
            scene = json.loads(Path(sp).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            bad += 1
            manifest.append({"image_id": iid, "dicom": dicom, "split": split, "ok": False})
            continue
        rc, bx, mk = region_concepts_from_scene(scene)
        region_concepts[i] = rc
        region_chexpert[i] = derive_region_chexpert(rc)
        boxes[i] = bx
        present[i] = mk
        if isinstance(labels, list) and len(labels) == C.NUM_CHEX:
            image_chex[i] = np.asarray(labels, dtype=np.int8)
        manifest.append({"image_id": iid, "dicom": dicom, "split": split,
                         "n_regions": int(mk.sum()), "ok": True})

    np.save(args.out_dir / "region_concepts.npy", region_concepts)
    np.save(args.out_dir / "region_chexpert.npy", region_chexpert)
    np.save(args.out_dir / "boxes.npy", boxes)
    np.save(args.out_dir / "present_mask.npy", present)
    np.save(args.out_dir / "image_chexpert.npy", image_chex)
    with open(args.out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")

    pos = int((region_concepts == 1).sum())
    neg = int((region_concepts == 0).sum())
    print(f"\n[DONE] {n:,} images  (bad scene reads: {bad})")
    print(f"  region_concepts: +1={pos:,}  0={neg:,}  -100={int((region_concepts == C.UNKNOWN).sum()):,}")
    print(f"  avg regions/image w/ box: {present.sum() / n:.1f}")
    print(f"  arrays + manifest.jsonl -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
