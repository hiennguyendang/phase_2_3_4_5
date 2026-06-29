"""[PREP — run local, no GPU] Build per-(region, disease) progression targets for M4.

For every MIMIC current image, parse the scene graph's per-region `comparison_cues` (which already
encode "how this finding compares to the prior", produced by ImaGenome's NLP). For region r and the
diseases that a cued phrase's findings feed, set the progression class:
    0 stable ("no change") | 1 improved | 2 worsened | -100 not mentioned (masked)

Output  data/m4_labels/ :
    progression.npy [N, 29, 14] int8   (class index or -100)
    manifest.jsonl                     (row i <-> image_id, + split, + n_cued cells)

Per-region present masks and the prior link are NOT stored here — M4's dataset reads present masks
from the m3 label arrays and the prior from m3_pairs.jsonl.

    python phase_4/labels.py --scene-root <dir> --metadata data/mimic_metadata_final.jsonl \
                             --out-dir data/m4_labels
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import constants as C

# canonical hedge detector (repo root) — shared with phase_2 + phase_3 so "uncertain" never diverges.
# Search ancestor dirs (robust to the code being copied/relocated on Kaggle).
for _cand in Path(__file__).resolve().parents:
    if (_cand / "hedge.py").exists():
        sys.path.insert(0, str(_cand))
        break
from hedge import is_hedged  # noqa: E402


def _phrase_is_uncertain(entry: dict, p_idx: int) -> bool:
    """Hedged source sentence (silver) OR an `uncertainty_cues` entry (LLM pseudo scene graph)."""
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


def dicom_id_from_image_id(image_id: str) -> str:
    parts = image_id.split("_", 3)
    return parts[3] if len(parts) == 4 else ""


def index_scene_graphs(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(_SCENE_SUFFIX):
                index.setdefault(fn[: -len(_SCENE_SUFFIX)], Path(dirpath) / fn)
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
    return (parts[0].strip(), parts[1].strip(), parts[2].strip()) if len(parts) == 3 else None


def _cue_class(cue_list) -> int | None:
    """A phrase's comparison_cues entry (list of 'comparison|yes|<label>') -> prog class or None.
    If several, the more salient change wins (worsened > improved > stable)."""
    best = None
    for s in (cue_list or []):
        t = parse_triplet(s)
        if t is None:
            continue
        cat, pol, label = t
        if cat != C.COMPARISON_CATEGORY or pol != "yes":
            continue
        cls = C.CUE_TO_PROG.get(label)
        if cls is None:
            continue
        if best is None or C.PROG_PRIORITY[cls] > C.PROG_PRIORITY[best]:
            best = cls
    return best


def progression_from_scene(scene: dict) -> tuple[np.ndarray, int]:
    """-> (progression [29,14] int8 in {0,1,2,-100}, n_cued_cells)."""
    prog = np.full((C.NUM_REGIONS, C.NUM_CHEX), C.UNKNOWN, dtype=np.int8)
    for entry in scene.get("attributes", []) or []:
        ri = C.REGION_INDEX.get(entry.get("bbox_name"))
        if ri is None:
            continue
        phrase_attrs = entry.get("attributes", []) or []      # list[ list[str] ]
        cues = entry.get("comparison_cues", []) or []          # parallel list[ list[str] ]
        for p_idx, attrs in enumerate(phrase_attrs):
            if _phrase_is_uncertain(entry, p_idx):             # hedged -> no confident progression
                continue
            cls = _cue_class(cues[p_idx] if p_idx < len(cues) else None)
            if cls is None:
                continue
            for s in (attrs or []):
                t = parse_triplet(s)
                if t is None:
                    continue
                cat, pol, label = t
                if cat not in C.CONCEPT_CATEGORIES or pol != "yes":
                    continue                                   # change applies to a present finding
                ci = C.CONCEPT_BY_CATLABEL.get((cat, label))
                if ci is None:
                    continue
                di = C.CONCEPT_TO_CHEX[ci]
                if di < 0:
                    continue
                cur = prog[ri, di]
                if cur == C.UNKNOWN or C.PROG_PRIORITY[cls] > C.PROG_PRIORITY[int(cur)]:
                    prog[ri, di] = cls
    return prog, int((prog != C.UNKNOWN).sum())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-region disease progression targets (M4)")
    p.add_argument("--metadata", type=Path, default=C.REPO_ROOT / "data" / "mimic_metadata_final.jsonl")
    p.add_argument("--scene-root", type=Path,
                   default=Path(r"C:\Users\Dang Hien\Downloads\chest-imagenome"))
    p.add_argument("--out-dir", type=Path, default=C.REPO_ROOT / "data" / "m4_labels")
    p.add_argument("--dataset", default="mimic")
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

    eligible = []
    for row in iter_jsonl(args.metadata):
        if str(row.get("dataset", "")).lower() != args.dataset:
            continue
        iid = str(row.get("image_id", "")).strip()
        sp = scene_index.get(dicom_id_from_image_id(iid))
        if not iid or sp is None:
            continue
        eligible.append((iid, str(row.get("split", "")), sp))
        if args.limit and len(eligible) >= args.limit:
            break
    n = len(eligible)
    print(f"  {n:,} eligible mimic images with a scene graph")
    if n == 0:
        raise SystemExit("[ERROR] nothing eligible")

    progression = np.full((n, C.NUM_REGIONS, C.NUM_CHEX), C.UNKNOWN, dtype=np.int8)
    manifest, bad, n_cued = [], 0, 0
    for i, (iid, split, sp) in enumerate(tqdm(eligible, desc="progression", unit="img")):
        try:
            scene = json.loads(Path(sp).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            bad += 1
            manifest.append({"image_id": iid, "split": split, "ok": False, "n_cued": 0})
            continue
        prog, cued = progression_from_scene(scene)
        progression[i] = prog
        n_cued += cued
        manifest.append({"image_id": iid, "split": split, "ok": True, "n_cued": cued})

    np.save(args.out_dir / "progression.npy", progression)
    with open(args.out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")

    counts = {C.PROG_NAMES[k]: int((progression == k).sum()) for k in range(C.NUM_PROG)}
    with_cue = sum(1 for m in manifest if m.get("n_cued", 0) > 0)
    print(f"\n[DONE] {n:,} images (bad reads {bad}); {with_cue:,} have >=1 cued cell")
    print(f"  cued cells: {n_cued:,}  classes={counts}  (rest -100)")
    print(f"  progression.npy + manifest.jsonl -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
