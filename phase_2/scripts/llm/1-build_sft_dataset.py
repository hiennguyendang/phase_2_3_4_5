"""Step 5 — build the chat SFT dataset for the FLAT scene-graph extractor LLM.

Each MIMIC row that has a scene graph becomes one chat sample:
    system    = SYSTEM_PROMPT          (sg_schema: task + the 69 allowed findings)
    user      = report + the regions available in the image
    assistant = the flat findings JSON (ground truth distilled from the silver scene graph)

The assistant target is the SIMPLE per-region structure (see sg_schema.py):
    { "<region>": [ {"finding","presence","progression"} , ... ] }
so a small (3B) model only copies a finding NAME, not a pipe-delimited relation string.

Split train/val/test by the metadata `split` field — the SAME canonical split YOLO (phase_2)
and phase_3/phase_4 use — so the whole pipeline shares one split and there is NO cross-module
leakage. (Do NOT re-split by hash: a patient in the official test could land in LLM-train.)
    split train      -> train
    split val/valid  -> val      (dev / overfit check / model selection)
    split test       -> test     (in-dataset held-out test, the reportable number)
    split gold       -> HELD OUT  (optional human-verified eval; ~800 imgs, not statistically
                                   powerful, so the silver `test` split is the primary test)
Empty-target samples (no findings) are down-sampled so the model doesn't learn to output {}.

Runs LOCALLY (no GPU). Example:
    python build_sft_dataset.py \
      --metadata data/mimic_metadata_final.jsonl \
      --scene-root "C:/Users/Dang Hien/Downloads/chest-imagenome" \
      --out phase_2/_work/sg_sft
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import json
import zlib
from pathlib import Path

import config
from scene_to_yolo import dicom_id_from_image_id, index_scene_graphs, iter_jsonl
from sg_lib import assemble_objects_from_scene, available_regions
from sg_schema import SYSTEM_PROMPT, build_user_prompt, dump_flat, flat_from_scene_graph

_DEFAULT_EXCLUDE = Path(__file__).resolve().parent / "gold_ids.txt"

# metadata `split` value -> output split (gold/unknown -> held out). Matches config.SPLIT_MAP
# except gold is EXCLUDED here (optional final eval), not folded into test.
_SPLIT_ROUTE = {"train": "train", "val": "val", "valid": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chat SFT dataset for the flat SG LLM")
    p.add_argument("--metadata", type=Path, default=config.DEFAULT_METADATA)
    p.add_argument("--scene-root", type=Path, default=config.DEFAULT_SCENE_ROOT)
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "sg_sft")
    p.add_argument("--keep-empty-frac", type=float, default=0.1)
    p.add_argument("--exclude-ids", type=Path, default=_DEFAULT_EXCLUDE,
                   help="newline-separated image_ids to hold out (default: gold_ids.txt). "
                        "Pass a non-existent path to disable.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def patient_bucket(key: str) -> float:
    """Deterministic [0,1) value (crc32)."""
    return (zlib.crc32(str(key).encode()) & 0xFFFFFFFF) / 0xFFFFFFFF


def load_exclude_ids(path: Path) -> set[str]:
    if not path or not path.exists():
        return set()
    ids = {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}
    print(f"excluding {len(ids):,} held-out ids from {path.name}")
    return ids


def main() -> int:
    args = parse_args()
    for label, path in (("metadata", args.metadata), ("scene-root", args.scene_root)):
        if not path.exists():
            raise SystemExit(f"[ERROR] {label} not found: {path}")

    exclude = load_exclude_ids(args.exclude_ids)

    print("Indexing scene graphs ...")
    scene_index = index_scene_graphs(args.scene_root)
    print(f"  {len(scene_index):,} scene graphs")

    args.out.mkdir(parents=True, exist_ok=True)
    f_train = open(args.out / "train.jsonl", "w", encoding="utf-8")
    f_val = open(args.out / "val.jsonl", "w", encoding="utf-8")
    f_test = open(args.out / "test.jsonl", "w", encoding="utf-8")

    writers = {"train": f_train, "val": f_val, "test": f_test}
    counts = {"train": 0, "val": 0, "test": 0}
    n_empty_kept = n_empty_drop = seen = no_scene = n_excluded = n_heldout = 0
    for row in iter_jsonl(args.metadata):
        if args.limit is not None and seen >= args.limit:
            break
        if str(row.get("dataset", "")).lower() not in ("mimic", ""):
            continue
        image_id = str(row.get("image_id", "")).strip()
        report = str(row.get("report", "")).strip()
        if not image_id or not report:
            continue
        if image_id in exclude:
            n_excluded += 1
            continue
        split = _SPLIT_ROUTE.get(str(row.get("split", "")).strip().lower())
        if split is None:                 # gold / unknown -> held out
            n_heldout += 1
            continue
        seen += 1

        scene_path = scene_index.get(dicom_id_from_image_id(image_id))
        if scene_path is None:
            sp = str(row.get("scene_path", "")).strip()
            if sp.endswith("_SceneGraph.json"):
                scene_path = scene_index.get(Path(sp).name[: -len("_SceneGraph.json")])
        if scene_path is None:
            no_scene += 1
            continue

        try:
            scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            no_scene += 1
            continue

        regions = available_regions(assemble_objects_from_scene(scene))
        if not regions:
            continue
        flat = flat_from_scene_graph(scene)
        # keep only findings whose region is actually available in the image's menu
        flat = {r: v for r, v in flat.items() if r in regions}

        is_empty = len(flat) == 0
        if is_empty:
            if patient_bucket(image_id) >= args.keep_empty_frac:   # down-sample empties
                n_empty_drop += 1
                continue
            n_empty_kept += 1

        sample = {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(report, regions)},
            {"role": "assistant", "content": dump_flat(flat)},
        ]}

        writers[split].write(json.dumps(sample, ensure_ascii=False) + "\n")   # metadata split
        counts[split] += 1

        if seen % 10000 == 0:
            print(f"  ...{seen:,} seen, train={counts['train']:,} "
                  f"val={counts['val']:,} test={counts['test']:,}")

    f_train.close()
    f_val.close()
    f_test.close()
    print("\n=== DONE ===")
    print(f"train samples   : {counts['train']:,}")
    print(f"val samples     : {counts['val']:,}")
    print(f"test samples    : {counts['test']:,}")
    print(f"empty kept/drop : {n_empty_kept:,} / {n_empty_drop:,}")
    print(f"held out (gold) : {n_heldout:,} by split  + {n_excluded:,} by id-list")
    print(f"rows seen       : {seen:,}  (no scene: {no_scene:,})")
    print(f"written         : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
