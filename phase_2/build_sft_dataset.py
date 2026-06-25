"""Step 5 — build the chat SFT dataset for the scene-graph LLM.

Each MIMIC row that has a scene graph becomes one chat sample:
    system    = SYSTEM_PROMPT
    user      = report + the list of regions available in the image
    assistant = compact target JSON (ground truth from the silver scene graph)

Split train/val by a hash of patient_id (no patient leakage). Empty-target
samples (no findings) are down-sampled so the model doesn't learn to output {}.

    python build_sft_dataset.py \
      --metadata /kaggle/input/<meta>/mimic_metadata_final.jsonl \
      --scene-root /kaggle/input/<scene> --keep-empty-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import config
from scene_to_yolo import dicom_id_from_image_id, index_scene_graphs, iter_jsonl
from sg_lib import (
    SYSTEM_PROMPT,
    assemble_objects_from_scene,
    available_regions,
    build_user_prompt,
    compact_target_from_scene_graph,
    dump_compact,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chat SFT dataset for SG LLM")
    p.add_argument("--metadata", type=Path, default=config.DEFAULT_METADATA)
    p.add_argument("--scene-root", type=Path, default=config.DEFAULT_SCENE_ROOT)
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "sg_sft")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--keep-empty-frac", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def patient_bucket(patient_id: str) -> float:
    """Deterministic [0,1) value per patient (crc32)."""
    return (zlib.crc32(str(patient_id).encode()) & 0xFFFFFFFF) / 0xFFFFFFFF


def main() -> int:
    args = parse_args()
    for label, path in (("metadata", args.metadata), ("scene-root", args.scene_root)):
        if not path.exists():
            raise SystemExit(f"[ERROR] {label} not found: {path}")

    print("Indexing scene graphs ...")
    scene_index = index_scene_graphs(args.scene_root)
    print(f"  {len(scene_index):,} scene graphs")

    args.out.mkdir(parents=True, exist_ok=True)
    f_train = open(args.out / "train.jsonl", "w", encoding="utf-8")
    f_val = open(args.out / "val.jsonl", "w", encoding="utf-8")

    n_train = n_val = n_empty_kept = n_empty_drop = seen = no_scene = 0
    for row in iter_jsonl(args.metadata):
        if args.limit is not None and seen >= args.limit:
            break
        if str(row.get("dataset", "")).lower() not in ("mimic", ""):
            continue
        image_id = str(row.get("image_id", "")).strip()
        report = str(row.get("report", "")).strip()
        if not image_id or not report:
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
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            no_scene += 1
            continue

        regions = available_regions(assemble_objects_from_scene(scene))
        if not regions:
            continue
        compact = compact_target_from_scene_graph(scene)
        # keep only findings whose region is actually available in the menu
        compact = {r: v for r, v in compact.items() if r in regions}

        is_empty = len(compact) == 0
        if is_empty:
            # down-sample empties deterministically
            if patient_bucket(image_id) >= args.keep_empty_frac:
                n_empty_drop += 1
                continue
            n_empty_kept += 1

        sample = {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(report, regions)},
            {"role": "assistant", "content": dump_compact(compact)},
        ]}

        if patient_bucket(str(row.get("patient_id", image_id))) < args.val_frac:
            f_val.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_val += 1
        else:
            f_train.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_train += 1

        if seen % 10000 == 0:
            print(f"  ...{seen:,} seen, train={n_train:,} val={n_val:,}")

    f_train.close()
    f_val.close()
    print("\n=== DONE ===")
    print(f"train samples   : {n_train:,}")
    print(f"val samples     : {n_val:,}")
    print(f"empty kept/drop : {n_empty_kept:,} / {n_empty_drop:,}")
    print(f"rows seen       : {seen:,}  (no scene: {no_scene:,})")
    print(f"written         : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
