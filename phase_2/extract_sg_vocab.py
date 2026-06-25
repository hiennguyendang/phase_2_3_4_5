"""Step 4 — scan ImaGenome scene graphs to build a controlled relation vocab.

Single source of truth for what the LLM is allowed to emit (used to snap/validate
LLM output in step 7). Writes sg_vocab.json:

    { "relations": [...],                       # all kept relation strings
      "regions": {region: [relations]},         # per-region allowed relations
      "cues": {comparison/temporal/severity/texture: [values]} }

    python extract_sg_vocab.py --scene-root /kaggle/input/<scene> --min-count 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import config
from constants import CLASS_NAMES
from scene_to_yolo import index_scene_graphs
from sg_lib import _CUE_COMPACT_TO_SCENE, compact_target_from_scene_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build sg_vocab.json from scene graphs")
    p.add_argument("--scene-root", type=Path, default=config.DEFAULT_SCENE_ROOT)
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "sg_vocab.json")
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--top-per-region", type=int, default=60)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.scene_root.exists():
        raise SystemExit(f"[ERROR] scene-root not found: {args.scene_root}")

    print(f"Indexing scene graphs under {args.scene_root} ...")
    scene_index = index_scene_graphs(args.scene_root)
    paths = list(scene_index.values())
    if args.limit:
        paths = paths[: args.limit]
    print(f"  {len(paths):,} scene graphs")

    region_rel = defaultdict(Counter)             # region -> relation -> count
    cue_values = {k: Counter() for k in _CUE_COMPACT_TO_SCENE}  # comparison/...

    for n, path in enumerate(paths):
        try:
            scene = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        compact = compact_target_from_scene_graph(scene)
        for region, findings in compact.items():
            for f in findings:
                for rel in f.get("rel", []):
                    region_rel[region][rel] += 1
                for ck in cue_values:
                    for v in f.get(ck, []):
                        cue_values[ck][v] += 1
        if (n + 1) % 5000 == 0:
            print(f"  ...{n + 1:,}")

    regions_out: dict[str, list[str]] = {}
    all_rel: set[str] = set()
    for region in CLASS_NAMES:
        counter = region_rel.get(region, Counter())
        kept = [r for r, c in counter.most_common(args.top_per_region) if c >= args.min_count]
        if kept:
            regions_out[region] = kept
            all_rel.update(kept)

    vocab = {
        "relations": sorted(all_rel),
        "regions": regions_out,
        "cues": {k: sorted(v) for k, v in cue_values.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"relations kept : {len(vocab['relations'])}")
    print(f"regions w/ rel : {len(regions_out)}/{len(CLASS_NAMES)}")
    for ck, vals in vocab["cues"].items():
        print(f"  cue {ck:<11}: {len(vals)} values -> {vals}")
    print(f"written        : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
