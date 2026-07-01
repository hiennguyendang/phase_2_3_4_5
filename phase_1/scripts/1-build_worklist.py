"""Step 1 — gather the image_ids M1 must extract and resolve each to a CXR jpg path.

worklist = (image_id in the m3 manifest)  ∪  (prior_image_id in the m4 pairs file).
Priors are included because M4 (temporal) needs BioViL-T features for the prior image too.
Writes one JSON object per line: {"image_id", "path", "source"} -> WORK_ROOT/worklist.jsonl.

    python 1-build_worklist.py --images-root data/mimic-cxr-448 \
        --manifest data/m3_labels/manifest.jsonl --pairs data/m4_labels/m3_pairs.jsonl
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_1/src

import argparse
import json
from pathlib import Path

import config
import constants as K


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue


def collect_ids(manifest: Path, pairs: Path) -> dict[str, str]:
    """-> {image_id: source} where source in {"manifest", "prior", "manifest+prior"}."""
    ids: dict[str, str] = {}

    def add(iid: str, src: str) -> None:
        if not iid:
            return
        if iid in ids and src not in ids[iid]:
            ids[iid] = f"{ids[iid]}+{src}"
        else:
            ids.setdefault(iid, src)

    if manifest.exists():
        for row in _iter_jsonl(manifest):
            add(str(row.get("image_id", "")), "manifest")
    else:
        print(f"[WARN] manifest not found: {manifest}")
    if pairs.exists():
        for row in _iter_jsonl(pairs):
            add(str(row.get("prior_image_id", "")), "prior")
    else:
        print(f"[WARN] pairs not found (priors skipped): {pairs}")
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the M1 extraction worklist")
    p.add_argument("--images-root", type=Path, default=config.DEFAULT_IMAGES_ROOT)
    p.add_argument("--manifest", type=Path, default=config.DEFAULT_MANIFEST)
    p.add_argument("--pairs", type=Path, default=config.DEFAULT_PAIRS)
    p.add_argument("--out", type=Path, default=config.DEFAULT_WORKLIST)
    p.add_argument("--index-walk", action="store_true",
                   help="walk images-root once to resolve paths (use if the sharded guess misses)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ids = collect_ids(args.manifest, args.pairs)
    print(f"image_ids to extract: {len(ids):,}  "
          f"(manifest U prior; from {args.manifest.name} + {args.pairs.name})")

    index = K.build_image_index(args.images_root) if args.index_walk else None
    if index is not None:
        print(f"walked images-root: {len(index):,} images indexed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    found = missing = 0
    missing_examples: list[str] = []
    with open(args.out, "w", encoding="utf-8") as fh:
        for iid, src in sorted(ids.items()):
            path = (index.get(iid) if index is not None
                    else K.guess_image_path(args.images_root, iid))
            if path is None and index is None:                 # guess missed -> last-ditch walk
                index = K.build_image_index(args.images_root)
                print(f"[info] sharded guess missed {iid}; walked root ({len(index):,} images)")
                path = index.get(iid)
            if path is None:
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(iid)
                continue
            fh.write(json.dumps({"image_id": iid, "path": str(path), "source": src}) + "\n")
            found += 1

    print(f"[DONE] worklist -> {args.out}")
    print(f"  resolved : {found:,}")
    print(f"  missing  : {missing:,}" + (f"  e.g. {missing_examples}" if missing else ""))
    if missing:
        print("  (missing = image_id had no jpg under --images-root; check the mount / --index-walk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
