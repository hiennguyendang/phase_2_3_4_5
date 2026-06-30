"""Scan every scene graph and rank the FINDINGS by frequency.

ImaGenome attribute strings are `category|polarity|label`, e.g.
`anatomicalfinding|yes|lung opacity`. This counts them **per category** so you can
separate imaging findings (`anatomicalfinding`) from disease names (`disease`) and
from cue types (`severity`/`texture`/`comparison`/`temporal`), plus the others
(`nlp`/`technicalassessment`/`tubesandlines`/`device`).

Two uses (both in the docstring of the user's request):
  1. seed the controlled vocab the LLM is allowed to emit (step 4/7), and
  2. propose the M3 output label space = top-K `anatomicalfinding` labels, if you
     prefer findings over the 14 CheXpert disease names.

Counting unit = **region-instance**: a (label, polarity) is counted once per
(image, region) even if several sentences mention it (so the number means "in how
many region boxes this finding was asserted", which is what M3 would predict).

    python scan_findings.py --scene-root <silver_scene_graph_dir> --min-count 20
    # run again with --scene-root <gold dir> --append   to fold gold in

Writes (to --out-dir):
  findings_frequency.csv   one row per (category,label): yes,no,total,n_images,top_regions
  findings_frequency.json  {category: [[label, total], ...] sorted desc}
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import config
from scene_to_yolo import index_scene_graphs

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

# every per-phrase list on an attribute entry that holds `category|pol|label` strings
ATTR_LIST_FIELDS = (
    "attributes", "comparison_cues", "temporal_cues", "severity_cues", "texture_cues",
)


def parse_triplet(s: str):
    """`anatomicalfinding|yes|lung opacity` -> ('anatomicalfinding','yes','lung opacity').
    Tolerant of labels that themselves contain '|' (rare) by capping the split."""
    parts = s.split("|", 2)
    if len(parts) != 3:
        return None
    cat, pol, label = (p.strip() for p in parts)
    if not cat or not label:
        return None
    return cat, pol, label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank scene-graph findings by frequency")
    p.add_argument("--scene-root", type=Path, default=config.DEFAULT_SCENE_ROOT)
    p.add_argument("--out-dir", type=Path, default=config.WORK_ROOT)
    p.add_argument("--min-count", type=int, default=1, help="drop labels rarer than this in CSV")
    p.add_argument("--limit", type=int, default=None, help="scan only first N scene graphs")
    p.add_argument("--append", action="store_true",
                   help="merge into an existing *_state.json (e.g. add gold after silver)")
    return p.parse_args()


def main() -> int:
    import sys
    try:  # Windows consoles default to cp1252; keep prints from crashing
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args()
    if not args.scene_root.exists():
        raise SystemExit(f"[ERROR] scene-root not found: {args.scene_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.out_dir / "findings_frequency_state.json"

    # counters keyed by (category, label)
    pol_counts: dict[tuple, Counter] = defaultdict(Counter)     # -> {'yes':n,'no':n}
    img_counts: dict[tuple, set] = defaultdict(set)             # -> set(image_id)
    region_counts: dict[tuple, Counter] = defaultdict(Counter)  # -> region -> n
    n_scanned = 0

    if args.append and state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        for key_s, d in st["pol"].items():
            key = tuple(key_s.split("\t"))
            pol_counts[key].update(d)
        for key_s, ims in st["img"].items():
            img_counts[tuple(key_s.split("\t"))].update(ims)
        for key_s, rd in st["region"].items():
            region_counts[tuple(key_s.split("\t"))].update(rd)
        n_scanned = st.get("n_scanned", 0)
        print(f"[append] loaded prior state ({n_scanned:,} scanned)")

    print(f"Indexing scene graphs under {args.scene_root} ...")
    paths = list(index_scene_graphs(args.scene_root).values())
    if args.limit:
        paths = paths[: args.limit]
    print(f"  {len(paths):,} scene graphs to scan")

    for path in tqdm(paths, desc="scan", unit="file"):
        try:
            scene = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        image_id = str(scene.get("image_id", path.stem))
        # dedup per (image, region, cat, pol, label)
        seen: set[tuple] = set()
        for entry in scene.get("attributes", []) or []:
            region = entry.get("bbox_name", "?")
            for field in ATTR_LIST_FIELDS:
                for per_phrase in entry.get(field, []) or []:
                    for s in (per_phrase or []):
                        t = parse_triplet(str(s))
                        if t is None:
                            continue
                        cat, pol, label = t
                        sig = (region, cat, pol, label)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        key = (cat, label)
                        pol_counts[key][pol] += 1
                        img_counts[key].add(image_id)
                        region_counts[key][region] += 1
        n_scanned += 1

    # ---- write CSV (sorted by category, then total desc) ----
    rows = []
    for (cat, label), pol in pol_counts.items():
        yes, no = pol.get("yes", 0), pol.get("no", 0)
        total = sum(pol.values())
        if total < args.min_count:
            continue
        top_regions = ";".join(f"{r}:{c}" for r, c in region_counts[(cat, label)].most_common(3))
        rows.append((cat, label, yes, no, total, len(img_counts[(cat, label)]), top_regions))
    rows.sort(key=lambda r: (r[0], -r[4]))

    csv_path = args.out_dir / "findings_frequency.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "finding", "yes", "no", "total", "n_images", "top_regions"])
        w.writerows(rows)

    # ---- write JSON: per-category ranked list ----
    by_cat: dict[str, list] = defaultdict(list)
    for cat, label, yes, no, total, n_img, _ in rows:
        by_cat[cat].append([label, total, yes, no, n_img])
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: -x[1])
    json_path = args.out_dir / "findings_frequency.json"
    json_path.write_text(json.dumps(by_cat, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- persist raw state for --append ----
    state = {
        "n_scanned": n_scanned,
        "pol": {"\t".join(k): dict(v) for k, v in pol_counts.items()},
        "img": {"\t".join(k): sorted(v) for k, v in img_counts.items()},
        "region": {"\t".join(k): dict(v) for k, v in region_counts.items()},
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    # ---- report ----
    print(f"\n=== DONE ({n_scanned:,} scene graphs) ===")
    cats = Counter()
    for (cat, _), pol in pol_counts.items():
        cats[cat] += 1
    print("distinct labels per category:")
    for cat, n in cats.most_common():
        print(f"  {cat:<20} {n} labels")
    print(f"\nTop 20 anatomicalfinding (the candidate M3 label space):")
    for label, total, yes, no, n_img in by_cat.get("anatomicalfinding", [])[:20]:
        print(f"  {total:>8,}  (+{yes:,}/-{no:,})  {label}")
    print(f"\nCSV  -> {csv_path}")
    print(f"JSON -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
