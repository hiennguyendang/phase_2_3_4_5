"""[UTILITY] Apply data-validated edits to the concept->CheXpert crosswalk, then re-derive
the per-region CheXpert targets from the EXISTING region_concepts.npy (no scene graphs needed).

The edits (from the MI/lift validation against the independent image-level CheXpert labels,
scratchpad/validate_crosswalk.py): add two clinically-correct edges the map was missing, drop
two benign nodule edges the data does not support.

  ADD  aspiration        -> Pneumonia     (aspiration pneumonia; lift x2.2)
  ADD  lung cancer       -> Lung Lesion    (a mass IS a lesion; correct direction)
  DROP calcified nodule  -> (None)         (benign granuloma; MI~0 with Lung Lesion)
  DROP cyst/bullae       -> (None)         (lucent, not a nodule/mass; MI~0)

Idempotent (keyed by concept name). Backs up both files. After running, RETRAIN B-faithful
(the mask rebuilds from constants; region_chexpert.npy targets are regenerated here).

    python phase_3/scripts/patch_crosswalk.py --labels-dir data/m3_labels
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_3/src

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

# concept name -> new CheXpert slot (None removes the edge). Keyed by name so it's robust.
EDITS: dict[str, str | None] = {
    "aspiration": "Pneumonia",
    "lung cancer": "Lung Lesion",
    "calcified nodule": None,
    "cyst/bullae": None,
}


def parse_args() -> argparse.Namespace:
    import config
    p = argparse.ArgumentParser(description="Patch concept->CheXpert crosswalk + re-derive region CheXpert")
    default_concept_space = Path(__file__).resolve().parents[1] / "src" / "m3_concept_space.json"
    p.add_argument("--concept-space", type=Path, default=default_concept_space)
    p.add_argument("--labels-dir", type=Path, default=config.DEFAULT_LABELS_DIR)
    p.add_argument("--dry-run", action="store_true", help="show the diff, write nothing")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ---- 1) edit the crosswalk JSON (plain json; do NOT import constants yet) ----
    cs = json.loads(args.concept_space.read_text(encoding="utf-8"))
    valid_chex = set(cs["chexpert14_index_order"])
    changed = []
    for c in cs["concepts"]:
        if c["name"] in EDITS:
            new = EDITS[c["name"]]
            if new is not None and new not in valid_chex:
                raise SystemExit(f"[ERROR] '{new}' not a valid CheXpert name")
            old = c.get("chexpert")
            if old != new:
                changed.append((c["name"], old, new))
                c["chexpert"] = new
    print("Crosswalk edits:")
    for name, old, new in changed:
        print(f"  {name:22} {str(old):14} -> {new}")
    if not changed:
        print("  (already applied — nothing to change)")

    if args.dry_run:
        print("[dry-run] no files written"); return 0

    if changed:
        shutil.copy2(args.concept_space, args.concept_space.with_suffix(".json.bak"))
        args.concept_space.write_text(json.dumps(cs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[write] {args.concept_space}  (backup .json.bak)")

    # ---- 2) re-derive region_chexpert.npy from region_concepts.npy with the NEW map ----
    import importlib
    import constants as C
    importlib.reload(C)                       # pick up the just-written JSON
    rc_path = args.labels_dir / "region_concepts.npy"
    rc = np.load(rc_path)                      # [N,29,69] in {1,0,-100}
    n = rc.shape[0]
    out = np.full((n, C.NUM_REGIONS, C.NUM_CHEX), C.UNKNOWN, dtype=np.int8)
    for xi, cis in C.CHEX_FROM_CONCEPTS.items():
        if not cis:
            continue
        sub = rc[:, :, cis]                    # [N,29,k]
        has_pos = (sub == 1).any(axis=-1)
        has_neg = (sub == 0).any(axis=-1)
        col = np.full((n, C.NUM_REGIONS), C.UNKNOWN, dtype=np.int8)
        col[has_neg] = 0
        col[has_pos] = 1                       # positive overrides
        out[:, :, xi] = col

    rchex_path = args.labels_dir / "region_chexpert.npy"
    old = np.load(rchex_path)
    diff = int((old != out).sum())
    print(f"[re-derive] region_chexpert {out.shape}: {diff:,} cells changed vs old "
          f"({100*diff/out.size:.3f}%)")
    shutil.copy2(rchex_path, rchex_path.with_suffix(".npy.bak"))
    np.save(rchex_path, out)
    print(f"[write] {rchex_path}  (backup .npy.bak)")

    # ---- 3) report the new mask for the four affected diseases ----
    for name in ("Pneumonia", "Lung Lesion"):
        xi = C.CHEX_INDEX[name]
        feats = [C.CONCEPT_NAMES[ci] for ci in C.CHEX_FROM_CONCEPTS[xi]]
        print(f"  [{name}] now fed by {len(feats)} concepts: {feats}")
    print("\n[NEXT] retrain B-faithful (mask auto-rebuilds from constants):")
    print("  python phase_3/scripts/4-train.py --mode B --disease-head faithful "
          "--name m3_Bfaithful_v2 --labels-dir data/m3_labels --features-root <feat> --device cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
