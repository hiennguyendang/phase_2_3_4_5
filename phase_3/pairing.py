"""[PREP — run local, no GPU] Build prior<->current image pairs.

For each image, find the same patient's most-recent EARLIER study and pick a
comparable image from it (prefer same ViewPosition, else a frontal AP/PA). This is
the temporal link M4 (T-KAN) needs; M3 stays single-image but we prepare the pairs
now so nothing blocks Kaggle.

Reads mimic-cxr-2.0.0-metadata.csv (dicom_id, subject_id, study_id, StudyDate,
StudyTime, ViewPosition). Writes m3_pairs.jsonl:
  {image_id, dicom, prior_image_id, prior_dicom, days_apart, same_view}

    python phase_3/pairing.py --cxr-meta data/mimic-cxr-2.0.0-metadata.csv \
                              --out data/m3_pairs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import constants as C

FRONTAL = {"AP", "PA"}


def image_id(subject: str, study: str, dicom: str) -> str:
    return f"MIMIC_p{subject}_s{study}_{dicom}"


def to_days(date_str: str) -> int | None:
    """StudyDate 'YYYYMMDD' -> ordinal-ish day count (deidentified dates still sort/diff)."""
    s = "".join(ch for ch in str(date_str) if ch.isdigit())
    if len(s) != 8:
        return None
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    return y * 365 + m * 31 + d  # approximate; fine for "days apart" magnitude


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prior<->current pairs")
    p.add_argument("--cxr-meta", type=Path, default=C.REPO_ROOT / "data" / "mimic-cxr-2.0.0-metadata.csv")
    p.add_argument("--out", type=Path, default=C.REPO_ROOT / "data" / "m3_pairs.jsonl")
    p.add_argument("--frontal-only", action="store_true",
                   help="only emit pairs where the CURRENT image is frontal (AP/PA)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cxr_meta.exists():
        raise SystemExit(f"[ERROR] cxr-meta not found: {args.cxr_meta}")

    # subject -> list of images
    by_subject: dict[str, list[dict]] = defaultdict(list)
    with open(args.cxr_meta, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            subj, study, dicom = row.get("subject_id"), row.get("study_id"), row.get("dicom_id")
            if not (subj and study and dicom):
                continue
            by_subject[subj].append({
                "dicom": dicom, "study": study, "view": (row.get("ViewPosition") or "").strip().upper(),
                "day": to_days(row.get("StudyDate")), "time": row.get("StudyTime") or "0",
            })
    print(f"patients: {len(by_subject):,}")

    def sort_key(im):
        try:
            t = float(im["time"])
        except (TypeError, ValueError):
            t = 0.0
        return (im["day"] if im["day"] is not None else 0, t)

    n_pairs = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for subj, imgs in by_subject.items():
            imgs.sort(key=sort_key)
            # group consecutive by study, keep study order
            studies: list[tuple[str, list[dict]]] = []
            for im in imgs:
                if studies and studies[-1][0] == im["study"]:
                    studies[-1][1].append(im)
                else:
                    studies.append((im["study"], [im]))
            # for each study (from the 2nd on), prior = previous study
            for k in range(1, len(studies)):
                prior_imgs = studies[k - 1][1]
                for cur in studies[k][1]:
                    if args.frontal_only and cur["view"] not in FRONTAL:
                        continue
                    # pick prior: same view > frontal > anything
                    same = [p for p in prior_imgs if p["view"] == cur["view"] and cur["view"]]
                    front = [p for p in prior_imgs if p["view"] in FRONTAL]
                    prior = (same or front or prior_imgs)[0]
                    da = None
                    if cur["day"] is not None and prior["day"] is not None:
                        da = cur["day"] - prior["day"]
                    out.write(json.dumps({
                        "image_id": image_id(subj, cur["study"], cur["dicom"]),
                        "dicom": cur["dicom"],
                        "prior_image_id": image_id(subj, prior["study"], prior["dicom"]),
                        "prior_dicom": prior["dicom"],
                        "days_apart": da,
                        "same_view": bool(same),
                    }) + "\n")
                    n_pairs += 1

    print(f"[DONE] {n_pairs:,} prior<->current pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
