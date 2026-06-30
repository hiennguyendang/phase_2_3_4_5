"""Dataset statistics for the paper. Writes m3_dataset_stats.md + .json.

Sections (each runs only if its inputs exist):
  A splits x datasets, #images/patients/studies
  B temporal pairing: #pairs, same-view, days-apart (median/IQR), studies/patient,
    patients with exactly 1 study, #currents with a prior
  C pairs whose BOTH images are concept-labeled (MIMIC+scene)
  D image-level CheXpert counts (pos/neg/unknown) overall + per split
  E comparison cues (improved/no change/worsened) from the findings scan
  F per-region concept stats (top concepts, mean/img)
  G report length (words): median + quartiles
  H concept->disease relevance: per-disease XGBoost AUC + top concepts

    python phase_3/dataset_stats.py            # uses repo defaults
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_3/src

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import constants as C

REPO = C.REPO_ROOT


def pct(a, qs=(25, 50, 75)):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def read_jsonl(p):
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-glob", default=str(REPO / "data" / "*_metadata_final.jsonl"))
    ap.add_argument("--cxr-meta", type=Path, default=REPO / "data" / "mimic-cxr-2.0.0-metadata.csv")
    ap.add_argument("--pairs", type=Path, default=REPO / "data" / "m3_pairs.jsonl")
    ap.add_argument("--labels-dir", type=Path, default=REPO / "data" / "m3_labels")
    ap.add_argument("--findings-json", type=Path,
                    default=Path(r"C:\Users\Dang Hien\Downloads\phase2_findings\findings_frequency.json"))
    ap.add_argument("--out", type=Path, default=REPO / "data" / "m3_dataset_stats")
    ap.add_argument("--xgb", action="store_true", default=True)
    args = ap.parse_args()

    S: dict = {}
    md = ["# M3 dataset statistics", ""]

    # ---- A. splits x datasets + report length (G) ----
    by_ds_split = defaultdict(Counter)
    patients, studies = defaultdict(set), defaultdict(set)
    report_words = defaultdict(list)
    n_empty_report = Counter()
    chex_counts = {sp: np.zeros((C.NUM_CHEX, 3), dtype=np.int64) for sp in ("train", "val", "test", "gold")}
    chex_total = np.zeros((C.NUM_CHEX, 3), dtype=np.int64)  # cols: pos, neg, unknown (mimic)
    for path in glob.glob(args.metadata_glob):
        for r in read_jsonl(path):
            ds = str(r.get("dataset", "?")).lower()
            sp = str(r.get("split", "?")).lower()
            by_ds_split[ds][sp] += 1
            patients[ds].add(r.get("patient_id"))
            studies[ds].add((r.get("patient_id"), r.get("study_id")))
            rep = str(r.get("report") or "")
            wc = len(rep.split())
            report_words[ds].append(wc)
            if wc == 0:
                n_empty_report[ds] += 1
            if ds == "mimic":
                lab = r.get("labels")
                if isinstance(lab, list) and len(lab) == C.NUM_CHEX:
                    arr = np.asarray(lab)
                    for c in range(C.NUM_CHEX):
                        col = 0 if arr[c] == 1 else (1 if arr[c] == 0 else 2)
                        chex_total[c, col] += 1
                        if sp in chex_counts:
                            chex_counts[sp][c, col] += 1
    S["A_splits"] = {ds: dict(cnt) for ds, cnt in by_ds_split.items()}
    S["A_images"] = {ds: int(sum(cnt.values())) for ds, cnt in by_ds_split.items()}
    S["A_patients"] = {ds: len(v - {None}) for ds, v in patients.items()}
    S["A_studies"] = {ds: len(v) for ds, v in studies.items()}

    md.append("## A. Images / patients / studies, by dataset & split")
    md.append("| dataset | images | patients | studies | train | val | test | gold |")
    md.append("|--|--|--|--|--|--|--|--|")
    for ds in by_ds_split:
        c = by_ds_split[ds]
        md.append(f"| {ds} | {S['A_images'][ds]:,} | {S['A_patients'][ds]:,} | {S['A_studies'][ds]:,} | "
                  f"{c.get('train',0):,} | {c.get('val',0):,} | {c.get('test',0):,} | {c.get('gold',0):,} |")

    # ---- G. report length ----
    S["G_report_words"] = {ds: {**pct(w), "mean": float(np.mean(w)) if w else 0,
                                "empty": int(n_empty_report[ds]), "n": len(w)}
                           for ds, w in report_words.items()}
    md.append("\n## G. Report length (words)")
    md.append("| dataset | n | median | p25 | p75 | mean | empty |")
    md.append("|--|--|--|--|--|--|--|")
    for ds, g in S["G_report_words"].items():
        md.append(f"| {ds} | {g['n']:,} | {g.get('p50',0):.0f} | {g.get('p25',0):.0f} | "
                  f"{g.get('p75',0):.0f} | {g['mean']:.0f} | {g['empty']:,} |")

    # ---- D. image-level CheXpert ----
    md.append("\n## D. MIMIC image-level CheXpert (pos / neg / unknown)")
    md.append("| # | label | pos | neg | unknown |")
    md.append("|--|--|--|--|--|")
    S["D_chexpert"] = {}
    for c in range(C.NUM_CHEX):
        p, n, u = (int(x) for x in chex_total[c])
        S["D_chexpert"][C.CHEX_NAMES[c]] = {"pos": p, "neg": n, "unknown": u}
        md.append(f"| {c} | {C.CHEX_NAMES[c]} | {p:,} | {n:,} | {u:,} |")

    # ---- B. studies/patient + pairing ----
    if args.cxr_meta.exists():
        studies_per = defaultdict(set)
        with open(args.cxr_meta, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("subject_id") and row.get("study_id"):
                    studies_per[row["subject_id"]].add(row["study_id"])
        counts = np.array([len(v) for v in studies_per.values()])
        S["B_studies_per_patient"] = {**pct(counts, (25, 50, 75, 90, 99)),
                                      "mean": float(counts.mean()), "max": int(counts.max()),
                                      "patients": int(len(counts)),
                                      "exactly_1_study": int((counts == 1).sum()),
                                      "ge_2_studies": int((counts >= 2).sum())}
        md.append("\n## B. Studies per patient (MIMIC-CXR)")
        b = S["B_studies_per_patient"]
        md.append(f"- patients: **{b['patients']:,}**  | exactly 1 study: **{b['exactly_1_study']:,}** "
                  f"({100*b['exactly_1_study']/b['patients']:.1f}%)  | ≥2 studies: **{b['ge_2_studies']:,}**")
        md.append(f"- studies/patient: median {b.get('p50',0):.0f}, p75 {b.get('p75',0):.0f}, "
                  f"p99 {b.get('p99',0):.0f}, max {b['max']}, mean {b['mean']:.2f}")

    labeled_ids = set()
    if (args.labels_dir / "manifest.jsonl").exists():
        labeled_ids = {m["image_id"] for m in read_jsonl(args.labels_dir / "manifest.jsonl")
                       if m.get("ok", True)}

    if args.pairs.exists():
        n_pairs = same_view = both_labeled = 0
        days = []
        cur_with_prior = set()
        for pr in read_jsonl(args.pairs):
            n_pairs += 1
            same_view += int(bool(pr.get("same_view")))
            if pr.get("days_apart") is not None:
                days.append(pr["days_apart"])
            cur_with_prior.add(pr["image_id"])
            if labeled_ids and pr["image_id"] in labeled_ids and pr["prior_image_id"] in labeled_ids:
                both_labeled += 1
        S["BC_pairs"] = {"pairs": n_pairs, "same_view": same_view,
                         "currents_with_prior": len(cur_with_prior),
                         "both_concept_labeled": both_labeled, "days_apart": pct(days)}
        md.append("\n## C. Temporal pairs (prior ↔ current)")
        md.append(f"- total pairs: **{n_pairs:,}**  | same ViewPosition: **{same_view:,}** "
                  f"({100*same_view/max(1,n_pairs):.1f}%)  | distinct current images with a prior: "
                  f"**{len(cur_with_prior):,}**")
        if labeled_ids:
            md.append(f"- pairs where BOTH images are concept-labeled (MIMIC+scene): **{both_labeled:,}**")
        dd = S["BC_pairs"]["days_apart"]
        if dd:
            md.append(f"- days apart: median {dd.get('p50',0):.0f}, p25 {dd.get('p25',0):.0f}, p75 {dd.get('p75',0):.0f}")

    # ---- E. comparison cues ----
    if args.findings_json.exists():
        fj = json.loads(args.findings_json.read_text(encoding="utf-8"))
        comp = {x[0]: x[1] for x in fj.get("comparison", [])}
        if comp:
            S["E_comparison"] = comp
            md.append("\n## E. Comparison cues (M4 progression supervision)")
            md.append("| cue | region-instances |")
            md.append("|--|--|")
            for k, v in sorted(comp.items(), key=lambda x: -x[1]):
                md.append(f"| {k} | {v:,} |")

    # ---- F + H. concept stats + concept->disease relevance ----
    rc_path = args.labels_dir / "region_concepts.npy"
    if rc_path.exists():
        rc = np.load(rc_path, mmap_mode="r")
        ic = np.load(args.labels_dir / "image_chexpert.npy")
        N = rc.shape[0]
        # image-level concept presence: any region == 1
        concept_present = np.zeros((N, C.NUM_CONCEPTS), dtype=np.int8)
        CH = 20000
        for s in range(0, N, CH):
            block = np.asarray(rc[s:s + CH])
            concept_present[s:s + CH] = (block == 1).any(axis=1).astype(np.int8)
        per_concept = concept_present.sum(0)
        S["F_concept_image_positives"] = {C.CONCEPT_NAMES[i]: int(per_concept[i])
                                          for i in np.argsort(-per_concept)}
        S["F_mean_pos_concepts_per_image"] = float(concept_present.sum(1).mean())
        md.append(f"\n## F. Concepts (image-level): mean **{S['F_mean_pos_concepts_per_image']:.1f}** "
                  f"positive concepts/image. Top 15:")
        md.append("| concept | images+ |")
        md.append("|--|--|")
        for i in np.argsort(-per_concept)[:15]:
            md.append(f"| {C.CONCEPT_NAMES[i]} | {int(per_concept[i]):,} |")

        if args.xgb:
            try:
                import xgboost as xgb
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import roc_auc_score
                md.append("\n## H. Concept → disease relevance (XGBoost, image-level)")
                md.append("How well the 69 concepts predict each CheXpert disease, + top driving concepts.")
                md.append("| disease | n(pos/neg) | AUC | top concepts (gain) |")
                md.append("|--|--|--|--|")
                X = concept_present.astype(np.float32)
                S["H_xgb"] = {}
                for c in range(C.NUM_CHEX):
                    try:
                        y = ic[:, c]
                        keep = (y == 0) | (y == 1)
                        yc_all = y[keep].astype(int)
                        if (yc_all == 1).sum() < 20 or (yc_all == 0).sum() < 20:
                            md.append(f"| {C.CHEX_NAMES[c]} | "
                                      f"{int((yc_all==1).sum()):,}/{int((yc_all==0).sum()):,} | "
                                      f"— (single-class) | — |")
                            continue
                        Xc, yc = X[keep], yc_all
                        Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=0, stratify=yc)
                        clf = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.2,
                                                subsample=0.8, eval_metric="logloss", n_jobs=4,
                                                verbosity=0)
                        clf.fit(Xtr, ytr)
                        auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
                        top = [C.CONCEPT_NAMES[i] for i in np.argsort(-clf.feature_importances_)[:4]]
                        S["H_xgb"][C.CHEX_NAMES[c]] = {"auc": auc, "n_pos": int((yc == 1).sum()),
                                                       "n_neg": int((yc == 0).sum()), "top": top}
                        md.append(f"| {C.CHEX_NAMES[c]} | {int((yc==1).sum()):,}/{int((yc==0).sum()):,} | "
                                  f"{auc:.3f} | {', '.join(top)} |")
                    except Exception as ee:  # noqa: BLE001 - one disease failing shouldn't kill the rest
                        md.append(f"| {C.CHEX_NAMES[c]} | — | err | {ee} |")
            except Exception as e:  # noqa: BLE001
                md.append(f"\n_(XGBoost section skipped: {e})_")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.with_suffix(".md")).write_text("\n".join(md), encoding="utf-8")
    (args.out.with_suffix(".json")).write_text(json.dumps(S, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] -> {args.out.with_suffix('.md')}  +  .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
