"""[STATS] Compute M4 progression-label statistics and render docs/VERA_phase_4_dataset_stats.md.

Reproducible: reads only data/m4_labels/ (progression.npy, manifest.jsonl, m3_pairs.jsonl) and the
present masks from data/m3_labels/. Documents EXACTLY how each number is derived (see the rendered
"How we count" section). Run:

    python phase_4/scripts/dataset_stats.py            # writes docs/VERA_phase_4_dataset_stats.md (+ .json)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
M4 = REPO / "data" / "m4_labels"
M3 = REPO / "data" / "m3_labels"
OUT_MD = REPO / "docs" / "VERA_phase_4_dataset_stats.md"
OUT_JSON = REPO / "docs" / "VERA_phase_4_dataset_stats.json"

SPLITS = ["train", "val", "test", "gold"]
NAMES = ["stable", "improved", "worsened"]     # class index 0/1/2 ; -100 = no cue (masked)
UNKNOWN = -100


def main() -> int:
    prog = np.load(M4 / "progression.npy", mmap_mode="r")           # [N,29,14] int8 in {-100,0,1,2}
    N, R, D = prog.shape
    man = [json.loads(l) for l in open(M4 / "manifest.jsonl", encoding="utf-8")]

    prior: dict[str, str] = {}
    for l in open(M4 / "m3_pairs.jsonl", encoding="utf-8"):
        l = l.strip()
        if l:
            r = json.loads(l)
            prior[r["image_id"]] = r["prior_image_id"]

    pm = np.load(M3 / "present_mask.npy", mmap_mode="r")            # [N3,29] bool-ish
    present: dict[str, np.ndarray] = {}
    for i, l in enumerate(open(M3 / "manifest.jsonl", encoding="utf-8")):
        m = json.loads(l)
        if m.get("ok", True):
            present[m["image_id"]] = np.asarray(pm[i], dtype=bool)

    # ---- funnel (image level) -------------------------------------------------
    funnel = {k: Counter() for k in ["all", "cued", "prior", "present"]}
    for m in man:
        s = str(m.get("split", "")).lower()
        ok = m.get("ok", True)
        if not ok:
            continue
        funnel["all"][s] += 1
        if m.get("n_cued", 0) > 0:
            funnel["cued"][s] += 1
            cid = m["image_id"]; pid = prior.get(cid)
            if pid is not None:
                funnel["prior"][s] += 1
                if cid in present and pid in present:
                    funnel["present"][s] += 1

    # ---- class distribution, both granularities, SAME present-both filter -----
    cell = {s: np.zeros(3, np.int64) for s in SPLITS}      # per (region,disease) present-cell
    stud = {s: np.zeros(3, np.int64) for s in SPLITS}      # collapse 29 regions -> 1 label/(study,disease)
    for i, m in enumerate(man):
        if not m.get("ok", True) or m.get("n_cued", 0) <= 0:
            continue
        s = str(m.get("split", "")).lower()
        if s not in cell:
            continue
        cid = m["image_id"]; pid = prior.get(cid)
        if pid is None or cid not in present or pid not in present:
            continue
        rmask = (present[cid] & present[pid])              # [29] present in BOTH
        p = np.asarray(prog[i])                            # [29,14]
        pm2 = p[rmask]                                     # [n_present,14]
        for k in range(3):
            cell[s][k] += int((pm2 == k).sum())
        # study x disease: a disease counts if any present region has a cue; priority W>I>S
        for d in range(D):
            col = pm2[:, d]
            has = col != UNKNOWN
            if not has.any():
                continue
            cls = 2 if (col == 2).any() else (1 if (col == 1).any() else 0)
            stud[s][cls] += 1

    # raw over full tensor (context only)
    flat = np.asarray(prog).reshape(-1)
    raw = {"masked": int((flat == UNKNOWN).sum()),
           "stable": int((flat == 0).sum()),
           "improved": int((flat == 1).sum()),
           "worsened": int((flat == 2).sum()),
           "total": int(flat.size)}

    stats = {
        "N_images": N, "regions": R, "diseases": D,
        "pairs_total": len(prior),
        "funnel": {k: dict(v) for k, v in funnel.items()},
        "cell": {s: cell[s].tolist() for s in SPLITS},
        "study_disease": {s: stud[s].tolist() for s in SPLITS},
        "raw": raw,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(stats), encoding="utf-8")
    print(f"[DONE] -> {OUT_MD}\n[DONE] -> {OUT_JSON}")
    return 0


def _pct_row(name: str, c: list[int]) -> str:
    tot = sum(c) or 1
    return (f"| {name} | {c[0]:,} ({100*c[0]/tot:.1f}%) | {c[1]:,} ({100*c[1]/tot:.1f}%) | "
            f"{c[2]:,} ({100*c[2]/tot:.1f}%) | {tot:,} |")


def render(st: dict) -> str:
    f = st["funnel"]
    tot = lambda d: sum(d.values())
    L = []
    L.append("# M4 (T-KAN) progression-label statistics\n")
    L.append("Per-`(region, disease)` temporal progression targets built by `phase_4/scripts/1-labels.py` "
             "from Chest ImaGenome `comparison_cues`. Regenerate with `python phase_4/scripts/dataset_stats.py`.\n")

    L.append("## How we count (definitions)\n")
    L.append("- **Source.** Labels come from the **current** study's scene-graph `comparison_cues` "
             "(ImaGenome NLP). A cued phrase's positive findings set the progression of the diseases "
             "they feed; conflicts resolve **worsened > improved > stable**.\n")
    L.append("- **Classes.** `0 stable` (\"no change\" cue) · `1 improved` · `2 worsened`. "
             "A cell with **no cue** is `-100` = **masked** (never trained, never scored).\n")
    L.append("- **Supervision gate (both granularities below use it).** A `(region, disease)` cell counts "
             "only if it has a cue **and** the region is **present in BOTH** the current and prior study "
             "(`REQUIRE_PRIOR_PRESENT`). No-prior / no-cue images carry no M4 signal and are excluded "
             "(they flow to M5's temporal guard, not a data error).\n")
    L.append("- **Unit 1 — cell** `(region × disease)`, 29×14: the exact unit the masked cross-entropy "
             "loss and macro-F1 operate on. **This is the primary distribution.**\n")
    L.append("- **Unit 2 — study×disease**: collapse the 29 regions of a disease within a study to ONE "
             "label (priority worsened>improved>stable). Reported for finding-level interpretability; a "
             "spatially spread finding (effusion/edema) occupies many region-cells, so cell-level "
             "over-weights it relative to this collapsed view.\n")
    L.append("- **`gold`** is the held-out split; note its cues are still NLP-silver (see caveat).\n")

    L.append("\n## A. Image funnel (how many studies survive each gate)\n")
    L.append("| gate | train | val | test | gold | total |")
    L.append("|--|--:|--:|--:|--:|--:|")
    labels = {"all": "ok rows", "cued": "+ has a cue (n_cued>0)",
              "prior": "+ has a prior (pairable)", "present": "+ region present in both (**used by M4**)"}
    for k in ["all", "cued", "prior", "present"]:
        d = f[k]
        L.append(f"| {labels[k]} | {d.get('train',0):,} | {d.get('val',0):,} | {d.get('test',0):,} | "
                 f"{d.get('gold',0):,} | {tot(d):,} |")
    L.append(f"\n- Temporal pairs available in `m3_pairs.jsonl` (curr→prior): **{st['pairs_total']:,}**. "
             f"Pairs actually **used by M4** (last funnel row): **{tot(f['present']):,}**.\n")

    L.append("\n## B. Class distribution — CELL `(region × disease)`, present-masked *(primary)*\n")
    L.append("| split | stable | improved | worsened | total cells |")
    L.append("|--|--:|--:|--:|--:|")
    grand = np.zeros(3, np.int64)
    for s in SPLITS:
        c = st["cell"][s]; grand += np.array(c)
        L.append(_pct_row(s, c))
    L.append(_pct_row("**all**", grand.tolist()))
    ch = int(grand[1] + grand[2]); g = int(grand.sum()) or 1
    L.append(f"\n- **change-only (improved + worsened)** = {ch:,} cells = **{100*ch/g:.1f}%**. "
             "accuracy ≈ predicting \"stable\" everywhere would therefore miss the majority of cells — "
             "read **change-only F1**, not accuracy.\n")

    L.append("\n## C. Class distribution — collapsed to study×disease *(finding-level view)*\n")
    L.append("| split | stable | improved | worsened | total labels |")
    L.append("|--|--:|--:|--:|--:|")
    grand2 = np.zeros(3, np.int64)
    for s in SPLITS:
        c = st["study_disease"][s]; grand2 += np.array(c)
        L.append(_pct_row(s, c))
    L.append(_pct_row("**all**", grand2.tolist()))

    r = st["raw"]
    L.append("\n## D. Context — raw tensor occupancy\n")
    L.append(f"- `progression.npy` shape **[{st['N_images']:,}, {st['regions']}, {st['diseases']}]** "
             f"(= {r['total']:,} cells). **{100*r['masked']/r['total']:.2f}%** are `-100` (no cue → masked). "
             "Only the remaining cued cells — further gated by present-in-both — enter B/C above.\n")

    L.append("\n## Caveat (OPEN decision B2)\n")
    L.append("These labels are **NLP-derived (silver)** from `comparison_cues`: fine to **train** on, but "
             "the improved/stable/worsened **test set must be human-annotated** before any faithful "
             "temporal claim. All three splits here share the same silver source, so their distributions "
             "are near-identical — that consistency is expected, **not** evidence of a clean eval. Numbers "
             "reported on this silver test are provisional pending a human temporal set (e.g. MS-CXR-T).\n")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
