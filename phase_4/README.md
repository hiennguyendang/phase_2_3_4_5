# phase_4 — Module 4: T-KAN (per-region temporal progression)

Per `(region, disease)` progression **improved / stable / worsened** → tensor `29×14×3`, supervised
by ImaGenome `comparison_cues`. Implements `docs/VERA_phase_3_4_5_spec.md` §4.

```
frozen M3 (precomputed)          per region, per disease
  curr ─┐                        [feat_curr ; feat_prior ; curr−prior]  (3×512)
        ├─ region cache ──► T-head ┤                                       ──► 29×14×3
  prior ┘                        [logit_curr ; logit_prior]             (2×14)
```

## Staged on a FROZEN M3 (spec 4.1)
M3 is frozen after Phase-3, so its region outputs are deterministic. `phase_3/precompute_regions.py`
caches them once (`<image_id>.npy` `[29, feat_dim+14]` = region features ‖ disease logits). M4 then
**only consumes the cache** — it never imports phase_3 or runs the backbone. The Siamese "shared
frozen branch" is a cache lookup. *Joint fine-tuning (letting M4 gradients reach the pool) is an
ablation only — it risks M3 faithfulness (α/softmax_r used by M5 would stop reflecting M3).*

Train M4 on the **mode-A** (shipping, where-faithful) M3 logits. The contract is mode-agnostic
(A/B/C all emit 14 soft logits, same shape), so B/C can be fed into the *same* M4 as an ablation —
no separate M4 per direction. Pass **soft logits**, not hard labels (magnitude signals the change).

## Layout (same split as phase_2 / phase_3: `src/` library + numbered `scripts/`)
```
phase_4/
  src/        config.py constants.py dataset.py heads.py losses.py model.py eval.py  (+ bundled JSONs)
  scripts/    1-labels.py  2-train.py  3-eval.py  4-infer.py     (numbered = run order)
  run_experiments.sh   notebooks/phase4_kaggle.ipynb   README.md
```
Numbered `scripts/` are the executable entry points; each prepends `src/` to `sys.path` then imports
the library modules. `src/eval.py` is a library (imported by `2-train.py` + `4-infer.py`) **and** an
entry point, so `3-eval.py` is a thin wrapper (same trick as phase_3's `5-eval.py`).

| File | Role | Needs GPU |
|------|------|-----------|
| `src/constants.py` | 29 regions + 14 CheXpert + concept→disease map + 3 progression classes | — |
| `src/config.py` | paths + hyperparams + ablation knobs (`INPUT_MODE`, `LOSS_TYPE`, `HEAD_TYPE`) | — |
| **Prep (local, no GPU):** | | |
| `scripts/1-labels.py` | scene-graph `comparison_cues` → `progression.npy [N,29,14]` (0/1/2/-100) + manifest | no |
| **Bridge (GPU, once) — lives in phase_3 (it freezes M3):** | | |
| `../phase_3/scripts/8-precompute_regions.py` | freeze M3 → cache region features + logits for every image | yes |
| **Model:** | | |
| `src/dataset.py` | pair curr↔prior, serve cached tensors + progression target | — |
| `src/heads.py` / `src/model.py` | T-head (`mlp` / `kan` / `linear`) + input-mode composition → `29×14×3` | — |
| `src/losses.py` | masked, class-weighted CE / focal (3 classes; "stable" dominates) | — |
| `scripts/2-train.py` / `src/eval.py` / `scripts/4-infer.py` | train (Drive-resumable) / **macro-F1** + change-only F1 / change-ledger JSON for M5 | yes |

## Run order
```bash
# 1) prep (local, no GPU) — progression targets; pairs come from phase_3/scripts/2-pairing.py
python phase_4/scripts/1-labels.py --scene-root <chest-imagenome> --out-dir data/m4_labels

# 2) bridge (GPU, once) — phase_3 script; needs the trained M3 ckpt + the BioViL-T feature cache
python phase_3/scripts/8-precompute_regions.py --ckpt data/m3_B_faithful/best.pt \
    --labels-dir data/m3_labels --features-root <feat> --out-dir data/m3_region_cache

# 3) train + eval + infer (or just: bash phase_4/run_experiments.sh for the full ablation grid)
python phase_4/scripts/2-train.py --region-cache data/m3_region_cache --m3-labels-dir data/m3_labels \
    --m4-labels-dir data/m4_labels --pairs data/m4_labels/m3_pairs.jsonl --device cuda --name m4_mlp
python phase_4/scripts/3-eval.py  --ckpt data/run/m4_mlp/best.pt --split test
python phase_4/scripts/4-infer.py --ckpt data/run/m4_mlp/best.pt --split test --out m4_pred.jsonl
```

## Ablation grid (`run_experiments.sh`)
Bridge runs **once**; all runs share the same B-faithful region cache. Axes: **head** (`mlp`/`kan`/`linear`),
**input signal** (`full`/`concat`/`diff`/`logits`/`feat`), **imbalance** (class-weight, focal, time-flip),
**supervision** (`--no-require-prior`). Read **change-only F1** as the headline; accuracy≈"stable" is a red flag.

## Notes
- ⚠️ **EVAL labels are a TRAIN-ONLY source (OPEN decision B2 — biggest risk; see
  `docs/VERA_methodology_concerns.md`).** `comparison_cues` are NLP-derived (weak), fine to **train**
  on but **must NOT be used to evaluate** M4. The improved/stable/worsened **test set must be
  human-annotated** (e.g. MS-CXR-T). Sourcing it is **TODO and high-priority** — the whole
  temporal-faithful claim rests on a clean human temporal eval set.
- **Labels** come from the **current** scene graph's `comparison_cues` (the NLP already encoded the
  comparison to prior). A cued phrase's positive findings set the progression of the diseases they
  feed; conflicts resolve worsened > improved > stable. Cells with no cue stay `-100` (masked).
- **Supervision contract:** a `(region, disease)` cell is trained only where the region is present in
  **both** current and prior (`REQUIRE_PRIOR_PRESENT`) and the cell has a cue.
- **No-prior images** (first study) carry no M4 signal — they are simply absent from the pairs and
  flow to M5's tier-4 temporal guard (language turned off), **not** a data error to filter.
- **Metric: macro-F1 over 3 classes + per-class + change-only F1.** accuracy ≈ "stable" is a red flag.
- **Prior features must use the same 448 center-crop preprocessing as current** — otherwise
  `curr − prior` is meaningless (the Siamese fails silently). This is on the M1 collaborator.
- **KAN swap:** `heads.py` is the only file to touch (same `make_head` interface).

## Current audit

Latest parsed `RUN/` + `LOGS/` summary lives in `docs/VERA_experiment_audit_roadmap.md`. Current
read: M4 silver-test scores are moderate, with `improved` as the persistent weak class. Treat these
as development numbers only until a human temporal eval set exists. Immediate work: confusion
matrix, per-disease/per-region breakdowns, and noise-robust target/loss experiments.

## 2026-07-08 update: MS-CXR-T audit and v4 hybrid

New entry points:

```bash
# External temporal audit on the local MS-CXR-T CSV.
python phase_4/scripts/5-mscxrt_audit.py \
  --ckpt data/run/m4v3_tf/best.pt \
  --csv data/MS_CXR_T_temporal_image_classification_v1.0.0.csv \
  --features-root data/features/frozen \
  --region-cache data/m4_region_cache \
  --m3-labels-dir data/m3_labels \
  --split all \
  --device cuda

# Server sweep: v3 controls plus v4 TempFuse+M3-delta variants.
bash phase_4/run_m4_retrain_matrix.sh --profile h100mini --eval-split test

# Optional adapter development on MS-CXR-T subject-hash train/val splits.
bash phase_4/run_m4_retrain_matrix.sh --profile h100mini --eval-split test --run-adapters
```

`--tempfuse-input-mode feat_logits` is the v4 hybrid path. It keeps TempFuse patch-level temporal
fusion, then feeds the head `[TempFuse region feature ; M3 current logits ; M3 prior logits ;
M3 delta logits]`. Default `feat` preserves old v3 checkpoints.

`--curriculum-same-view-epochs N` trains the first N epochs on same-view pairs, then switches back
to all pairs. This is different from `--same-view`, which filters the entire train/eval set.

`--flip-consistency-weight W` adds a symmetric KL regularizer: the prediction for `(current, prior)`
must match the prediction for `(prior, current)` after swapping improved and worsened, while stable
stays stable. This doubles the train-time forward cost for affected batches, so it belongs on the
server sweep unless using a tiny local smoke run.

Metric clarification: M4 does not choose the disease identity from scratch. The disease axis is the
fixed 14-label M3/CheXpert space; M4 predicts the temporal class stable/improved/worsened for each
valid `(region, disease)` cell. `change-only F1` averages improved+worsened only, so stable-heavy
behavior cannot hide weak temporal-change performance.

Local MS-CXR-T audit ran for `RUN/m4v3_tf/best.pt` using `data/frozen`: 964/1,045 usable pairs.
Best aggregation (`lse`) scored macro-F1 0.5695 and change-only F1 0.6463. This is an image-level
external audit via explicit region-to-image aggregation, not a region-level localization score.

## 2026-07-09 full rerun plan

Use repo-root `phase_4.sh` for the complete Phase-4 rerun/audit. It expects Phase 3 to have already
finished the faithful ship checkpoint and faithfulness audit.

```bash
# Recommended order.
bash phase_3.sh --profile h100mini --tag xwalk_v2
bash phase_4.sh --profile h100mini --tag xwalk_v2

# Audit existing tagged runs only.
bash phase_4.sh --profile h100mini --tag xwalk_v2 --skip-train

# Include optional MS-CXR-T adapter fine-tunes.
bash phase_4.sh --profile h100mini --tag xwalk_v2 --run-adapters

# Local smoke.
bash phase_4.sh --profile local4060 --tag smoke --epochs 2 --audit-splits gold
```

`phase_4.sh` runs dataset stats, frozen-M3 region-cache generation, the full v3/v4 retrain matrix,
silver diagnostics on `val test gold`, MS-CXR-T external audit, SVG/CSV plots, M4 inference JSONL
for Phase 5, and optional M5 assembly when `data/m3_pred.jsonl` exists.

Key outputs:

```text
data/run/m4v4_tf_m3delta_kl005_xwalk_v2/best.pt      # one likely candidate
data/m4_pred.test.xwalk_v2.jsonl
data/m4_pred.jsonl
data/m4_region_cache_xwalk_v2/
artifacts/phase4_xwalk_v2/
artifacts/diagnostics/m4*.xwalk_v2*.json
```

Server upload minimum for Phase 4:

```text
data/features/frozen/
data/m3_labels/
data/m4_labels/
data/m4_labels/m3_pairs.jsonl
data/run/m3_B_faithful_xwalk_v2/best.pt
data/MS_CXR_T_temporal_image_classification_v1.0.0.csv   # recommended external audit
```

If `data/m4_region_cache_xwalk_v2/` is not uploaded, `phase_4.sh` will generate it from the M3
checkpoint. Do not start the final Phase-4 run until the Phase-3 `m3_B_faithful_xwalk_v2` checkpoint
has passed faithfulness; M4 is explicitly staged on a frozen faithful M3.
