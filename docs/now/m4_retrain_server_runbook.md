# M4 Detector-Box Retraining Runbook

> **Dependency:** M2 detector refresh and detector-box M3 retraining must be
> complete first. The M4 cache and every M4 checkpoint must be tied to the new
> M3 checkpoint hash and to `data/m3_labels/detector_provenance.json`.
>
> **Implementation update (2026-08-03):** the detector-mask blocker is repaired:
> detector runs use `present_mask_det.npy`, including MS-CXR-T, while the GT
> oracle uses `present_mask.npy`. The training/grid/test campaign may run. Final
> report-facing temporal calibration remains deferred until the readout policy
> in `m4_temporal_calibration_and_readout_policy.md` is frozen.

## 1. Dependency and scope

M4 may start as soon as the M3 v2 **main faithful checkpoint** has passed the
acceptance checks in `m3_retrain_server_runbook.md`; the remaining M3 ablation
rows do not block the first M4 coefficient grid. The frozen dependency is:

```text
data/run/m3v2_vera_graph_lse_det/best.pt
```

This campaign replaces every temporal number retained in the main paper and
appendix. It does not calibrate Stage-4 report thresholds, generate final
present/absent text, or modify the report renderer.

## 2. Upload manifest

Preserve repository-relative paths.

### Code

```text
phase_3/src/                         # required to reconstruct frozen M3
phase_3/scripts/8-precompute_regions.py
phase_4/src/                         # complete directory
phase_4/scripts/2-train.py
phase_4/scripts/3-eval.py
phase_4/scripts/4-infer.py
phase_4/scripts/5-mscxrt_audit.py
phase_4/scripts/7-temporal_consistency.py
phase_4/run_paper_m4_v2.sh
```

Uploading complete `phase_3/` and `phase_4/` directories is recommended because
the train/eval scripts use phase-local flat imports.

### Required weights and data

```text
data/run/m3v2_vera_graph_lse_det/best.pt
data/run/m3v2_vera_graph_lse_gt/best.pt       # required only for the GT oracle
data/features/frozen/**/<image_id>.npy or .pt
data/m3_labels/manifest.jsonl
data/m3_labels/boxes_det.npy
data/m3_labels/present_mask_det.npy
data/m3_labels/boxes.npy
data/m3_labels/present_mask.npy
data/m4_labels/manifest.jsonl
data/m4_labels/progression.npy
data/m4_labels/m3_pairs.jsonl
data/MS_CXR_T_temporal_image_classification_v1.0.0.csv
```

The remaining M3 arrays may be uploaded with the directory and are harmless,
but the temporal dataset directly requires the files listed above. Raw chest
X-rays, scene graphs, detector weights, and old M4 checkpoints are unnecessary.

M4 must use the same detector-box protocol as the final M3 run: the fresh
`boxes_det.npy` and `present_mask_det.npy` generated from the refreshed YOLO
checkpoint, and the M3 checkpoint trained with those arrays. The YOLO internal
inference size may be `1024`, but all downstream boxes remain in the shared
448-pixel coordinate frame.

### Generated cache and run outputs

```text
data/m4_region_cache_m3v2_detector/
data/m4_region_cache_m3v2_gt_oracle/
data/run/m4v2_*/
logs/m4_paper_v2/
artifacts/diagnostics/m4_paper_v2/
```

Each cache contains a `.m3_source.json` marker with the M3 checkpoint SHA-256
and box source. The launcher aborts on a provenance mismatch instead of reusing
stale arrays.

## 3. Selection policy and run order

### Step A: detector validation grid

Train nine 40-epoch `TempFuse + Disease-Delta` models:

```text
KL = {0.025, 0.050, 0.075}
Distance = {0.350, 0.500, 0.650}
```

Here `KL` is the temporal flip-consistency term and `Distance` is the ordinal
distance penalty. All nine use seed 42, detector boxes, the same frozen M3,
time-flip augmentation, and validation change-F1 checkpoint selection. Do not
run MS-CXR-T during this search. The initial matrix uses one controlled seed per
row; additional seeds are reserved for the selected final model if needed.

The launcher compares `val_change_f1` in the nine `best.pt` files and writes:

```text
artifacts/diagnostics/m4_paper_v2/selected_coefficients.env
```

Review this file before continuing. It records `KL_WEIGHT`, `DIST_WEIGHT`,
`MAIN_RUN`, and the winning validation change-F1.

### Step B: paper ablations after coefficients are frozen

Only the following additional models are trained.

Architecture table, all under the selected KL and distance coefficients:

| Paper label | Configuration |
|---|---|
| RegionDiff | frozen M3 region features/logits with explicit difference |
| TempFuse | patch-level temporal fusion without M3 disease logits |
| VERA (TempFuse + Disease-Delta) | selected grid run; no duplicate training |

Regularization appendix, all using detector boxes and TempFuse + Disease-Delta:

| Paper label | Added terms |
|---|---|
| TempFuse + Disease-Delta | none |
| KL only | selected KL |
| Distance only | selected distance |
| KL + Distance | selected grid run |
| Smooth005 | label smoothing 0.05 |
| Smooth005 + KL | smoothing 0.05 + selected KL |
| Smooth005 + Distance | smoothing 0.05 + selected distance |
| Smooth005 + KL + Distance | all three selected terms |

One additional GT-box oracle uses the complete selected VERA configuration and
the matched M3 faithful checkpoint trained with GT boxes. It must not reuse the
detector-trained M3 checkpoint with the box source switched only at inference.
Removed variants such as Dist010, Smooth005+KL005+Dist010, two-stage heads,
two-block fusion, same-view filtering, FTCB, adapters, and KAN are not run.

This is 19 unique M4 trainings: 9 grid runs, 7 additional loss rows, 2
additional architecture rows, and 1 GT oracle.

### Step C: final audits

After the detector main model is fixed:

1. internal test evaluation and confusion diagnostics;
2. full MS-CXR-T external audit;
3. full temporal swap test;
4. full identical-image stable/null test;
5. raw detector-box test inference including stable predictions.

Stage-4 calibration is deliberately deferred until these artifacts are signed
off.

## 4. Commands

Install/check the environment from the repository root:

```bash
source .venv/bin/activate
pip install torch numpy scikit-learn matplotlib pandas
python -m py_compile phase_3/src/*.py phase_3/scripts/8-precompute_regions.py \
  phase_4/src/*.py phase_4/scripts/*.py
```

Verify the M3 checkpoint contract and all required temporal inputs without
building a cache:

```bash
bash phase_4/run_paper_m4_v2.sh --profile h100mini --scope preflight
```

Run the coefficient search first:

```bash
bash phase_4/run_paper_m4_v2.sh --profile h100mini --scope grid
cat artifacts/diagnostics/m4_paper_v2/selected_coefficients.env
```

This detector-only grid may begin immediately after
`m3v2_vera_graph_lse_det/best.pt` passes acceptance. It does not require the
other M3 ablations or the M3 GT-box checkpoint. The later `--scope paper` step
does require `m3v2_vera_graph_lse_gt/best.pt` because it includes the matched
GT-box oracle row.

After reviewing the selected coefficients, run the retained paper variants:

```bash
bash phase_4/run_paper_m4_v2.sh --profile h100mini --scope paper
```

Then run the final external and consistency audits:

```bash
bash phase_4/run_paper_m4_v2.sh --profile h100mini --scope final
```

`--scope all` executes the same three steps sequentially, but the staged form is
preferred because it provides an explicit coefficient-review checkpoint.

To override the automatic choice after inspecting the grid:

```bash
KL_WEIGHT=0.050 DIST_WEIGHT=0.500 \
MAIN_RUN=m4v2_grid_kl0050_dist0500_det \
bash phase_4/run_paper_m4_v2.sh --profile h100mini --scope paper
```

Server tuning and remote checkpoint sync:

```bash
W=8 EVAL_W=8 BATCH=64 \
SYNC_REMOTE='remote:VERA/m4v2' \
bash phase_4/run_paper_m4_v2.sh --profile h100mini --scope grid
```

Every training writes `last.pt` each epoch and resumes automatically. A run is
skipped only after `last.pt` records all requested epochs. The M3 region-cache
builder is also resumable at image-file granularity.

## 5. Paper mapping

Use internal test values for the main temporal and architecture/regularization
tables after the validation choice is frozen. Use MS-CXR-T only for the selected
detector main row in the external SOTA comparison. The GT run appears only in
the detector-versus-GT sensitivity table and must be labeled oracle.

Required metrics/artifacts are:

- internal accuracy, progression macro-F1, change-F1, per-class F1, and
  confusion matrices;
- per-finding and average MS-CXR-T accuracy/balanced accuracy/F1;
- swap consistency, swap change consistency, swapped-target change accuracy,
  symmetric KL, identical-image stable rate, and identical-image change rate;
- training logs/curves and the selected-coefficient artifact;
- raw detector main predictions for later Stage-4 calibration.

Never copy the old GT-box values into detector rows. Never select coefficients,
architectures, or checkpoints using MS-CXR-T or internal test results.

## 6. Stop conditions

Stop and repair the setup if any condition occurs:

- the M3 checkpoint contract is not mode B + faithful + detach + derived No
  Finding + no global bypass + LSE;
- a detector run loads `present_mask.npy` instead of `present_mask_det.npy` for
  either the current or prior region mask;
- cache provenance does not match the M3 SHA-256 or box source;
- a detector checkpoint records `box_source=gt` or vice versa;
- any grid run is missing before coefficient selection;
- an external audit is attempted before `selected_coefficients.env` exists;
- Stage-4 threshold fitting appears in the M4 training logs.
