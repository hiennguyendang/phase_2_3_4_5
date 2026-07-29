# VERA Kaggle Handoff (No Server)

This document is the handoff for a new Codex session that must continue the
paper experiments on Kaggle. It is operational: read this first, then inspect
the source files listed below before writing or running notebooks.

## 1. Read In This Order

1. AGENTS.md
2. docs/now/paper.md (the dated decision log is the project source of truth)
3. docs/now/m2_yolo_server_inference_runbook.md
4. docs/now/m3_retrain_server_runbook.md
5. docs/now/m4_retrain_server_runbook.md
6. docs/now/full_pipeline_weights_and_report_demo.md
7. docs/now/confidence_calibration_policy.md
8. docs/now/phase5_report_sample.md and docs/now/VERA_phase5_prep.md
9. phase_3/README.md, phase_4/README.md, and phase_5/README.md
10. docs/latex/contents/4-experiments.tex, 5-results.tex, and
    9-appendix.tex to see which cells are still placeholders.

Do not trust older notebook prose without checking the current Python and shell
entry points. The current architecture is the v2 detector-first protocol.

## 2. Current Project State

The repository uses the following names to avoid confusion between the paper's
four stages and the old phase folders:

| Component | Meaning | Current state |
|---|---|---|
| M1 / paper Stage 1 | Frozen BioViL-T image features and the shared spatial frame | Feature extraction is assumed to be done elsewhere; the feature cache is not present in this local checkout. |
| M2 / detector part of Stage 1 | YOLOv8m, 29 anatomical regions | Final checkpoint and validation numbers exist locally. Full inference with this checkpoint has not been completed locally. |
| M3 / paper Stage 2 | 29-region concept and disease prediction | Old pre-xwalk runs exist, and an xwalk-v2 checkpoint exists, but the final nine detector-box v2 runs must be regenerated after new YOLO boxes. |
| M4 / paper Stage 3 | Temporal progression for each disease and region | The available KL005 + Dist050 checkpoint is a GT-box oracle. No final detector-box M4 campaign has been completed. |
| M5 / paper Stage 4 | Deterministic report assembler | Local HTML demos work, but thresholds and report calibration must be refit after final detector-box M3/M4 inference. |

### M2 artifact already available

~~~
weight/detect/det29_ft1024_s2/weights/last.pt
weight/detect/det29_ft1024_s2/results.csv
weight/audit_report.json
~~~

Protocol: YOLOv8m, full detector data, imgsz=1024, batch 16, 30 epochs.
Recorded validation values are precision 0.93657, recall 0.89551,
mAP50 0.94025, mAP50-95 0.71935. The matching full-data audit reports
mean region IoU 0.8207 on 21,335 validation images. The checkpoint hash and
detector provenance are documented in the M2 runbook.

The following output does not exist yet and must be created in Kaggle:

~~~
data/phase2_detector_server_retrained/predictions.jsonl
data/m3_labels/boxes_det.npy
data/m3_labels/present_mask_det.npy
data/m3_labels/detector_provenance.json
~~~

The existing boxes_det.npy and present_mask_det.npy must be treated as stale
until regenerated from the final YOLO checkpoint and aligned to the authoritative
M3 manifest.

### M3 state

Legacy development runs under data/run_before_xwalk/ are mechanism diagnostics
only. They must not be copied into the final Results tables. The local
weight/m3_B_faithful_xwalk_v2/best.pt is also a legacy/reference checkpoint,
not a replacement for retraining on the new detector boxes.

The final M3 launcher defines nine retained rows:

~~~
m3v2_vera_graph_lse_det       main VERA: graph-constrained, no global bypass, LSE
m3v2_no_concept_det           direct regional disease head
m3v2_concept_mlp_det          free concept-to-disease MLP
m3v2_graph_global_fusion_det graph plus direct global fusion
m3v2_global_only_det          BioViL-T global token only
m3v2_graph_attention_det     attention aggregation
m3v2_graph_mean_det          mean aggregation
m3v2_graph_max_det           max aggregation
m3v2_vera_graph_lse_gt        matched GT-box oracle
~~~

The detector-box main row and the retained ablations are the missing M3
numbers. The GT row is an oracle sensitivity analysis and is never the deployed
main result.

### M4 state

~~~
weight/m4v4_tf_m3delta40_kl005_dist050_xwalk_v2/best_acc.pt
~~~

This checkpoint was trained with GT boxes and is kept only for debugging and
oracle comparison. Do not call it the final VERA temporal result.

The current paper plan is a detector-box M4 campaign after the faithful M3
checkpoint is frozen:

1. 3x3 KL-weight by distance-penalty validation grid.
2. Selected coefficient run becomes the VERA temporal model.
3. Retained regularization and architecture ablations.
4. Test split, MS-CXR-T external audit, and flip/null consistency checks.

The current retained plan is 18 unique detector-box M4 trainings: the 9-cell
coefficient grid, 2 additional architecture rows, and 7 additional loss rows.
A separate matched GT-box oracle makes 19 total trainings. The paper launcher
is authoritative for the exact list.

The exact run names and coefficient values must come from
phase_4/run_paper_m4_v2.sh; do not infer a numeric value from an old run name.
In particular, the old name kl005 represented a flip-KL weight of 0.05,
not 0.005. Record the actual checkpoint configuration JSON in every result.

## 3. Code Map For The New Session

### M2 detector inference

Read:

~~~
phase_2/src/constants.py
phase_2/src/config.py
phase_2/scripts/yolo/5-infer_yolo.py
phase_3/scripts/3-boxes_from_pred.py
~~~

The current script is under phase_2/scripts/yolo/; the old Kaggle notebook may
call a flat infer_yolo.py, which is stale. The output JSONL must be converted
using the M3 manifest, not by assuming JSONL order equals manifest order.

### M3 training and evaluation

Read:

~~~
phase_3/src/config.py
phase_3/src/dataset.py
phase_3/src/model.py
phase_3/src/heads.py
phase_3/src/pooling.py
phase_3/src/losses.py
phase_3/scripts/4-train.py
phase_3/scripts/5-eval.py
phase_3/scripts/6-faithfulness.py
phase_3/scripts/7-infer.py
phase_3/scripts/8-precompute_regions.py
phase_3/scripts/10-region-report.py
phase_3/run_paper_m3_v2.sh
~~~

The main M3 contract is mode B, faithful graph head, detached concept-to-disease
path, no global disease bypass, normalized LSE aggregation, detector boxes, and
derived No Finding. The launcher stores the architecture snapshot in each
checkpoint and automatically resumes from last.pt.

### M4 training and evaluation

Read:

~~~
phase_4/src/config.py
phase_4/src/dataset.py
phase_4/src/model.py
phase_4/src/losses.py
phase_4/scripts/2-train.py
phase_4/scripts/3-eval.py
phase_4/scripts/4-infer.py
phase_4/scripts/5-mscxrt_audit.py
phase_4/scripts/7-temporal_consistency.py
phase_4/run_paper_m4_v2.sh
phase_4/run_m4_retrain_matrix.sh
~~~

Before launching the final campaign, also read
`docs/now/m4_temporal_calibration_and_readout_policy.md`. Its 2026-07-28 audit
found that the current M4 dataset uses the GT `present_mask.npy` even for
detector-box runs, and that disease-level majority calibration conflicts with
mixed regional directions. These are stop conditions, not post-run cleanup.

M4 is staged on a frozen M3. It needs the M3 region cache and, for TempFuse,
the original frozen BioViL-T patch grids. Use --box-source detector. The
launcher must reject a checkpoint/cache whose box-source provenance is GT.

### M5 report and calibration

Read:

~~~
phase_5/calibrate.py
phase_5/assemble.py
phase_5/render_report.py
phase_5/run.py
phase_5/verify.py
phase_5/README.md
~~~

M5 has no learned training step and does not fit M3 thresholds. The M3
post-training stage fits validation-only disease thresholds and concept gates;
M5 strictly consumes those frozen artifacts alongside final M3/M4 predictions
and renders the classification/progression/ground-truth HTML report. Existing
GT-box calibration artifacts are provisional and must be regenerated after
detector-box M3/M4 are accepted.

## 4. What Must Be Uploaded To Kaggle

Use separate Kaggle datasets rather than one enormous archive. Kaggle input
datasets are read-only, so each notebook must copy labels/checkpoints into
/kaggle/working before replacing or resuming them.

### Dataset A: code

Preferred: clone the current repository in a notebook with Internet enabled.
Otherwise upload the complete folders below as a code dataset:

~~~
phase_2/
phase_3/
phase_4/
phase_5/
~~~

Do not use the old notebooks as the source of truth; update them to call the
current numbered scripts. The notebook should print the Git commit/hash or the
uploaded source version before running.

### Dataset B: detector inference input

Only the M2 notebook needs raw images:

~~~
data/mimic-cxr-448/                    # about 4.0 GB locally
weight/detect/det29_ft1024_s2/weights/last.pt
data/m3_labels/manifest.jsonl
~~~

The M2 notebook returns predictions.jsonl, boxes_det.npy,
present_mask_det.npy, and detector_provenance.json. Save them as a new Kaggle
dataset or copy them to Drive before ending the session.

### Dataset C: M3 training input

For the current Kaggle handoff, upload the fresh M2 result folder as a separate
dataset named **`vera-v2-m2-detector-outputs`**. The M3-v2 paper notebook is
`kaggle_notebooks/phase3_paper_v2_kaggle.ipynb`. Do not use the legacy notebook
under `phase_3/notebooks/` for this campaign.

The M3 notebook expects these canonical mounts:

```text
/kaggle/input/datasets/nguynnghin/vera-v2-inputs
/kaggle/input/datasets/nguynnghin/vera-v2-m2-detector-outputs
/kaggle/input/datasets/nguynnghin/frozen
```

The M2 output dataset must retain `boxes_det.npy`, `present_mask_det.npy`, and
`detector_provenance.json` under
`vera-v2-m2-detector-output/m3_labels_detector_v2/`; keep
`predictions.jsonl` as the detector audit trail. The authoritative
`manifest.jsonl` remains in `vera-v2-inputs/m3_labels_base`. The notebook copies
that label bundle into `/kaggle/working`, overlays the detector arrays, and
checks their row count and shape against the manifest before training.

~~~
data/m3_labels/                         # about 0.65 GB locally
data/m3_concept_space.json
phase_3/src/m3_concept_space.json
phase_4/src/m3_concept_space.json
frozen BioViL-T features for every manifest image
~~~

The three concept-space JSON files must be identical and contain 69 concepts.
The feature cache is missing from this local checkout. The old notebooks refer
to a Kaggle dataset similar to mimic-biovilt-feats, but the new session must
verify the actual slug, file format, image coverage, and feature shape before
training. Expected feature tensors are per-image [197, 512] or [196, 512],
depending on whether the global token was retained. Do not start M3 until the
loader successfully reads several files and the feature dimension matches the
checkpoint/model contract.

Raw JPEGs, DICOMs, YOLO labels, and scene graphs are not needed for M3 once
m3_labels and detector arrays are available.

### Dataset D: M4 training input

~~~
data/m3_labels/                         # final detector arrays included
data/m4_labels/                         # progression.npy, manifest, pairs
data/MS_CXR_T_temporal_image_classification_v1.0.0.csv
frozen BioViL-T features for current and prior images
data/run/m3v2_vera_graph_lse_det/best.pt  # produced by M3 notebook
~~~

M4 does not need raw CXRs or YOLO weights. The M3 notebook must first produce a
detector-box faithful checkpoint. The M4 notebook then runs
phase_3/scripts/8-precompute_regions.py once to create a fresh cache whose
marker records the M3 checkpoint hash and box_source=detector.

### Optional Dataset E: report demo

For final M5 examples, also provide:

~~~
data/mimic-metadata/mimic_metadata_final.jsonl
data/mimic-cxr-2.0.0-metadata.csv
data/demo/                              # optional selected pair assets
~~~

This is not required to train M3 or M4.

## 5. Notebook Plan

Create four new notebooks. They may share helper cells, but each notebook must
be restartable and must save outputs before the Kaggle session ends.

Target a single Kaggle GPU (preferably T4 or another CUDA GPU with enough
memory), keep DataLoader workers at 2--4, and start M3/M4 with conservative
batch sizes such as 8--16. Do not copy the H100-mini batch/worker profile into
Kaggle. Split the run matrix across sessions and persist every run directory
before the session expires.

### Notebook 1: phase2_infer_boxes_v2_kaggle.ipynb

Cells:

1. Print GPU, Python, package versions, source commit, image count, and YOLO
   checkpoint SHA-256.
2. Mount the image and detector-weight datasets; install the pinned Ultralytics
   version used by the checkpoint.
3. Run phase_2/scripts/yolo/5-infer_yolo.py over all 448 images with the
   agreed imgsz, confidence, IoU, batch, and --no-per-image settings.
4. Run phase_3/scripts/3-boxes_from_pred.py against the authoritative M3
   manifest and validate shapes [N,29,4] and [N,29].
5. Write detector provenance, coverage statistics, and a small overlay sample.
6. Save the four outputs as a new Kaggle dataset or Drive artifact.

Do not implement resume by appending to an existing JSONL. If a session dies,
restart into a fresh output directory or partition the image tree explicitly.

### Notebook 2: phase3_paper_v2_kaggle.ipynb

Cells:

1. Mount code, M3 labels, detector arrays from Notebook 1, and the feature cache.
2. Copy m3_labels to /kaggle/working/m3_labels_detector_v2 and place the new
   detector arrays there; never mutate /kaggle/input.
3. Run the launcher preflight and a one-batch model/feature smoke test.
4. Run run_paper_m3_v2.sh --profile local4060 --scope main first.
5. The main launcher automatically regenerates stale validation dumps, fits
   pair-specific disease thresholds and concept gates, writes both CSV audits,
   and creates `m3v2_vera_graph_lse_det.calibration.SUCCESS.json` only on
   completion. Training checkpoints sync every epoch and completed diagnostics
   sync after each split, so a dead session can resume without discarding prior
   work. Save best.pt, last.pt, logs, metrics, diagnostics, predictions,
   regional audit, faithfulness JSON, the four calibration outputs, and marker.
6. Resume with --scope all to obtain the eight remaining retained rows. If a
   Kaggle session is too short, expose a RUN_SCOPE/RUN_INDEX setting so one or
   two rows can be run per session while preserving the same run directory.
7. Upload the complete M3 run directories, especially the main checkpoint and
   its config/metrics/diagnostics, as the next Kaggle dataset.

The notebook must not use the old phase_3/notebooks/phase3_kaggle.ipynb
configuration (m3_A/m3_B, macro-F1 selection, or old C-KAN wording).

### Notebook 3: phase4_paper_v2_kaggle.ipynb

Cells:

1. Mount final detector-box M3, M3/M4 labels, pairs, and the frozen feature
   cache. Verify M3 checkpoint metadata says mode B, faithful head, no global
   bypass, LSE, and box_source=detector.
2. Run 8-precompute_regions.py with the final M3 checkpoint and a fresh output
   directory. Verify .m3_source.json before training.
3. Run the M4 coefficient grid first. Use the current
   phase_4/run_paper_m4_v2.sh scope grid; do not select coefficients using test
   or MS-CXR-T labels.
4. Persist the selected-coefficient file, grid diagnostics, and run directories.
5. Run the retained paper ablations with the selected frozen coefficient and
   detector boxes. Use the launcher's paper scope, then final for test,
   MS-CXR-T, inference, and temporal consistency.
6. Save all checkpoints, config JSONs, diagnostics, confusion matrices, plots,
   M4 inference JSONL, and the cache provenance marker.

If a Kaggle session cannot finish the entire grid, make the notebook resume
individual named runs from last.pt; never silently reduce the epoch count or
change box source between sessions.

### Notebook 4: phase5_report_calibration_v2_kaggle.ipynb

This is optional and comes only after M3/M4 acceptance. Run validation-only
calibration, generate test predictions, assemble the three report blocks, run
the structural verifier, and render the HTML previews. The notebook must keep
the reference report as metadata text and must not use an LLM to add clinical
content.

## 6. Acceptance Checks Before Copying Numbers Into The Paper

### M2

~~~
YOLO class count = 29
prediction coverage matches the M3 manifest
boxes_det.npy shape = [N, 29, 4]
present_mask_det.npy shape = [N, 29]
all coordinates finite and in the 448 coordinate frame
detector provenance records checkpoint hash and inference settings
~~~

### M3

~~~
main checkpoint was trained with the new detector arrays
checkpoint box_source = detector
mode = B, disease_head = faithful
DETACH_CONCEPT_FOR_DISEASE = true
USE_GLOBAL_HEAD = false
REGION_AGG = lse
DERIVE_NO_FINDING = true
validation and test diagnostics exist for every retained row
faithfulness is run for every mode-B row
schema-v2 validation dump records IDs, patient clusters, and checkpoint/manifest/box provenance
main-row report_thresholds.json and concept_gate.json exist with matching provenance
both calibration CSV audits and calibration.SUCCESS.json exist
per-pair disease thresholds and concept gates are not claimed before those audit artifacts exist
~~~

Do not use a legacy or GT-box checkpoint merely because it has a convenient
metric. The old values in data/run_before_xwalk/ are not final paper values.

### M4

~~~
fresh region cache marker points to the final M3 checkpoint hash
cache and checkpoint both use box_source = detector
coefficient selection uses validation only
Support Devices is excluded from progression reporting
test and MS-CXR-T are run only after coefficient selection
flip and identical-pair consistency audits are saved
~~~

### M5

~~~
temperature and display thresholds are fitted on validation only
calibration is regenerated after detector-box M3/M4
present/absent and temporal display rules are documented
HTML structural verifier passes
~~~

## 7. Common Failure Modes

- Do not use the GT-box M4 checkpoint as the main result.
- Do not evaluate old M3 weights on new YOLO boxes without retraining.
- Do not use a stale boxes_det.npy just because its shape is correct.
- Do not train M4 until the faithful detector-box M3 checkpoint is frozen.
- Do not use test/MS-CXR-T results to choose KL or distance coefficients.
- Do not assume the old Kaggle notebooks match current script paths or model
  flags; they predate the v2 launcher.
- Do not upload the entire repository and all raw images to every notebook.
  Split datasets by dependency to keep Kaggle startup and storage manageable.
- If the feature cache cannot be located, stop and ask the user for its Kaggle
  dataset slug or for a way to regenerate it. M3/M4 cannot run without it.

## 8. Files To Return To The Local Workspace

At minimum, bring back:

~~~
M2: predictions.jsonl, boxes_det.npy, present_mask_det.npy, detector_provenance.json
M3: all retained run dirs, especially main best.pt/last.pt, metrics, diagnostics, faithfulness,
    schema-v2 validation NPZ, report thresholds, concept gate, both CSV audits, success marker
M4: fresh detector cache, selected_coefficients.env, retained run dirs, test/MS-CXR-T diagnostics
M5: calibrated thresholds, final prediction JSONL, verifier output, selected HTML report
~~~

After downloading artifacts, update docs/now/paper.md first, then fill the
blank cells in docs/latex/contents/5-results.tex and the appendix. Never copy
numbers into the paper without recording the exact checkpoint, box source,
split, coefficient settings, and metric convention.
