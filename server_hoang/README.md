# VERA server handoff for Hoàng

This folder is the single server entry point for the final detector-first
campaign:

```text
M2: infer the 29 anatomical boxes once
  -> align detector boxes to the authoritative M3 manifest
  -> M3: train/evaluate/calibrate all nine retained rows
  -> M4: build frozen-M3 caches, run the coefficient grid, select on validation,
         run the retained paper rows, then produce final audits
```

Use `run_all.sh`; do not manually call old notebooks or legacy run folders.
The script is restartable and writes no generated artifact into the uploaded
input directory.

## 1. What must be copied to the server

Copy the **entire current repository**, including `.git` when possible, to one
directory on the server. The source commit should contain this folder and the
current `phase_2/`, `phase_3/`, and `phase_4/` directories.

Large data may be copied directly into `server_hoang/input/` or symlinked from
another server disk. The default layout is:

```text
<repo>/
├── phase_2/
├── phase_3/
├── phase_4/
├── data/
│   └── m3_concept_space.json
├── server_hoang/
│   ├── README.md
│   ├── run_all.sh
│   └── input/
│       ├── mimic-cxr-448/
│       │   └── p10/p10000032/MIMIC_p10000032_....jpg
│       ├── frozen/
│       │   └── <image_id>.pt
│       ├── yolo/
│       │   └── last.pt
│       ├── m3_labels/
│       │   ├── manifest.jsonl
│       │   ├── region_concepts.npy
│       │   ├── region_chexpert.npy
│       │   ├── image_chexpert.npy
│       │   ├── boxes.npy
│       │   └── present_mask.npy
│       ├── m4_labels/
│       │   ├── manifest.jsonl
│       │   ├── progression.npy
│       │   └── m3_pairs.jsonl
│       └── external/
│           └── MS_CXR_T_temporal_image_classification_v1.0.0.csv
└── ...
```

### Raw images

`mimic-cxr-448/` must contain the resized images using the layout expected by
the detector:

```text
<image root>/<patient prefix>/<patient id>/<image_id>.jpg
```

For example:

```text
mimic-cxr-448/p10/p10000032/MIMIC_p10000032_s50414267_<dicom-id>.jpg
```

The image extension may be `.jpg`, `.jpeg`, or `.png`. M2 resolves the images
from `m3_labels/manifest.jsonl`; the filenames and manifest `image_id` values
must agree exactly.

### Frozen BioViL-T features

`frozen/` contains one file per image ID. Flat and nested layouts are accepted:

```text
<image_id>.pt       # normally torch.float16 [197,512]
<image_id>.npy      # also accepted; [197,512] or [196,512]
```

All current and prior images used by M3/M4 must be covered. Do not upload the
raw BioViL-T model; only the already frozen features are needed.

### YOLO checkpoint

The expected file is the final 29-region checkpoint:

```text
server_hoang/input/yolo/last.pt
```

Expected SHA-256:

```text
71d4b4e3b173cc046fc45c7120b6cf4489c384ceaaec9f08231182108a40da56
```

The script refuses a different hash by default. If a checkpoint is
intentionally replaced, use a new `OUTPUT_ROOT` and explicitly set
`EXPECTED_YOLO_SHA256`; never mix its boxes with an old M3/M4 output tree.

### M3 label bundle

Upload the base label bundle, not the detector result from an older run:

| File | Contract |
|---|---|
| `manifest.jsonl` | authoritative row order and train/val/test split |
| `region_concepts.npy` | `[N,29,69]`, values `1/0/-100` |
| `region_chexpert.npy` | `[N,29,14]`, values `1/0/-100` |
| `image_chexpert.npy` | `[N,14]`, values `1/0/-100` |
| `boxes.npy` | GT/oracle boxes `[N,29,4]` in the 448 coordinate frame |
| `present_mask.npy` | GT/oracle region mask `[N,29]` |

`boxes_det.npy`, `present_mask_det.npy`, and `detector_provenance.json` do not
need to be uploaded. This server campaign regenerates them once from the final
YOLO checkpoint.

### M4 labels and external benchmark

Upload:

```text
m4_labels/manifest.jsonl
m4_labels/progression.npy
m4_labels/m3_pairs.jsonl
external/MS_CXR_T_temporal_image_classification_v1.0.0.csv
```

The M4 run begins only after the final M3 detector and GT-oracle checkpoints
exist. Raw images and the YOLO checkpoint are not read again by M4.

### Concept graph copies supplied by the source repository

These three files are source/configuration, not large uploads. They must be
identical and contain 69 concepts:

```text
data/m3_concept_space.json
phase_3/src/m3_concept_space.json
phase_4/src/m3_concept_space.json
```

## 2. Environment preparation

The server must be Linux with Bash, `nohup`, `setsid`, `flock`, NVIDIA drivers,
and enough persistent disk for checkpoints, diagnostics, and M4 region caches.
The local computer may disconnect, but the **server itself must remain on**.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy scipy scikit-learn matplotlib pandas pillow pyyaml ultralytics
chmod +x server_hoang/run_all.sh
```

Install the CUDA build of PyTorch appropriate for the server if the generic
command installs a CPU-only build. Confirm `python -c 'import torch;
print(torch.cuda.is_available())'` prints `True`.

The full `phase_2/requirements.txt` also includes packages for the unrelated
LLM scene-graph branch. They are not required for this M2→M3→M4 campaign.

## 3. Paths and tuning

The default input and output roots are:

```text
INPUT_ROOT=<repo>/server_hoang/input
OUTPUT_ROOT=<repo>/server_hoang/output
```

For a large server disk, symlinks are acceptable:

```bash
ln -s /data/vera/mimic-cxr-448 server_hoang/input/mimic-cxr-448
ln -s /data/vera/frozen server_hoang/input/frozen
```

Alternatively export absolute roots before every command:

```bash
export INPUT_ROOT=/data/vera/input
export OUTPUT_ROOT=/data/vera/output_2026_08
export GPU_IDS=0,1,2,3
```

Important tuning variables:

| Variable | Default | Meaning |
|---|---:|---|
| `GPU_IDS` | all GPUs reported by `nvidia-smi` | physical GPU indices used by the campaign |
| `M2_BATCH` | 16 | YOLO inference batch per GPU |
| `M2_SHARDS_PER_GPU` | 16 | resumable M2 shards assigned to each GPU |
| `M3_BATCH` | 64 | batch for each single-GPU M3 run |
| `M3_WORKERS` | 8 | feature-loading workers per concurrent M3 run |
| `M4_BATCH` | 128 | batch for M4/cache/evaluation |
| `M4_WORKERS` | 16 | M4 feature-loading workers |
| `REMOTE_ROOT` | empty | optional rclone destination, e.g. `drive:VERA_SERVER` |
| `RESULTS_INTERVAL` | 60 | seconds between automatic table refreshes |

M3 assigns independent runs round-robin across the GPUs. For example, with
four GPUs, four M3 runs train concurrently. `M3_BATCH` is a **per-run,
single-GPU batch**, not a global batch. Reduce batch/workers if the server runs
out of GPU memory or RAM. A restart resumes completed epochs.

M4 currently uses the first ID in `GPU_IDS` and follows the authoritative
launcher sequentially so coefficient selection cannot race incomplete grid
runs. The M4 dataset is much smaller than M3; correctness of the selection
order takes priority over manually parallelizing it.

If `REMOTE_ROOT` is set, install and configure `rclone` on the server before
preflight. Local persistent outputs remain authoritative; the remote is an
additional checkpoint copy, not a replacement for `OUTPUT_ROOT`.

## 4. Validate before starting

Run:

```bash
bash server_hoang/run_all.sh preflight
```

It checks:

- source entrypoints and the three concept-space copies;
- required images, feature cache, M3/M4 arrays, and MS-CXR-T CSV;
- Python packages and CUDA visibility;
- the YOLO checkpoint SHA-256;
- available GPUs and output-disk capacity.

Do not start if this command fails.

## 5. Start in the background

```bash
bash server_hoang/run_all.sh start
```

`start` runs preflight synchronously and launches the actual worker through
`nohup + setsid`. Once it prints a worker PID, closing SSH, VS Code, or the
personal computer does not stop the server process.

Check it from a later SSH session:

```bash
bash server_hoang/run_all.sh status
bash server_hoang/run_all.sh logs
```

`logs` follows the log. Pressing `Ctrl-C` stops only `tail`; it does not stop
training.

To request a graceful stop:

```bash
bash server_hoang/run_all.sh stop
```

To resume later:

```bash
bash server_hoang/run_all.sh start
# "resume" is an alias:
bash server_hoang/run_all.sh resume
```

Do not launch a second copy manually. `flock` and the worker PID protect the
same output tree from concurrent supervisors.

## 6. Resume and exactly-once behavior

### M2

M2 writes deterministic shards under:

```text
output/m2_detector/shards/shard_0000/
output/m2_detector/shards/shard_0001/
...
```

Each shard gets `SUCCESS.json` only after its JSONL line count matches its
expected manifest slice. After interruption, only missing/incomplete shards
are overwritten. Completed shards are never inferred again.

When all shards finish, the script merges them atomically, rejects duplicate,
missing, or extra image IDs, aligns the boxes to the full manifest, and writes
an immutable inference contract plus `M2.SUCCESS.json`. A completed M2 stage is
therefore skipped on every later start. If weights, manifest, inference
thresholds, resolution, or shard count change, the script refuses to reuse the
tree. Choose a new `OUTPUT_ROOT` rather than deleting provenance.

### M3

Every run stores `last.pt` after each completed epoch and resumes automatically.
Completed 40-epoch runs and already generated evaluations are skipped. The nine
retained runs are:

```text
m3v2_vera_graph_lse_det
m3v2_no_concept_det
m3v2_concept_mlp_det
m3v2_graph_global_fusion_det
m3v2_global_only_det
m3v2_graph_attention_det
m3v2_graph_mean_det
m3v2_graph_max_det
m3v2_vera_graph_lse_gt
```

The main detector row also produces validation-selected disease thresholds,
the concept gate, CSV audits, and a calibration success marker.

### M4

The frozen-M3 region cache skips image files already present. M4 checkpoints
resume from `last.pt` at the last completed epoch. The launcher runs:

1. the nine-cell validation coefficient grid;
2. validation-only coefficient selection;
3. retained architecture/loss rows and the matched GT-box oracle;
4. internal test, MS-CXR-T, temporal consistency, and raw test inference.

`M3_SYNC_EVERY` and `M4_SYNC_EVERY` default to `0`. This deliberately avoids a
mid-epoch checkpoint that could be mistaken for a completed epoch. On an
interruption, at most the current partial epoch is repeated.

## 7. Output names and locations

All generated artifacts live under `OUTPUT_ROOT`:

```text
output/
├── m2_detector/
│   ├── contract.json
│   ├── predictions.jsonl
│   ├── boxes_det.npy
│   ├── present_mask_det.npy
│   ├── detector_provenance.json
│   ├── M2.SUCCESS.json
│   └── shards/shard_*/
├── labels/m3/
│   ├── <copied base labels>
│   ├── boxes_det.npy
│   ├── present_mask_det.npy
│   └── detector_provenance.json
├── runs/
│   ├── m3v2_*/{config.json,metrics.jsonl,last.pt,best.pt}
│   └── m4v2_*/{last.pt,best.pt,best_acc.pt,best_prog.pt,best_change.pt}
├── cache/
│   ├── m4_region_detector/
│   └── m4_region_gt_oracle/
├── diagnostics/
│   ├── m3/
│   └── m4/
│       ├── selected_coefficients.env
│       ├── <selected-main>.test.json
│       ├── <selected-main>.mscxrt.json
│       ├── <selected-main>.temporal_consistency.json
│       └── <selected-main>.test.raw_predictions.jsonl
├── logs/
│   ├── server_hoang.supervisor.log
│   ├── m2/
│   ├── m3/
│   └── m4/
├── results/
│   ├── runs.csv
│   ├── runs.md
│   └── summary.json
└── state/
    ├── current_stage
    ├── run_manifest.json
    ├── stages/*.SUCCESS.json
    └── PIPELINE.SUCCESS
```

The collector refreshes the three files under `results/` every minute and
again whenever a run/stage completes. It reads checkpoints defensively, so
observing a checkpoint while it is being replaced cannot kill training.

Metric columns mean:

- M3 primary: image macro-AUC; M3 auxiliary: image macro-F1;
- M4 primary: change-only macro-F1; M4 auxiliary: three-class progression
  macro-F1;
- validation values are for selection; test values are for final reporting.

Refresh the table manually without affecting training:

```bash
bash server_hoang/run_all.sh collect
```

## 8. What must be returned after completion

Copy back the complete `OUTPUT_ROOT`. At minimum retain:

```text
m2_detector/{predictions.jsonl,boxes_det.npy,present_mask_det.npy,detector_provenance.json,contract.json}
labels/m3/{boxes_det.npy,present_mask_det.npy,detector_provenance.json}
runs/m3v2_*/
runs/m4v2_*/
cache/m4_region_detector/.m3_source.json
cache/m4_region_gt_oracle/.m3_source.json
diagnostics/m3/
diagnostics/m4/
results/{runs.csv,runs.md,summary.json}
logs/
state/PIPELINE.SUCCESS
```

Do not copy numbers into the paper solely from terminal output. Use the
collected table together with the corresponding diagnostic JSON and exact
checkpoint/config provenance.

## 9. Failure handling

- If `status` says `failed`, read the last lines of the supervisor log and fix
  the cause. Then run `start` again; do not remove completed runs.
- For out-of-memory errors, reduce `M3_BATCH` or `M4_BATCH`, keep the same
  output root, and resume. Record the final batch setting.
- If the YOLO/manifest contract changes, use a **new output root**.
- Resume an existing output tree with the same Git commit recorded in
  `state/run_manifest.json`; a changed source commit requires a new output root
  so checkpoints from different code are not silently mixed.
- If a cache provenance error says the M3 hash changed, use a new cache/output
  root. Never reuse a cache from another M3 checkpoint.
- Never run old GT-box checkpoints as the detector main result.
- Never choose the M4 grid winner from test or MS-CXR-T metrics. The launcher
  selects using internal validation change-F1 only.
