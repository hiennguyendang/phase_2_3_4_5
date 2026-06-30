# phase_2 — Module 2: 29-region detector + report → scene-graph

Module 2 of VERA turns a report + detected regions into the per-region findings scene graph.
There are now **three** branches here:

- **BBOX branch (detector)** — YOLOv8 detects the **29 anatomical regions**; feeds
  the bboxes that phase_3 (C-KAN) ROI-pools over.
- **Rule parser (active attribute path — `src/` + `scripts/`)** — a deterministic, CPU,
  glass-box report parser that replicates ImaGenome's NLP pipeline WITHOUT a model. Full 22k val:
  **with-region F1 0.860 / P 0.894 / R 0.829 / macro 0.779 / finding-only 0.924** (region-perfect
  ceiling 0.936; the residual is cross-sentence laterality, which ImaGenome's authors also leave
  unsolved). Preferred over the LLM for the glass-box thesis (no GPU, auditable, no hallucination).
- **Attribute/relationship branch (LLM, optional)** — a QLoRA-fine-tuned Qwen reads the report +
  regions and emits the same flat schema. Kept for CheXplus pseudo-labels, but the rule parser
  matches/beats it without the cost.

```
detector:   scene graphs → YOLO labels → train YOLOv8 → infer 29 boxes
rule parser: report → src/rule_parser.parse_report → flat findings scene graph   (no GPU)
LLM:        scene graphs → SFT data → QLoRA Qwen → pseudo scene graphs
```

## Directory layout (reorganised 2026-06-30)
```
src/                       # flat shared LIBRARY (one import root) + bundled data
  config.py constants.py scene_to_yolo.py sg_schema.py sg_lib.py sg_eval_lib.py
  rule_parser.py                                   # the rule parser
  m3_concept_space.json                            # 69-concept space
  rule_finding_triggers.json rule_region_priors.json rule_concept_parents.json   # parser data
scripts/                   # runnable entry points, numbered in run order; each adds ../../src to path
  yolo/   1-build_yolo_dataset  2-link_yolo_images  3-train_yolo  4-eval_yolo  5-infer_yolo
          + visualize viz_boxes audit_yolo scan_findings           (utilities, no number)
  llm/    1-build_sft_dataset  2-finetune_sg_llm  3-merge_lora  4-eval_sg_llm  5-build_pseudo_scene_graph
          + kaggle_sync  extract_sg_vocab (LEGACY)
  rule_parser/  1-mine_region_priors  2-mine_finding_lexicon  3-eval_rule_parser  4-eval_gold
                5-diagnose_rule_parser  + rethreshold_lexicon (utility)
notebooks/                 # all .ipynb (Kaggle / audit)
_work/  runs/              # intermediate outputs (eval jsons, rich sidecar, YOLO runs) — not source
gold_ids.txt  requirements.txt  README.md
```
> `src/` is kept FLAT on purpose: every module imports the others by bare name (`import config`,
> `from sg_schema import ...`), so one path entry (`src/`) makes the whole library importable. The
> per-subsystem split is in `scripts/` (and `notebooks/`). Run scripts by path, e.g.
> `python scripts/rule_parser/3-eval_rule_parser.py`. The numeric prefix = run order.

### Reproduce the rule parser end-to-end (no GPU)
```bash
python scripts/rule_parser/1-mine_region_priors.py                                   # -> src/rule_region_priors.json + rule_concept_parents.json
python scripts/rule_parser/2-mine_finding_lexicon.py --scene-root <chest-imagenome>  # -> src/rule_finding_triggers.json (+ _work rich sidecar)
python scripts/rule_parser/3-eval_rule_parser.py --val _work/sg_sft/val.jsonl --limit 0   # vs silver  -> F1 0.860
python scripts/rule_parser/4-eval_gold.py       --scene-root <chest-imagenome>            # vs gold    -> F1 0.771
python scripts/rule_parser/5-diagnose_rule_parser.py --val _work/sg_sft/val.jsonl         # FP/FN decomposition + ceiling 0.936
```
Tuning is via env flags in `src/rule_parser.py`, all locked to best defaults (`RULE_SECTION_FILTER=1`,
`RULE_DEFAULT_THRESH=0.55`, `RULE_ALLOW_THRESH=0.05`, `RULE_CLAUSE_LOC=1`, `RULE_FWD_NEG=1`;
tested-and-OFF: `RULE_LOC_ONLY`, `RULE_ZONE_CLIP`, `RULE_REGION_CONTAINMENT`, `RULE_LAST_WINS`).

## Proposed deletions (NOT deleted — confirm first)
- `notebooks/yolo-output.ipynb` (67 KB) — looks like a saved notebook-output dump, not source.
- `scripts/llm/extract_sg_vocab.py` + `sg_lib.py`'s compact-prompt/parse/snap functions — LEGACY,
  only fed the old compact target, unused since the flat schema.
- (review) `notebooks/audit.ipynb`, `notebooks/phase2_kaggle.ipynb` if superseded.

## Files
| File | Role |
|------|------|
| `constants.py` | the 29 region names; class id = alphabetical index (single source of truth) |
| `config.py` | paths (Kaggle defaults) + hyperparameters — **edit the `KAGGLE_*` paths** |
| **Detector branch** | |
| `scene_to_yolo.py` | core: one `*_SceneGraph.json` → normalized YOLO lines (filter 29, drop sentinel/degenerate, clip) |
| `build_yolo_dataset.py` | step 0 — build `images/labels/{train,val,test}` + `dataset.yaml` |
| `train_yolo.py` | step 1 — train YOLOv8l (anatomy-safe aug, resumable) |
| `eval_yolo.py` | mAP on a split (test/gold) |
| `infer_yolo.py` | step 2 — run `best.pt` → one JSON of 29 boxes per image |
| `visualize.py` | sanity: draw converted GT boxes on a few images |
| **LLM branch** | |
| `sg_lib.py` | core: scene graph ↔ compact target, prompts, vocab snap, assemble |
| `extract_sg_vocab.py` | step 4 — controlled relation vocab → `sg_vocab.json` |
| `build_sft_dataset.py` | step 5 — chat SFT dataset (report+regions → compact JSON) |
| `finetune_sg_llm.py` | step 6 — QLoRA Qwen2.5-7B (completion-only loss) |
| `merge_lora.py` | (optional) merge LoRA → standalone weights for step 7 |
| `build_pseudo_scene_graph.py` | step 7 — LLM → pseudo `*_SceneGraph.json` for CheXplus |
| `kaggle_sync.py` | durable checkpointing to a Drive (rclone) remote — survives Kaggle session death |

> **`phase2_kaggle_train.ipynb`** = the detector training notebook with **Drive-resumable
> checkpoints**. It pushes the run dir to rclone (`--sync-remote`) every `--sync-every`
> steps + each epoch; re-run with `RESUME=1` and it pulls `last.pt` from Drive and
> continues. Use this when training the real (full) dataset across multiple Kaggle sessions.

> **`phase2_kaggle.ipynb`** packages every command below as ready-to-run cells —
> edit the CONFIG cell paths and run the detector / LLM sections (separate GPU
> sessions). The CLI steps below are the same thing by hand.

## Data expected on Kaggle
- **images**: the resized CXR images, filename stem == `image_id`
  (`MIMIC_<patient>_<study>_<dicom>.jpg`).
- **scene graphs**: `<dicom>_SceneGraph.json` files (ImaGenome format).
- **metadata**: `mimic_metadata_final.jsonl` (gives `split` and which images have a
  scene graph).

Images and bboxes must already be resized/scaled together — the converter
normalizes boxes by the **actual image width/height** (read with PIL).

## Recommended flow: build labels LOCALLY, link on Kaggle

The slow step (open every image for W/H + convert boxes) only needs the scene graphs + metadata —
do it **once locally** and upload a small `labels/` dataset, so Kaggle never re-builds (and you don't
upload the 11 GB scene-graph tree or the metadata there). The notebook `phase2_kaggle_train.ipynb`
already follows this. Because the images are center-cropped to a square, `--fixed-size 448` skips
PIL entirely — **you don't even need the images locally**, just scene graphs + metadata.

```bash
# LOCAL (no GPU, no images needed): scene graphs + metadata -> labels/ + dataset.yaml
python build_yolo_dataset.py --labels-only --fixed-size 448 \
  --metadata data/mimic_metadata_final.jsonl \
  --scene-root <chest-imagenome> --out yolo_labels
# -> upload the `yolo_labels/` folder as a Kaggle dataset named `yolo-labels`

# KAGGLE (in the notebook): just symlink the matching images (fast, no PIL)
python link_yolo_images.py --labels-dir /kaggle/input/yolo-labels \
  --images-root /kaggle/input/<images> --out /kaggle/working/yolo_ds
```

> Fallback (build entirely on Kaggle, slower): `python build_yolo_dataset.py --images-root ... \
> --scene-root ... --metadata ... --out /kaggle/working/yolo_ds` (opens every image for W/H).

## Train on Kaggle (GPU)
```bash
pip install -q ultralytics

# (recommended) eyeball alignment before training
python visualize.py --split val --n 12     # check /kaggle/working/viz

# 1) train (imgsz=448 matches the cropped images; bump to 640 to upsample; smaller batch if you OOM)
python train_yolo.py --imgsz 448 --batch -1 --epochs 100
python train_yolo.py --resume               # continue in a later session

# eval mAP on the held-out / gold split
python eval_yolo.py --weights /kaggle/working/runs/det29/weights/best.pt --split test

# 2) inference -> 29-box JSON per image
python infer_yolo.py \
  --weights /kaggle/working/runs/det29/weights/best.pt \
  --source  /kaggle/input/<images> --out /kaggle/working/pred --jsonl
```

## Run the LLM branch (GPU)
```bash
pip install -q transformers trl peft bitsandbytes accelerate datasets

# 4) controlled relation vocab from the silver scene graphs
python extract_sg_vocab.py --scene-root /kaggle/input/<scene> --min-count 5

# 5) chat SFT dataset (report + regions -> compact findings JSON)
python build_sft_dataset.py \
  --metadata /kaggle/input/<meta>/mimic_metadata_final.jsonl \
  --scene-root /kaggle/input/<scene> --keep-empty-frac 0.1

# 6) QLoRA fine-tune Qwen2.5-7B (single 16GB GPU OK; resumes from output_dir)
python finetune_sg_llm.py --data-dir /kaggle/working/sg_sft \
  --out /kaggle/working/sg_lora --epochs 2

# 7) pseudo scene graphs for CheXplus (needs detector boxes from step 2 + reports)
python infer_yolo.py --weights .../best.pt \
  --source /kaggle/input/<chexplus-images> --out /kaggle/working/pred_chexplus
python build_pseudo_scene_graph.py \
  --pred-dir /kaggle/working/pred_chexplus \
  --metadata /kaggle/input/<chexplus>/chexplpus_metadata_final.jsonl \
  --vocab /kaggle/working/sg_vocab.json --lora /kaggle/working/sg_lora \
  --out /kaggle/working/pseudo_sg \
  --update-metadata /kaggle/working/chexplus_with_scene.jsonl
```
NIH has no report → no cues → it is intentionally skipped by step 7 (its boxes
still come from the detector for C-KAN).

## Notes / things you can change
- **Class order** is alphabetical (`constants.CLASS_NAMES`). If phase_3
  `REGION_NAMES` ever uses a different order, change it in ONE place here and
  rebuild — `dataset.yaml` is generated from it.
- **Splits**: `gold` is routed to `test` (clean human-verified eval set). Edit
  `config.SPLIT_MAP` to change.
- **Multi-session training**: checkpoints every `SAVE_PERIOD` epochs in
  `runs/det29/weights/`. Keep that under `/kaggle/working` so it persists,
  then `--resume`.
- A scene graph commonly has **>29 objects** (e.g. `descending aorta`,
  `left/right cardiac silhouette`, `left/right upper abdomen`); the converter
  keeps only the 29 canonical regions and reports how many it dropped.
