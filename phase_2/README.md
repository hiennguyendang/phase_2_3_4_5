# phase_2 — Module 2: 29-region detector + scene-graph LLM

Module 2 of KAN-TRaCE has two branches, both here:

- **BBOX branch (detector)** — YOLOv8l detects the **29 anatomical regions**; feeds
  the bboxes that phase_3 (C-KAN) ROI-pools over.
- **Attribute/relationship branch (LLM)** — a QLoRA-fine-tuned Qwen2.5-7B reads the
  report + detected regions and emits per-region findings (with `comparison_cues`
  → phase_4 T-KAN). Produces ImaGenome-style pseudo scene graphs for CheXplus.

```
detector:  scene graphs → YOLO labels → train YOLOv8l → infer 29 boxes
LLM:       scene graphs → vocab + SFT data → QLoRA Qwen → pseudo scene graphs
```

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

## Run on Kaggle (GPU)
```bash
pip install -q ultralytics

# 0) point config.py KAGGLE_* paths at your dataset slugs, OR pass flags below.
#    (build_yolo_dataset.py also auto-detects roots under /kaggle/input.)
python build_yolo_dataset.py \
  --metadata   /kaggle/input/<meta>/mimic_metadata_final.jsonl \
  --images-root /kaggle/input/<images> \
  --scene-root  /kaggle/input/<scene> \
  --link-mode symlink

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
