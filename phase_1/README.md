# phase_1 — Module 1: FROZEN BioViL-T feature extraction (M3's input)

Extracts BioViL-T grid features for every CXR and writes the **cache that `phase_3` loads**.
One file per image:

```
<image_id>.pt   =  torch.save( tensor [197, 512] float16 )
  row 0       = projected_global_embedding            (BioViL-T's own global vector)
  rows 1..196 = projected_patch_embeddings [512,14,14] flattened, index = y*14 + x
```

This is the exact format `phase_3/src/features.py` expects — **do not drift from it**. The encoder
is loaded once, `eval()`, `no_grad()`, never trained; the cache is deterministic.

```
CXR jpg 448x448 ──BioViL-T (frozen)──> projected_global [512] ─┐
                                        projected_patch [512,14,14] ─flatten y*14+x─> [196,512]
                                                                   └──cat──> [197,512] f16 -> <id>.pt
```

## Alignment (risk #1 — handled, and verified)
The m3 boxes (`data/m3_labels/boxes.npy`, `0..448`) live in a **448×448 stretched frame** — the
`mimic-cxr-448` jpgs are exactly 448×448 (a straight stretch). So M1 feeds those jpgs **as-is, with
no geometric resize/crop** (`TRANSFORM_MODE="stretch448"`), *not* BioViL-T's default
`Resize(512)+CenterCrop(448)` (which would re-frame the image and desync the boxes). cell = 448/14 =
32 px, matching `pooling.py`. `scripts/3-verify_features.py` proves alignment with an **overlay**:
it draws a region bbox on the 14×14 grid so a human confirms the cells sit on the anatomy.

## Frozen, self-consistent (we do NOT match the fine-tuned reference)
The `docs/*.pt` / `hoangtimothy/biovilt-features` cache was made with a **fine-tuned** BioViL-T.
We want the **FROZEN pretrained** encoder, so our features are *intentionally different* — the whole
cache is produced by one frozen model, one preprocessing. The go/no-go is therefore **not** a
reference match; it's `check_sanity` (the frozen encoder is alive & discriminative: feature std>0
and it separates different images) plus the alignment overlay. `--compare-reference <pt>` still
prints the cosine vs the fine-tuned reference, but only as information (low is expected).

## Layout (mirrors phase_2 / phase_3)
```
phase_1/
  src/        importable libs (clean names — NOT numbered)
  scripts/    numbered run-order entries (1- 2- 3-); each self-inserts ../src on sys.path
  notebooks/  m1_kaggle.ipynb
```

**`src/` — libraries:**
| File | Role |
|------|------|
| `config.py` | paths + 448/14/512 geometry + `TRANSFORM_MODE` + Kaggle/Drive defaults |
| `constants.py` | `image_id` helpers + sharded CXR path resolution (`p<pid[:2]>/p<pid>/<id>.jpg`) |
| `biovilt.py` | load FROZEN BioViL-T encoder + photometric transform + forward → `[197,C]` f16 |
| `io_features.py` | save `.pt` + rclone Drive flush (delete-local) + resume done-set |

**`scripts/` — run-order entries:**
| File | Role | GPU |
|------|------|-----|
| `1-build_worklist.py` | gather `image_id ∪ prior_image_id`, resolve each to a jpg path → `worklist.jsonl` | no |
| `2-extract_features.py` | the extraction loop (batched forward + Drive flush + resume) | yes |
| `3-verify_features.py` | structure / naming / coverage + frozen-encoder sanity + alignment overlay | yes\* |

\* `3-verify` runs the structural + alignment checks on CPU; only the **encoder sanity** (`[4]`)
needs the GPU/encoder, and it skips cleanly if `health_multimodal` is absent.

## Run order (Kaggle — see `notebooks/m1_kaggle.ipynb`)
```bash
# data/ is gitignored: attach the Kaggle datasets (mimic-cxr-448 + m3/m4 labels), not the repo clone.
pip install hi-ml-multimodal          # BioViL-T

# 1) worklist = manifest images ∪ pairs prior_image_id
python scripts/1-build_worklist.py --images-root <mimic-cxr-448> \
       --manifest <m3_labels>/manifest.jsonl --pairs <m4_labels>/m3_pairs.jsonl --out worklist.jsonl

# 2) GO/NO-GO: frozen-encoder sanity + alignment BEFORE the big run (both must PASS)
python scripts/3-verify_features.py --images-root <mimic-cxr-448> --features-root features \
       --manifest <..>/manifest.jsonl --pairs <..>/m3_pairs.jsonl --labels-dir <m3_labels>

# 3) extract (resumable; flushes to Drive, frees local). Re-run after a dead session.
python scripts/2-extract_features.py --worklist worklist.jsonl --out-dir features \
       --remote dhint:CHEX-DATA/biovilt_features --device cuda --batch 48

# 4) verify the cache (structure/naming/coverage + alignment overlay)
python scripts/3-verify_features.py --features-root features --labels-dir <m3_labels> \
       --images-root <mimic-cxr-448> --manifest <..>/manifest.jsonl --pairs <..>/m3_pairs.jsonl
```

## Notes
- **Feature source** = `config.FEATURE_SOURCE="backbone"` → `img_embedding` [512] + `patch_embeddings`
  [512,14,14] (the 512-d PRE-projection features), **not** the 128-d `projected_*` head. `C` is
  auto-detected; the whole cache shares one `C` (512).
- **Sync/resume** uses the same rclone **OAuth** remote as phase_2 (`GDRIVE_TOKEN` secret — *not* a
  service account, which 403s on My-Drive upload). `done_ids` = `rclone lsf` ∪ local staging.
- **Priors are included** in the worklist because M4 (temporal) needs the prior image's features too.
- `docs/*.pt` / `hoangtimothy/biovilt-features` are **fine-tuned** BioViL-T features — deliberately
  NOT reused. We run the frozen encoder for a clean, reproducible, self-consistent cache.
