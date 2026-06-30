# phase_3 — Module 3: C-KAN (per-region concept + disease)

BioViL-T grid → **bbox-masked attention-pool 29 anatomical regions** → **neck 512→128** →
**69 concepts** (43 finding + 10 disease + 12 tubes + 4 device) → **14 CheXpert**. The
region-level path is fused with a **global head** through a learned gate for relational findings.
Per-region outputs feed M4 (temporal) and M5 (report). Implements `docs/VERA_phase_3_4_5_spec.md`.

```
features 196×512 ─pool(mask bbox)→ 29×512 ─concept→ 29×69 ─disease→ 29×14 ─agg─┐
                                  (neck off; 512 kept)                          ├gate→ 14 (image)
                            global vec ─GlobalHead→ 14 ───────────────────────────┘
```

## Three directions (config.HEAD_MODE / --mode) — spec 3.3 letters
| mode | disease head input | faithfulness | trade-off |
|------|--------------------|--------------|-----------|
| **A** Direct | region features | **where-faithful, unconditional** (safe fallback) | accuracy ceiling, no "why" |
| **B** CBM | 69 concepts only | the **only** "why"-faithful path (if it passes 3.4 tests) | small accuracy cost |
| **C** Hybrid | features ⊕ 69 concepts (leaky) | **CBM-leakage risk** — concepts may be decorative | highest accuracy, must pass leakage test |

Run all three; **faithfulness numbers (not accuracy) decide which gets the "why" claim** — see
`scripts/6-faithfulness.py` (spec 3.4).

## Layout (mirrors phase_2)
```
phase_3/
  src/         importable libs (clean names — imported across modules, so NOT numbered)
  scripts/     numbered run-order entries (1-… 8-…); each self-inserts ../src on sys.path
  notebooks/   phase3_kaggle.ipynb
```
Library modules cannot be numbered (`import 4-dataset` is invalid), so the executable scripts that
import them live in `scripts/` with `N-` prefixes and the libs stay clean-named in `src/` — exactly
the phase_2 `src` vs `scripts` split.

## Files
**`src/` — libraries (no GPU, imported, not run directly):**
| File | Role |
|------|------|
| `constants.py` | 29 regions + 69 concepts + 14 CheXpert + maps (JSONs bundled alongside) |
| `config.py` | paths + hyperparams + toggles (mode, neck, mask, global head, pos_weight, agg) |
| `features.py` | loader for the cached grids (`.pt`/`.npy`) = the **format contract** with M1 |
| `pooling.py` | attention-pool 196→29, **masked to each region's bbox**, returns α (grounding) |
| `heads.py` | MLP heads now; FastKAN swap is one config word later |
| `model.py` | `CKAN` — neck + modes A/B/C + region→image agg + global-head gate fusion |
| `losses.py` | masked-BCE (ignores -100) + RADAR log-scale pos_weight for imbalance |
| `dataset.py` | join features + labels (+boxes) by image_id, split filter |
| `eval.py` | macro-F1 + AUC metrics + `evaluate()` (imported by train/faithfulness; CLI via `5-eval.py`) |

**`scripts/` — run-order entries:**
| File | Role | Needs GPU |
|------|------|-----------|
| `1-labels.py` | scene graphs → per-region concept/CheXpert label arrays + boxes + manifest | no |
| `2-pairing.py` | prior↔current pairs from the CXR metadata (for M4) | no |
| `3-boxes_from_pred.py` | YOLO predictions → detector boxes aligned to the manifest (`boxes_det.npy`) | no |
| `4-train.py` | train loop (checkpoint on val image macro-F1) | yes |
| `5-eval.py` | **macro-F1** + AUC report (thin CLI over `src/eval.py`) | yes |
| `6-faithfulness.py` | spec 3.4 tests: go/no-go concept-from-image, intervention (B), leakage (C) | yes |
| `7-infer.py` | per-image JSON for M4/M5 (+α cells) | yes |
| `8-precompute_regions.py` | freeze M3, dump per-image region features + logits cache for M4 | yes |
| `dataset_stats.py` | dataset statistics for the paper (utility, unnumbered) | no |

**Features (M1 — EXTERNAL):** BioViL-T extraction is **implemented & run separately by a
collaborator** (not in this repo). It writes one `<image_id>.pt` (or `.npy`) `[197,C]` per image;
`src/features.py` only LOADS it.

## Run order
> **STATUS (2026-06-29): phase_3 training is DEFERRED** (waiting on M1 BioViL-T features). The only
> step being done now is the **detector-box prerequisite** — step 0b (`phase2_infer_boxes.ipynb` →
> `predictions.jsonl`). Steps 0/0b-align/1/2+ below are the full order for when phase_3 resumes;
> re-run `scripts/1-labels.py` then (it now hedge-masks). `scripts/3-boxes_from_pred.py` only runs once the manifest exists.

```bash
# 0) one-time prep (local, no GPU) — upload the outputs to Kaggle
python phase_3/scripts/1-labels.py  --scene-root <chest-imagenome> --out-dir data/m3_labels
python phase_3/scripts/2-pairing.py  # -> data/m3_pairs.jsonl

# 0b) detector boxes (BOX_SOURCE="detector"): run YOLO over all MIMIC (GPU; Kaggle notebook
#     phase_2/.../phase2_infer_boxes.ipynb pulls best.pt from Drive), then align to the manifest:
python phase_2/scripts/yolo/5-infer_yolo.py --weights best.pt --source <mimic-448> --out pred --no-per-image
python phase_3/scripts/3-boxes_from_pred.py --pred pred/predictions.jsonl \
       --manifest data/m3_labels/manifest.jsonl --out-dir data/m3_labels  # -> boxes_det.npy

# 1) features (M1, EXTERNAL): your collaborator extracts BioViL-T grids and gives you the
#    cache <image_id>.pt|.npy [197,C]. phase_3 only LOADS it (src/features.py is the format contract).

# 2) train each direction (A=safe fallback first), eval, then DECIDE by faithfulness
python phase_3/scripts/4-train.py --mode A --labels-dir data/m3_labels --features-root <feat> --device cuda
python phase_3/scripts/4-train.py --mode B ...
python phase_3/scripts/4-train.py --mode C ...
python phase_3/scripts/5-eval.py         --ckpt <run>/m3_B/best.pt --split test
python phase_3/scripts/6-faithfulness.py --ckpt <run>/m3_B/best.pt --split val   # B: intervention test
python phase_3/scripts/6-faithfulness.py --ckpt <run>/m3_C/best.pt --split val   # C: leakage test
python phase_3/scripts/7-infer.py        --ckpt <run>/m3_A/best.pt --split test --out m3_pred.jsonl --topk-cells 3
```

## Notes
- **Letters follow the spec** (A=Direct safe fallback, B=pure CBM, C=Hybrid). This is the
  intended run order; do not confuse with any earlier C/A/B labelling.
- **Headline metric = macro-F1 + per-class** (spec 3.6); checkpoint selection is on val image-F1.
  AUC is reported alongside. Accuracy is deliberately not used (majority class dominates).
- **bbox-masked pooling** (`MASK_BBOX=True`): each region query attends only its bbox cells, so
  α is a faithful within-region "where" signal. Absent boxes fall back to full-grid (no NaN).
- **Neck DISABLED by choice** (`NECK_DIM=None`): we keep the full **512-d** region feature
  (richer signal). It is the contract shared with M4 → `region_in_dim = 512×3 + 14×2 = 1564`
  (still light). Set `NECK_DIM=128` to re-enable the neck (→ `412`).
- **Global head + gate** (`USE_GLOBAL_HEAD`): relational findings (cardiomegaly, diffuse edema,
  low lung volumes) come from a GAP/global vector, fused per-disease via `g=σ(gate(global))`.
- **Imbalance** (`USE_POS_WEIGHT`): RADAR log-scale `α_i = log(1+|D|/pos_i)` on every BCE term.
- **Labels:** `1` positive / `0` negative / `-100` not-mentioned (never collapse -100→0).
  Per-region CheXpert is **derived** from concepts via the map in `constants.py`.
- **M4 hook:** `model.py` returns `region_feats [B,29,512]` (128 if neck on) and `region_attn [B,29,196]`.
- **Boxes (B1, see `docs/VERA_methodology_concerns.md`):** `config.BOX_SOURCE` (default
  **`"detector"`**) selects the ROI-pool box source — YOLO detector boxes (`boxes_det.npy`, same
  source at train & launch) vs silver **GT boxes** (`boxes.npy`). Build the detector boxes with
  `phase_2/scripts/yolo/5-infer_yolo.py` → `phase_3/scripts/3-boxes_from_pred.py` (aligned to the
  label manifest). Flip to `--box-source gt` (train/eval) for the **gold-vs-detector oracle ablation**.
- **Uncertainty (shared with M2/M4):** a finding asserted in a HEDGED sentence ("possible", "no
  definite", "cannot exclude" — `hedge.py::is_hedged`) is **masked** at M3 (`-100`, not trained as a
  confident finding). Detected from silver `phrases` or the LLM's `uncertainty_cues`, so the silver
  and LLM-pseudo paths mask identically.
