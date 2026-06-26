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
`faithfulness.py` (spec 3.4).

## Files
| File | Role | Needs GPU |
|------|------|-----------|
| `constants.py` | 29 regions + 69 concepts + 14 CheXpert + maps (from `data/m3_concept_space.json`) | — |
| `config.py` | paths + hyperparams + toggles (mode, neck, mask, global head, pos_weight, agg) | — |
| **Prep (run local, before Kaggle):** | | |
| `labels.py` | scene graphs → per-region concept/CheXpert label arrays + boxes + manifest | no |
| `pairing.py` | prior↔current pairs from the CXR metadata (for M4) | no |
| **Features (M1 — EXTERNAL):** | | |
| _(BioViL-T extraction)_ | **implemented & run separately by a collaborator** — not in this repo | yes |
| `features.py` | loader for the cached grids = the **format contract** with M1 | — |
| **Model:** | | |
| `pooling.py` | attention-pool 196→29, **masked to each region's bbox**, returns α (grounding) | — |
| `heads.py` | MLP heads now; FastKAN swap is one config word later | — |
| `model.py` | `CKAN` — neck + modes A/B/C + region→image agg + global-head gate fusion | — |
| `losses.py` | masked-BCE (ignores -100) + RADAR log-scale pos_weight for imbalance | — |
| `dataset.py` | join features + labels (+boxes) by image_id, split filter | — |
| `train.py` / `eval.py` / `infer.py` | train loop / **macro-F1**+AUC / per-image JSON (+α cells) | yes |
| `faithfulness.py` | spec 3.4 tests: go/no-go concept-from-image, intervention (B), leakage (C) | yes |

## Run order
```bash
# 0) one-time prep (local, no GPU) — upload the outputs to Kaggle
python phase_3/labels.py  --scene-root <chest-imagenome> --out-dir data/m3_labels
python phase_3/pairing.py  # -> data/m3_pairs.jsonl

# 1) features (M1, EXTERNAL): your collaborator extracts BioViL-T grids and gives you the
#    cache <image_id>.npy [197,C]. phase_3 only LOADS it (features.py is the format contract).

# 2) train each direction (A=safe fallback first), eval, then DECIDE by faithfulness
python phase_3/train.py --mode A --labels-dir data/m3_labels --features-root <feat> --device cuda
python phase_3/train.py --mode B ...
python phase_3/train.py --mode C ...
python phase_3/eval.py         --ckpt <run>/m3_B/best.pt --split test
python phase_3/faithfulness.py --ckpt <run>/m3_B/best.pt --split val   # B: intervention test
python phase_3/faithfulness.py --ckpt <run>/m3_C/best.pt --split val   # C: leakage test
python phase_3/infer.py        --ckpt <run>/m3_A/best.pt --split test --out m3_pred.jsonl --topk-cells 3
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
- **M4 hook:** `model.py` returns `region_feats [B,29,128]` and `region_attn [B,29,196]`.
- Boxes for training = scene-graph GT (MIMIC). For CheXplus/NIH, feed detector boxes
  into a dataset variant (same array shapes).
