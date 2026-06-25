# phase_3 — Module 3: C-KAN (per-region concept + disease)

BioViL-T grid → **attention-pool 29 anatomical regions** → **69 concepts** (43 finding +
10 disease + 12 tubes + 4 device) → **14 CheXpert**, with a faithful Chain-of-Explanation
intermediate (the concepts). Per-region outputs feed M4 (temporal) and M5 (report).

```
features 196×512 ─pool→ 29×512 ─concept head→ 29×69 ─disease head→ 29×14 ─agg→ 14 (image)
```

## Three directions (config.HEAD_MODE / --mode)
| mode | disease head input | trade-off |
|------|--------------------|-----------|
| **C** | region features | direct baseline = accuracy ceiling |
| **A** | 69 concepts only | pure concept bottleneck → faithful causal explanation, small accuracy cost |
| **B** | features ⊕ 69 concepts (leaky) | hybrid → keeps accuracy, partial explanation |
Run all three on the same data to measure the accuracy-vs-explainability trade-off.

## Files
| File | Role | Needs GPU |
|------|------|-----------|
| `constants.py` | 29 regions + 69 concepts + 14 CheXpert + maps (from `data/m3_concept_space.json`) | — |
| `config.py` | paths + hyperparams + toggles (mode, global, head type, agg) | — |
| **Prep (run local, before Kaggle):** | | |
| `labels.py` | scene graphs → per-region concept/CheXpert label arrays + manifest | no |
| `pairing.py` | prior↔current pairs from the CXR metadata (for M4) | no |
| **Features (M1 — EXTERNAL):** | | |
| _(BioViL-T extraction)_ | **implemented & run separately by a collaborator** — not in this repo | yes |
| `features.py` | loader for the cached grids = the **format contract** with M1 | — |
| **Model:** | | |
| `pooling.py` | attention-pool 196→29 (each region attends the full grid; optional global) | — |
| `heads.py` | MLP heads now; FastKAN swap is one config word later | — |
| `model.py` | `CKAN` — modes A/B/C + region→image attention aggregation | — |
| `losses.py` | masked-BCE (ignores -100) for concept + region + image | — |
| `dataset.py` | join features + labels by image_id, split filter | — |
| `train.py` / `eval.py` / `infer.py` | train loop / AUC metrics / per-image JSON for M4-M5 | yes |

## Run order
```bash
# 0) one-time prep (local, no GPU) — upload the outputs to Kaggle
python phase_3/labels.py  --scene-root <chest-imagenome> --out-dir data/m3_labels
python phase_3/pairing.py  # -> data/m3_pairs.jsonl

# 1) features (M1, EXTERNAL): your collaborator extracts BioViL-T grids and gives you the
#    cache <image_id>.npy [197,C]. phase_3 only LOADS it (features.py is the format contract).

# 2) train each direction + eval
python phase_3/train.py --mode C --labels-dir data/m3_labels --features-root <feat> --device cuda
python phase_3/train.py --mode A ...
python phase_3/train.py --mode B ...
python phase_3/eval.py  --ckpt <run>/m3_B/best.pt --split test
python phase_3/infer.py --ckpt <run>/m3_B/best.pt --split test --out m3_pred.jsonl
```

## Notes
- **Labels:** `1` positive / `0` negative / `-100` not-mentioned (masked-BCE; never collapse -100→0).
  Per-region CheXpert is **derived** from concepts via the map in `constants.py`.
- **Global node:** off by default — each region query already attends the whole grid.
  BioViL-T's `projected_global_embedding` is cached as row 0 and used as the 30th region
  only if `--use-global`.
- **M4 hook:** `model.py` returns `region_feats [B,29,C]`; T-KAN consumes `[prior; curr; curr−prior]`.
- Boxes used for training = scene-graph GT (MIMIC). For CheXplus/NIH, feed detector boxes
  into a dataset variant (same array shapes).
- Feature caching dominates storage; if you switch to fixed ROI-pooling you could pre-pool
  to `29×C` (~7GB) instead, but that gives up the learned attention pool.
