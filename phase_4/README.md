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

## Files
| File | Role | Needs GPU |
|------|------|-----------|
| `constants.py` | 29 regions + 14 CheXpert + concept→disease map + 3 progression classes | — |
| `config.py` | paths + hyperparams | — |
| **Prep (local, no GPU):** | | |
| `labels.py` | scene-graph `comparison_cues` → `progression.npy [N,29,14]` (0/1/2/-100) + manifest | no |
| **Bridge (GPU, once):** | | |
| `../phase_3/precompute_regions.py` | freeze M3 → cache region features + logits for every image | yes |
| **Model:** | | |
| `dataset.py` | pair curr↔prior, serve cached tensors + progression target | — |
| `heads.py` / `model.py` | MLP T-head (KAN ablation) → `29×14×3` | — |
| `losses.py` | masked, class-weighted CE (3 classes; "stable" dominates) | — |
| `train.py` / `eval.py` / `infer.py` | train (Drive-resumable) / **macro-F1** + change-only F1 / change-ledger JSON for M5 | yes |

## Run order
```bash
# 0) prep (local, no GPU) — progression targets; pairs come from phase_3/pairing.py
python phase_4/labels.py --scene-root <chest-imagenome> --out-dir data/m4_labels

# 1) bridge (GPU, once) — needs the trained M3 ckpt + the BioViL-T feature cache
python phase_3/precompute_regions.py --ckpt <run>/m3_A/best.pt \
    --labels-dir data/m3_labels --features-root <feat> --out-dir data/m3_region_cache

# 2) train + eval + infer
python phase_4/train.py --region-cache data/m3_region_cache --m3-labels-dir data/m3_labels \
    --m4-labels-dir data/m4_labels --pairs data/m3_pairs.jsonl --device cuda
python phase_4/eval.py  --ckpt <run>/m4/best.pt --split test
python phase_4/infer.py --ckpt <run>/m4/best.pt --split test --out m4_pred.jsonl
```

## Notes
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
