# phase_5 — Module 5: faithful assembler

Turns M3 (per-region disease) + M4 (progression) predictions into a report that is a **readout of a
verifiable table — no diagnosis is generated**. Implements `docs/VERA_phase_3_4_5_spec.md` §5.
Mostly deterministic (CPU); runs locally on the `infer.py` JSONL from M3/M4.

```
m3_pred.jsonl ─┐
               ├─ assemble (tiers 1-4) ─► realize (template) ─► verify (round-trip + coverage)
m4_pred.jsonl ─┘        provenance per finding                       deterministic, NOT an LLM
```

## The six tiers (which are built vs pluggable)
| tier | what | status here |
|------|------|-------------|
| 1 structured core | M3 region/image logits + M4 progression → assert/hedge/abstain/omit | **built** (`assemble.py`) |
| 2 grounding "where" | lead region per disease + **α cells** in provenance + **29-region coverage map** | **built** |
| 3 calibration + abstention | **per-class temperature** (`calibrate.py`) + τ bands → hedge / abstain | **built** (T=1 until fit) |
| 4 temporal guard | no prior ⇒ NO temporal language *by construction*; else readout of M4 argmax | **built, structural** |
| 5 realize | template (faithful default) **·** constrained paraphraser (LLM) | template built; LLM = interface |
| 6 verify | round-trip label re-extraction + coverage | **built** (hard-match; CheXbert = interface) |

**Status bands (tier 1+3):** `assert` (p≥τ_assert) · `hedge` ("there may be…", p≥τ_uncertain) ·
`abstain` ("…cannot be excluded", p≥τ_abstain, defer to radiologist) · `omit` (silent).

## Faithfulness, by construction
- **Temporal guard:** a progression clause is emitted *only* when an M4 cell exists for the image.
  No prior → no M4 row → there is **no code path** that produces temporal words (`assemble.temporal_of`
  returns `None`). This is the paper's "temporal-halluc = 0 by construction".
- **Verifier is deterministic** (`verify.extract_labels`) — hard string match now, the single seam
  where **CheXbert/RadGraph** plugs in. **Never an LLM** (an LLM verifier hallucinates, defeating the
  point). It catches `out_of_table` (added findings), `coverage_miss` (dropped asserts), `temporal_halluc`.
- **Paraphraser is prose-from-table** (`paraphrase.py`): may only rephrase listed findings; its output
  is re-verified and **falls back to the template** if it drifts. Default backend = None = template.
- **Provenance:** every finding carries a pointer back to its source cells (`m3_image_prob`,
  `m3_region_probs`, `m4`) — this is what feeds per-sentence provenance / coverage-map visualization.

## Files
| File | Role |
|------|------|
| `constants.py` | CheXpert order + progression + disease→phrase vocab (self-contained) |
| `config.py` | τ thresholds, temperature, realize mode |
| `assemble.py` | tiers 1-4 + template realize; report object w/ provenance + coverage map |
| `calibrate.py` | tier 3 per-class temperature fit (BCE/ECE on a val split) → `m5_temperature.json` |
| `verify.py` | tier 6 deterministic round-trip + coverage (CheXbert seam) |
| `paraphrase.py` | tier 5 constrained LLM paraphraser interface (default = identity) |
| `run.py` | CLI: join m3/m4 pred JSONL (+temperature) → reports.jsonl + faithfulness stats |
| `demo.py` | self-contained synthetic demo (no model needed) |

## Run
```bash
python phase_5/demo.py                               # synthetic, no data needed
python phase_5/run.py --m3-pred data/m3_pred.jsonl \ # real M3/M4 predictions
    --m4-pred data/m4_pred.jsonl --out data/m5_reports.jsonl
# omit --m4-pred entirely -> every report is single-image (no temporal language), guaranteed
```

## Output (one JSON line per image)
```
{ image_id, prior_image_id, has_prior, normal,
  findings: [ {disease, status, prob, lead_region, regions, temporal, text,
               provenance:{m3_image_prob, m3_lead_region, m3_region_probs, m3_cells, m4}} ],
  coverage_map: {region: "abnormal"|"normal"|"not_assessable"},   # all 29
  text, verify: {ok, out_of_table, coverage_miss, temporal_halluc, spoken, extractor} }
```

## Calibrate (run after M3 inference; CPU, no GPU)
```bash
python phase_5/calibrate.py --m3-pred data/m3_pred_val.jsonl --split val --out data/m5_temperature.json
python phase_5/run.py --m3-pred ... --m4-pred ... --temperature data/m5_temperature.json --out ...
```

## TODO when the externals arrive
- swap `verify.extract_labels` → CheXbert/RadGraph (keep the same return type — NEVER an LLM).
- wire a real `backend` into `paraphrase.paraphrase` (re-verify + fallback already handled).
- visualization (provenance-per-sentence, the 29-region coverage map, change-ledger) reads these
  JSON lines directly — `m3_cells` + `coverage_map` are already emitted for it.
