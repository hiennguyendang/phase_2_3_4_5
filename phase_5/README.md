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
| 1 structured core | M3 region/image logits + M4 progression -> present/absent/unknown | **built** (`assemble.py`) |
| 2 grounding "where" | all confident **(disease, region)** cells + **α cells** in provenance + **29-region coverage map** | **built** |
| 3 calibration + abstention | **per-class temperature** (`calibrate.py`) + validation-fitted dual thresholds | **built** (provisional until artifacts are fit) |
| 4 temporal guard | no prior ⇒ NO temporal language; calibrated M4 argmax must clear the confidence gate | **built, structural** |
| 5 realize | template (faithful default) **·** constrained paraphraser (optional) | template built; paraphrase is off by default |
| 6 verify | round-trip label re-extraction + coverage | **built** (hard-match; CheXbert = interface) |

Only `present` and `absent` are visible. The dual-threshold middle band is
internal `unknown` and is omitted; the report never renders `possible`,
`hedge`, or `abstain` states.

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
  `m3_region_probs`, `m3_concepts`, `m4`) — this is what feeds per-sentence provenance / coverage-map visualization.

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
    --m4-pred data/m4_pred.jsonl --thresholds data/m5_disease_thresholds.json \
    --out data/m5_reports.jsonl
# optional qualitative audit: append GT tables and one metadata report text
python phase_5/run.py --m3-pred data/m3_pred.jsonl --m4-pred data/m4_pred.jsonl \
    --thresholds data/m5_disease_thresholds.json \
    --ground-truth-metadata data/mimic-metadata/mimic_metadata_final.jsonl \
    --out data/m5_reports_with_gt.jsonl
# omit --m4-pred entirely -> every report is single-image (no temporal language), guaranteed
```

## Output (one JSON line per image)
```
{ image_id, prior_image_id, has_prior,
  classification: {
    image: {image_id, path, boxes:[{disease,region,state,bbox}]},
    table: [{disease,region,state,evidence,confidence,bbox}],
    text: "rule-based current text" },
  progression: {
    images: {prior:{image_id,path,boxes}, current:{image_id,path,boxes}},
    table: [{disease,change,regions,lead_region,lead_bbox,confidence}],
    text: "rule-based interval text" },
  ground_truth: {
    classification: {image,table}, progression:{images,table},
    text: "exactly one metadata report" },
  coverage_map: {region: "abnormal"|"normal"|"not_assessable"},   # all 29
  text, verify: {ok, out_of_table, coverage_miss, temporal_halluc, spoken, extractor} }
```

## Calibrate (run after M3 inference; CPU, no GPU)
```bash
python phase_5/calibrate.py --m3-pred data/m3_pred_val.jsonl --split val --out data/m5_temperature.json
python phase_5/run.py --m3-pred ... --m4-pred ... --temperature data/m5_temperature.json --out ...
```

Before report generation, fit pair-specific disease thresholds and concept
display gates from the complete schema-v2 M3 validation dump:

```bash
python phase_3/scripts/11-calibrate_report.py \
  --pred-dump artifacts/diagnostics/m3_paper_v2/m3v2_vera_graph_lse_det.val.pred.npz \
  --thresholds-json data/m5_disease_thresholds.json \
  --concept-gate-json data/m3_concept_gate.json \
  --disease-audit-csv data/m3_report_threshold_audit.csv \
  --concept-audit-csv data/m3_concept_gate_audit.csv
python phase_5/run.py --m3-pred data/m3_pred.jsonl --m4-pred data/m4_pred.jsonl \
  --thresholds data/m5_disease_thresholds.json \
  --concept-gate data/m3_concept_gate.json \
  --out data/m5_reports_with_absent.jsonl
```

Benchmark present thresholds maximize validation F1. Report-facing present
thresholds additionally require validation PPV and specificity constraints;
the absent threshold maximizes
coverage subject to NPV >= 0.95 and at least 30 validation absent calls.
The state is `present` above the positive threshold, `absent` below the negative threshold, and an
internal unknown/abstain band between them. Unknown diseases are omitted from the report rather
than rendered as a third class. The visible confidence is confidence in the reported state:
calibrated `p(disease)` for present and `1-p(disease)` for absent. Thresholds remain in provenance.
M3 inference must use `--all-region-diseases` so every confident `(disease, region)` cell can be
retained and should pass `--concept-gate` so visible evidence contains calibrated,
graph-valid concepts ranked by learned edge contribution. Absent region cells have no concepts,
but their location is still available from the
region head. Without a threshold artifact, the CLI stops by default. The fixed
0.10/0.50 fallback is available only through `--allow-fixed-threshold-fallback`
for smoke demos; final paper output must use validation-selected thresholds.
Thresholds must come from validation; do not derive them from the test split.
Missing or unsupported region/disease and region/concept pairs abstain without a
pooled fallback. M5 verifies checkpoint, label-manifest, and box-source
provenance before consuming the artifacts.

Temporal confidence is separately temperature-scaled per disease using M4
validation readouts and regional-majority validation targets. A temporal row is
rendered only when it clears its validation-fitted disease/change gate. The
fixed `0.60` gate is only the explicit no-artifact fallback; `Support Devices`
is excluded from progression.
Classification evidence is intentionally model-derived: it is the predicted,
graph-valid concept output for the disease-region cell, not text copied from the
reference report.

## Visual report preview

Render self-contained HTML pages with embedded CXRs, bbox overlays, structured
tables, rule-based text, and the separate ground-truth block:

```bash
python phase_5/render_report.py \
  --reports data/demo/m5_reports.report_candidates_v2.gt.jsonl \
  --out-dir data/demo/report_previews
```

The visible classification table uses up to two high-confidence regions per
disease so a provisional uncalibrated report remains readable. The full table
is preserved in an expandable audit block and in the source JSONL. Rows may be
assigned the CSS classes `match`, `partial`, or `mismatch` for later qualitative
comparison with ground truth. Print CSS is included for paper-oriented export.

## TODO when the externals arrive
- swap `verify.extract_labels` → CheXbert/RadGraph (keep the same return type — NEVER an LLM).
- wire a real `backend` into `paraphrase.paraphrase` (re-verify + fallback already handled).
- visualization (provenance-per-sentence, the 29-region coverage map, change-ledger) reads these
  JSON lines directly — `m3_cells` + `coverage_map` are already emitted for it.

## Methodology TODO (docs/VERA_methodology_concerns.md)
- **B4 calibration evidence:** `calibrate.py` already reports per-class ECE before/after; add a
  **reliability diagram** and treat poorly-calibrated rare classes as default-hedge.
- **B5 global-finding grounding:** relational findings (cardiomegaly, diffuse edema) come from the
  M3 GlobalHead, not a box. Label them in the report as **global grounding**, not a fake region cell.

## Current audit

Latest parsed `RUN/` + `LOGS/` summary lives in `docs/VERA_experiment_audit_roadmap.md`. Immediate
M5 work is to consume calibrated M3/M4 outputs, report verify statistics, and mark GlobalHead
findings as global-grounded rather than fake region-grounded.
