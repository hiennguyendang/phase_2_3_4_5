# VERA Confidence, Calibration, and Threshold Policy

Last audited: 2026-07-28

This note records the implemented threshold and confidence policy. Benchmark
decision thresholds and conservative report-display thresholds are distinct.
The approved but not-yet-implemented per-pair redesign is specified separately
in `docs/now/m3_report_threshold_and_concept_policy.md`.

## 1. Terms that must not be conflated

- **Raw probability/score** is the sigmoid or softmax output of a trained model.
- **Calibration** transforms a raw score so that a displayed probability better
  matches empirical correctness on held-out validation data.
- **Threshold** converts a score into a decision. VERA uses separate absent and
  present thresholds, with an internal unknown band between them.
- **Per-case confidence** is the number shown for one reported state.
- **NPV** is a dataset-level reliability metric for negative predictions. It is
  not a per-case confidence and is not itself a threshold.

For an absent call, NPV is

\[
\mathrm{NPV}=\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FN}},
\]

where the denominator contains all cases predicted absent. In the current M3
diagnostics this same quantity is exported under the name `absent_precision`,
because absence is temporarily treated as the positive class when computing the
diagnostic table.

## 2. Paper Stage 2 (repository `phase_3`): current implementation

The M3 disease head emits image logits `[B,14]` and region logits `[B,29,14]`.
Inference applies sigmoid:

\[
p_d=\sigma(z_d), \qquad p_{r,d}=\sigma(z_{r,d}).
\]

### Image-level state and confidence

`phase_5/calibrate.py` fits one scalar temperature per disease on the validation
split by minimizing binary cross-entropy. This is post-hoc fitting, not neural
network retraining:

\[
q_d=\sigma\!\left(\frac{\operatorname{logit}(p_d)}{T_d}\right).
\]

The report compares `q_d` with correspondingly transformed thresholds. The
current state and displayed confidence are:

\[
s_d=\begin{cases}
\text{present},&q_d\geq\widetilde\tau_d^+,\\
\text{absent},&q_d\leq\widetilde\tau_d^-,\\
\text{unknown},&\text{otherwise},
\end{cases}
\qquad
c_d=\begin{cases}
q_d,&s_d=\text{present},\\
1-q_d,&s_d=\text{absent}.
\end{cases}
\]

Unknown is retained internally and omitted from the report. The confidence is
therefore confidence in the displayed state, not always confidence in presence.

Historical change already made on 2026-07-21: absent rows previously risked
showing the disease-presence probability `q_d`; `phase_5/assemble.py` now shows
`1-q_d` for absence. This was a report-readout correction, not M3 retraining.

### Image and region thresholds

`phase_3/src/eval.py` retains the maximum-present-F1 threshold for benchmark
evaluation. For report display, it also fits the lowest threshold satisfying
PPV >= 0.90, specificity >= 0.90, at least 30 retained positive calls, and at
least 30 explicit negative labels. The report abstains if validation cannot
support a reliable positive display threshold. It then selects the largest
lower threshold whose validation NPV is at least
0.95 and whose predicted-absent support is at least 30, thereby maximizing
absent coverage under the reliability constraint. These defaults are exposed as
`--target-absent-npv` and `--min-absent-support`. If no candidate qualifies,
the exported absent threshold is `null` and that disease cannot produce an
explicit absent statement. The legacy `absent_precision` field is retained as
an alias of the clearer `absent_npv` field.

Image thresholds are disease-specific. The final design requires region
thresholds for each `(disease, region)` pair because pooling all 29 regions can
cause anatomically unsupported regions to inherit a disease-wide threshold.
However, the current standard `5-eval.py`/Kaggle-notebook path still pools
detected regions for its top-level regional diagnostics and reports named-region
breakdowns at `0.5`. Pair-specific diagnostics are partially supported by
`diagnostics_from_pred.py` and `export_thresholds.py`, but are not yet wired into
the final Kaggle campaign. Do not describe pair-specific threshold production
as implemented until the decision record above is completed.

Concept evidence is no longer selected unconditionally by top-k. The approved
design keeps continuous concept probabilities inside the M3 disease forward
path and applies thresholds only as post-hoc report-display gates. A visible
concept must pass validation reliability, `(region, concept)` support, and the
actual concept-to-disease graph edge; unsupported concepts are omitted rather
than replaced by pooled or low-confidence evidence. The current inference and
Kaggle paths do not yet implement this complete pair-specific gate. On the prior
detector-box validation run, zero concepts met the older pooled conservative
display gate; final values must be regenerated for the frozen detector-box M3.

### Region confidence limitation

Region rows retain raw `p_{r,d}` or `1-p_{r,d}` as `region_score` for thresholding
and provenance. The visible `confidence` column uses the calibrated image-level
disease-state confidence, so repeated regions for one disease share the same
displayed confidence. Region scores are not presented as calibrated patient-level
confidence.

Recommended final policy: fit one region-level temperature per disease using
all valid validation `(region,disease)` cells, then apply the same state-aware
confidence formula. This requires existing M3 weights plus validation inference;
it does not require retraining M3.

## 3. Paper Stage 3 (repository `phase_4`): current implementation

M4 emits three logits per `(region,disease)` cell for stable, improved, and
worsened. `phase_4/scripts/4-infer.py` currently applies a raw softmax:

\[
p_{r,d,k}=\operatorname{softmax}_k(o_{r,d,k}).
\]

`phase_4/scripts/4-infer.py` now exports disease-level LSE probabilities and
exact lead regions. `phase_4/scripts/calibrate_temporal.py` fits one temperature
per disease on validation outputs using regional-majority targets. It then fits
a gate per `(disease, selected change class)` as the lowest confidence with
precision >= 0.90 and at least 30 retained predictions. A class with no
qualifying gate emits no temporal row. The fixed 0.60 threshold is retained only
as an explicit fallback when no temporal calibration artifact is supplied.
Support Devices and No Finding are
excluded from disease progression.

Readout correction applied on 2026-07-22: `interval_changes[].confidence` now
means change confidence. For `improved`/`worsened`, it is the selected M4
calibrated softmax probability; for `new`/`resolved`, it is the minimum of current and
prior Stage-2 disease-state confidence. The current-finding table continues to
use Stage-2 disease confidence. This is a readout formula change, not M4
retraining. The currently fitted local temporal artifact uses the provisional
GT-box M4 checkpoint and must be regenerated for the final detector-box model.

The prose readout is also separated by table: `current_text` renders only the
single-image disease-region table, while `interval_text` renders only the
prior-current disease-change table. Temporal language is no longer inserted
into the single-image text.

Implemented policy:

1. Aggregate raw valid-region logits with the paper-defined LSE to obtain
   disease-level temporal logits `O[d,k]`.
2. Apply softmax over the three temporal classes.
3. Fit a scalar temporal temperature per disease on internal validation labels.
4. Fit a per-disease, per-selected-class confidence gate on the same validation split.
5. Display that calibrated selected-class probability as **change confidence**.
6. Keep **disease-state confidence** from Stage 2 as a separate field.
7. Derive the lead-region attribution from the raw LSE logits so calibration
   does not rewrite the model's attribution evidence.

This needs no M4 retraining, but it must be rerun whenever the final checkpoint
or box source changes.

## 4. What calibration does

A model can assign confidence 0.90 to many predictions while only 0.75 of those
predictions are correct. It is then overconfident. Calibration learns a simple
post-hoc mapping from raw score to empirical probability on held-out validation
data. Temperature scaling changes certainty but preserves within-disease score
ordering. A temperature above 1 softens probabilities toward uncertainty; a
temperature below 1 sharpens them.

Calibration requires labels and predictions on a held-out validation split. It
must not be fitted on the test set. It does not update BioViL-T, M3, or M4
weights. Raw sigmoid/softmax scores are produced automatically at inference;
calibration parameters and thresholds require a one-time validation fit after
the final checkpoint is frozen.

## 5. Recommended reporting of reliability

- Show calibrated per-case confidence in the clinical table.
- Report NPV, absent coverage, unknown rate, and confidence intervals in Results.
- Do not use NPV as the confidence shown for an individual patient.
- State the target NPV, minimum support, validation split, achieved NPV, and
  achieved absent coverage explicitly.
- For present and temporal display gates, also report achieved precision,
  specificity where applicable, support, and abstention coverage.

## 6. Current local validation artifacts (2026-07-22)

- M3 predictions: `data/demo/m3_pred.val.xwalk_v2.detector.jsonl`
  (22,136 validation images; detector boxes).
- M3 diagnostics: `data/demo/m3_diagnostics.val.xwalk_v2.detector.json`.
- Image temperatures: `data/demo/m5_temperature.val.xwalk_v2.detector.json`.
- Conservative disease/region thresholds:
  `data/demo/m5_disease_thresholds.val.xwalk_v2.detector.precision90.json`.
- Concept gate: `data/demo/m3_concept_gate.val.xwalk_v2.detector.precision90.json`
  (0 concepts currently allowed for report evidence).
- M4 predictions: `data/demo/m4_pred.val.kl005_dist050.gtbox.jsonl`
  (9,344 validation pairs; provisional GT-box checkpoint).
- Temporal temperatures and gates:
  `data/demo/m5_temporal_temperature.val.kl005_dist050.gtbox.json`.
- Conservative candidate reports:
  `data/demo/m5_reports.report_candidates_v2.precision90.gt.jsonl` and
  `data/demo/report_previews_precision90/`.

The M3 audit produced image macro-AUC 0.831, region macro-AUC 0.874,
concept macro-AUC 0.531, and concept macro-F1 0.171. The sparse negative-label
policy makes maximum-F1 thresholds unsuitable as a clinical display policy:
the original three candidates contained 213--334 classification rows. The
conservative per-pair policy reduces these to 6--18 rows. It still does not make
`No Finding` directly learnable because its validation labels contain no
negative examples; normal remains a strict derived state.

The current temporal gates retain only validation-supported calls. No
`improved` class reaches the 0.90 precision and 30-call support requirement.
These numbers must be regenerated after the final detector-box M4 checkpoint
is trained.

## 7. Change-control rule

Any future confidence change must be announced and logged here. The same change
must update: (1) inference/assembly code, (2) Section 3 equations and wording,
(3) Section 4 protocol, (4) Section 5 calibration results, and (5) generated
report artifacts. Until that synchronized update is made, proposed formulas are
not part of the final VERA system.
