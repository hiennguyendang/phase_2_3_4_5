# M3 Report Threshold and Concept-Evidence Policy

Status: **design approved; implementation not started**  
Decision date: 2026-07-28  
Applies to: final detector-box M3, its validation artifacts, and M5 report readout

This document is the implementation contract to follow before changing the M3
Kaggle notebook or report-generation code. It separates model training,
post-hoc calibration, benchmark metrics, and report display. No statement in
this document means that the corresponding code path is already wired.

## 1. Ownership and stage boundary

Threshold fitting belongs to the **M3 post-training evaluation/calibration
stage**, because it requires frozen M3 predictions and validation labels. M5
must only consume the resulting immutable artifact.

~~~text
M3 train on train
  -> freeze best.pt
  -> M3 inference on validation
  -> fit/audit thresholds and concept gates using validation labels
  -> freeze threshold + gate artifacts
  -> M3 inference on test/report cases
  -> M5 applies frozen artifacts and assembles text
~~~

M5 must never:

- inspect validation or test ground truth while assembling a report;
- optimize a threshold;
- silently replace an unsupported pair-specific threshold with another
  operating point; or
- reinterpret `unknown` as `absent`.

## 2. Current implementation gap

The current M3 model already emits image disease logits `[B,14]`, regional
disease logits `[B,29,14]`, and concept logits `[B,29,69]`.

The current standard `phase_3/scripts/5-eval.py` path:

- computes image diagnostics per disease;
- pools all detected regions for its top-level regional disease diagnostics;
- reports named-region metric breakdowns at a fixed threshold of `0.5`; and
- does **not** create final report thresholds for every `(region, disease)`.

`phase_3/scripts/diagnostics_from_pred.py` and
`phase_3/scripts/export_thresholds.py` contain parts of a pair-specific route,
but the final M3 Kaggle notebook does not yet run the complete route. Therefore
the repository must not currently claim that the Kaggle campaign automatically
produces final pair-specific report thresholds.

## 3. Disease outputs and decision states

For every detector-present region, M3 produces the same 14-dimensional
CheXpert probability vector. This is a multi-label vector, not a single-class
choice and not a region-specific disease vocabulary.

The final report uses two thresholds:

~~~text
p <= tau_minus                 -> absent
tau_minus < p < tau_plus       -> unknown / abstain
p >= tau_plus                  -> present
~~~

The middle band is a real internal state. It is omitted from prose, not turned
into a negative.

Threshold scope:

- image-level thresholds are per disease;
- report-facing regional thresholds are per `(named region, disease)`;
- a threshold is valid only for the checkpoint, label manifest, box source,
  split, and calibration policy recorded in its provenance; and
- an unsupported pair has null thresholds and must abstain.

Pooling all regions into one disease-wide regional threshold may remain a
benchmark diagnostic, but it is not the final report operating point. Score
distributions and label support differ materially by anatomical region.

## 4. Fitting disease thresholds

Thresholds are not neural-network parameters. They are fitted after `best.pt`
is frozen and do not update M3 weights.

For each image-level disease or eligible `(region, disease)` pair:

1. Collect frozen validation probabilities and explicit targets only.
2. Exclude target `-100`; it means unknown, not negative.
3. Sweep candidate thresholds on validation. The first implementation may keep
   the current reproducible grid `0.01, 0.02, ..., 0.99`.
4. Fit a present threshold and an absent threshold separately.
5. Store raw `TP`, `FP`, `TN`, and `FN` counts, point estimates, support,
   coverage, and uncertainty intervals.
6. Freeze the artifact before test evaluation or report generation.

### 4.1 Present threshold

The conservative report target remains:

- validation PPV at least `0.90`;
- validation specificity at least `0.90`; and
- sufficient retained present calls and explicit negative labels.

Among qualifying candidates, select the lowest threshold to maximize retained
present coverage. This differs from the benchmark threshold that maximizes F1.

### 4.2 Absent threshold

The conservative report target remains:

- the threshold must be below the selected present threshold;
- validation NPV at least `0.95`; and
- sufficient retained absent calls.

Among qualifying candidates, select the largest threshold to maximize absent
coverage. If no candidate qualifies, explicit absence is unavailable for that
pair.

### 4.3 What the support value 30 means

`30` is retained only as a preliminary small-sample rejection floor. It is not
evidence that a measured PPV, specificity, or NPV is precise. It means 30
retained validation calls for the state being assessed, not 30 epochs and not
necessarily 30 independent patients.

Examples of why `30` is insufficient as a final justification:

- `27/30 = 90%` has a wide 95% binomial interval;
- `29/30 = 96.7%` also has substantial uncertainty; and
- repeated images from one patient reduce the effective independent sample
  size further.

The implementation must therefore report uncertainty rather than treating
`n >= 30` as proof of reliability.

### 4.4 Statistical support policy to implement

The first implementation must:

- keep `30` as a configurable exploratory floor, not a paper claim;
- record both call count and unique-patient count;
- report numerator/denominator as well as percentages;
- compute two-sided 95% intervals for PPV, specificity, NPV, sensitivity, and
  call coverage;
- prefer patient-cluster bootstrap intervals because MIMIC may contain repeated
  images/studies for one patient;
- also permit Wilson score intervals as a deterministic sensitivity analysis;
- mark a pair unsupported when its explicit positive/negative labels or unique
  patients are insufficient; and
- report final, frozen-threshold performance and intervals on test without
  retuning on test.

A maximum acceptable confidence-interval width has **not yet been frozen**.
It must be selected from a validation support audit, recorded before inspecting
test results, and exposed as configuration. Until then, `30` must not be
described as a statistically sufficient sample size.

## 5. Paper metrics versus report frequency

Different operating points intentionally produce different positive-call
frequencies:

- AUC is threshold-free;
- `F1@0.5` describes a fixed benchmark operating point;
- F1-optimal thresholds describe the best validation balance of misses and
  false alarms; and
- conservative report thresholds trade coverage for more reliable statements.

The report may therefore mention fewer findings than a table evaluated at
`0.5`. This is acceptable only if the paper separates model discrimination
from report operation.

The paper must report at least:

- AUC and clearly labelled F1 convention;
- frozen-threshold PPV, NPV, sensitivity/agreement, and specificity/agreement;
- present, absent, and unknown/abstention rates;
- raw numerators and denominators plus 95% intervals;
- number of supported and unsupported `(region, disease)` pairs; and
- checkpoint, manifest, box-source, split, and threshold-artifact provenance.

Report emission frequency is not disease prevalence. In addition, silver or
weak reference labels must be described as agreement targets rather than an
unqualified clinical gold standard.

## 6. Concept bottleneck contract

The final M3 main row is a **continuous concept bottleneck**:

~~~text
region feature
  -> 69 concept logits
  -> sigmoid concept probabilities
  -> fixed concept-to-disease graph mask
     + learned non-negative allowed-edge weights
     + disease bias
  -> 14 regional disease logits
~~~

The disease head consumes continuous concept probabilities. No hard concept
threshold is inserted into the forward path. This remains a concept bottleneck
because the main disease head receives concepts rather than raw region features.
The classifier intercept/bias does not create a raw-feature bypass.

The main row's detached concept path further means that disease loss trains the
allowed concept-to-disease weights but does not rewrite the concept extractor.
Concept intervention remains meaningful: changing a concept probability and
propagating it through the head changes disease logits only along graph-allowed,
non-negative edges.

This follows the standard CBM structure in which concepts are predicted first
and the downstream task is predicted from those concepts. A bottleneck need not
be hard-binary; probabilistic/continuous concept bottlenecks are established
variants:

- Koh et al., *Concept Bottleneck Models*, ICML 2020:
  https://proceedings.mlr.press/v119/koh20a.html
- Kim et al., *Probabilistic Concept Bottleneck Models*, ICML 2023:
  https://proceedings.mlr.press/v202/kim23g.html

## 7. Concept thresholds are display gates, not disease inputs

Concept thresholding has a separate purpose: decide whether a predicted concept
is reliable enough to be shown as explanation evidence. It must not affect the
disease probability already produced by M3.

The report order is:

1. Use frozen disease thresholds to determine the disease state.
2. Only for a reportable present disease, inspect concepts from the same region.
3. Keep only concepts connected to that disease by the actual M3 graph mask.
4. Apply the concept reliability gate.
5. Rank surviving concepts by their real contribution to the disease logit,
   `concept_probability * nonnegative_edge_weight`, not merely by probability.
6. Show a small top-k set, with probability, contribution, region, and graph
   edge provenance.

Selecting a disease first for report rendering does not reverse the model's
causal computation. The disease score was already computed from the concepts.
The later selection only answers: "which reliable bottleneck variables may be
shown as evidence for this already-selected disease statement?"

If a disease is reportable but no concept passes its gate, the disease may be
shown with empty concept evidence and an explicit provenance flag. M5 must not
invent, relax, or substitute a concept merely to fill an explanation column.

## 8. Scope of concept gates

M3 computes all 69 concept probabilities for every detector-present region.
All probabilities are needed during calibration. The current inference behavior
that truncates concepts with a hard `0.5` and top-k before final gating must not
be used as the calibration source.

The final report-facing gate is per `(region, concept)` when validation support
is adequate. There are up to `29 * 69 = 2,001` such pairs, so sparsity is much
more severe than for regional diseases. The policy is therefore:

- fit and audit `(region, concept)` present gates only; reports do not make
  explicit "concept absent" statements;
- exclude `-100` labels;
- require explicit positive and negative support, unique-patient support, and
  uncertainty reporting;
- if a pair is unsupported, do not fall back silently to a pooled threshold for
  final paper reports;
- pooled per-concept thresholds may be reported as diagnostics, but not used to
  claim a specific regional concept without pair-level support; and
- it is acceptable for few or zero concepts to pass. Empty evidence is more
  faithful than a low-reliability explanation.

Concept gate quality and disease-head contribution answer different questions:

- the gate asks whether the concept prediction agrees reliably with concept
  labels; and
- the contribution asks how strongly that concept affected the M3 disease
  logit through the trained graph edge.

Both are required for visible explanation evidence.

## 9. Required artifacts

The M3 post-training stage must eventually produce:

~~~text
validation_predictions.npz
  image IDs and patient IDs
  image disease probabilities/targets
  region disease probabilities/targets/region indices
  concept probabilities/targets/region indices

m3_report_thresholds.json
  image thresholds per disease
  region thresholds per (region, disease)
  support, 2x2 counts, confidence intervals, provenance

m3_concept_gate.json
  supported (region, concept) present thresholds
  graph-independent concept reliability statistics
  support, confidence intervals, provenance

m3_threshold_audit.csv
m3_concept_gate_audit.csv
  one auditable row per candidate pair, including unsupported pairs
~~~

M5 must store the threshold and gate hashes in every generated report artifact.

## 10. Implementation sequence after approval

No notebook change should precede this decision record. When implementation is
authorized, proceed in this order:

1. Extend M3 validation prediction dumps with image/patient identity and full
   non-truncated concept scores.
2. Implement disease threshold fitting, support accounting, and uncertainty.
3. Implement concept pair gates and graph-edge contribution export.
4. Export JSON/CSV artifacts with complete provenance.
5. Add the post-training calibration stage to the M3 Kaggle notebook and Drive
   synchronization/resume behavior.
6. Make M5 consume artifacts strictly and abstain on missing/unsupported pairs.
7. Add synthetic tests for threshold boundaries, unknown handling, graph masks,
   unsupported pairs, and provenance mismatches.
8. Produce validation audits, freeze policy parameters, then run test once.

## 11. Acceptance criteria

- Threshold fitting uses validation only; test never changes policy.
- Every threshold is scoped to an exact checkpoint/manifest/box source.
- Unknown labels never become negatives.
- Unsupported pairs produce abstention, not fallback claims.
- Disease inference uses continuous concepts without display thresholds.
- Visible concepts pass both reliability and graph-edge checks.
- Concept ranking uses actual faithful-head contribution.
- Disease statements may exist without visible concept evidence.
- M5 does not learn or optimize anything.
- Paper tables distinguish benchmark performance from conservative report
  operation and report coverage/abstention with uncertainty.

