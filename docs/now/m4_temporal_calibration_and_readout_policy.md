# M4 Temporal Progression, Calibration, and Readout Policy

- Status: **implementation audit complete; policy decisions and repairs pending**
- Audit date: 2026-07-28
- Applies to: final detector-box M4 training, validation calibration, test
  evaluation, M4 inference, and M5 temporal readout

This is the M4 counterpart of
`docs/now/m3_report_threshold_and_concept_policy.md`. It distinguishes what the
current code actually does from the intended final protocol. No final
detector-box M4 campaign should start until the blockers and decisions in this
record are resolved.

## 1. Stage ownership

M4 is a temporal classifier staged on a frozen M3. It does not determine
whether a disease exists from scratch. The disease axis is the fixed 14-label
CheXpert/M3 space; M4 predicts one of three temporal classes for each
`(region, disease)` cell:

~~~text
0 stable | 1 improved | 2 worsened | -100 unknown/not supervised
~~~

The intended boundary is:

~~~text
freeze accepted detector-box M3
  -> build a cache tied to the exact M3 checkpoint and detector arrays
  -> select M4 hyperparameters using silver internal validation only
  -> freeze M4 checkpoint and temporal readout policy
  -> fit report-facing temporal calibration on validation only
  -> evaluate frozen M4/readout on internal test and MS-CXR-T
  -> M5 consumes immutable M3 + M4 + calibration artifacts
~~~

M4 training must not use test or MS-CXR-T to choose coefficients, architecture,
checkpoint, thresholds, or lead-region rules. M5 must not fit temporal
temperatures or gates.

## 2. Temporal labels

`phase_4/scripts/1-labels.py` builds `progression.npy [N,29,14]` from the
current study's Chest ImaGenome `comparison_cues`. The cues already describe
the comparison with the prior study. A positive concept/finding attached to a
cue is mapped to its CheXpert disease.

Important rules:

- hedged source phrases are excluded;
- only positive mapped findings receive a temporal label;
- no cue means `-100`, not stable;
- within one `(region,disease)` cell, conflicting cues resolve by
  `worsened > improved > stable`;
- training/evaluation exclude `-100`; and
- a cell is intended to be valid only when its anatomical region is available
  in both current and prior studies.

These labels are NLP-derived silver supervision. They are suitable for model
development but are agreement targets, not human clinical ground truth.
MS-CXR-T supplies human image-level temporal labels for five diseases and must
remain separated from hyperparameter selection if used as the final external
test.

## 3. Current intended main architecture

The paper launcher defines the main row as **TempFuse + Disease-Delta**, not the
FTCB concept-delta model:

~~~text
current BioViL-T patches [196,512]
prior BioViL-T patches   [196,512]
  -> current-to-prior cross-attention fusion
  -> current detector bbox-guided pooling into 29 region vectors
  -> concatenate, per region:
       fused temporal feature [512]
       current M3 disease logits [14]
       prior M3 disease logits [14]
       current-prior M3 disease-logit delta [14]
  -> shared MLP
  -> logits [29,14,3]
~~~

The M3 disease logits are continuous inputs. M3 display thresholds are not used
inside M4. The head is a free MLP, so the main M4 is not a temporal concept
bottleneck and does not support a faithful concept-level "why changed" claim.

### 3.1 M3-to-M4 disease-logit contract

The current main protocol intentionally passes the complete 14-dimensional M3
disease-logit vector for current, prior, and their difference into M4. It does
**not** hard-mask the vector to diseases that passed an M3 report threshold, and
it does not select diseases using their aggregate validation F1. This is a
training/inference contract, not a report decision:

- M4 trains on all valid `(region, disease)` temporal cells, including diseases
  for which M3 is weak on average.
- A validation F1 is a property of a disease/model/split, not evidence that a
  particular study contains that disease. It therefore cannot be used as a
  per-study feature selector.
- Hard masking would create an error cascade and a train/inference mismatch:
  an M3 miss would remove the temporal model's input before M4 could learn from
  visual evidence, while an M3 false positive could still be carried as a hard
  positive feature.

The all-logit main row must therefore be audited rather than silently replaced.
The following separate ablations are permitted and must not overwrite the main
M4 campaign:

1. **Visual-only M4:** remove current/prior/delta M3 disease logits.
2. **Soft-confidence M4.5:** retain all 14 channels but attenuate each disease's
   M3 channels with the frozen, calibrated per-study M3 probabilities. The same
   transformation must be used in train, validation, test, and inference.
3. **Hard-gated M4.5:** zero a disease channel only for report-time analysis;
   this is not a primary training protocol because it is brittle and cannot
   recover an M3 miss.

M4.5 is selected only after comparing temporal accuracy, selective precision,
coverage, and false temporal-call rate. It must not change the existing M4
checkpoint or headline results.

There are two explicit operating modes:

- **Benchmark mode:** emit one of the three temporal classes for every external
  benchmark item, without M3 display gating, so comparison with MS-CXR-T,
  BioViL-T, and CoCa-CXR is apples-to-apples.
- **Report mode:** first require the M3 disease gate on the current/prior
  study, then require regional M3 relevance and M4 temporal confidence gates.
  A disease that fails these gates cannot receive a temporal sentence in M5,
  even if raw M4 logits have a high argmax.

The repository also contains `FTCBTKAN` and
`FaithfulTemporalConceptHead`, which route signed M3 concept deltas through a
masked non-negative map. That branch is excluded from the final launcher after
the concept-severity assumption produced poor performance and was judged
clinically unreliable. It is an appendix/negative result, not the shipped
temporal model.

The defensible M4 explanation claim is therefore **where changed**, using the
regional logits, not **why changed**, using concepts.

## 4. Current training objective

The final grid currently uses:

- class-weighted cross-entropy over valid cells;
- training-only time-flip augmentation, where current/prior are swapped and
  improved/worsened are exchanged;
- a symmetric flip-consistency KL term;
- an expected ordinal-distance penalty on the clinical axis
  `improved < stable < worsened`;
- detector boxes;
- checkpoint selection by validation change-F1; and
- one seed (`42`) per coefficient-grid row.

The coefficient search is:

~~~text
flip-KL = {0.025, 0.050, 0.075}
distance = {0.350, 0.500, 0.650}
~~~

The selected grid run becomes the main VERA row. The complete current launcher
contains 18 detector-box trainings (9 grid + 2 architecture + 7 additional loss
rows) and one matched GT-box oracle, for 19 unique trainings total.

Benchmark metrics operate on valid regional cells and include accuracy,
three-class macro-F1, improved/worsened change-F1, per-class/per-disease/
per-region F1, confusion matrices, QWK, opposite-direction error, mean ordinal
distance, and stable-prediction rate. Accuracy is not the selection metric.

## 5. Detector-mask blocker

The 2026-07-28 audit found that `_present_by_image()` always loaded
`present_mask.npy`. Consequently a run marked `box_source=detector` used
detector boxes but the GT region-presence mask for row validity and
supervision.

This is a protocol mismatch. A GT-present/detector-missing region has an empty
detector box; the pooling code deliberately lets an empty box attend the whole
grid to avoid NaNs, and the GT mask can then incorrectly make that whole-image
vector supervised as a named detector region.

Local audit on the current arrays:

- mask shape: `[222155,29]`;
- GT/detector cell disagreement: `400975` cells (`6.22%`);
- images with at least one disagreement: `64648` (`29.10%`);
- GT-only present cells: `344928`;
- detector-only present cells: `56047`;
- validation valid cells under current GT mask: `80510`;
- validation valid cells under detector mask: `79674`.

**Implemented 2026-08-03:** `_present_by_image(m3_labels_dir, box_source)` now
selects `present_mask_det.npy` for detector runs and `present_mask.npy` for the
GT oracle. `box_source` is passed through the region-cache, concept-cache,
TempFuse, and MS-CXR-T dataset constructors; inference and consistency reuse
those same constructors. This removes the detector-mask stop condition. The
mixed-direction/report-calibration decisions below remain separate from M4
training and benchmark evaluation.

## 6. Current output and regional aggregation

For each valid region `r`, disease `d`, and temporal class `k`, M4 emits raw
logits `o[r,d,k]`. Regional probabilities are:

~~~text
p[r,d,:] = softmax_k(o[r,d,:])
~~~

Current inference collapses the valid regions at the logit level:

~~~text
O[d,k] = logsumexp_r(o[r,d,k])
P[d,:] = softmax_k(O[d,:])
class[d] = argmax_k P[d,k]
~~~

All 14 disease readouts are exported. M5 excludes `No Finding` and
`Support Devices` from temporal reporting.

For a non-stable selected class, current inference defines the lead region as:

~~~text
lead_region = argmax_r o[r,d,selected_class]
~~~

For fixed `(d,k)`, the exact LSE regional attribution share is:

~~~text
share[r,d,k] = exp(o[r,d,k]) / sum_j exp(o[j,d,k])
~~~

These shares sum to one and are the exact derivative/contribution weights of
the LSE pooled class logit. However, the current JSON exports only the lead
region name. It does not export shares, top-1/top-2 dominance gap, entropy, or
an attribution-abstention decision. Therefore the full documented
lead-region policy is not yet implemented.

## 7. Mixed regional directions

A single disease can have different silver directions in different regions.
The current calibration converts regional labels to one disease-level target by
majority vote. The dataset-statistics document instead describes a
`worsened > improved > stable` priority collapse. These policies are not
equivalent and neither explicitly represents a mixed-direction case.

Detector-mask validation audit over `24625` usable `(study,disease)` targets:

- `1878` (`7.63%`) contain more than one regional class;
- `357` (`1.45%`) contain both improved and worsened regions;
- majority and priority collapse disagree for `909` (`3.69%`); and
- majority voting has a tie for `463` (`1.88%`).

This choice can materially change fitted temperatures, gates, reported class
frequencies, and lead-region correctness. It must be frozen before the final
validation calibration. It must not be silently inherited from the current
`counts.argmax()` implementation.

Recommended policy for discussion: keep all valid cells for regional training
and regional benchmark metrics, but fit a scalar disease-level report gate only
on validation cases whose valid regional labels agree on one direction. Treat
mixed-direction cases as lacking an unambiguous scalar calibration target. At
report time, either abstain from a single disease-level temporal sentence when
strong regional directions conflict, or explicitly support region-specific
temporal rows. Do not force a mixed case into an undocumented priority class.

## 8. Current calibration and its gaps

`phase_4/scripts/calibrate_temporal.py` currently:

1. consumes validation M4 disease-level LSE probabilities;
2. creates one majority-vote target per `(study,disease)`;
3. fits one scalar temperature per disease by grid-searching multiclass NLL;
4. fits a gate for each `(disease, selected temporal class)` as the lowest
   confidence with validation precision at least `0.90` and at least `30`
   retained predictions; and
5. excludes `No Finding` and `Support Devices`.

The following gaps prevent it from being a final-paper artifact:

- it looks for `present_mask.npy` inside `m4_labels`, where the standard
  artifact does not contain it, so its disease-level target ignores the
  current/prior detector-valid region mask;
- it uses majority targets despite the unresolved mixed-direction policy;
- it records no image IDs, patient IDs, unique-patient support, 2x2/3x3 raw
  counts, uncertainty intervals, checkpoint hash, M3 source hash, manifest
  hash, box-source hash, or prediction-file hash;
- it emits non-standard JSON `NaN` values when a gate is unsupported;
- it has no schema/resume marker and is not run by the paper launcher/notebook;
- the minimum support `30` is only an exploratory floor and has no interval
  precision guarantee; and
- M5 can still use a fixed `0.60` fallback or fall back when an artifact is
  incomplete.

The old local temperature/gate JSON was fitted from a GT-box oracle and is not
valid for the detector-box campaign.

## 9. Recommended final temporal calibration contract

Subject to approval, the M4 post-training stage should mirror the strict M3
contract:

1. Freeze the accepted detector-box M4 checkpoint and the disease-level
   aggregation formula.
2. Produce a full schema-versioned validation dump containing image/prior/
   patient IDs, regional logits/probabilities/targets/masks, disease-level
   logits/probabilities/targets, lead shares, and provenance.
3. Exclude `-100` and any region unavailable under the exact box-source mask.
4. Resolve or abstain on mixed regional targets according to one preregistered
   rule.
5. Fit one temperature per disease on validation only.
6. Fit separate report gates per `(disease, stable/improved/worsened)` with a
   target agreement precision, call support, unique-patient support, Wilson
   intervals, and patient-cluster bootstrap intervals.
7. Unsupported classes receive null gates and never use a pooled/fixed
   fallback in final reports.
8. Export lead-region shares and apply an explicit dominance/dispersion gate
   before naming one region; otherwise omit localization or label the evidence
   diffuse/multifocal.
9. Freeze artifact hashes before internal test and MS-CXR-T evaluation.
10. Make M5 verify exact M4 checkpoint, M3 source, manifest, detector arrays,
    box source, aggregation, and calibration hashes.

The target `0.90` and support floor `30` may remain exploratory defaults, but
the paper must report achieved numerator/denominator, coverage/abstention, and
uncertainty. They are not proof of clinical reliability.

## 10. M5 temporal semantics

M5 currently combines M3 and M4 rather than rendering M4 blindly:

- no usable prior M3 record means no temporal language;
- current M3 present + prior M3 absent becomes `new`;
- current M3 absent + prior M3 present becomes `resolved`;
- these presence-crossing states use M3 confidence, not M4 confidence;
- otherwise a gated M4 class yields `stable`, `improved`, or `worsened`; and
- M4 confidence is the calibrated selected-class probability.

Thus `new/resolved` are not fourth/fifth M4 classes. They are deterministic M3
state transitions in M5. This distinction must remain explicit in Methods,
tables, calibration, and provenance.

## 11. Provenance and resume gaps

Current strengths:

- M3 region-cache generation resumes at individual image files;
- M4 training writes `last.pt` and syncs each epoch;
- grid rows and completed eval JSON files can be skipped; and
- Kaggle can run independent rows on two T4 GPUs.

Required hardening:

- the cache marker currently records only M3 checkpoint hash and box-source
  name; it must also hash the M3 manifest, selected boxes/present-mask arrays,
  and detector provenance;
- M4 checkpoints must embed the cache/provenance hashes;
- inference/calibration outputs must embed the M4 checkpoint hash and all
  upstream hashes;
- validation/test inference should be resumable or atomically sharded;
- diagnostics must sync after each completed row/split rather than only after
  a whole notebook stage; and
- a calibration success marker must be written only after all JSON/CSV outputs
  pass hash validation.

## 12. Decisions required before implementation

1. **Main architecture:** keep TempFuse + Disease-Delta as the final M4 and
   retain FTCB only as a negative/appendix result? Recommended: yes.
2. **Mixed regional targets:** consensus-only scalar calibration with abstention
   on strong directional conflict, or a documented majority/priority collapse?
   Recommended: consensus-only for scalar claims.
3. **MS-CXR-T role:** final untouched external test, or calibration/development
   data? Recommended: final external test only; never tune on it.
4. **Stable statements:** allow `unchanged` only when its own disease/stable
   validation gate passes? Recommended: yes.
5. **Lead-region wording:** name one region only after a validated dominance
   rule, otherwise omit location or say diffuse/multifocal? Recommended: yes.
6. **Selected-model repeats:** rerun the chosen main configuration with
   additional seeds after the one-seed grid? Recommended: at least three total
   seeds for the final main estimate if compute permits.

## 13. Acceptance criteria before final test

- Detector runs use detector masks everywhere; GT masks appear only in the
  explicit oracle.
- Every M4 cache/checkpoint/prediction/calibration artifact has complete
  upstream provenance.
- Hyperparameter and checkpoint selection use internal validation only.
- Disease-level target aggregation and mixed-direction handling are explicit.
- Calibration uses exact valid-region masks, patient support, uncertainty, and
  null unsupported gates.
- Final M5 has no temporal fixed-threshold fallback.
- Lead-region shares and abstention are exported and auditable.
- Internal test and MS-CXR-T are evaluated only after policy freeze.
- Regional silver metrics are described as agreement with weak labels.
- MS-CXR-T is described as image-level external temporal evaluation, not
  region-level localization validation.
