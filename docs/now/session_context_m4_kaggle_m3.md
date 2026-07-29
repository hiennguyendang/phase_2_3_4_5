# Session context: M4 contract and Kaggle M3 handoff

This note is a short restart point for the current discussion. The detailed
policies remain in the linked documents; this file is intentionally compact.

## Decisions discussed in this session

- Main M4 continues to train all valid `(region, disease)` temporal cells.
- Main M4 receives all 14 continuous M3 disease logits for current, prior, and
  delta. M3 report thresholds are not fed into the M4 loss.
- A visual-only M4 and a soft-confidence M4.5 are separate ablations. They do
  not replace or alter the all-logit main M4 result.
- M5 report mode gates temporal language through M3 disease/regional relevance
  and M4 confidence. External benchmark mode remains forced three-class output
  for fair comparison with MS-CXR-T/BioViL-T/CoCa-CXR.
- No `mixed` training class is added. Regional cells retain their explicit
  `stable/improved/worsened` tags. A compact report may show one lead region per
  direction when a disease has reliable opposing regional calls.
- `new/resolved` remain a report-level M3 presence transition in the main path;
  a learned five-state temporal model is an optional M4.5 experiment.

See `m4_temporal_calibration_and_readout_policy.md` for the complete M3→M4
contract, detector-mask blocker, calibration, and aggregation details.

## Files created or materially updated in the M4 audit

- `docs/now/m4_temporal_calibration_and_readout_policy.md`
- `docs/now/m4_retrain_server_runbook.md`
- `docs/now/kaggle_no_server_handoff.md`
- `docs/now/confidence_calibration_policy.md`
- this restart note

The earlier M3 implementation work is recorded by commits
`4d45dcd` and `6f56361`; the M4 audit/runbook commit is `c50b4da`.

## Current Kaggle dependency order

```text
M2 detector output
  -> detector-aligned boxes_det.npy + present_mask_det.npy
  -> detector-box M3 v2 training
  -> frozen M3 region cache
  -> detector-box M4 training
  -> temporal calibration and M5 report
```

The final Kaggle entry point is
`kaggle_notebooks/phase3_paper_v2_kaggle.ipynb`. It calls
`phase_3/run_paper_m3_v2.sh`, runs the faithful detector-box main row first,
and leaves the remaining paper ablations opt-in. The notebook under
`phase_3/notebooks/` is legacy and is not used for this campaign.
