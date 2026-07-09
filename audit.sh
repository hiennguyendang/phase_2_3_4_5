#!/usr/bin/env bash
# Sequential VERA audit runner.
#
# Profiles:
#   local4060  - RTX 4060 8GB-ish, 4 workers, conservative batch sizes
#   h100mini   - H100/MIG ~20GB, 16 workers, larger batch sizes
#
# Examples:
#   bash audit.sh --profile local4060 --split gold
#   bash audit.sh --profile h100mini --split test --run-full
#   bash audit.sh --profile h100mini --run-m4-train-grid
#   bash audit.sh --profile h100mini --run-phase4-full

set -euo pipefail

PROFILE="local4060"
SPLIT="gold"
RUN_FULL=0
RUN_M4_TRAIN_GRID=0
RUN_PHASE4_FULL=0
DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --split) SPLIT="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --run-full) RUN_FULL=1; shift ;;
    --run-m4-train-grid) RUN_M4_TRAIN_GRID=1; shift ;;
    --run-phase4-full) RUN_PHASE4_FULL=1; shift ;;
    -h|--help)
      sed -n '1,28p' "$0"
      exit 0
      ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  local4060)
    WORKERS=4
    M3_BATCH=64
    M4_BATCH=24
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/frozen}"
    ;;
  h100mini)
    WORKERS=16
    M3_BATCH=512
    M4_BATCH=128
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/features/frozen}"
    ;;
  *)
    echo "[ERROR] profile must be local4060 or h100mini" >&2
    exit 2
    ;;
esac

if [[ -x ".venv/Scripts/python.exe" ]]; then
  PY="${PY:-.venv/Scripts/python.exe}"
elif [[ -x ".venv/bin/python" ]]; then
  PY="${PY:-.venv/bin/python}"
else
  PY="${PY:-python3}"
fi

pick_first() {
  for p in "$@"; do
    if [[ -e "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  echo "$1"
}

M3_CKPT="${M3_CKPT:-$(pick_first RUN/m3_B_faithful/best.pt data/run/m3_B_faithful/best.pt)}"
M4_CKPT="${M4_CKPT:-$(pick_first RUN/m4v3_tf/best.pt data/run/m4v3_tf/best.pt)}"
M3LAB="${M3LAB:-data/m3_labels}"
M4LAB="${M4LAB:-data/m4_labels}"
PAIRS="${PAIRS:-data/m4_labels/m3_pairs.jsonl}"
CACHE="${CACHE:-data/m4_region_cache}"
MS_CSV="${MS_CSV:-data/MS_CXR_T_temporal_image_classification_v1.0.0.csv}"
OUTDIR="${OUTDIR:-artifacts/diagnostics}"
mkdir -p "$OUTDIR"

echo "===== VERA audit profile=$PROFILE split=$SPLIT device=$DEVICE workers=$WORKERS ====="
echo "python=$PY"
echo "features=$FEAT"

echo
echo "===== 0) syntax/import compile checks ====="
"$PY" -m py_compile \
  phase_3/src/eval.py \
  phase_3/scripts/6-faithfulness.py \
  phase_3/scripts/plot_diagnostics.py \
  phase_4/src/model.py \
  phase_4/src/dataset.py \
  phase_4/src/eval.py \
  phase_4/src/mscxrt.py \
  phase_4/scripts/5-mscxrt_audit.py \
  phase_4/scripts/6-mscxrt_adapter.py \
  phase_4/scripts/plot_diagnostics.py \
  phase_5/assemble.py \
  phase_5/run.py

echo
echo "===== 1) M3 eval diagnostics ($SPLIT) ====="
M3_DIAG="$OUTDIR/m3_B_faithful.$SPLIT.diagnostics.json"
"$PY" phase_3/scripts/5-eval.py \
  --ckpt "$M3_CKPT" \
  --labels-dir "$M3LAB" \
  --features-root "$FEAT" \
  --split "$SPLIT" \
  --batch "$M3_BATCH" \
  --workers "$WORKERS" \
  --box-source detector \
  --device "$DEVICE" \
  --diagnostics-json "$M3_DIAG"

echo
echo "===== 2) M3 reliability plots/tables ====="
"$PY" phase_3/scripts/plot_diagnostics.py \
  --diagnostics "$M3_DIAG" \
  --out-dir "$OUTDIR/m3_B_faithful.$SPLIT.plots" \
  --top-ece 8

echo
echo "===== 3) M3 faithfulness diagnostics ($SPLIT) ====="
"$PY" phase_3/scripts/6-faithfulness.py \
  --ckpt "$M3_CKPT" \
  --labels-dir "$M3LAB" \
  --features-root "$FEAT" \
  --split "$SPLIT" \
  --batch "$M3_BATCH" \
  --workers "$WORKERS" \
  --box-source detector \
  --device "$DEVICE" \
  --diagnostics-json "$OUTDIR/m3_B_faithful.$SPLIT.faithfulness.json"

echo
echo "===== 4) M4 diagnostics ($SPLIT) ====="
"$PY" phase_4/scripts/3-eval.py \
  --ckpt "$M4_CKPT" \
  --region-cache "$CACHE" \
  --features-root "$FEAT" \
  --m3-labels-dir "$M3LAB" \
  --m4-labels-dir "$M4LAB" \
  --pairs "$PAIRS" \
  --split "$SPLIT" \
  --batch "$M4_BATCH" \
  --device "$DEVICE" \
  --diagnostics-json "$OUTDIR/m4v3_tf.$SPLIT.diagnostics.json"

echo
echo "===== 5) MS-CXR-T external temporal audit if CSV exists ====="
if [[ -f "$MS_CSV" ]]; then
  "$PY" phase_4/scripts/5-mscxrt_audit.py \
    --ckpt "$M4_CKPT" \
    --csv "$MS_CSV" \
    --region-cache "$CACHE" \
    --features-root "$FEAT" \
    --m3-labels-dir "$M3LAB" \
    --split all \
    --batch "$M4_BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --out-json "$OUTDIR/m4v3_tf.mscxrt.diagnostics.json"
else
  echo "[skip] no $MS_CSV"
fi

echo
echo "===== 5b) M4 diagnostic plots/tables ====="
PLOT_ARGS=(--diagnostics "$OUTDIR/m4v3_tf.$SPLIT.diagnostics.json" --out-dir "artifacts/phase4_audit")
if [[ -f "$OUTDIR/m4v3_tf.mscxrt.diagnostics.json" ]]; then
  PLOT_ARGS+=(--mscxrt "$OUTDIR/m4v3_tf.mscxrt.diagnostics.json")
fi
"$PY" phase_4/scripts/plot_diagnostics.py "${PLOT_ARGS[@]}" || true

echo
echo "===== 6) M5 deterministic smoke demo ====="
"$PY" phase_5/demo.py

echo
echo "===== 7) M5 report verify stats if prediction JSONL exists ====="
M3_PRED="${M3_PRED:-data/m3_pred.jsonl}"
M4_PRED="${M4_PRED:-data/m4_pred.jsonl}"
if [[ -f "$M3_PRED" ]]; then
  "$PY" phase_5/run.py \
    --m3-pred "$M3_PRED" \
    --m4-pred "$M4_PRED" \
    --out "$OUTDIR/m5_reports.jsonl" \
    --stats-json "$OUTDIR/m5_stats.json"
else
  echo "[skip] no $M3_PRED; run phase_3/scripts/7-infer.py first to audit real M5 outputs."
fi

if [[ "$RUN_M4_TRAIN_GRID" == "1" ]]; then
  echo
  echo "===== 8) M4 retrain matrix ====="
  bash phase_4/run_m4_retrain_matrix.sh --profile "$PROFILE" --device "$DEVICE" --eval-split "$SPLIT"
fi

if [[ "$RUN_PHASE4_FULL" == "1" ]]; then
  echo
  echo "===== 9) Phase 4 full pipeline ====="
  bash phase_4.sh --profile "$PROFILE" --device "$DEVICE" --audit-splits "$SPLIT"
fi

if [[ "$RUN_FULL" == "1" && "$SPLIT" != "test" ]]; then
  echo
  echo "[note] --run-full was set, but --split is '$SPLIT'. Re-run with --split test for full-test audit."
fi

echo
echo "===== DONE ====="
echo "Diagnostics written under $OUTDIR"
