#!/usr/bin/env bash
# Run a small selected Phase-3 ablation set without re-running the faithful ship grid.

set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

PROFILE="h100mini"
TAG="${TAG:-xwalk_v2}"
DEVICE=""
EP=""
TRAIN=1
FORCE=0
AUDIT_SPLITS="${AUDIT_SPLITS:-val test gold}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --epochs) EP="${2:?}"; shift 2 ;;
    --audit-splits) AUDIT_SPLITS="${2:?}"; shift 2 ;;
    --skip-train) TRAIN=0; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  local4060)
    W="${W:-4}"
    BATCH="${BATCH:-24}"
    EVAL_BATCH="${EVAL_BATCH:-64}"
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/frozen}"
    EP="${EP:-5}"
    ;;
  h100mini)
    W="${W:-16}"
    BATCH="${BATCH:-512}"
    EVAL_BATCH="${EVAL_BATCH:-512}"
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/features/frozen}"
    EP="${EP:-30}"
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

LABELS="${LABELS:-data/m3_labels}"
RUNS="${RUNS:-data/run}"
LOGDIR="${LOGDIR:-logs/phase3_$TAG}"
DIAGDIR="${DIAGDIR:-artifacts/diagnostics}"
mkdir -p "$RUNS" "$LOGDIR" "$DIAGDIR"

echo "===== Phase 3 selected ablations profile=$PROFILE tag=$TAG device=$DEVICE workers=$W ====="
echo "python=$PY"
echo "features=$FEAT"
echo "labels=$LABELS"
echo "runs=$RUNS"
echo "audit_splits=$AUDIT_SPLITS"

run_one() {
  local base="$1"; local box="$2"; shift 2
  local name="${base}_${TAG}"
  local ck="$RUNS/$name/best.pt"
  echo
  echo "===== selected ablation train/eval $name box=$box $* ====="
  if [[ "$TRAIN" == "1" ]]; then
    if [[ -f "$ck" && "$FORCE" != "1" ]]; then
      echo "[skip train] $ck exists; use --force or a new --tag to retrain"
    else
      "$PY" phase_3/scripts/4-train.py \
        --labels-dir "$LABELS" \
        --features-root "$FEAT" \
        --out "$RUNS" \
        --name "$name" \
        --epochs "$EP" \
        --batch "$BATCH" \
        --workers "$W" \
        --device "$DEVICE" \
        --box-source "$box" \
        --select-by auc \
        "$@" 2>&1 | tee "$LOGDIR/$name.train.log"
    fi
  else
    echo "[skip train] --skip-train set"
  fi

  if [[ ! -f "$ck" ]]; then
    echo "[warn] missing $ck; skipping audits for $name" >&2
    return 1
  fi

  local audits_complete=1
  for split in $AUDIT_SPLITS; do
    [[ -f "$DIAGDIR/$name.$split.diagnostics.json" ]] || audits_complete=0
    [[ -f "$DIAGDIR/$name.$split.faithfulness.json" ]] || audits_complete=0
  done
  if [[ "$audits_complete" == "1" && "$FORCE" != "1" ]]; then
    echo "[skip audits] diagnostics/faithfulness already exist for $name"
    return 0
  fi

  "$PY" phase_3/scripts/5-eval.py \
    --ckpt "$ck" \
    --labels-dir "$LABELS" \
    --features-root "$FEAT" \
    --split test \
    --box-source "$box" \
    --batch "$EVAL_BATCH" \
    --workers "$W" \
    --device "$DEVICE" 2>&1 | tee "$LOGDIR/$name.eval.log"

  for split in $AUDIT_SPLITS; do
    local diag="$DIAGDIR/$name.$split.diagnostics.json"
    "$PY" phase_3/scripts/5-eval.py \
      --ckpt "$ck" \
      --labels-dir "$LABELS" \
      --features-root "$FEAT" \
      --split "$split" \
      --box-source "$box" \
      --batch "$EVAL_BATCH" \
      --workers "$W" \
      --device "$DEVICE" \
      --diagnostics-json "$diag" \
      2>&1 | tee "$LOGDIR/$name.$split.diagnostics.log"

    "$PY" phase_3/scripts/plot_diagnostics.py \
      --diagnostics "$diag" \
      --out-dir "$DIAGDIR/$name.$split.plots" \
      --top-ece 8 2>&1 | tee "$LOGDIR/$name.$split.plots.log"

    "$PY" phase_3/scripts/6-faithfulness.py \
      --ckpt "$ck" \
      --labels-dir "$LABELS" \
      --features-root "$FEAT" \
      --split "$split" \
      --box-source "$box" \
      --batch "$EVAL_BATCH" \
      --workers "$W" \
      --device "$DEVICE" \
      --diagnostics-json "$DIAGDIR/$name.$split.faithfulness.json" \
      2>&1 | tee "$LOGDIR/$name.$split.faithfulness.log"
  done
}

run_one m3_A detector --mode A
run_one m3_global_only detector --mode A --global-only
run_one m3_B detector --mode B --disease-head mlp
run_one m3_B_linear detector --mode B --disease-head linear
