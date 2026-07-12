#!/usr/bin/env bash
# Resume the interrupted xwalk_v2 Phase-4 matrix, then continue the normal full pipeline.

set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PROFILE="${PROFILE:-h100mini}"
TAG="${TAG:-xwalk_v2}"
PY="${PY:-python3}"
LOGDIR="${LOGDIR:-logs/phase4_${TAG}}"
RESTART_LOG="${RESTART_LOG:-$LOGDIR/restart_xwalk_v2.log}"

if [[ "${RESTART_XWALK_LOG_ACTIVE:-0}" != "1" ]]; then
  mkdir -p "$LOGDIR"
  export RESTART_XWALK_LOG_ACTIVE=1
  exec > >(tee -a "$RESTART_LOG") 2>&1
fi

echo
echo "===== restart_xwalk_v2 START $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid=$$ ====="
echo "profile=$PROFILE tag=$TAG cmd=$0 $*"
on_exit() {
  local rc=$?
  echo "===== restart_xwalk_v2 END $(date -u '+%Y-%m-%d %H:%M:%S UTC') rc=$rc ====="
}
trap on_exit EXIT

case "$PROFILE" in
  h100mini)
    W="${W:-16}"
    EVAL_W="${EVAL_W:-32}"
    BATCH="${BATCH:-128}"
    EVAL_BATCH="${EVAL_BATCH:-$BATCH}"
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/features/frozen}"
    ;;
  local4060)
    W="${W:-4}"
    EVAL_W="${EVAL_W:-$W}"
    BATCH="${BATCH:-16}"
    EVAL_BATCH="${EVAL_BATCH:-$BATCH}"
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/frozen}"
    ;;
  *)
    echo "[ERROR] profile must be h100mini or local4060" >&2
    exit 2
    ;;
esac

M3_TAG="${M3_TAG:-$TAG}"
M3_CKPT="${M3_CKPT:-data/run/m3_B_faithful_${M3_TAG}/best.pt}"
M3LAB="${M3LAB:-data/m3_labels}"
M4LAB="${M4LAB:-data/m4_labels}"
PAIRS="${PAIRS:-data/m4_labels/m3_pairs.jsonl}"
RUNS="${RUNS:-data/run}"
CACHE="${CACHE:-data/m4_region_cache_${M3_TAG}}"
MS_CSV="${MS_CSV:-data/MS_CXR_T_temporal_image_classification_v1.0.0.csv}"
DIAGDIR="${DIAGDIR:-artifacts/diagnostics}"
PLOTDIR="${PLOTDIR:-artifacts/phase4_${TAG}/matrix}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
FORCE="${FORCE:-0}"
M4_MATRIX_SCOPE="${M4_MATRIX_SCOPE:-m3delta40}"
RUN_LEGACY_RESUMES="${RUN_LEGACY_RESUMES:-0}"
mkdir -p "$LOGDIR/matrix"

resume_if_needed() {
  local name="$1"; shift
  local ck_last="$RUNS/$name/last.pt"
  local train_log="$LOGDIR/matrix/$name.train.log"
  if [[ ! -f "$ck_last" ]]; then
    echo "[resume] no last.pt for $name; skip explicit resume"
    return 0
  fi
  if [[ -f "$train_log" ]] && grep -q "\\[DONE\\]" "$train_log"; then
    echo "[resume] $name already has [DONE] in $train_log"
    return 0
  fi
  echo "[resume] completing interrupted $name from $ck_last"
  "$PY" phase_4/scripts/2-train.py \
    --region-cache "$CACHE" \
    --features-root "$FEAT" \
    --m3-labels-dir "$M3LAB" \
    --m4-labels-dir "$M4LAB" \
    --pairs "$PAIRS" \
    --out "$RUNS" \
    --name "$name" \
    --epochs 30 \
    --batch "$BATCH" \
    --workers "$W" \
    --device "$DEVICE" \
    --select-metric change \
    --patience 8 \
    --resume \
    "$@" 2>&1 | tee -a "$train_log"
}

eval_if_needed() {
  local name="$1"
  local ck="$RUNS/$name/best.pt"
  local diag="$DIAGDIR/$name.$EVAL_SPLIT.diagnostics.json"
  local msdiag="$DIAGDIR/$name.mscxrt.json"
  local plot_summary="$PLOTDIR/$name/m4_run_summary.csv"
  if [[ ! -f "$ck" ]]; then
    echo "[eval-first] no best.pt for $name; skip"
    return 0
  fi
  if [[ -s "$diag" && "$FORCE" != "1" ]]; then
    echo "[eval-first] $diag exists; skip test eval"
  else
    echo "[eval-first] $name $EVAL_SPLIT diagnostics -> $diag"
    "$PY" phase_4/scripts/3-eval.py \
      --ckpt "$ck" \
      --region-cache "$CACHE" \
      --features-root "$FEAT" \
      --m3-labels-dir "$M3LAB" \
      --m4-labels-dir "$M4LAB" \
      --pairs "$PAIRS" \
      --split "$EVAL_SPLIT" \
      --batch "$EVAL_BATCH" \
      --workers "$EVAL_W" \
      --device "$DEVICE" \
      --diagnostics-json "$diag" \
      2>&1 | tee -a "$LOGDIR/matrix/$name.$EVAL_SPLIT.eval.log"
  fi
  if [[ -f "$MS_CSV" ]]; then
    if [[ -s "$msdiag" && "$FORCE" != "1" ]]; then
      echo "[eval-first] $msdiag exists; skip MS-CXR-T"
    else
      echo "[eval-first] $name MS-CXR-T -> $msdiag"
      "$PY" phase_4/scripts/5-mscxrt_audit.py \
        --ckpt "$ck" \
        --csv "$MS_CSV" \
        --region-cache "$CACHE" \
        --features-root "$FEAT" \
        --m3-labels-dir "$M3LAB" \
        --split all \
        --batch "$EVAL_BATCH" \
        --workers "$W" \
        --device "$DEVICE" \
        --out-json "$msdiag" \
        2>&1 | tee -a "$LOGDIR/matrix/$name.mscxrt.eval.log"
    fi
  fi
  if [[ -s "$plot_summary" && "$FORCE" != "1" ]]; then
    echo "[eval-first] $plot_summary exists; skip plots"
  else
    local plot_args=(--diagnostics "$diag" --out-dir "$PLOTDIR/$name")
    [[ -f "$msdiag" ]] && plot_args+=(--mscxrt "$msdiag")
    [[ -f "$LOGDIR/matrix/$name.train.log" ]] && plot_args+=(--train-logs "$LOGDIR/matrix/$name.train.log")
    echo "[eval-first] $name plots -> $PLOTDIR/$name"
    "$PY" phase_4/scripts/plot_diagnostics.py "${plot_args[@]}" \
      2>&1 | tee -a "$LOGDIR/matrix/$name.plots.log"
  fi
}

if [[ "$RUN_LEGACY_RESUMES" == "1" ]]; then
  echo "[legacy-resume] RUN_LEGACY_RESUMES=1; checking older explicit resume hooks"
  resume_if_needed m4v3_tf_smooth005_${TAG} --arch tempfuse --label-smoothing 0.05
  resume_if_needed m4v3_tf_dist010_${TAG} --arch tempfuse --distance-penalty-weight 0.10
  eval_if_needed m4v3_tf_dist010_${TAG}
  resume_if_needed m4v3_tf_opp025_${TAG} --arch tempfuse --opposite-penalty-weight 0.25
else
  echo "[legacy-resume] skipped; matrix resume/skip logic will handle runs in scope"
fi

echo "[continue] running Phase 4 wrapper (M4_MATRIX_SCOPE=$M4_MATRIX_SCOPE)"
M3_CKPT="$M3_CKPT" M3LAB="$M3LAB" M4LAB="$M4LAB" PAIRS="$PAIRS" CACHE="$CACHE" FEAT="$FEAT" RUNS="$RUNS" \
  BATCH="$BATCH" EVAL_BATCH="$EVAL_BATCH" W="$W" DEVICE="$DEVICE" MS_CSV="$MS_CSV" \
  EVAL_W="$EVAL_W" \
  M4_MATRIX_SCOPE="$M4_MATRIX_SCOPE" \
  DIAGDIR="$DIAGDIR" PLOTDIR="artifacts/phase4_${TAG}" \
  bash phase_4.sh --profile "$PROFILE" --tag "$TAG"
