#!/usr/bin/env bash
# Sequential M4 retraining matrix for server sweeps.
#
# Default profile targets the H100-mini setup. Override paths with env vars when needed:
#   FEAT=data/features/frozen M3LAB=data/m3_labels M4LAB=data/m4_labels RUNS=data/run ...
#
# Examples:
#   bash phase_4/run_m4_retrain_matrix.sh --profile h100mini
#   bash phase_4/run_m4_retrain_matrix.sh --profile h100mini --run-adapters
#   bash phase_4/run_m4_retrain_matrix.sh --profile local4060 --epochs 3 --eval-split gold

set -euo pipefail

PROFILE="h100mini"
DEVICE=""
EP=""
RUN_ADAPTERS=0
EVAL_SPLIT="${EVAL_SPLIT:-test}"
FORCE="${FORCE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --epochs) EP="${2:?}"; shift 2 ;;
    --eval-split) EVAL_SPLIT="${2:?}"; shift 2 ;;
    --run-adapters) RUN_ADAPTERS=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help)
      sed -n '1,28p' "$0"
      exit 0
      ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  local4060)
    W="${W:-4}"
    BATCH="${BATCH:-16}"
    ADAPT_BATCH="${ADAPT_BATCH:-16}"
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/frozen}"
    EP="${EP:-8}"
    PAT="${PAT:-4}"
    ;;
  h100mini)
    W="${W:-16}"
    BATCH="${BATCH:-128}"
    ADAPT_BATCH="${ADAPT_BATCH:-64}"
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/features/frozen}"
    EP="${EP:-30}"
    PAT="${PAT:-8}"
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

M3_CKPT="${M3_CKPT:-$(pick_first data/run/m3_B_faithful/best.pt RUN/m3_B_faithful/best.pt)}"
M3LAB="${M3LAB:-data/m3_labels}"
M4LAB="${M4LAB:-data/m4_labels}"
PAIRS="${PAIRS:-data/m4_labels/m3_pairs.jsonl}"
CACHE="${CACHE:-data/m4_region_cache}"
RUNS="${RUNS:-data/run}"
MS_CSV="${MS_CSV:-data/MS_CXR_T_temporal_image_classification_v1.0.0.csv}"
LOGDIR="${LOGDIR:-logs/m4_matrix}"
DIAGDIR="${DIAGDIR:-artifacts/diagnostics}"
PLOTDIR="${PLOTDIR:-artifacts/phase4_matrix}"
COMMON="--select-metric change --patience $PAT"
mkdir -p "$RUNS" "$LOGDIR" "$DIAGDIR" "$PLOTDIR"

echo "===== M4 retrain matrix profile=$PROFILE device=$DEVICE workers=$W batch=$BATCH epochs=$EP ====="
echo "features=$FEAT"
echo "region_cache=$CACHE"
echo "runs=$RUNS"
echo "run_suffix=$RUN_SUFFIX"

echo
echo "===== bridge: M3 region cache for regiondiff and tempfuse+M3-delta ====="
"$PY" phase_3/scripts/8-precompute_regions.py \
  --ckpt "$M3_CKPT" \
  --labels-dir "$M3LAB" \
  --features-root "$FEAT" \
  --out-dir "$CACHE" \
  --batch "$BATCH" \
  --workers "$W" \
  --device "$DEVICE" 2>&1 | tee "$LOGDIR/00_bridge.log"

eval_one() {
  local name="$1"
  local ck="$RUNS/$name/best.pt"
  local diag="$DIAGDIR/$name.$EVAL_SPLIT.diagnostics.json"
  local msdiag="$DIAGDIR/$name.mscxrt.json"
  "$PY" phase_4/scripts/3-eval.py \
    --ckpt "$ck" \
    --region-cache "$CACHE" \
    --features-root "$FEAT" \
    --m3-labels-dir "$M3LAB" \
    --m4-labels-dir "$M4LAB" \
    --pairs "$PAIRS" \
    --split "$EVAL_SPLIT" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --diagnostics-json "$diag" \
    2>&1 | tee "$LOGDIR/$name.$EVAL_SPLIT.eval.log"

  if [[ -f "$MS_CSV" ]]; then
    "$PY" phase_4/scripts/5-mscxrt_audit.py \
      --ckpt "$ck" \
      --csv "$MS_CSV" \
      --region-cache "$CACHE" \
      --features-root "$FEAT" \
      --m3-labels-dir "$M3LAB" \
      --split all \
      --batch "$BATCH" \
      --workers "$W" \
      --device "$DEVICE" \
      --out-json "$msdiag" \
      2>&1 | tee "$LOGDIR/$name.mscxrt.eval.log"
  fi

  local plot_args=(--diagnostics "$diag" --out-dir "$PLOTDIR/$name")
  [[ -f "$msdiag" ]] && plot_args+=(--mscxrt "$msdiag")
  [[ -f "$LOGDIR/$name.train.log" ]] && plot_args+=(--train-logs "$LOGDIR/$name.train.log")
  "$PY" phase_4/scripts/plot_diagnostics.py "${plot_args[@]}" \
    2>&1 | tee "$LOGDIR/$name.plots.log"
}

train_one() {
  local base="$1"; shift
  local name="${base}${RUN_SUFFIX}"
  local ck="$RUNS/$name/best.pt"
  echo
  echo "===== train $name $* ====="
  if [[ -f "$ck" && "$FORCE" != "1" ]]; then
    echo "[skip train] $ck exists; set FORCE=1 or pass --force to retrain"
  else
    "$PY" phase_4/scripts/2-train.py \
      --region-cache "$CACHE" \
      --features-root "$FEAT" \
      --m3-labels-dir "$M3LAB" \
      --m4-labels-dir "$M4LAB" \
      --pairs "$PAIRS" \
      --out "$RUNS" \
      --name "$name" \
      --epochs "$EP" \
      --batch "$BATCH" \
      --workers "$W" \
      --device "$DEVICE" \
      "$@" 2>&1 | tee "$LOGDIR/$name.train.log"
  fi
  eval_one "$name"
}

adapter_one() {
  local base="$1"; shift
  local scope="$1"; shift
  local base_run="${base}${RUN_SUFFIX}"
  local name="${base_run}_mscxrt_${scope}"
  local ck="$RUNS/$name/best.pt"
  [[ -f "$MS_CSV" ]] || return 0
  echo
  echo "===== MS-CXR-T adapter $name from $base scope=$scope ====="
  if [[ -f "$ck" && "$FORCE" != "1" ]]; then
    echo "[skip adapter] $ck exists"
  else
    "$PY" phase_4/scripts/6-mscxrt_adapter.py \
      --base-ckpt "$RUNS/$base_run/best.pt" \
      --csv "$MS_CSV" \
      --region-cache "$CACHE" \
      --features-root "$FEAT" \
      --m3-labels-dir "$M3LAB" \
      --out "$RUNS" \
      --name "$name" \
      --train-scope "$scope" \
      --epochs 12 \
      --batch "$ADAPT_BATCH" \
      --workers "$W" \
      --device "$DEVICE" \
      2>&1 | tee "$LOGDIR/$name.train.log"
  fi
  eval_one "$name"
}

# Controls: v2/v3 ideas already implemented (time flip is default, same-view and two-stage are flags).
train_one m4v3_tf_retrain              $COMMON --arch tempfuse
train_one m4v3_tf_smooth003            $COMMON --arch tempfuse --label-smoothing 0.03
train_one m4v3_tf_smooth005            $COMMON --arch tempfuse --label-smoothing 0.05
train_one m4v3_tf_smooth010            $COMMON --arch tempfuse --label-smoothing 0.10
train_one m4v3_tf_focal                $COMMON --arch tempfuse --loss focal
train_one m4v3_tf_kl005                $COMMON --arch tempfuse --flip-consistency-weight 0.05
train_one m4v3_tf_kl010                $COMMON --arch tempfuse --flip-consistency-weight 0.10
train_one m4v3_tf_kl005_noaug          $COMMON --arch tempfuse --flip-consistency-weight 0.05 --no-augment
train_one m4v3_tf_twostage_retrain     $COMMON --arch tempfuse --head-mode twostage
train_one m4v3_tf_sv2stage_retrain     $COMMON --arch tempfuse --same-view --head-mode twostage
train_one m4v3_tf_svcur3               $COMMON --arch tempfuse --curriculum-same-view-epochs 3
train_one m4v3_tf_svcur5_twostage      $COMMON --arch tempfuse --curriculum-same-view-epochs 5 --head-mode twostage
train_one m4v3_tf_2blocks_retrain      $COMMON --arch tempfuse --fuse-blocks 2
train_one m4v3_tf_detbox_retrain       $COMMON --arch tempfuse --box-source detector

# New v4 idea: TempFuse region feature + M3 current/prior/delta disease logits.
HYB="$COMMON --arch tempfuse --tempfuse-input-mode feat_logits"
train_one m4v4_tf_m3delta              $HYB
train_one m4v4_tf_m3delta_smooth003    $HYB --label-smoothing 0.03
train_one m4v4_tf_m3delta_smooth005    $HYB --label-smoothing 0.05
train_one m4v4_tf_m3delta_smooth010    $HYB --label-smoothing 0.10
train_one m4v4_tf_m3delta_focal        $HYB --loss focal
train_one m4v4_tf_m3delta_kl005        $HYB --flip-consistency-weight 0.05
train_one m4v4_tf_m3delta_kl010        $HYB --flip-consistency-weight 0.10
train_one m4v4_tf_m3delta_kl005_noaug  $HYB --flip-consistency-weight 0.05 --no-augment
train_one m4v4_tf_m3delta_kl005_2st    $HYB --flip-consistency-weight 0.05 --head-mode twostage
train_one m4v4_tf_m3delta_twostage     $HYB --head-mode twostage
train_one m4v4_tf_m3delta_sv2stage     $HYB --same-view --head-mode twostage
train_one m4v4_tf_m3delta_svcur3       $HYB --curriculum-same-view-epochs 3
train_one m4v4_tf_m3delta_svcur5_2st   $HYB --curriculum-same-view-epochs 5 --head-mode twostage
train_one m4v4_tf_m3delta_2blocks      $HYB --fuse-blocks 2
train_one m4v4_tf_m3delta_detbox       $HYB --box-source detector
train_one m4v4_tf_m3delta_linear       $HYB --head-type linear
train_one m4v4_tf_m3delta_kan          $HYB --head-type kan

# Regiondiff controls for checking whether the M3-logit signal alone carries the gain.
train_one m4v2_regiondiff_full_retrain $COMMON --arch regiondiff --input-mode full
train_one m4v2_regiondiff_logits       $COMMON --arch regiondiff --input-mode logits
train_one m4v2_regiondiff_diff         $COMMON --arch regiondiff --input-mode diff

if [[ "$RUN_ADAPTERS" == "1" ]]; then
  adapter_one m4v3_tf_retrain head
  adapter_one m4v4_tf_m3delta head
  adapter_one m4v4_tf_m3delta pool-head
  adapter_one m4v4_tf_m3delta_smooth005 head
fi

echo
echo "===== summary: eval logs under $LOGDIR; diagnostics under artifacts/diagnostics ====="
