#!/usr/bin/env bash
# Full Phase-4 rerun/audit pipeline.
#
# Prerequisite: Phase 3 must have produced a faithful ship checkpoint first.
# Recommended order:
#   bash phase_3.sh --profile h100mini --tag xwalk_v2
#   bash phase_4.sh --profile h100mini --tag xwalk_v2
#
# This script runs everything Phase-4-related:
#   1) optional M4 label rebuild
#   2) M4 dataset stats
#   3) frozen-M3 region cache bridge for regiondiff + v4 TempFuse+M3-delta
#   4) full M4 retrain matrix
#   5) silver diagnostics on val/test/gold
#   6) MS-CXR-T external audit
#   7) plots/tables
#   8) M4 inference JSONL for Phase 5
#   9) optional M5 assembly if m3_pred.jsonl is available

set -euo pipefail

PROFILE="h100mini"
TAG="${TAG:-xwalk_v2}"
M3_TAG="${M3_TAG:-}"
DEVICE=""
EP=""
TRAIN=1
FORCE=0
RUN_ADAPTERS=0
REBUILD_LABELS=0
RUN_CACHE=1
RUN_INFER=1
RUN_M5_IF_READY=1
AUDIT_SPLITS="${AUDIT_SPLITS:-val test gold}"
INFER_SPLITS="${INFER_SPLITS:-test gold}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; if [[ -z "$M3_TAG" ]]; then M3_TAG="$TAG"; fi; shift 2 ;;
    --m3-tag) M3_TAG="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --epochs) EP="${2:?}"; shift 2 ;;
    --audit-splits) AUDIT_SPLITS="${2:?}"; shift 2 ;;
    --infer-splits) INFER_SPLITS="${2:?}"; shift 2 ;;
    --skip-train) TRAIN=0; shift ;;
    --force) FORCE=1; shift ;;
    --run-adapters) RUN_ADAPTERS=1; shift ;;
    --rebuild-labels) REBUILD_LABELS=1; shift ;;
    --skip-cache) RUN_CACHE=0; shift ;;
    --skip-infer) RUN_INFER=0; shift ;;
    --skip-m5) RUN_M5_IF_READY=0; shift ;;
    -h|--help)
      sed -n '1,36p' "$0"
      exit 0
      ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$M3_TAG" ]]; then
  M3_TAG="$TAG"
fi

case "$PROFILE" in
  local4060)
    W="${W:-4}"
    BATCH="${BATCH:-16}"
    EVAL_BATCH="${EVAL_BATCH:-24}"
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/frozen}"
    EP="${EP:-6}"
    ;;
  h100mini)
    W="${W:-16}"
    BATCH="${BATCH:-128}"
    EVAL_BATCH="${EVAL_BATCH:-128}"
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

pick_first() {
  for p in "$@"; do
    if [[ -e "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  echo "$1"
}

RUN_SUFFIX="${RUN_SUFFIX:-_$TAG}"
M3_CKPT="${M3_CKPT:-$(pick_first data/run/m3_B_faithful_${M3_TAG}/best.pt data/run/m3_B_faithful/best.pt RUN/m3_B_faithful/best.pt)}"
M3_FAITH="${M3_FAITH:-$(pick_first artifacts/diagnostics/m3_B_faithful_${M3_TAG}.test.faithfulness.json artifacts/diagnostics/m3_B_faithful_${M3_TAG}.gold.faithfulness.json)}"
M3LAB="${M3LAB:-data/m3_labels}"
M4LAB="${M4LAB:-data/m4_labels}"
PAIRS="${PAIRS:-data/m4_labels/m3_pairs.jsonl}"
RUNS="${RUNS:-data/run}"
CACHE="${CACHE:-data/m4_region_cache_${M3_TAG}}"
MS_CSV="${MS_CSV:-data/MS_CXR_T_temporal_image_classification_v1.0.0.csv}"
SCENE_ROOT="${SCENE_ROOT:-}"
LOGDIR="${LOGDIR:-logs/phase4_$TAG}"
DIAGDIR="${DIAGDIR:-artifacts/diagnostics}"
PLOTDIR="${PLOTDIR:-artifacts/phase4_$TAG}"
M4_PRED_DIR="${M4_PRED_DIR:-data}"
M3_PRED="${M3_PRED:-data/m3_pred.jsonl}"
mkdir -p "$RUNS" "$LOGDIR" "$DIAGDIR" "$PLOTDIR"

echo "===== Phase 4 full profile=$PROFILE tag=$TAG m3_tag=$M3_TAG ====="
echo "python=$PY"
echo "features=$FEAT"
echo "m3_ckpt=$M3_CKPT"
echo "m3_faith=$M3_FAITH"
echo "m3_labels=$M3LAB"
echo "m4_labels=$M4LAB"
echo "pairs=$PAIRS"
echo "cache=$CACHE"
echo "runs=$RUNS run_suffix=$RUN_SUFFIX"
echo "audit_splits=$AUDIT_SPLITS"

echo
echo "===== 0) preflight: Phase 3 must be faithful first ====="
if [[ ! -f "$M3_CKPT" ]]; then
  echo "[ERROR] missing M3 checkpoint: $M3_CKPT" >&2
  echo "Run Phase 3 first, e.g. bash phase_3.sh --profile h100mini --tag $M3_TAG" >&2
  exit 2
fi
if [[ -f "$M3_FAITH" ]]; then
  "$PY" - "$M3_FAITH" <<'PY'
import json, sys
p=sys.argv[1]
d=json.load(open(p, encoding="utf-8"))
ok=bool(d.get("why_faithful_allowed"))
print(f"[M3 faithfulness] {p}: why_faithful_allowed={ok}")
raise SystemExit(0 if ok else 3)
PY
else
  echo "[warn] no M3 faithfulness JSON found. Recommended: finish phase_3.sh faithfulness before Phase 4."
fi

echo
echo "===== 1) compile Phase-4 scripts ====="
"$PY" -m py_compile \
  phase_4/src/model.py \
  phase_4/src/dataset.py \
  phase_4/src/eval.py \
  phase_4/src/losses.py \
  phase_4/src/mscxrt.py \
  phase_4/scripts/1-labels.py \
  phase_4/scripts/2-train.py \
  phase_4/scripts/3-eval.py \
  phase_4/scripts/4-infer.py \
  phase_4/scripts/5-mscxrt_audit.py \
  phase_4/scripts/6-mscxrt_adapter.py \
  phase_4/scripts/dataset_stats.py \
  phase_4/scripts/plot_diagnostics.py

if [[ "$REBUILD_LABELS" == "1" ]]; then
  echo
  echo "===== 2) rebuild M4 labels from scene graphs ====="
  if [[ -z "$SCENE_ROOT" ]]; then
    echo "[ERROR] --rebuild-labels requires SCENE_ROOT=/path/to/chest-imagenome" >&2
    exit 2
  fi
  "$PY" phase_4/scripts/1-labels.py \
    --scene-root "$SCENE_ROOT" \
    --out-dir "$M4LAB" 2>&1 | tee "$LOGDIR/01_labels.log"
else
  echo
  echo "===== 2) M4 labels: using existing $M4LAB ====="
fi

echo
echo "===== 3) M4 dataset stats ====="
"$PY" phase_4/scripts/dataset_stats.py 2>&1 | tee "$LOGDIR/02_dataset_stats.log"

if [[ "$RUN_CACHE" == "1" ]]; then
  echo
  echo "===== 4) frozen-M3 region cache bridge ====="
  "$PY" phase_3/scripts/8-precompute_regions.py \
    --ckpt "$M3_CKPT" \
    --labels-dir "$M3LAB" \
    --features-root "$FEAT" \
    --out-dir "$CACHE" \
    --batch "$EVAL_BATCH" \
    --workers "$W" \
    --device "$DEVICE" 2>&1 | tee "$LOGDIR/03_region_cache.log"
fi

if [[ "$TRAIN" == "1" ]]; then
  echo
  echo "===== 5) full M4 retrain matrix ====="
  matrix_args=(--profile "$PROFILE" --device "$DEVICE" --epochs "$EP" --eval-split test)
  [[ "$RUN_ADAPTERS" == "1" ]] && matrix_args+=(--run-adapters)
  [[ "$FORCE" == "1" ]] && matrix_args+=(--force)
  RUN_SUFFIX="$RUN_SUFFIX" \
  M3_CKPT="$M3_CKPT" M3LAB="$M3LAB" M4LAB="$M4LAB" PAIRS="$PAIRS" CACHE="$CACHE" \
  FEAT="$FEAT" RUNS="$RUNS" MS_CSV="$MS_CSV" LOGDIR="$LOGDIR/matrix" \
  DIAGDIR="$DIAGDIR" PLOTDIR="$PLOTDIR/matrix" \
    bash phase_4/run_m4_retrain_matrix.sh "${matrix_args[@]}"
else
  echo
  echo "===== 5) train matrix skipped ====="
fi

eval_run_split() {
  local run="$1"; local split="$2"
  local ck="$RUNS/$run/best.pt"
  [[ -f "$ck" ]] || return 0
  local diag="$DIAGDIR/$run.$split.diagnostics.json"
  local msdiag="$DIAGDIR/$run.mscxrt.json"
  "$PY" phase_4/scripts/3-eval.py \
    --ckpt "$ck" \
    --region-cache "$CACHE" \
    --features-root "$FEAT" \
    --m3-labels-dir "$M3LAB" \
    --m4-labels-dir "$M4LAB" \
    --pairs "$PAIRS" \
    --split "$split" \
    --batch "$EVAL_BATCH" \
    --device "$DEVICE" \
    --diagnostics-json "$diag" \
    2>&1 | tee "$LOGDIR/$run.$split.eval.log"
  if [[ -f "$MS_CSV" && "$split" == "test" ]]; then
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
      2>&1 | tee "$LOGDIR/$run.mscxrt.log"
  fi
}

echo
echo "===== 6) full diagnostics over tagged runs ====="
shopt -s nullglob
run_dirs=("$RUNS"/m4*"$RUN_SUFFIX")
if [[ ${#run_dirs[@]} -eq 0 ]]; then
  echo "[warn] no tagged runs found under $RUNS/*$RUN_SUFFIX. Falling back to existing m4 runs."
  run_dirs=("$RUNS"/m4* RUN/m4*)
fi
for d in "${run_dirs[@]}"; do
  [[ -d "$d" && -f "$d/best.pt" ]] || continue
  run="$(basename "$d")"
  for split in $AUDIT_SPLITS; do
    eval_run_split "$run" "$split"
  done
done

echo
echo "===== 7) plots/tables ====="
diag_files=("$DIAGDIR"/m4*"$RUN_SUFFIX".*.diagnostics.json)
ms_files=("$DIAGDIR"/m4*"$RUN_SUFFIX".mscxrt.json)
train_logs=("$LOGDIR"/matrix/m4*"$RUN_SUFFIX".train.log "$LOGDIR"/m4*"$RUN_SUFFIX".train.log LOGS/m4*.train.log)
plot_args=(--out-dir "$PLOTDIR")
[[ ${#diag_files[@]} -gt 0 ]] && plot_args+=(--diagnostics "${diag_files[@]}")
[[ ${#ms_files[@]} -gt 0 ]] && plot_args+=(--mscxrt "${ms_files[@]}")
[[ ${#train_logs[@]} -gt 0 ]] && plot_args+=(--train-logs "${train_logs[@]}")
"$PY" phase_4/scripts/plot_diagnostics.py "${plot_args[@]}" \
  2>&1 | tee "$LOGDIR/phase4_$TAG.plots.log"

pick_ship_ckpt() {
  local candidates=(
    "${SHIP_M4_NAME:-}"
    "m4v4_tf_m3delta_kl005_2st${RUN_SUFFIX}"
    "m4v4_tf_m3delta_kl005${RUN_SUFFIX}"
    "m4v4_tf_m3delta_smooth005${RUN_SUFFIX}"
    "m4v4_tf_m3delta${RUN_SUFFIX}"
    "m4v3_tf_retrain${RUN_SUFFIX}"
    "m4v3_tf${RUN_SUFFIX}"
    "m4v3_tf"
    "m4v3_tf_sv2stage"
  )
  for r in "${candidates[@]}"; do
    [[ -n "$r" ]] || continue
    if [[ -f "$RUNS/$r/best.pt" ]]; then
      echo "$RUNS/$r/best.pt"; return 0
    fi
    if [[ -f "RUN/$r/best.pt" ]]; then
      echo "RUN/$r/best.pt"; return 0
    fi
  done
  echo ""
}

SHIP_CKPT="${SHIP_CKPT:-$(pick_ship_ckpt)}"
if [[ "$RUN_INFER" == "1" && -n "$SHIP_CKPT" && -f "$SHIP_CKPT" ]]; then
  echo
  echo "===== 8) M4 inference JSONL for Phase 5 ====="
  ship_name="$(basename "$(dirname "$SHIP_CKPT")")"
  for split in $INFER_SPLITS; do
    out="$M4_PRED_DIR/m4_pred.$split.$TAG.jsonl"
    "$PY" phase_4/scripts/4-infer.py \
      --ckpt "$SHIP_CKPT" \
      --region-cache "$CACHE" \
      --features-root "$FEAT" \
      --m3-labels-dir "$M3LAB" \
      --m4-labels-dir "$M4LAB" \
      --pairs "$PAIRS" \
      --split "$split" \
      --out "$out" \
      --batch "$EVAL_BATCH" \
      --device "$DEVICE" \
      2>&1 | tee "$LOGDIR/$ship_name.infer.$split.log"
    if [[ "$split" == "test" ]]; then
      cp "$out" "$M4_PRED_DIR/m4_pred.jsonl"
      echo "[m5-default] copied $out -> $M4_PRED_DIR/m4_pred.jsonl"
    fi
  done
else
  echo
  echo "[skip infer] no ship checkpoint found or --skip-infer set. SHIP_CKPT='$SHIP_CKPT'"
fi

if [[ "$RUN_M5_IF_READY" == "1" && -f "$M3_PRED" && -f "$M4_PRED_DIR/m4_pred.jsonl" ]]; then
  echo
  echo "===== 9) optional M5 report assembly ====="
  "$PY" phase_5/run.py \
    --m3-pred "$M3_PRED" \
    --m4-pred "$M4_PRED_DIR/m4_pred.jsonl" \
    --out "$PLOTDIR/m5_reports.jsonl" \
    --stats-json "$PLOTDIR/m5_stats.json" \
    2>&1 | tee "$LOGDIR/phase4_$TAG.m5.log"
fi

echo
echo "===== DONE Phase 4 full ====="
echo "m3_ckpt=$M3_CKPT"
echo "cache=$CACHE"
echo "ship_ckpt=$SHIP_CKPT"
echo "m4_pred=$M4_PRED_DIR/m4_pred.jsonl"
echo "plots=$PLOTDIR"
