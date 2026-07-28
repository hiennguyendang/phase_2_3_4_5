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
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PROFILE="h100mini"
DEVICE=""
EP=""
RUN_ADAPTERS=0
EVAL_SPLIT="${EVAL_SPLIT:-test}"
FORCE="${FORCE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
M4_MATRIX_SCOPE="${M4_MATRIX_SCOPE:-full}"
M4_BOX_SOURCE="${M4_BOX_SOURCE:-gt}"
RUN_MS_CXRT="${RUN_MS_CXRT:-0}"

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
    EVAL_W="${EVAL_W:-$W}"
    BATCH="${BATCH:-16}"
    ADAPT_BATCH="${ADAPT_BATCH:-16}"
    DEVICE="${DEVICE:-cuda}"
    FEAT="${FEAT:-data/frozen}"
    EP="${EP:-8}"
    PAT="${PAT:-4}"
    ;;
  h100mini)
    W="${W:-16}"
    EVAL_W="${EVAL_W:-32}"
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

if [[ "$M4_MATRIX_SCOPE" == "m3delta40" || "$M4_MATRIX_SCOPE" == "m3delta_refine" ||
      "$M4_MATRIX_SCOPE" == "m3delta_kl_dist_refine" ||
      "$M4_MATRIX_SCOPE" == "m3delta_smooth_dist_refine" ||
      "$M4_MATRIX_SCOPE" == detector_* ]]; then
  EP="${M3DELTA40_EPOCHS:-40}"
fi

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
COMMON="--select-metric change --patience $PAT --box-source $M4_BOX_SOURCE"
HYB="$COMMON --arch tempfuse --tempfuse-input-mode feat_logits"
mkdir -p "$RUNS" "$LOGDIR" "$DIAGDIR" "$PLOTDIR"

has_done_marker() {
  local log="$1"
  [[ -f "$log" ]] && grep -q "\\[DONE\\]" "$log"
}

log_header() {
  local log="$1"; shift
  {
    echo
    echo "===== $* $(date -u '+%Y-%m-%d %H:%M:%S UTC') ====="
  } >> "$log"
}

echo "===== M4 retrain matrix profile=$PROFILE device=$DEVICE workers=$W batch=$BATCH epochs=$EP ====="
echo "eval_workers=$EVAL_W"
echo "features=$FEAT"
echo "region_cache=$CACHE"
echo "runs=$RUNS"
echo "run_suffix=$RUN_SUFFIX"
echo "matrix_scope=$M4_MATRIX_SCOPE"
echo "box_source=$M4_BOX_SOURCE"
echo "run_mscxrt=$RUN_MS_CXRT"

echo
echo "===== bridge: M3 region cache for regiondiff and tempfuse+M3-delta ====="
"$PY" phase_3/scripts/8-precompute_regions.py \
  --ckpt "$M3_CKPT" \
  --labels-dir "$M3LAB" \
  --features-root "$FEAT" \
  --box-source "$M4_BOX_SOURCE" \
  --out-dir "$CACHE" \
  --batch "$BATCH" \
  --workers "$W" \
  --device "$DEVICE" 2>&1 | tee -a "$LOGDIR/00_bridge.log"

eval_one() {
  local name="$1"
  local variants=()
  if [[ -f "$RUNS/$name/best_acc.pt" || -f "$RUNS/$name/best_prog.pt" || -f "$RUNS/$name/best_change.pt" ]]; then
    [[ -f "$RUNS/$name/best_acc.pt" ]] && variants+=("best_acc:best_acc.pt")
    [[ -f "$RUNS/$name/best_prog.pt" ]] && variants+=("best_prog:best_prog.pt")
    [[ -f "$RUNS/$name/best_change.pt" ]] && variants+=("best_change:best_change.pt")
  else
    variants+=("selected:best.pt")
  fi

  local entry variant ckfile ck suffix stem diag msdiag plot_summary
  local plot_args
  for entry in "${variants[@]}"; do
    variant="${entry%%:*}"
    ckfile="${entry#*:}"
    ck="$RUNS/$name/$ckfile"
    [[ -f "$ck" ]] || { echo "[skip eval] missing checkpoint $ck"; continue; }
    actual_box=$("$PY" -c 'import sys, torch; c=torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(c.get("box_source", "gt"))' "$ck")
    if [[ "$actual_box" != "$M4_BOX_SOURCE" ]]; then
      echo "[ERROR] checkpoint $ck has box_source=$actual_box; expected $M4_BOX_SOURCE" >&2
      exit 2
    fi
    suffix=""
    [[ "$variant" != "selected" ]] && suffix=".$variant"
    stem="$name$suffix"
    diag="$DIAGDIR/$stem.$EVAL_SPLIT.diagnostics.json"
    msdiag="$DIAGDIR/$stem.mscxrt.json"
    if [[ -s "$diag" && "$FORCE" != "1" ]]; then
      echo "[skip eval] $diag exists; set FORCE=1 or pass --force to rerun"
    else
      log_header "$LOGDIR/$stem.$EVAL_SPLIT.eval.log" "eval $stem split=$EVAL_SPLIT ckpt=$ckfile"
      "$PY" phase_4/scripts/3-eval.py \
        --ckpt "$ck" \
        --region-cache "$CACHE" \
        --features-root "$FEAT" \
        --m3-labels-dir "$M3LAB" \
        --m4-labels-dir "$M4LAB" \
        --pairs "$PAIRS" \
        --split "$EVAL_SPLIT" \
        --batch "$BATCH" \
        --workers "$EVAL_W" \
        --device "$DEVICE" \
        --diagnostics-json "$diag" \
        2>&1 | tee -a "$LOGDIR/$stem.$EVAL_SPLIT.eval.log"
    fi

    if [[ "$RUN_MS_CXRT" == "1" && -f "$MS_CSV" ]]; then
      if [[ -s "$msdiag" && "$FORCE" != "1" ]]; then
        echo "[skip MS-CXR-T] $msdiag exists; set FORCE=1 or pass --force to rerun"
      else
        log_header "$LOGDIR/$stem.mscxrt.eval.log" "MS-CXR-T $stem ckpt=$ckfile"
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
          2>&1 | tee -a "$LOGDIR/$stem.mscxrt.eval.log"
      fi
    fi

    plot_summary="$PLOTDIR/$stem/m4_run_summary.csv"
    plot_args=(--diagnostics "$diag" --out-dir "$PLOTDIR/$stem")
    [[ -f "$msdiag" ]] && plot_args+=(--mscxrt "$msdiag")
    [[ -f "$LOGDIR/$name.train.log" ]] && plot_args+=(--train-logs "$LOGDIR/$name.train.log")
    if [[ -s "$plot_summary" && "$FORCE" != "1" ]]; then
      echo "[skip plots] $plot_summary exists; set FORCE=1 or pass --force to rerun"
    else
      log_header "$LOGDIR/$stem.plots.log" "plots $stem"
      "$PY" phase_4/scripts/plot_diagnostics.py "${plot_args[@]}" \
        2>&1 | tee -a "$LOGDIR/$stem.plots.log"
    fi
  done
}

train_one() {
  local base="$1"; shift
  local name="${base}${RUN_SUFFIX}"
  local ck="$RUNS/$name/best.pt"
  local last="$RUNS/$name/last.pt"
  local train_log="$LOGDIR/$name.train.log"
  echo
  echo "===== train $name $* ====="
  if has_done_marker "$train_log" && [[ "$FORCE" != "1" ]]; then
    echo "[skip train] $train_log has [DONE]; set FORCE=1 or pass --force to retrain"
  elif [[ -f "$last" && "$FORCE" != "1" ]]; then
    echo "[resume train] $name from $last; appending to $train_log"
    log_header "$train_log" "resume train $name"
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
      --resume \
      "$@" 2>&1 | tee -a "$train_log"
  elif [[ -f "$ck" && "$FORCE" != "1" ]]; then
    echo "[skip train] $ck exists but no resumable last.pt was found; set FORCE=1 or pass --force to retrain"
  else
    log_header "$train_log" "fresh train $name"
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
      "$@" 2>&1 | tee -a "$train_log"
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
      2>&1 | tee -a "$LOGDIR/$name.train.log"
  fi
  eval_one "$name"
}

ensure_ftcb_cache() {
  CONCEPT_CACHE="${CONCEPT_CACHE:-data/m4_concept_cache${RUN_SUFFIX}}"
  FTCB_M3_CKPT="${FTCB_M3_CKPT:-$(pick_first data/run/m3_B_faithful${RUN_SUFFIX}/best.pt data/run/m3_B_faithful/best.pt)}"
  if [[ ! -d "$CONCEPT_CACHE" || -z "$(ls -A "$CONCEPT_CACHE" 2>/dev/null)" ]]; then
    echo
    echo "===== precompute FTCB concept cache -> $CONCEPT_CACHE (from $FTCB_M3_CKPT) ====="
    "$PY" phase_3/scripts/8-precompute_regions.py \
      --ckpt "$FTCB_M3_CKPT" --labels-dir "$M3LAB" --features-root "$FEAT" \
      --out-dir "$CACHE" --concept-cache-out "$CONCEPT_CACHE" \
      --batch "$BATCH" --workers "$W" --device "$DEVICE"
  fi
}

train_ftcb_one() {
  local base="$1"; shift
  ensure_ftcb_cache
  train_one "$base" $COMMON --arch ftcb --concept-cache "$CONCEPT_CACHE" --no-augment "$@"
}

run_m3delta_promising_matrix() {
  # User-requested m3delta triage: test only the strongest non-m3delta ideas first.
  HYB="$COMMON --arch tempfuse --tempfuse-input-mode feat_logits"
  train_one m4v4_tf_m3delta              $HYB
  train_one m4v4_tf_m3delta_smooth003    $HYB --label-smoothing 0.03
  train_one m4v4_tf_m3delta_smooth005    $HYB --label-smoothing 0.05
  train_one m4v4_tf_m3delta_kl005        $HYB --flip-consistency-weight 0.05
  train_one m4v4_tf_m3delta_dist050      $HYB --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta_dist100      $HYB --distance-penalty-weight 1.00
}

run_cdwh_matrix() {
  # Hybrid CE + lambda*CDW-CE (alpha=5): safety-sensitivity frontier.
  train_one m4v3_tf_cdwh010              $COMMON --arch tempfuse --cdw-weight 0.10 --cdw-alpha 5
  train_one m4v3_tf_cdwh025              $COMMON --arch tempfuse --cdw-weight 0.25 --cdw-alpha 5
  train_one m4v3_tf_cdwh050              $COMMON --arch tempfuse --cdw-weight 0.50 --cdw-alpha 5
}

run_tempfuse_combo_matrix() {
  # Combinations of the current three promising signals: smoothing, flip-KL, and distance penalty.
  train_one m4v3_tf_smooth005_kl005       $COMMON --arch tempfuse --label-smoothing 0.05 --flip-consistency-weight 0.05
  train_one m4v3_tf_smooth005_dist050     $COMMON --arch tempfuse --label-smoothing 0.05 --distance-penalty-weight 0.50
  train_one m4v3_tf_kl005_dist050         $COMMON --arch tempfuse --flip-consistency-weight 0.05 --distance-penalty-weight 0.50
  train_one m4v3_tf_smooth005_kl005_dist050 $COMMON --arch tempfuse --label-smoothing 0.05 --flip-consistency-weight 0.05 --distance-penalty-weight 0.50
}

run_ftcb_matrix() {
  # FTCB variants: faithful concept bottleneck with the same 1/2/3-way regularizer combinations.
  # Distill is intentionally excluded.
  train_ftcb_one m4_ftcb                  --label-smoothing 0.05 --distance-penalty-weight 0.10
  train_ftcb_one m4_ftcb_smooth005        --label-smoothing 0.05
  train_ftcb_one m4_ftcb_kl005            --flip-consistency-weight 0.05
  train_ftcb_one m4_ftcb_dist050          --distance-penalty-weight 0.50
  train_ftcb_one m4_ftcb_smooth005_kl005  --label-smoothing 0.05 --flip-consistency-weight 0.05
  train_ftcb_one m4_ftcb_smooth005_dist050 --label-smoothing 0.05 --distance-penalty-weight 0.50
  train_ftcb_one m4_ftcb_kl005_dist050    --flip-consistency-weight 0.05 --distance-penalty-weight 0.50
  train_ftcb_one m4_ftcb_smooth005_kl005_dist050 --label-smoothing 0.05 --flip-consistency-weight 0.05 --distance-penalty-weight 0.50
}

run_staged_matrix() {
  run_m3delta_promising_matrix
  run_cdwh_matrix
  run_tempfuse_combo_matrix
  echo "[scope] staged queue stops before FTCB; concept bottleneck is out of the active plan"
}

run_m3delta40_matrix() {
  # M3Delta-only long run: 40 epochs, no early stop, save/evaluate best_acc/prog/change.
  EP="${M3DELTA40_EPOCHS:-40}"
  M3D40="--select-metric change --patience 0 --arch tempfuse --tempfuse-input-mode feat_logits"
  train_one m4v4_tf_m3delta40_base               $M3D40
  train_one m4v4_tf_m3delta40_smooth005          $M3D40 --label-smoothing 0.05
  train_one m4v4_tf_m3delta40_kl005              $M3D40 --flip-consistency-weight 0.05
  train_one m4v4_tf_m3delta40_dist050            $M3D40 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_dist100            $M3D40 --distance-penalty-weight 1.00
  train_one m4v4_tf_m3delta40_smooth005_kl005    $M3D40 --label-smoothing 0.05 --flip-consistency-weight 0.05
  train_one m4v4_tf_m3delta40_smooth005_dist050  $M3D40 --label-smoothing 0.05 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_smooth005_dist100  $M3D40 --label-smoothing 0.05 --distance-penalty-weight 1.00
  train_one m4v4_tf_m3delta40_kl005_dist050      $M3D40 --flip-consistency-weight 0.05 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_kl005_dist100      $M3D40 --flip-consistency-weight 0.05 --distance-penalty-weight 1.00
  train_one m4v4_tf_m3delta40_smooth005_kl005_dist050 $M3D40 --label-smoothing 0.05 --flip-consistency-weight 0.05 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_smooth005_kl005_dist100 $M3D40 --label-smoothing 0.05 --flip-consistency-weight 0.05 --distance-penalty-weight 1.00
  run_m3delta_refine_matrix
}

run_m3delta_refine_matrix() {
  run_m3delta_kl_dist_refine_matrix
  run_m3delta_smooth_dist_refine_matrix
}

run_m3delta_kl_dist_refine_matrix() {
  # Focused sweep around the current KL=0.05, distance=0.50 candidate.
  # The center KL=0.05, distance=0.50 run is
  # already complete, so this fills the other eight cells of the local 3x3 grid.
  EP="${M3DELTA40_EPOCHS:-40}"
  M3DREF="--select-metric change --patience 0 --arch tempfuse --tempfuse-input-mode feat_logits"

  # KL x distance local grid: run the four axis neighbors first, then the corners.
  train_one m4v4_tf_m3delta40_kl005_dist035       $M3DREF --flip-consistency-weight 0.05  --distance-penalty-weight 0.35
  train_one m4v4_tf_m3delta40_kl005_dist065       $M3DREF --flip-consistency-weight 0.05  --distance-penalty-weight 0.65
  train_one m4v4_tf_m3delta40_kl0025_dist050     $M3DREF --flip-consistency-weight 0.025 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_kl0075_dist050     $M3DREF --flip-consistency-weight 0.075 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_kl0025_dist035     $M3DREF --flip-consistency-weight 0.025 --distance-penalty-weight 0.35
  train_one m4v4_tf_m3delta40_kl0025_dist065     $M3DREF --flip-consistency-weight 0.025 --distance-penalty-weight 0.65
  train_one m4v4_tf_m3delta40_kl0075_dist035     $M3DREF --flip-consistency-weight 0.075 --distance-penalty-weight 0.35
  train_one m4v4_tf_m3delta40_kl0075_dist065     $M3DREF --flip-consistency-weight 0.075 --distance-penalty-weight 0.65
}

run_m3delta_smooth_dist_refine_matrix() {
  EP="${M3DELTA40_EPOCHS:-40}"
  M3DREF="--select-metric change --patience 0 --arch tempfuse --tempfuse-input-mode feat_logits"

  # Smoothing x distance local grid: run likely axis improvements first, then the corners.
  train_one m4v4_tf_m3delta40_smooth003_dist050  $M3DREF --label-smoothing 0.03 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_smooth005_dist035  $M3DREF --label-smoothing 0.05 --distance-penalty-weight 0.35
  train_one m4v4_tf_m3delta40_smooth005_dist065  $M3DREF --label-smoothing 0.05 --distance-penalty-weight 0.65
  train_one m4v4_tf_m3delta40_smooth0075_dist050 $M3DREF --label-smoothing 0.075 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_smooth003_dist035  $M3DREF --label-smoothing 0.03 --distance-penalty-weight 0.35
  train_one m4v4_tf_m3delta40_smooth003_dist065  $M3DREF --label-smoothing 0.03 --distance-penalty-weight 0.65
  train_one m4v4_tf_m3delta40_smooth0075_dist035 $M3DREF --label-smoothing 0.075 --distance-penalty-weight 0.35
  train_one m4v4_tf_m3delta40_smooth0075_dist065 $M3DREF --label-smoothing 0.075 --distance-penalty-weight 0.65

  # One safety-heavy edge beyond the local grid; run last because dist=1.0 already over-predicted stable.
  train_one m4v4_tf_m3delta40_smooth005_dist075  $M3DREF --label-smoothing 0.05 --distance-penalty-weight 0.75
}

require_detector_scope() {
  if [[ "$M4_BOX_SOURCE" != "detector" ]]; then
    echo "[ERROR] $M4_MATRIX_SCOPE requires M4_BOX_SOURCE=detector" >&2
    exit 2
  fi
}

run_detector_kl_dist_matrix() {
  require_detector_scope
  EP="${M3DELTA40_EPOCHS:-40}"
  local cfg="--select-metric change --patience 0 --box-source $M4_BOX_SOURCE --arch tempfuse --tempfuse-input-mode feat_logits"
  local kl dist ktag dtag kentry dentry
  local kls=("0.025:0025" "0.05:005" "0.075:0075")
  local dists=("0.35:035" "0.50:050" "0.65:065")
  for kentry in "${kls[@]}"; do
    kl="${kentry%%:*}"
    ktag="${kentry#*:}"
    for dentry in "${dists[@]}"; do
      dist="${dentry%%:*}"
      dtag="${dentry#*:}"
      train_one "m4v4_tf_m3delta40_kl${ktag}_dist${dtag}" $cfg \
        --flip-consistency-weight "$kl" --distance-penalty-weight "$dist"
    done
  done
}

run_detector_appendix_matrix() {
  require_detector_scope
  EP="${M3DELTA40_EPOCHS:-40}"
  local cfg="--select-metric change --patience 0 --box-source $M4_BOX_SOURCE --arch tempfuse --tempfuse-input-mode feat_logits"
  train_one m4v4_tf_m3delta40_base                    $cfg
  train_one m4v4_tf_m3delta40_kl005                   $cfg --flip-consistency-weight 0.05
  train_one m4v4_tf_m3delta40_dist050                 $cfg --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_kl005_dist050           $cfg --flip-consistency-weight 0.05 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_smooth005               $cfg --label-smoothing 0.05
  train_one m4v4_tf_m3delta40_smooth005_kl005         $cfg --label-smoothing 0.05 --flip-consistency-weight 0.05
  train_one m4v4_tf_m3delta40_smooth005_dist050       $cfg --label-smoothing 0.05 --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta40_smooth005_kl005_dist050 $cfg --label-smoothing 0.05 --flip-consistency-weight 0.05 --distance-penalty-weight 0.50
}

run_detector_architecture_matrix() {
  require_detector_scope
  EP="${M3DELTA40_EPOCHS:-40}"
  local cfg="--select-metric change --patience 0 --box-source $M4_BOX_SOURCE"
  train_one m4arch_regiondiff40       $cfg --arch regiondiff --input-mode full
  train_one m4arch_tempfuse40         $cfg --arch tempfuse --tempfuse-input-mode feat
  train_one m4arch_m3delta_twostage40 $cfg --arch tempfuse --tempfuse-input-mode feat_logits --head-mode twostage
  train_one m4arch_m3delta_2blocks40  $cfg --arch tempfuse --tempfuse-input-mode feat_logits --fuse-blocks 2
  train_one m4arch_m3delta_sv2stage40 $cfg --arch tempfuse --tempfuse-input-mode feat_logits --same-view --head-mode twostage
}

run_broad_matrix() {
  # Broader controls and v4 ideas. Keep this out of the default restart path
  # so a safety-loss resume does not silently spend time on older broad sweeps.
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
  train_one m4v4_tf_m3delta_opp010       $HYB --opposite-penalty-weight 0.10
  train_one m4v4_tf_m3delta_opp025       $HYB --opposite-penalty-weight 0.25
  train_one m4v4_tf_m3delta_opp050       $HYB --opposite-penalty-weight 0.50
  train_one m4v4_tf_m3delta_dist010      $HYB --distance-penalty-weight 0.10
  train_one m4v4_tf_m3delta_dist025      $HYB --distance-penalty-weight 0.25
  train_one m4v4_tf_m3delta_dist050      $HYB --distance-penalty-weight 0.50
  train_one m4v4_tf_m3delta_dist100      $HYB --distance-penalty-weight 1.00
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

  # Regiondiff controls for checking whether the M3-logit signal alone carries the gain.
  train_one m4v2_regiondiff_full_retrain $COMMON --arch regiondiff --input-mode full
  train_one m4v2_regiondiff_logits       $COMMON --arch regiondiff --input-mode logits
  train_one m4v2_regiondiff_diff         $COMMON --arch regiondiff --input-mode diff

  # FTCB-distill is intentionally not part of the active matrix.
}

case "$M4_MATRIX_SCOPE" in
  m3delta40)
    run_m3delta40_matrix
    echo "[scope] M4_MATRIX_SCOPE=m3delta40; skipping staged/broad/FTCB queues"
    ;;
  m3delta_refine)
    run_m3delta_refine_matrix
    echo "[scope] M4_MATRIX_SCOPE=m3delta_refine; focused coefficient sweep complete"
    ;;
  m3delta_kl_dist_refine)
    run_m3delta_kl_dist_refine_matrix
    echo "[scope] M4_MATRIX_SCOPE=m3delta_kl_dist_refine; KL x distance sweep complete"
    ;;
  m3delta_smooth_dist_refine)
    run_m3delta_smooth_dist_refine_matrix
    echo "[scope] M4_MATRIX_SCOPE=m3delta_smooth_dist_refine; smoothing x distance sweep complete"
    ;;
  detector_kl_dist)
    run_detector_kl_dist_matrix
    echo "[scope] detector KL x distance validation grid complete"
    ;;
  detector_appendix)
    run_detector_appendix_matrix
    echo "[scope] detector appendix regularization rows complete"
    ;;
  detector_architecture)
    run_detector_architecture_matrix
    echo "[scope] detector architecture ablations complete"
    ;;
  detector_paper)
    run_detector_kl_dist_matrix
    run_detector_appendix_matrix
    run_detector_architecture_matrix
    echo "[scope] complete detector paper queue finished"
    ;;
  staged|safety)
    run_staged_matrix
    echo "[scope] M4_MATRIX_SCOPE=$M4_MATRIX_SCOPE; skipping broad controls/v4 matrix"
    ;;
  full)
    run_staged_matrix
    run_broad_matrix
    ;;
  broad)
    run_broad_matrix
    ;;
  *)
    echo "[ERROR] unknown M4_MATRIX_SCOPE=$M4_MATRIX_SCOPE (see detector_kl_dist, detector_appendix, detector_architecture, detector_paper)" >&2
    exit 2
    ;;
esac

if [[ "$RUN_ADAPTERS" == "1" ]]; then
  adapter_one m4v3_tf_retrain head
  adapter_one m4v4_tf_m3delta head
  adapter_one m4v4_tf_m3delta pool-head
  adapter_one m4v4_tf_m3delta_smooth005 head
fi

echo
echo "===== summary: eval logs under $LOGDIR; diagnostics under artifacts/diagnostics ====="
