#!/usr/bin/env bash
# Retrain the Stage-2/M3 paper matrix under the final detector-first protocol.
# Run from the repository root. All paths can be overridden with environment variables.

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

PROFILE="h100mini"
SCOPE="all"
EP=""
FORCE_EVAL="${FORCE_EVAL:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --scope) SCOPE="${2:?}"; shift 2 ;;
    --epochs) EP="${2:?}"; shift 2 ;;
    --force-eval) FORCE_EVAL=1; shift ;;
    -h|--help)
      echo "Usage: bash phase_3/run_paper_m3_v2.sh [--profile h100mini|local4060] [--scope preflight|all|train|eval|main|gt] [--epochs N] [--force-eval]"
      exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  h100mini)
    W="${W:-16}"; EVAL_W="${EVAL_W:-24}"; BATCH="${BATCH:-128}"; EP="${EP:-40}" ;;
  local4060)
    W="${W:-4}"; EVAL_W="${EVAL_W:-4}"; BATCH="${BATCH:-16}"; EP="${EP:-40}" ;;
  *) echo "[ERROR] profile must be h100mini or local4060" >&2; exit 2 ;;
esac

if [[ -x .venv/bin/python ]]; then
  PY="${PY:-.venv/bin/python}"
elif [[ -x .venv/Scripts/python.exe ]]; then
  PY="${PY:-.venv/Scripts/python.exe}"
else
  PY="${PY:-python3}"
fi

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LABELS="${LABELS:-data/m3_labels}"
FEAT="${FEAT:-data/features/frozen}"
RUNS="${RUNS:-data/run}"
LOGDIR="${LOGDIR:-logs/m3_paper_v2}"
DIAGDIR="${DIAGDIR:-artifacts/diagnostics/m3_paper_v2}"
SYNC_REMOTE="${SYNC_REMOTE:-}"
SYNC_EVERY="${SYNC_EVERY:-0}"
mkdir -p "$RUNS" "$LOGDIR" "$DIAGDIR"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1" >&2; exit 2; }; }

for f in manifest.jsonl region_concepts.npy region_chexpert.npy image_chexpert.npy \
         boxes.npy present_mask.npy boxes_det.npy present_mask_det.npy; do
  require_file "$LABELS/$f"
done
require_dir "$FEAT"
require_file data/m3_concept_space.json
require_file phase_3/src/m3_concept_space.json
require_file phase_4/src/m3_concept_space.json

"$PY" - <<'PY'
import json
from pathlib import Path
paths = [Path("data/m3_concept_space.json"), Path("phase_3/src/m3_concept_space.json"),
         Path("phase_4/src/m3_concept_space.json")]
objects = [json.loads(p.read_text(encoding="utf-8-sig")) for p in paths]
if not all(x == objects[0] for x in objects[1:]):
    raise SystemExit("[ERROR] concept-space JSON files differ; synchronize them before training")
concepts = objects[0].get("concepts", objects[0])
if len(concepts) != 69:
    raise SystemExit(f"[ERROR] expected 69 concepts, found {len(concepts)}")
print("[preflight] concept-space copies agree (69 concepts)")
PY

if [[ "$SCOPE" == "preflight" ]]; then
  echo "[DONE] M3 preflight passed; no training was started."
  exit 0
fi

declare -a RUN_NAMES=()
declare -A RUN_BOX
declare -A RUN_ARGS

add_run() {
  local name="$1" box="$2"; shift 2
  RUN_NAMES+=("$name")
  RUN_BOX["$name"]="$box"
  RUN_ARGS["$name"]="$*"
}

# Main and paper ablations. Shared settings are explicit so config defaults cannot drift.
MAIN_FLAGS="--mode B --head-type mlp --disease-head faithful --detach-concept --derive-no-finding --no-global-head --region-agg lse"
add_run m3v2_vera_graph_lse_det detector $MAIN_FLAGS
add_run m3v2_no_concept_det detector --mode A --head-type mlp --derive-no-finding --no-global-head --region-agg lse
add_run m3v2_concept_mlp_det detector --mode B --head-type mlp --disease-head mlp --detach-concept --derive-no-finding --no-global-head --region-agg lse
add_run m3v2_graph_global_fusion_det detector --mode B --head-type mlp --disease-head faithful --detach-concept --derive-no-finding --global-head --region-agg lse
add_run m3v2_global_only_det detector --mode A --head-type mlp --global-only --derive-no-finding --no-global-head --region-agg lse
add_run m3v2_graph_attention_det detector --mode B --head-type mlp --disease-head faithful --detach-concept --derive-no-finding --no-global-head --region-agg attention
add_run m3v2_graph_mean_det detector --mode B --head-type mlp --disease-head faithful --detach-concept --derive-no-finding --no-global-head --region-agg mean
add_run m3v2_graph_max_det detector --mode B --head-type mlp --disease-head faithful --detach-concept --derive-no-finding --no-global-head --region-agg max
add_run m3v2_vera_graph_lse_gt gt $MAIN_FLAGS

if [[ "$SCOPE" == "main" ]]; then
  RUN_NAMES=(m3v2_vera_graph_lse_det)
elif [[ "$SCOPE" == "gt" ]]; then
  RUN_NAMES=(m3v2_vera_graph_lse_gt)
elif [[ "$SCOPE" != "all" && "$SCOPE" != "train" && "$SCOPE" != "eval" ]]; then
  echo "[ERROR] scope must be preflight, all, train, eval, main, or gt" >&2; exit 2
fi

# Kaggle can run independent retained rows on separate GPUs.  Keep the paper
# matrix in one place and optionally select one row without changing flags.
if [[ -n "${RUN_NAME:-}" ]]; then
  found=0
  for name in "${RUN_NAMES[@]}"; do [[ "$name" == "$RUN_NAME" ]] && found=1; done
  [[ "$found" == 1 ]] || { echo "[ERROR] RUN_NAME not in selected M3 scope: $RUN_NAME" >&2; exit 2; }
  RUN_NAMES=("$RUN_NAME")
fi
if [[ -n "${RUN_INDEX:-}" ]]; then
  [[ "$RUN_INDEX" =~ ^[0-9]+$ ]] || { echo "[ERROR] RUN_INDEX must be a zero-based integer" >&2; exit 2; }
  (( RUN_INDEX < ${#RUN_NAMES[@]} )) || { echo "[ERROR] RUN_INDEX out of range: $RUN_INDEX" >&2; exit 2; }
  RUN_NAMES=("${RUN_NAMES[$RUN_INDEX]}")
fi

is_complete() {
  local last="$1"
  [[ -f "$last" ]] || return 1
  local completed
  completed=$("$PY" - "$last" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(c.get("epoch", -1)) + 1)
PY
)
  [[ "$completed" -ge "$EP" ]]
}

train_one() {
  local name="$1" box="${RUN_BOX[$1]}" args="${RUN_ARGS[$1]}"
  local run="$RUNS/$name" log="$LOGDIR/$name.train.log"
  if is_complete "$run/last.pt"; then
    echo "[skip train] $name already reached at least $EP epochs"
    return
  fi
  if [[ -f "$run/best.pt" && ! -f "$run/last.pt" ]]; then
    echo "[ERROR] $run has best.pt but no last.pt; refusing to overwrite a non-resumable run" >&2
    exit 2
  fi
  local resume=() sync=()
  # On an ephemeral Kaggle session the local run directory starts empty.  If
  # a Drive remote is configured, --resume first pulls the remote run and then
  # continues from last.pt (or starts fresh when the remote has no checkpoint).
  [[ -f "$run/last.pt" || -n "$SYNC_REMOTE" ]] && resume=(--resume)
  [[ -n "$SYNC_REMOTE" ]] && sync=(--sync-remote "$SYNC_REMOTE" --sync-every "$SYNC_EVERY")
  echo "===== train $name (box=$box) =====" | tee -a "$log"
  # RUN_ARGS contains only controlled flags declared above.
  # shellcheck disable=SC2086
  "$PY" phase_3/scripts/4-train.py \
    --labels-dir "$LABELS" --features-root "$FEAT" --out "$RUNS" --name "$name" \
    --epochs "$EP" --batch "$BATCH" --workers "$W" --device "$DEVICE" \
    --box-source "$box" --select-by auc --seed "$SEED" $args "${resume[@]}" "${sync[@]}" \
    2>&1 | tee -a "$log"
}

eval_one() {
  local name="$1" box="${RUN_BOX[$1]}" ck="$RUNS/$name/best.pt"
  require_file "$ck"
  for split in val test; do
    local diag="$DIAGDIR/$name.$split.json" pred="$DIAGDIR/$name.$split.pred.npz"
    if [[ -s "$diag" && -s "$pred" && "$FORCE_EVAL" != "1" ]]; then
      echo "[skip eval] $name/$split artifacts already exist"
    else
      "$PY" phase_3/scripts/5-eval.py \
        --ckpt "$ck" --labels-dir "$LABELS" --features-root "$FEAT" \
        --box-source "$box" --split "$split" --batch "$BATCH" --workers "$EVAL_W" \
        --device "$DEVICE" --diagnostics-json "$diag" --pred-dump "$pred" \
        --min-region-pos 30 --min-region-neg 30 \
        2>&1 | tee "$LOGDIR/$name.$split.eval.log"
    fi
    if [[ "${RUN_ARGS[$name]}" != *"--global-only"* ]]; then
      local report_dir="$DIAGDIR/$name.$split.regions"
      if [[ ! -s "$report_dir/regional_audit.md" || "$FORCE_EVAL" == "1" ]]; then
        "$PY" phase_3/scripts/10-region-report.py \
          --diagnostics "$diag" --out-dir "$report_dir" \
          2>&1 | tee "$LOGDIR/$name.$split.region-report.log"
      fi
    fi
  done
  if [[ "${RUN_ARGS[$name]}" == *"--mode B"* ]]; then
    "$PY" phase_3/scripts/6-faithfulness.py \
      --ckpt "$ck" --labels-dir "$LABELS" --features-root "$FEAT" --box-source "$box" \
      --split test --batch "$BATCH" --workers "$EVAL_W" --device "$DEVICE" \
      --diagnostics-json "$DIAGDIR/$name.faithfulness.json" \
      2>&1 | tee "$LOGDIR/$name.faithfulness.log"
  fi
}

echo "[M3 paper v2] profile=$PROFILE scope=$SCOPE epochs=$EP batch=$BATCH workers=$W"
echo "[M3 paper v2] features=$FEAT labels=$LABELS runs=$RUNS"

if [[ "$SCOPE" != "eval" ]]; then
  for name in "${RUN_NAMES[@]}"; do train_one "$name"; done
fi
if [[ "$SCOPE" != "train" ]]; then
  for name in "${RUN_NAMES[@]}"; do eval_one "$name"; done
fi

echo "[DONE] M3 paper matrix complete. Diagnostics: $DIAGDIR"
