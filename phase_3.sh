#!/usr/bin/env bash
# Full Phase-3 rerun after the crosswalk/calibration/faithfulness improvements.
#
# This is intentionally bigger than phase_3/run_experiments.sh. It runs:
#   1) crosswalk patch + label re-derive
#   2) full M3 training grid under a fresh tag
#   3) eval + diagnostics + reliability CSV/SVG + faithfulness JSON
#   4) thresholds + concept explanation gate for the ship run
#   5) bootstrap CI from prediction dumps
#   6) M3 inference JSONL for Phase 5
#   7) frozen M3 region cache for Phase 4
#
# Examples:
#   bash phase_3.sh --profile h100mini --tag xwalk_v2
#   bash phase_3.sh --profile h100mini --tag xwalk_v2 --skip-train
#   bash phase_3.sh --profile local4060 --tag smoke --epochs 2 --audit-splits gold

set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

PROFILE="h100mini"
TAG="${TAG:-xwalk_v2}"
DEVICE=""
EP=""
APPLY_CROSSWALK=1
TRAIN=1
FORCE=0
FAITHFUL_ONLY="${FAITHFUL_ONLY:-0}"
RUN_INFER=1
RUN_M4_CACHE=1
RUN_M5_IF_READY=1
RESUME=0
SYNC_REMOTE="${SYNC_REMOTE:-}"
AUDIT_SPLITS="${AUDIT_SPLITS:-val test gold}"
INFER_SPLITS="${INFER_SPLITS:-test gold}"
BOOT_N="${BOOT_N:-1000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --epochs) EP="${2:?}"; shift 2 ;;
    --audit-splits) AUDIT_SPLITS="${2:?}"; shift 2 ;;
    --infer-splits) INFER_SPLITS="${2:?}"; shift 2 ;;
    --boot-n) BOOT_N="${2:?}"; shift 2 ;;
    --no-crosswalk) APPLY_CROSSWALK=0; shift ;;
    --skip-train) TRAIN=0; shift ;;
    --force) FORCE=1; shift ;;
    --faithful-only) FAITHFUL_ONLY=1; shift ;;
    --skip-infer) RUN_INFER=0; shift ;;
    --skip-m4-cache) RUN_M4_CACHE=0; shift ;;
    --skip-m5) RUN_M5_IF_READY=0; shift ;;
    --resume) RESUME=1; shift ;;
    --sync-remote) SYNC_REMOTE="${2:?}"; shift 2 ;;
    -h|--help)
      sed -n '1,32p' "$0"
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
PRED_DUMPS="${PRED_DUMPS:-artifacts/predictions}"
CALIB_DIR="${CALIB_DIR:-artifacts/calibration}"
DATASET_OUT="${DATASET_OUT:-artifacts/dataset/m3_dataset_stats_$TAG}"
CACHE="${CACHE:-data/m4_region_cache_$TAG}"
M3_PRED_DIR="${M3_PRED_DIR:-data}"
M4_PRED="${M4_PRED:-data/m4_pred.jsonl}"
mkdir -p "$RUNS" "$LOGDIR" "$DIAGDIR" "$PRED_DUMPS" "$CALIB_DIR" "$(dirname "$DATASET_OUT")"

echo "===== Phase 3 full profile=$PROFILE tag=$TAG device=$DEVICE workers=$W ====="
echo "python=$PY"
echo "features=$FEAT"
echo "labels=$LABELS"
echo "runs=$RUNS"
echo "audit_splits=$AUDIT_SPLITS"
echo "faithful_only=$FAITHFUL_ONLY"

echo
echo "===== 0) compile Phase-3 scripts ====="
"$PY" -m py_compile \
  phase_3/src/eval.py \
  phase_3/scripts/4-train.py \
  phase_3/scripts/5-eval.py \
  phase_3/scripts/6-faithfulness.py \
  phase_3/scripts/7-infer.py \
  phase_3/scripts/8-precompute_regions.py \
  phase_3/scripts/patch_crosswalk.py \
  phase_3/scripts/dataset_stats.py \
  phase_3/scripts/plot_metrics.py \
  phase_3/scripts/plot_diagnostics.py \
  phase_3/scripts/export_thresholds.py \
  phase_3/scripts/bootstrap_metrics.py

echo
echo "===== 1) dataset stats snapshot ====="
"$PY" phase_3/scripts/dataset_stats.py --labels-dir "$LABELS" --out "$DATASET_OUT" \
  2>&1 | tee "$LOGDIR/00_dataset_stats.log"

echo
echo "===== 2) crosswalk patch ====="
"$PY" phase_3/scripts/patch_crosswalk.py --labels-dir "$LABELS" --dry-run \
  2>&1 | tee "$LOGDIR/01_crosswalk.dryrun.log"
if [[ "$APPLY_CROSSWALK" == "1" ]]; then
  "$PY" phase_3/scripts/patch_crosswalk.py --labels-dir "$LABELS" \
    2>&1 | tee "$LOGDIR/02_crosswalk.apply.log"
else
  echo "[skip] --no-crosswalk set"
fi

train_one() {
  local base="$1"; local box="$2"; shift 2
  local name="${base}_${TAG}"
  local ck="$RUNS/$name/best.pt"
  echo
  echo "===== train/eval $name box=$box $* ====="
  if [[ "$TRAIN" == "1" ]]; then
    if [[ -f "$ck" && "$FORCE" != "1" ]]; then
      echo "[skip train] $ck exists; use --force or a new --tag to retrain"
    else
      local extra=()
      [[ "$RESUME" == "1" ]] && extra+=(--resume)
      [[ -n "$SYNC_REMOTE" ]] && extra+=(--sync-remote "$SYNC_REMOTE")
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
        "${extra[@]}" \
        "$@" 2>&1 | tee "$LOGDIR/$name.train.log"
    fi
  else
    echo "[skip train] --skip-train set"
  fi

  if [[ ! -f "$ck" ]]; then
    echo "[warn] missing $ck; skipping audits for $name" >&2
    return 0
  fi

  local audits_complete=1
  for split in $AUDIT_SPLITS; do
    [[ -f "$DIAGDIR/$name.$split.diagnostics.json" ]] || audits_complete=0
    [[ -f "$DIAGDIR/$name.$split.faithfulness.json" ]] || audits_complete=0
    if [[ "$base" == "m3_B_faithful" || "$base" == m3_Bf_* ]]; then
      [[ -f "$PRED_DUMPS/$name.$split.predictions.npz" ]] || audits_complete=0
    fi
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
    local pred_args=()
    if [[ "$base" == "m3_B_faithful" || "$base" == m3_Bf_* ]]; then
      pred_args+=(--pred-dump "$PRED_DUMPS/$name.$split.predictions.npz")
    fi
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
      "${pred_args[@]}" 2>&1 | tee "$LOGDIR/$name.$split.diagnostics.log"

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

select_faithful_ship() {
  local picked
  local selection="$DIAGDIR/phase3_$TAG.faithful_selection.json"
  if picked=$("$PY" - "$DIAGDIR" "$RUNS" "$TAG" "$selection" "$@" <<'PY'
import json
import math
import sys
from pathlib import Path

diagdir = Path(sys.argv[1])
runs = Path(sys.argv[2])
tag = sys.argv[3]
selection = Path(sys.argv[4])
bases = sys.argv[5:]
rows = []
for base in bases:
    name = f"{base}_{tag}"
    ckpt = runs / name / "best.pt"
    diag_path = diagdir / f"{name}.val.diagnostics.json"
    faith_path = diagdir / f"{name}.val.faithfulness.json"
    test_diag_path = diagdir / f"{name}.test.diagnostics.json"
    row = {
        "name": name,
        "checkpoint": str(ckpt),
        "val_diagnostics": str(diag_path),
        "val_faithfulness": str(faith_path),
        "checkpoint_exists": ckpt.exists(),
        "faithfulness_pass": False,
    }
    if diag_path.exists():
        d = json.loads(diag_path.read_text(encoding="utf-8"))
        row["val_image_auc_macro"] = d.get("image_auc_macro")
        row["val_image_f1_macro"] = d.get("image_f1_macro")
        row["val_concept_f1_macro"] = d.get("concept_f1_macro")
    if test_diag_path.exists():
        d = json.loads(test_diag_path.read_text(encoding="utf-8"))
        row["test_image_auc_macro"] = d.get("image_auc_macro")
        row["test_image_f1_macro"] = d.get("image_f1_macro")
        row["test_concept_f1_macro"] = d.get("concept_f1_macro")
    if faith_path.exists():
        f = json.loads(faith_path.read_text(encoding="utf-8"))
        row["faithfulness_pass"] = bool(f.get("why_faithful_allowed"))
        row["val_concept_go_no_go"] = bool(f.get("concept_go_no_go"))
        row["val_intervention_pass"] = bool(f.get("intervention_pass"))
    rows.append(row)

def good_number(x):
    return isinstance(x, (int, float)) and not math.isnan(float(x))

eligible = [
    r for r in rows
    if r["checkpoint_exists"] and r["faithfulness_pass"] and good_number(r.get("val_image_auc_macro"))
]
eligible.sort(
    key=lambda r: (
        float(r.get("val_image_auc_macro", -1.0)),
        float(r.get("val_image_f1_macro", -1.0)) if good_number(r.get("val_image_f1_macro")) else -1.0,
        float(r.get("val_concept_f1_macro", -1.0)) if good_number(r.get("val_concept_f1_macro")) else -1.0,
    ),
    reverse=True,
)
out = {
    "selection_rule": "highest val_image_auc_macro among detector-box B-faithful variants with why_faithful_allowed=true on val",
    "selected": eligible[0]["name"] if eligible else None,
    "candidates": rows,
}
selection.parent.mkdir(parents=True, exist_ok=True)
selection.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
if not eligible:
    raise SystemExit(1)
print(eligible[0]["name"])
PY
  ); then
    SHIP="$picked"
    SHIP_CKPT="$RUNS/$SHIP/best.pt"
    echo "[select] faithful ship=$SHIP"
    echo "[select] summary=$selection"
  else
    SHIP="m3_B_faithful_$TAG"
    SHIP_CKPT="$RUNS/$SHIP/best.pt"
    echo "[warn] no eligible faithful variant found; falling back to $SHIP" >&2
    echo "[warn] selection summary=$selection" >&2
  fi
}

echo
echo "===== 3) faithful detector variants first ====="
FAITHFUL_CANDIDATES=(
  m3_B_faithful
  m3_Bf_aggmax
  m3_Bf_aggmean
  m3_Bf_noglobal
)
train_one m3_B_faithful  detector  --mode B --disease-head faithful
train_one m3_Bf_aggmax   detector  --mode B --disease-head faithful --region-agg max
train_one m3_Bf_aggmean  detector  --mode B --disease-head faithful --region-agg mean
train_one m3_Bf_noglobal detector  --mode B --disease-head faithful --no-global-head

if [[ -n "${SHIP_M3_NAME:-}" ]]; then
  SHIP="$SHIP_M3_NAME"
  SHIP_CKPT="$RUNS/$SHIP/best.pt"
  echo "[select] SHIP_M3_NAME override: $SHIP"
else
  select_faithful_ship "${FAITHFUL_CANDIDATES[@]}"
fi

if [[ "$FAITHFUL_ONLY" == "1" ]]; then
  echo
  echo "===== 3b) skipped non-faithful/full-grid ablations (--faithful-only) ====="
else
  echo
  echo "===== 3b) remaining full-grid ablations ====="
  train_one m3_A           detector  --mode A
  train_one m3_B           detector  --mode B --disease-head mlp
  train_one m3_B_linear    detector  --mode B --disease-head linear
  train_one m3_C           detector  --mode C
  train_one m3_Bf_gtbox    gt        --mode B --disease-head faithful
fi

echo
echo "===== 4) thresholds, concept gates, bootstrap for ship run ====="
SHIP_TEST_DIAG="$DIAGDIR/$SHIP.test.diagnostics.json"
if [[ -f "$SHIP_TEST_DIAG" ]]; then
  "$PY" phase_3/scripts/export_thresholds.py \
    --diagnostics "$SHIP_TEST_DIAG" \
    --thresholds-json "$CALIB_DIR/$SHIP.thresholds.json" \
    --concept-gate-json "$CALIB_DIR/$SHIP.concept_gate.json" \
    2>&1 | tee "$LOGDIR/$SHIP.thresholds.log"
fi

for split in $AUDIT_SPLITS; do
  dump="$PRED_DUMPS/$SHIP.$split.predictions.npz"
  if [[ -f "$dump" ]]; then
    "$PY" phase_3/scripts/bootstrap_metrics.py \
      --pred-dump "$dump" \
      --out-json "$DIAGDIR/$SHIP.$split.bootstrap.image.json" \
      --level image \
      --n "$BOOT_N" 2>&1 | tee "$LOGDIR/$SHIP.$split.bootstrap.image.log"
    "$PY" phase_3/scripts/bootstrap_metrics.py \
      --pred-dump "$dump" \
      --out-json "$DIAGDIR/$SHIP.$split.bootstrap.region.json" \
      --level region \
      --n "$BOOT_N" 2>&1 | tee "$LOGDIR/$SHIP.$split.bootstrap.region.log"
    "$PY" phase_3/scripts/bootstrap_metrics.py \
      --pred-dump "$dump" \
      --out-json "$DIAGDIR/$SHIP.$split.bootstrap.concept.json" \
      --level concept \
      --n "$BOOT_N" 2>&1 | tee "$LOGDIR/$SHIP.$split.bootstrap.concept.log"
  fi
done

echo
echo "===== 5) training curves / comparison ====="
runs=("$RUNS"/*_"$TAG")
if [[ -e "${runs[0]}" ]]; then
  "$PY" phase_3/scripts/plot_metrics.py "${runs[@]}" \
    --out "$DIAGDIR/phase3_$TAG.compare.png" 2>&1 | tee "$LOGDIR/phase3_$TAG.plot_metrics.log"
fi

if [[ "$RUN_INFER" == "1" && -f "$SHIP_CKPT" ]]; then
  echo
  echo "===== 6) M3 inference JSONL for Phase 5 ====="
  for split in $INFER_SPLITS; do
    out="$M3_PRED_DIR/m3_pred.$split.$TAG.jsonl"
    "$PY" phase_3/scripts/7-infer.py \
      --ckpt "$SHIP_CKPT" \
      --labels-dir "$LABELS" \
      --features-root "$FEAT" \
      --split "$split" \
      --box-source detector \
      --out "$out" \
      --topk-concepts 8 \
      --topk-cells 3 \
      --batch "$EVAL_BATCH" \
      --workers "$W" \
      --device "$DEVICE" 2>&1 | tee "$LOGDIR/$SHIP.infer.$split.log"
    if [[ "$split" == "test" ]]; then
      cp "$out" "$M3_PRED_DIR/m3_pred.jsonl"
      echo "[m5-default] copied $out -> $M3_PRED_DIR/m3_pred.jsonl"
    fi
  done
fi

if [[ "$RUN_M4_CACHE" == "1" && -f "$SHIP_CKPT" ]]; then
  echo
  echo "===== 7) frozen M3 region cache for Phase 4 ====="
  "$PY" phase_3/scripts/8-precompute_regions.py \
    --ckpt "$SHIP_CKPT" \
    --labels-dir "$LABELS" \
    --features-root "$FEAT" \
    --out-dir "$CACHE" \
    --batch "$EVAL_BATCH" \
    --workers "$W" \
    --device "$DEVICE" 2>&1 | tee "$LOGDIR/$SHIP.precompute_regions.log"
fi

if [[ "$RUN_M5_IF_READY" == "1" && -f "$M3_PRED_DIR/m3_pred.jsonl" && -f "$M4_PRED" ]]; then
  echo
  echo "===== 8) optional M5 report assembly if M4 prediction exists ====="
  "$PY" phase_5/run.py \
    --m3-pred "$M3_PRED_DIR/m3_pred.jsonl" \
    --m4-pred "$M4_PRED" \
    --out "$DIAGDIR/$SHIP.m5_reports.jsonl" \
    --stats-json "$DIAGDIR/$SHIP.m5_stats.json" \
    2>&1 | tee "$LOGDIR/$SHIP.m5.log"
fi

echo
echo "===== DONE Phase 3 full ====="
echo "ship_run=$SHIP"
echo "ship_ckpt=$SHIP_CKPT"
echo "thresholds=$CALIB_DIR/$SHIP.thresholds.json"
echo "concept_gate=$CALIB_DIR/$SHIP.concept_gate.json"
echo "m3_pred=$M3_PRED_DIR/m3_pred.jsonl"
echo "m4_region_cache=$CACHE"
