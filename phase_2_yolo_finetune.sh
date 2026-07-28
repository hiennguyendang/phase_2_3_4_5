#!/usr/bin/env bash
# Phase-2 detector WARM-START fine-tune: continue from an existing best.pt at a
# higher resolution on a data subset, then eval + B1 audit.
#
# Why warm-start (NOT --resume):
#   `train_yolo.py --resume` reloads last.pt WITH its optimizer state and keeps the
#   SAME imgsz/epoch budget -- it cannot change resolution or fraction. To bump imgsz
#   720 and switch to 1/3 data we instead pass best.pt as the initial --model and run
#   a fresh short fine-tune (this is the imgsz-640/720 warm-start recommended in
#   docs/VERA_phase2_yolo_results.md for the small landmark regions).
#
# Examples:
#   bash phase_2_yolo_finetune.sh --profile h100mini
#   bash phase_2_yolo_finetune.sh --profile local4060 --epochs 10
#   bash phase_2_yolo_finetune.sh --profile h100mini --weights weight/dect/best.pt \
#        --yolo-ds data/yolo_ds --imgsz 720 --fraction 0.333

set -euo pipefail

PROFILE="h100mini"
WEIGHTS="${WEIGHTS:-weight/dect/best.pt}"   # existing checkpoint to warm-start from
YOLO_DS="${YOLO_DS:-data/yolo_ds}"          # dir with images/ labels/ dataset.yaml
DATA=""                                     # dataset.yaml (defaults to $YOLO_DS/dataset.yaml)
IMGSZ="${IMGSZ:-720}"                       # NOTE: YOLO rounds to a multiple of 32 (720 -> 736)
FRACTION="${FRACTION:-0.333}"               # 1/3 of the train set
EPOCHS="${EPOCHS:-15}"                      # warm-start is short; convergence is fast from best.pt
NAME="${NAME:-det29_ft${IMGSZ}}"
DEVICE=""
BATCH=""
WORKERS=""
EVAL_SPLITS="${EVAL_SPLITS:-val test}"      # 'gold' is routed to test via config.SPLIT_MAP
RUN_AUDIT=1
RUN_TRAIN=1
SYNC_REMOTE="${SYNC_REMOTE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?}"; shift 2 ;;
    --weights) WEIGHTS="${2:?}"; shift 2 ;;
    --yolo-ds) YOLO_DS="${2:?}"; shift 2 ;;
    --data) DATA="${2:?}"; shift 2 ;;
    --imgsz) IMGSZ="${2:?}"; NAME="det29_ft${IMGSZ}"; shift 2 ;;
    --fraction) FRACTION="${2:?}"; shift 2 ;;
    --epochs) EPOCHS="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --batch) BATCH="${2:?}"; shift 2 ;;
    --workers) WORKERS="${2:?}"; shift 2 ;;
    --eval-splits) EVAL_SPLITS="${2:?}"; shift 2 ;;
    --skip-train) RUN_TRAIN=0; shift ;;
    --skip-audit) RUN_AUDIT=0; shift ;;
    --sync-remote) SYNC_REMOTE="${2:?}"; shift 2 ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# imgsz 720 costs ~2.5x the memory of 448, so batches are smaller than the base train.
case "$PROFILE" in
  local4060)
    DEVICE="${DEVICE:-0}"; BATCH="${BATCH:-4}"; WORKERS="${WORKERS:-4}" ;;
  h100mini)
    DEVICE="${DEVICE:-0}"; BATCH="${BATCH:-16}"; WORKERS="${WORKERS:-16}" ;;
  *) echo "[ERROR] profile must be local4060 or h100mini" >&2; exit 2 ;;
esac

DATA="${DATA:-$YOLO_DS/dataset.yaml}"

if [[ -x ".venv/Scripts/python.exe" ]]; then
  PY="${PY:-.venv/Scripts/python.exe}"
elif [[ -x ".venv/bin/python" ]]; then
  PY="${PY:-.venv/bin/python}"
else
  PY="${PY:-python3}"
fi

TRAIN_PY="phase_2/scripts/yolo/3-train_yolo.py"
EVAL_PY="phase_2/scripts/yolo/4-eval_yolo.py"
AUDIT_PY="phase_2/scripts/yolo/audit_yolo.py"
RUNS="${RUNS:-phase_2/_work/runs}"
LOGDIR="${LOGDIR:-logs/phase2_ft_${IMGSZ}}"
AUDITDIR="${AUDITDIR:-artifacts/phase2_audit}"
mkdir -p "$RUNS" "$LOGDIR" "$AUDITDIR"

FT_CKPT="$RUNS/$NAME/weights/best.pt"

echo "===== Phase-2 YOLO fine-tune profile=$PROFILE ====="
echo "python       = $PY"
echo "warm-start   = $WEIGHTS"
echo "dataset.yaml = $DATA"
echo "yolo-ds      = $YOLO_DS"
echo "imgsz        = $IMGSZ  (YOLO rounds to nearest 32: 720 -> 736)"
echo "fraction     = $FRACTION"
echo "epochs       = $EPOCHS   batch=$BATCH   workers=$WORKERS   device=$DEVICE"
echo "out ckpt     = $FT_CKPT"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[ERROR] warm-start weights not found: $WEIGHTS" >&2; exit 1
fi
if [[ ! -f "$DATA" ]]; then
  echo "[ERROR] dataset.yaml not found: $DATA (build it with build_yolo_dataset.py / link_yolo_images.py)" >&2; exit 1
fi

echo
echo "===== 0) syntax check ====="
"$PY" -m py_compile "$TRAIN_PY" "$EVAL_PY" "$AUDIT_PY"

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo
  echo "===== 1) warm-start fine-tune ($IMGSZ, fraction $FRACTION) ====="
  extra=()
  [[ -n "$SYNC_REMOTE" ]] && extra+=(--sync-remote "$SYNC_REMOTE")
  "$PY" "$TRAIN_PY" \
    --model "$WEIGHTS" \
    --data "$DATA" \
    --runs "$RUNS" \
    --name "$NAME" \
    --imgsz "$IMGSZ" \
    --fraction "$FRACTION" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    "${extra[@]}" 2>&1 | tee "$LOGDIR/train.log"
else
  echo "[skip] --skip-train set"
fi

if [[ ! -f "$FT_CKPT" ]]; then
  echo "[warn] fine-tuned checkpoint missing: $FT_CKPT (skipping eval/audit)" >&2
  exit 0
fi

echo
echo "===== 2) mAP eval (at fine-tune imgsz $IMGSZ) ====="
for split in $EVAL_SPLITS; do
  "$PY" "$EVAL_PY" \
    --weights "$FT_CKPT" \
    --data "$DATA" \
    --split "$split" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device "$DEVICE" 2>&1 | tee "$LOGDIR/eval.$split.log"
done

if [[ "$RUN_AUDIT" == "1" ]]; then
  echo
  echo "===== 3) B1 audit (static-prior + atypicality strata + perturbation) ====="
  "$PY" "$AUDIT_PY" \
    --weights "$FT_CKPT" \
    --yolo-ds "$YOLO_DS" \
    --split val \
    --imgsz "$IMGSZ" \
    --device "$DEVICE" \
    --out "$AUDITDIR/$NAME.audit.json" 2>&1 | tee "$LOGDIR/audit.log"
fi

echo
echo "===== DONE Phase-2 fine-tune ====="
echo "fine_tuned_ckpt = $FT_CKPT"
echo "eval_logs       = $LOGDIR/eval.*.log"
[[ "$RUN_AUDIT" == "1" ]] && echo "audit_json      = $AUDITDIR/$NAME.audit.json"
echo "compare vs base 448: mAP50 0.931 / mAP50-95 0.694 / IoU 0.807"
