#!/usr/bin/env bash
# Retrain the Stage-3/M4 paper matrix from the frozen final M3 checkpoint.
# Run from the repository root. Threshold calibration and Stage-4 report rendering are excluded.

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
      echo "Usage: bash phase_4/run_paper_m4_v2.sh [--profile h100mini|local4060] [--scope preflight|all|grid|paper|final] [--epochs N] [--force-eval]"
      exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  h100mini)
    W="${W:-16}"; EVAL_W="${EVAL_W:-24}"; BATCH="${BATCH:-128}"; EP="${EP:-40}" ;;
  local4060)
    W="${W:-4}"; EVAL_W="${EVAL_W:-4}"; BATCH="${BATCH:-12}"; EP="${EP:-40}" ;;
  *) echo "[ERROR] profile must be h100mini or local4060" >&2; exit 2 ;;
esac
case "$SCOPE" in preflight|all|grid|paper|final) ;; *) echo "[ERROR] invalid scope: $SCOPE" >&2; exit 2 ;; esac

if [[ -x .venv/bin/python ]]; then
  PY="${PY:-.venv/bin/python}"
elif [[ -x .venv/Scripts/python.exe ]]; then
  PY="${PY:-.venv/Scripts/python.exe}"
else
  PY="${PY:-python3}"
fi

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
M3_CKPT="${M3_CKPT:-data/run/m3v2_vera_graph_lse_det/best.pt}"
M3_GT_CKPT="${M3_GT_CKPT:-data/run/m3v2_vera_graph_lse_gt/best.pt}"
M3LAB="${M3LAB:-data/m3_labels}"
M4LAB="${M4LAB:-data/m4_labels}"
PAIRS="${PAIRS:-data/m4_labels/m3_pairs.jsonl}"
FEAT="${FEAT:-data/features/frozen}"
CACHE_DET="${CACHE_DET:-data/m4_region_cache_m3v2_detector}"
CACHE_GT="${CACHE_GT:-data/m4_region_cache_m3v2_gt_oracle}"
RUNS="${RUNS:-data/run}"
LOGDIR="${LOGDIR:-logs/m4_paper_v2}"
DIAGDIR="${DIAGDIR:-artifacts/diagnostics/m4_paper_v2}"
SELECTED_ENV="${SELECTED_ENV:-$DIAGDIR/selected_coefficients.env}"
MS_CSV="${MS_CSV:-data/MS_CXR_T_temporal_image_classification_v1.0.0.csv}"
SYNC_REMOTE="${SYNC_REMOTE:-}"
SYNC_EVERY="${SYNC_EVERY:-0}"
mkdir -p "$RUNS" "$LOGDIR" "$DIAGDIR"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1" >&2; exit 2; }; }
require_file "$M3_CKPT"
require_dir "$FEAT"
for f in manifest.jsonl boxes.npy present_mask.npy boxes_det.npy present_mask_det.npy; do require_file "$M3LAB/$f"; done
for f in manifest.jsonl progression.npy; do require_file "$M4LAB/$f"; done
require_file "$PAIRS"

verify_m3_ckpt() {
  local ckpt="$1" expected_box="$2"
  "$PY" - "$ckpt" "$expected_box" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = c.get("cfg", {})
expected = {
    "DISEASE_HEAD": "faithful", "DETACH_CONCEPT_FOR_DISEASE": True,
    "DERIVE_NO_FINDING": True, "USE_GLOBAL_HEAD": False, "REGION_AGG": "lse",
}
bad = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
if c.get("mode") != "B" or bad:
    raise SystemExit(f"[ERROR] M4 requires final M3 mode B and v2 main config; mode={c.get('mode')} mismatches={bad}")
if c.get("box_source") != sys.argv[2]:
    raise SystemExit(f"[ERROR] M3 checkpoint box_source={c.get('box_source')} expected={sys.argv[2]}")
print(f"[preflight] final M3 checkpoint contract verified (box={sys.argv[2]})")
PY
}
verify_m3_ckpt "$M3_CKPT" detector

if [[ "$SCOPE" == "preflight" ]]; then
  echo "[DONE] M4 preflight passed; no cache or training was started."
  exit 0
fi

prepare_cache() {
  local box="$1" cache="$2" m3_ckpt="$3" marker="$2/.m3_source.json"
  mkdir -p "$cache"
  "$PY" - "$m3_ckpt" "$box" "$cache" "$marker" <<'PY'
import hashlib, json, sys
from pathlib import Path
ckpt, box, cache, marker = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
h = hashlib.sha256()
with ckpt.open("rb") as f:
    for block in iter(lambda: f.read(8 << 20), b""):
        h.update(block)
want = {"checkpoint": str(ckpt), "sha256": h.hexdigest(), "box_source": box}
if marker.exists():
    got = json.loads(marker.read_text(encoding="utf-8"))
    if got != want:
        raise SystemExit(f"[ERROR] stale cache provenance in {cache}; choose a new CACHE_{box.upper()} path")
elif any(cache.glob("*.npy")):
    raise SystemExit(f"[ERROR] unverified existing cache {cache}; choose an empty/new cache path")
marker.write_text(json.dumps(want, indent=2), encoding="utf-8")
print(f"[cache provenance] {box}: {want['sha256'][:12]} -> {cache}")
PY
  "$PY" phase_3/scripts/8-precompute_regions.py \
    --ckpt "$m3_ckpt" --labels-dir "$M3LAB" --features-root "$FEAT" \
    --box-source "$box" --out-dir "$cache" --batch "$BATCH" --workers "$W" \
    --device "$DEVICE" 2>&1 | tee -a "$LOGDIR/00_cache_${box}.log"
}

slug() { "$PY" - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.3f}".replace(".", ""))
PY
}

declare -a GRID_NAMES=()
declare -a PAPER_NAMES=()
declare -A RUN_BOX RUN_CACHE RUN_ARGS ADDED

add_run() {
  local list="$1" name="$2" box="$3" cache="$4"; shift 4
  if [[ -n "${ADDED[$name]:-}" ]]; then return; fi
  ADDED["$name"]=1; RUN_BOX["$name"]="$box"; RUN_CACHE["$name"]="$cache"; RUN_ARGS["$name"]="$*"
  if [[ "$list" == grid ]]; then GRID_NAMES+=("$name"); else PAPER_NAMES+=("$name"); fi
}

COMMON="--head-type mlp --loss ce --select-metric change --patience 0"
KL_VALUES=(0.025 0.050 0.075)
DIST_VALUES=(0.350 0.500 0.650)
for kl in "${KL_VALUES[@]}"; do
  for dist in "${DIST_VALUES[@]}"; do
    name="m4v2_grid_kl$(slug "$kl")_dist$(slug "$dist")_det"
    add_run grid "$name" detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits \
      --flip-consistency-weight "$kl" --distance-penalty-weight "$dist"
  done
done

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
  local name="$1" box="${RUN_BOX[$1]}" cache="${RUN_CACHE[$1]}" args="${RUN_ARGS[$1]}"
  local run="$RUNS/$name" log="$LOGDIR/$name.train.log"
  if is_complete "$run/last.pt"; then echo "[skip train] $name already reached $EP epochs"; return; fi
  if [[ -f "$run/best.pt" && ! -f "$run/last.pt" ]]; then
    echo "[ERROR] $run has best.pt but no last.pt; refusing to overwrite it" >&2; exit 2
  fi
  local resume=() sync=()
  # A new Kaggle session has no local checkpoint.  --resume also pulls the
  # named run from SYNC_REMOTE, and safely starts fresh if it is absent there.
  [[ -f "$run/last.pt" || -n "$SYNC_REMOTE" ]] && resume=(--resume)
  [[ -n "$SYNC_REMOTE" ]] && sync=(--sync-remote "$SYNC_REMOTE" --sync-every "$SYNC_EVERY")
  echo "===== train $name (box=$box) =====" | tee -a "$log"
  # shellcheck disable=SC2086
  "$PY" phase_4/scripts/2-train.py \
    --region-cache "$cache" --features-root "$FEAT" --box-source "$box" \
    --m3-labels-dir "$M3LAB" --m4-labels-dir "$M4LAB" --pairs "$PAIRS" \
    --out "$RUNS" --name "$name" --epochs "$EP" --batch "$BATCH" --workers "$W" --seed "$SEED" \
    --device "$DEVICE" $args "${resume[@]}" "${sync[@]}" 2>&1 | tee -a "$log"
}

eval_one() {
  local name="$1" split="$2" cache="${RUN_CACHE[$1]}" ck="$RUNS/$name/best.pt"
  require_file "$ck"
  local diag="$DIAGDIR/$name.$split.json"
  if [[ -s "$diag" && "$FORCE_EVAL" != 1 ]]; then echo "[skip eval] $diag exists"; return; fi
  "$PY" - "$ck" "${RUN_BOX[$name]}" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if c.get("box_source", "gt") != sys.argv[2]:
    raise SystemExit(f"[ERROR] checkpoint box_source={c.get('box_source')} expected={sys.argv[2]}")
PY
  "$PY" phase_4/scripts/3-eval.py \
    --ckpt "$ck" --region-cache "$cache" --features-root "$FEAT" \
    --m3-labels-dir "$M3LAB" --m4-labels-dir "$M4LAB" --pairs "$PAIRS" \
    --split "$split" --batch "$BATCH" --workers "$EVAL_W" --device "$DEVICE" \
    --diagnostics-json "$diag" 2>&1 | tee "$LOGDIR/$name.$split.eval.log"
}

select_grid() {
  "$PY" - "$RUNS" "$SELECTED_ENV" "${GRID_NAMES[@]}" <<'PY'
import sys, torch
from pathlib import Path
runs, out, *names = sys.argv[1:]
rows = []
for name in names:
    p = Path(runs) / name / "best.pt"
    if not p.exists():
        raise SystemExit(f"[ERROR] grid incomplete; missing {p}")
    c = torch.load(p, map_location="cpu", weights_only=False)
    rows.append((float(c["val_change_f1"]), name,
                 float(c.get("flip_consistency_weight", 0.0)),
                 float(c.get("distance_penalty_weight", 0.0))))
score, name, kl, dist = max(rows)
dest = Path(out); dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(f"KL_WEIGHT={kl}\nDIST_WEIGHT={dist}\nMAIN_RUN={name}\nVAL_CHANGE_F1={score}\n", encoding="utf-8")
print(f"[selected] {name}: val change-F1={score:.6f}, KL={kl:g}, distance={dist:g}")
print(f"[selected] wrote {dest}")
PY
}

load_selected() {
  if [[ -n "${KL_WEIGHT:-}" && -n "${DIST_WEIGHT:-}" ]]; then
    MAIN_RUN="${MAIN_RUN:-m4v2_grid_kl$(slug "$KL_WEIGHT")_dist$(slug "$DIST_WEIGHT")_det}"
  else
    require_file "$SELECTED_ENV"
    # This file is generated internally and contains only four scalar assignments.
    # shellcheck disable=SC1090
    source "$SELECTED_ENV"
  fi
  export KL_WEIGHT DIST_WEIGHT MAIN_RUN
  echo "[locked coefficients] KL=$KL_WEIGHT distance=$DIST_WEIGHT main=$MAIN_RUN"
}

define_paper_runs() {
  load_selected
  local ktag dtag suffix
  ktag=$(slug "$KL_WEIGHT"); dtag=$(slug "$DIST_WEIGHT"); suffix="kl${ktag}_dist${dtag}"
  # Selected grid run is VERA; do not retrain it under a second name.
  add_run paper "$MAIN_RUN" detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits \
    --flip-consistency-weight "$KL_WEIGHT" --distance-penalty-weight "$DIST_WEIGHT"
  # Architecture ablation under the selected regularization.
  add_run paper "m4v2_regiondiff_${suffix}_det" detector "$CACHE_DET" $COMMON --arch regiondiff --input-mode full \
    --flip-consistency-weight "$KL_WEIGHT" --distance-penalty-weight "$DIST_WEIGHT"
  add_run paper "m4v2_tempfuse_${suffix}_det" detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat \
    --flip-consistency-weight "$KL_WEIGHT" --distance-penalty-weight "$DIST_WEIGHT"
  # Loss ablation. The selected KL+distance row is MAIN_RUN above.
  add_run paper m4v2_reg_base_det detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits
  add_run paper "m4v2_reg_kl${ktag}_det" detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits \
    --flip-consistency-weight "$KL_WEIGHT"
  add_run paper "m4v2_reg_dist${dtag}_det" detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits \
    --distance-penalty-weight "$DIST_WEIGHT"
  add_run paper m4v2_reg_smooth005_det detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits --label-smoothing 0.05
  add_run paper "m4v2_reg_smooth005_kl${ktag}_det" detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits \
    --label-smoothing 0.05 --flip-consistency-weight "$KL_WEIGHT"
  add_run paper "m4v2_reg_smooth005_dist${dtag}_det" detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits \
    --label-smoothing 0.05 --distance-penalty-weight "$DIST_WEIGHT"
  add_run paper "m4v2_reg_smooth005_${suffix}_det" detector "$CACHE_DET" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits \
    --label-smoothing 0.05 --flip-consistency-weight "$KL_WEIGHT" --distance-penalty-weight "$DIST_WEIGHT"
  # Oracle sensitivity only; the deployed/main model remains detector-box.
  add_run paper "m4v2_vera_${suffix}_gt_oracle" gt "$CACHE_GT" $COMMON --arch tempfuse --tempfuse-input-mode feat_logits \
    --flip-consistency-weight "$KL_WEIGHT" --distance-penalty-weight "$DIST_WEIGHT"
}

final_audits() {
  load_selected
  local ck="$RUNS/$MAIN_RUN/best.pt"
  require_file "$ck"; require_file "$MS_CSV"
  eval_one "$MAIN_RUN" test
  "$PY" phase_4/scripts/5-mscxrt_audit.py \
    --ckpt "$ck" --csv "$MS_CSV" --region-cache "$CACHE_DET" --features-root "$FEAT" \
    --m3-labels-dir "$M3LAB" --split all --batch "$BATCH" --workers "$EVAL_W" \
    --device "$DEVICE" --out-json "$DIAGDIR/$MAIN_RUN.mscxrt.json" \
    2>&1 | tee "$LOGDIR/$MAIN_RUN.mscxrt.log"
  "$PY" phase_4/scripts/7-temporal_consistency.py \
    --ckpt "$ck" --region-cache "$CACHE_DET" --features-root "$FEAT" \
    --m3-labels-dir "$M3LAB" --m4-labels-dir "$M4LAB" --pairs "$PAIRS" \
    --split test --batch "$BATCH" --workers "$EVAL_W" --device "$DEVICE" \
    --out "$DIAGDIR/$MAIN_RUN.temporal_consistency.json" \
    2>&1 | tee "$LOGDIR/$MAIN_RUN.temporal_consistency.log"
  "$PY" phase_4/scripts/4-infer.py \
    --ckpt "$ck" --region-cache "$CACHE_DET" --features-root "$FEAT" \
    --m3-labels-dir "$M3LAB" --m4-labels-dir "$M4LAB" --pairs "$PAIRS" \
    --split test --include-stable --batch "$BATCH" --workers "$EVAL_W" --device "$DEVICE" \
    --out "$DIAGDIR/$MAIN_RUN.test.raw_predictions.jsonl" \
    2>&1 | tee "$LOGDIR/$MAIN_RUN.infer.log"
  echo "[NOTE] No report threshold or Stage-4 calibration was fitted by this script."
}

echo "[M4 paper v2] profile=$PROFILE scope=$SCOPE epochs=$EP batch=$BATCH workers=$W"
if [[ "${SKIP_CACHE:-0}" != 1 && "${SKIP_DET_CACHE:-0}" != 1 ]]; then
  prepare_cache detector "$CACHE_DET" "$M3_CKPT"
else
  require_file "$CACHE_DET/.m3_source.json"
  echo "[skip cache] reusing verified detector cache: $CACHE_DET"
fi

if [[ "$SCOPE" == all || "$SCOPE" == grid ]]; then
  matched=0
  for name in "${GRID_NAMES[@]}"; do
    if [[ -z "${RUN_NAME:-}" || "$name" == "$RUN_NAME" ]]; then
      matched=1; train_one "$name"; eval_one "$name" val
    fi
  done
  [[ "$matched" == 1 ]] || { echo "[ERROR] RUN_NAME is not a grid row: ${RUN_NAME:-}" >&2; exit 2; }
  if [[ -z "${RUN_NAME:-}" ]]; then
    select_grid
  else
    echo "[grid partial] completed $RUN_NAME; coefficient selection waits for the unfiltered grid pass"
  fi
fi

if [[ "$SCOPE" == all || "$SCOPE" == paper ]]; then
  define_paper_runs
  require_file "$M3_GT_CKPT"
  verify_m3_ckpt "$M3_GT_CKPT" gt
  if [[ "${SKIP_CACHE:-0}" != 1 && "${SKIP_GT_CACHE:-0}" != 1 ]]; then
    prepare_cache gt "$CACHE_GT" "$M3_GT_CKPT"
  else
    require_file "$CACHE_GT/.m3_source.json"
    echo "[skip cache] reusing verified GT oracle cache: $CACHE_GT"
  fi
  matched=0
  for name in "${PAPER_NAMES[@]}"; do
    if [[ -z "${RUN_NAME:-}" || "$name" == "$RUN_NAME" ]]; then matched=1; train_one "$name"; fi
  done
  [[ "$matched" == 1 ]] || { echo "[ERROR] RUN_NAME is not a paper row: ${RUN_NAME:-}" >&2; exit 2; }
  for name in "${PAPER_NAMES[@]}"; do
    if [[ -z "${RUN_NAME:-}" || "$name" == "$RUN_NAME" ]]; then eval_one "$name" val; eval_one "$name" test; fi
  done
fi

if [[ "$SCOPE" == all || "$SCOPE" == final ]]; then
  # Reconstruct the selected run metadata when entering directly with --scope final.
  if [[ -z "${RUN_CACHE[${MAIN_RUN:-}]:-}" ]]; then define_paper_runs; fi
  final_audits
fi

echo "[DONE] M4 paper campaign complete. Diagnostics: $DIAGDIR"
