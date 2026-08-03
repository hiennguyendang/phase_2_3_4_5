#!/usr/bin/env bash
# VERA server supervisor: resumable M2 inference -> M3 paper matrix -> M4 paper matrix.
# Run from any directory. See server_hoang/README.md before the first launch.

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PIPELINE_TOOLS="$SCRIPT_DIR/pipeline_tools.py"
RESULT_COLLECTOR="$SCRIPT_DIR/collect_results.py"

# Optional persistent, non-secret server configuration. This is loaded before
# defaults so start/status/resume in later SSH sessions resolve the same tree.
SERVER_ENV="${SERVER_ENV:-$SCRIPT_DIR/server.env}"
if [[ -f "$SERVER_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SERVER_ENV"
  set +a
fi

# All paths and tuning knobs may be overridden as environment variables.
INPUT_ROOT="${INPUT_ROOT:-$SCRIPT_DIR/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
IMAGE_ROOT="${IMAGE_ROOT:-$INPUT_ROOT/mimic-cxr-448}"
FEATURE_ROOT="${FEATURE_ROOT:-$INPUT_ROOT/frozen}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-$INPUT_ROOT/yolo/last.pt}"
M3_LABELS_INPUT="${M3_LABELS_INPUT:-$INPUT_ROOT/m3_labels}"
M4_LABELS="${M4_LABELS:-$INPUT_ROOT/m4_labels}"
MS_CSV="${MS_CSV:-$INPUT_ROOT/external/MS_CXR_T_temporal_image_classification_v1.0.0.csv}"

M3_LABELS_WORK="${M3_LABELS_WORK:-$OUTPUT_ROOT/labels/m3}"
RUNS_ROOT="${RUNS_ROOT:-$OUTPUT_ROOT/runs}"
M2_ROOT="${M2_ROOT:-$OUTPUT_ROOT/m2_detector}"
M3_LOGDIR="${M3_LOGDIR:-$OUTPUT_ROOT/logs/m3}"
M4_LOGDIR="${M4_LOGDIR:-$OUTPUT_ROOT/logs/m4}"
M3_DIAGDIR="${M3_DIAGDIR:-$OUTPUT_ROOT/diagnostics/m3}"
M4_DIAGDIR="${M4_DIAGDIR:-$OUTPUT_ROOT/diagnostics/m4}"
CACHE_DET="${CACHE_DET:-$OUTPUT_ROOT/cache/m4_region_detector}"
CACHE_GT="${CACHE_GT:-$OUTPUT_ROOT/cache/m4_region_gt_oracle}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_ROOT/results}"
STATE_DIR="${STATE_DIR:-$OUTPUT_ROOT/state}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-$OUTPUT_ROOT/logs/server_hoang.supervisor.log}"

YOLO_IMGSZ="${YOLO_IMGSZ:-1024}"
YOLO_CONF="${YOLO_CONF:-0.25}"
YOLO_IOU="${YOLO_IOU:-0.50}"
M2_BATCH="${M2_BATCH:-16}"
M2_SHARDS_PER_GPU="${M2_SHARDS_PER_GPU:-16}"
EXPECTED_YOLO_SHA256="${EXPECTED_YOLO_SHA256:-71d4b4e3b173cc046fc45c7120b6cf4489c384ceaaec9f08231182108a40da56}"

M3_EPOCHS="${M3_EPOCHS:-40}"
M3_BATCH="${M3_BATCH:-64}"
M3_WORKERS="${M3_WORKERS:-8}"
M3_EVAL_WORKERS="${M3_EVAL_WORKERS:-8}"
M3_AMP="${M3_AMP:-1}"
M3_SYNC_EVERY="${M3_SYNC_EVERY:-0}"

M4_EPOCHS="${M4_EPOCHS:-40}"
M4_BATCH="${M4_BATCH:-128}"
M4_WORKERS="${M4_WORKERS:-16}"
M4_EVAL_WORKERS="${M4_EVAL_WORKERS:-16}"
M4_SYNC_EVERY="${M4_SYNC_EVERY:-0}"

REMOTE_ROOT="${REMOTE_ROOT:-}"
RESULTS_INTERVAL="${RESULTS_INTERVAL:-60}"
GPU_IDS="${GPU_IDS:-}"

M3_RUN_NAMES=(
  m3v2_vera_graph_lse_det
  m3v2_no_concept_det
  m3v2_concept_mlp_det
  m3v2_graph_global_fusion_det
  m3v2_global_only_det
  m3v2_graph_attention_det
  m3v2_graph_mean_det
  m3v2_graph_max_det
  m3v2_vera_graph_lse_gt
)

usage() {
  cat <<'EOF'
Usage: bash server_hoang/run_all.sh COMMAND

First server setup: run server_hoang/download_kaggle.py to download inputs and
generate server_hoang/server.env before using preflight/start.

Commands:
  preflight   Validate environment, GPUs, source and all uploaded inputs.
  start       Preflight, then start detached with nohup+setsid (safe after SSH closes).
  resume      Alias for start; completed stages/runs/shards are skipped.
  foreground  Run in the current terminal (debug only).
  status      Show PID, current stage, latest result summary and log tail.
  logs        Follow the supervisor log (Ctrl-C only stops tail, not training).
  collect     Refresh output/results/{runs.csv,runs.md,summary.json} now.
  stop        Gracefully stop the detached process group; a later start resumes.

Important environment variables:
  INPUT_ROOT, OUTPUT_ROOT, GPU_IDS=0,1,..., M3_BATCH, M4_BATCH,
  M3_WORKERS, M4_WORKERS, REMOTE_ROOT (optional rclone destination).
EOF
}

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
say() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
die() { say "ERROR: $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "command not found: $1"; }

resolve_python() {
  if [[ -n "${PY:-}" ]]; then
    :
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    PY="$REPO_ROOT/.venv/Scripts/python.exe"
  else
    PY="python3"
  fi
  export PY
}

resolve_gpus() {
  local detected
  if [[ -z "$GPU_IDS" ]]; then
    detected="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
    GPU_IDS="$detected"
  fi
  [[ -n "$GPU_IDS" ]] || die "no GPU detected; set GPU_IDS explicitly after fixing CUDA"
  IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
  ((${#GPU_ARRAY[@]} > 0)) || die "GPU_IDS is empty"
  export GPU_IDS
}

sha256_file() {
  "$PY" "$PIPELINE_TOOLS" sha256 "$1"
}

preflight() {
  cd "$REPO_ROOT"
  resolve_python
  require_command nvidia-smi
  require_command nohup
  require_command setsid
  require_command flock
  if [[ -n "$REMOTE_ROOT" ]]; then require_command rclone; fi
  if [[ "$PY" == */* ]]; then
    [[ -x "$PY" ]] || die "Python is not executable: $PY"
  else
    require_command "$PY"
  fi
  resolve_gpus

  require_file "$REPO_ROOT/phase_2/scripts/yolo/5-infer_yolo.py"
  require_file "$REPO_ROOT/phase_3/scripts/3-boxes_from_pred.py"
  require_file "$REPO_ROOT/phase_3/run_paper_m3_v2.sh"
  require_file "$REPO_ROOT/phase_4/run_paper_m4_v2.sh"
  require_file "$PIPELINE_TOOLS"
  require_file "$RESULT_COLLECTOR"
  require_file "$REPO_ROOT/data/m3_concept_space.json"
  require_file "$REPO_ROOT/phase_3/src/m3_concept_space.json"
  require_file "$REPO_ROOT/phase_4/src/m3_concept_space.json"

  require_dir "$IMAGE_ROOT"
  require_dir "$FEATURE_ROOT"
  require_file "$YOLO_WEIGHTS"
  require_dir "$M3_LABELS_INPUT"
  require_dir "$M4_LABELS"
  require_file "$MS_CSV"

  local name
  for name in manifest.jsonl region_concepts.npy region_chexpert.npy image_chexpert.npy boxes.npy present_mask.npy; do
    require_file "$M3_LABELS_INPUT/$name"
  done
  for name in manifest.jsonl progression.npy m3_pairs.jsonl; do
    require_file "$M4_LABELS/$name"
  done
  [[ -n "$(find "$FEATURE_ROOT" -type f \( -name '*.pt' -o -name '*.npy' \) -print -quit)" ]] \
    || die "no .pt/.npy feature file under $FEATURE_ROOT"
  [[ -n "$(find "$IMAGE_ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)" ]] \
    || die "no image under $IMAGE_ROOT"

  "$PY" "$PIPELINE_TOOLS" preflight --gpu-ids "$GPU_IDS" --repo-root "$REPO_ROOT"

  local got_sha
  got_sha="$(sha256_file "$YOLO_WEIGHTS")"
  if [[ "$EXPECTED_YOLO_SHA256" != "any" && "${got_sha,,}" != "${EXPECTED_YOLO_SHA256,,}" ]]; then
    die "YOLO SHA-256 mismatch: got=$got_sha expected=$EXPECTED_YOLO_SHA256"
  fi
  say "preflight OK | GPUs=$GPU_IDS | YOLO_SHA256=$got_sha"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  df -h "$OUTPUT_ROOT" 2>/dev/null || df -h "$(dirname "$OUTPUT_ROOT")"
}

write_stage() {
  mkdir -p "$STATE_DIR"
  printf '%s\n' "$1" > "$STATE_DIR/current_stage.tmp"
  mv -f "$STATE_DIR/current_stage.tmp" "$STATE_DIR/current_stage"
}

mark_stage_complete() {
  local stage="$1"
  mkdir -p "$STATE_DIR/stages"
  "$PY" "$PIPELINE_TOOLS" mark-stage \
    --dest "$STATE_DIR/stages/$stage.SUCCESS.json" --stage "$stage" --completed-at "$(timestamp)"
}

stage_complete() { [[ -s "$STATE_DIR/stages/$1.SUCCESS.json" ]]; }

collect_results() {
  resolve_python
  mkdir -p "$RESULTS_DIR"
  (
    exec 8>"$RESULTS_DIR/collector.lock"
    flock -w 30 8 || exit 0
    "$PY" "$RESULT_COLLECTOR" \
      --runs-root "$RUNS_ROOT" --m2-root "$M2_ROOT" \
      --m3-diagdir "$M3_DIAGDIR" --m4-diagdir "$M4_DIAGDIR" \
      --results-dir "$RESULTS_DIR" --m3-epochs "$M3_EPOCHS" --m4-epochs "$M4_EPOCHS"
  )
}

collector_loop() {
  while true; do
    collect_results || say "collector warning: refresh failed; will retry"
    sleep "$RESULTS_INTERVAL"
  done
}

prepare_m3_labels() {
  local marker="$STATE_DIR/m3_base_labels.SUCCESS"
  mkdir -p "$M3_LABELS_WORK" "$STATE_DIR"
  if [[ ! -s "$marker" ]]; then
    say "copying immutable M3 base labels into writable output"
    cp -a "$M3_LABELS_INPUT/." "$M3_LABELS_WORK/"
    for name in manifest.jsonl region_concepts.npy region_chexpert.npy image_chexpert.npy boxes.npy present_mask.npy; do
      require_file "$M3_LABELS_WORK/$name"
    done
    printf '%s\n' "$(timestamp)" > "$marker.tmp"
    mv -f "$marker.tmp" "$marker"
  fi
}

write_m2_contract() {
  local contract="$M2_ROOT/contract.json"
  mkdir -p "$M2_ROOT"
  "$PY" "$PIPELINE_TOOLS" m2-contract \
    --dest "$contract" --weights "$YOLO_WEIGHTS" \
    --manifest "$M3_LABELS_WORK/manifest.jsonl" --image-root "$IMAGE_ROOT" \
    --imgsz "$YOLO_IMGSZ" --conf "$YOLO_CONF" --iou "$YOLO_IOU" \
    --batch "$M2_BATCH" --num-shards "$M2_NUM_SHARDS"
}

m2_shard_complete() {
  local shard="$1" marker="$M2_ROOT/shards/shard_$(printf '%04d' "$shard")/SUCCESS.json"
  [[ -s "$marker" ]] || return 1
  "$PY" "$PIPELINE_TOOLS" shard-check --marker "$marker" \
    --predictions "$M2_ROOT/shards/shard_$(printf '%04d' "$shard")/predictions.jsonl" \
    --manifest "$M3_LABELS_WORK/manifest.jsonl" --shard "$shard" \
    --num-shards "$M2_NUM_SHARDS" >/dev/null 2>&1
}

run_m2_gpu_worker() {
  local worker="$1" gpu="$2" shard out log expected rows
  for ((shard=worker; shard<M2_NUM_SHARDS; shard+=GPU_COUNT)); do
    out="$M2_ROOT/shards/shard_$(printf '%04d' "$shard")"
    log="$OUTPUT_ROOT/logs/m2/shard_$(printf '%04d' "$shard").log"
    mkdir -p "$out" "$(dirname "$log")"
    if m2_shard_complete "$shard"; then
      say "[M2] skip complete shard $shard/$M2_NUM_SHARDS"
      continue
    fi
    rm -f "$out/SUCCESS.json"
    say "[M2] GPU $gpu -> shard $shard/$M2_NUM_SHARDS"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$REPO_ROOT/phase_2/scripts/yolo/5-infer_yolo.py" \
      --weights "$YOLO_WEIGHTS" --source "$IMAGE_ROOT" \
      --manifest "$M3_LABELS_WORK/manifest.jsonl" --out "$out" \
      --imgsz "$YOLO_IMGSZ" --conf "$YOLO_CONF" --iou "$YOLO_IOU" \
      --batch "$M2_BATCH" --device 0 --no-per-image \
      --shard-index "$shard" --num-shards "$M2_NUM_SHARDS" \
      2>&1 | tee -a "$log"
    "$PY" "$PIPELINE_TOOLS" shard-mark --predictions "$out/predictions.jsonl" \
      --manifest "$M3_LABELS_WORK/manifest.jsonl" --shard "$shard" \
      --num-shards "$M2_NUM_SHARDS" --dest "$out/SUCCESS.json"
  done
}

merge_m2_shards() {
  "$PY" "$PIPELINE_TOOLS" merge-shards --root "$M2_ROOT" \
    --manifest "$M3_LABELS_WORK/manifest.jsonl" --num-shards "$M2_NUM_SHARDS"
}

write_detector_provenance() {
  "$PY" "$PIPELINE_TOOLS" detector-provenance \
    --weights "$YOLO_WEIGHTS" --predictions "$M2_ROOT/predictions.jsonl" \
    --labels "$M3_LABELS_WORK" --imgsz "$YOLO_IMGSZ" --conf "$YOLO_CONF" \
    --iou "$YOLO_IOU" --batch "$M2_BATCH" --num-shards "$M2_NUM_SHARDS"
}

run_m2() {
  if stage_complete m2; then
    require_file "$M2_ROOT/M2.SUCCESS.json"
    require_file "$M2_ROOT/predictions.jsonl"
    require_file "$M3_LABELS_WORK/boxes_det.npy"
    require_file "$M3_LABELS_WORK/present_mask_det.npy"
    require_file "$M3_LABELS_WORK/detector_provenance.json"
    say "[M2] stage already complete"
    return
  fi
  write_stage m2_detector_inference
  prepare_m3_labels
  if [[ -z "${M2_NUM_SHARDS:-}" && -s "$M2_ROOT/contract.json" ]]; then
    M2_NUM_SHARDS="$($PY "$PIPELINE_TOOLS" contract-num-shards --contract "$M2_ROOT/contract.json")"
  fi
  M2_NUM_SHARDS="${M2_NUM_SHARDS:-$((GPU_COUNT * M2_SHARDS_PER_GPU))}"
  ((M2_NUM_SHARDS >= GPU_COUNT)) || die "M2_NUM_SHARDS must be >= GPU count"
  export M2_NUM_SHARDS
  write_m2_contract
  local pids=() worker failed=0
  for ((worker=0; worker<GPU_COUNT; worker++)); do
    run_m2_gpu_worker "$worker" "${GPU_ARRAY[$worker]}" &
    pids+=("$!")
  done
  for worker in "${pids[@]}"; do
    if ! wait "$worker"; then failed=1; fi
  done
  ((failed == 0)) || die "one or more M2 GPU shard workers failed"
  merge_m2_shards
  "$PY" "$REPO_ROOT/phase_3/scripts/3-boxes_from_pred.py" \
    --pred "$M2_ROOT/predictions.jsonl" --manifest "$M3_LABELS_WORK/manifest.jsonl" \
    --out-dir "$M3_LABELS_WORK" --input-res 448 \
    2>&1 | tee -a "$OUTPUT_ROOT/logs/m2/align.log"
  write_detector_provenance
  cp -f "$M3_LABELS_WORK/detector_provenance.json" "$M2_ROOT/detector_provenance.json"
  cp -f "$M3_LABELS_WORK/boxes_det.npy" "$M2_ROOT/boxes_det.npy"
  cp -f "$M3_LABELS_WORK/present_mask_det.npy" "$M2_ROOT/present_mask_det.npy"
  "$PY" "$PIPELINE_TOOLS" m2-success --dest "$M2_ROOT/M2.SUCCESS.json" \
    --predictions "$M2_ROOT/predictions.jsonl" --labels "$M3_LABELS_WORK"
  mark_stage_complete m2
  collect_results
}

m3_remote() { [[ -n "$REMOTE_ROOT" ]] && printf '%s/m3_runs' "${REMOTE_ROOT%/}" || true; }
m4_remote() { [[ -n "$REMOTE_ROOT" ]] && printf '%s/m4_runs' "${REMOTE_ROOT%/}" || true; }

write_run_manifest() {
  local git_commit="unavailable"
  git_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unavailable)"
  "$PY" "$PIPELINE_TOOLS" run-manifest --dest "$STATE_DIR/run_manifest.json" \
    --git-commit "$git_commit" --repo-root "$REPO_ROOT" --input-root "$INPUT_ROOT" \
    --output-root "$OUTPUT_ROOT" --gpu-ids "$GPU_IDS" --m3-epochs "$M3_EPOCHS" \
    --m3-batch "$M3_BATCH" --m4-epochs "$M4_EPOCHS" --m4-batch "$M4_BATCH" \
    --started-at "$(timestamp)"
}

run_m3_gpu_worker() {
  local worker="$1" gpu="$2" i name remote
  remote="$(m3_remote)"
  for ((i=worker; i<${#M3_RUN_NAMES[@]}; i+=GPU_COUNT)); do
    name="${M3_RUN_NAMES[$i]}"
    say "[M3] GPU $gpu -> $name"
    env CUDA_VISIBLE_DEVICES="$gpu" PY="$PY" DEVICE=cuda:0 RUN_NAME="$name" \
      BATCH="$M3_BATCH" W="$M3_WORKERS" EVAL_W="$M3_EVAL_WORKERS" EP="$M3_EPOCHS" \
      AMP="$M3_AMP" DATA_PARALLEL=0 LOG_EVERY=100 \
      FEAT="$FEATURE_ROOT" LABELS="$M3_LABELS_WORK" RUNS="$RUNS_ROOT" \
      LOGDIR="$M3_LOGDIR" DIAGDIR="$M3_DIAGDIR" \
      SYNC_REMOTE="$remote" SYNC_DIAG_REMOTE="" SYNC_EVERY="$M3_SYNC_EVERY" \
      bash "$REPO_ROOT/phase_3/run_paper_m3_v2.sh" --profile h100mini --scope all --epochs "$M3_EPOCHS"
    collect_results
  done
}

run_m3() {
  if stage_complete m3; then
    local completed_name
    for completed_name in "${M3_RUN_NAMES[@]}"; do
      require_file "$RUNS_ROOT/$completed_name/best.pt"
      require_file "$RUNS_ROOT/$completed_name/last.pt"
      require_file "$M3_DIAGDIR/$completed_name.val.json"
      require_file "$M3_DIAGDIR/$completed_name.test.json"
    done
    require_file "$M3_DIAGDIR/m3v2_vera_graph_lse_det.calibration.SUCCESS.json"
    say "[M3] stage already complete"
    return
  fi
  write_stage m3_paper_matrix
  env PY="$PY" FEAT="$FEATURE_ROOT" LABELS="$M3_LABELS_WORK" RUNS="$RUNS_ROOT" \
    LOGDIR="$M3_LOGDIR" DIAGDIR="$M3_DIAGDIR" \
    bash "$REPO_ROOT/phase_3/run_paper_m3_v2.sh" --profile h100mini --scope preflight
  local pids=() worker failed=0
  for ((worker=0; worker<GPU_COUNT; worker++)); do
    run_m3_gpu_worker "$worker" "${GPU_ARRAY[$worker]}" &
    pids+=("$!")
  done
  for worker in "${pids[@]}"; do
    if ! wait "$worker"; then failed=1; fi
  done
  ((failed == 0)) || die "one or more M3 GPU workers failed; rerun start to resume"
  local name
  for name in "${M3_RUN_NAMES[@]}"; do
    require_file "$RUNS_ROOT/$name/best.pt"
    require_file "$RUNS_ROOT/$name/last.pt"
    require_file "$M3_DIAGDIR/$name.val.json"
    require_file "$M3_DIAGDIR/$name.test.json"
  done
  require_file "$M3_DIAGDIR/m3v2_vera_graph_lse_det.calibration.SUCCESS.json"
  mark_stage_complete m3
  collect_results
}

run_m4() {
  if stage_complete m4; then
    require_file "$M4_DIAGDIR/selected_coefficients.env"
    say "[M4] stage already complete"
    return
  fi
  write_stage m4_grid_paper_final
  local remote first_gpu
  remote="$(m4_remote)"; first_gpu="${GPU_ARRAY[0]}"
  env CUDA_VISIBLE_DEVICES="$first_gpu" PY="$PY" DEVICE=cuda:0 \
    M3_CKPT="$RUNS_ROOT/m3v2_vera_graph_lse_det/best.pt" \
    M3_GT_CKPT="$RUNS_ROOT/m3v2_vera_graph_lse_gt/best.pt" \
    M3LAB="$M3_LABELS_WORK" M4LAB="$M4_LABELS" PAIRS="$M4_LABELS/m3_pairs.jsonl" \
    FEAT="$FEATURE_ROOT" CACHE_DET="$CACHE_DET" CACHE_GT="$CACHE_GT" \
    RUNS="$RUNS_ROOT" LOGDIR="$M4_LOGDIR" DIAGDIR="$M4_DIAGDIR" \
    SELECTED_ENV="$M4_DIAGDIR/selected_coefficients.env" MS_CSV="$MS_CSV" \
    BATCH="$M4_BATCH" W="$M4_WORKERS" EVAL_W="$M4_EVAL_WORKERS" EP="$M4_EPOCHS" \
    SYNC_REMOTE="$remote" SYNC_EVERY="$M4_SYNC_EVERY" \
    bash "$REPO_ROOT/phase_4/run_paper_m4_v2.sh" --profile h100mini --scope all --epochs "$M4_EPOCHS"
  require_file "$M4_DIAGDIR/selected_coefficients.env"
  local main_run
  main_run="$(sed -n 's/^MAIN_RUN=//p' "$M4_DIAGDIR/selected_coefficients.env" | tail -n 1)"
  [[ -n "$main_run" ]] || die "selected_coefficients.env has no MAIN_RUN"
  require_file "$RUNS_ROOT/$main_run/best.pt"
  require_file "$M4_DIAGDIR/$main_run.test.json"
  require_file "$M4_DIAGDIR/$main_run.mscxrt.json"
  require_file "$M4_DIAGDIR/$main_run.temporal_consistency.json"
  require_file "$M4_DIAGDIR/$main_run.test.raw_predictions.jsonl"
  mark_stage_complete m4
  collect_results
}

worker_main() {
  cd "$REPO_ROOT"
  resolve_python
  resolve_gpus
  GPU_COUNT="${#GPU_ARRAY[@]}"; export GPU_COUNT
  mkdir -p "$OUTPUT_ROOT/logs" "$RUNS_ROOT" "$M3_LOGDIR" "$M4_LOGDIR" \
    "$M3_DIAGDIR" "$M4_DIAGDIR" "$RESULTS_DIR" "$STATE_DIR/stages"
  exec 9>"$STATE_DIR/pipeline.lock"
  flock -n 9 || die "another server_hoang worker already holds $STATE_DIR/pipeline.lock"
  echo "$$" > "$STATE_DIR/worker.pid.tmp"; mv -f "$STATE_DIR/worker.pid.tmp" "$STATE_DIR/worker.pid"
  write_run_manifest
  local collector_pid=""
  worker_cleanup() {
    local code=$?
    if [[ -n "$collector_pid" ]]; then kill "$collector_pid" 2>/dev/null || true; wait "$collector_pid" 2>/dev/null || true; fi
    collect_results || true
    if ((code == 0)); then
      write_stage complete
      printf '%s\n' "$(timestamp)" > "$STATE_DIR/PIPELINE.SUCCESS"
      rm -f "$STATE_DIR/PIPELINE.FAILED"
    else
      printf '%s exit=%s stage=%s\n' "$(timestamp)" "$code" "$(cat "$STATE_DIR/current_stage" 2>/dev/null || true)" \
        > "$STATE_DIR/PIPELINE.FAILED"
    fi
    rm -f "$STATE_DIR/worker.pid"
  }
  trap worker_cleanup EXIT
  trap 'say "termination requested; stopping children"; exit 143' TERM INT
  collector_loop & collector_pid=$!
  say "pipeline worker started | pid=$$ | repo=$REPO_ROOT | output=$OUTPUT_ROOT | GPUs=$GPU_IDS"
  run_m2
  run_m3
  run_m4
  say "pipeline complete"
}

is_running() {
  local pid=""
  [[ -s "$STATE_DIR/worker.pid" ]] && pid="$(cat "$STATE_DIR/worker.pid")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start_detached() {
  mkdir -p "$OUTPUT_ROOT/logs" "$STATE_DIR"
  if is_running; then die "pipeline already running with PID $(cat "$STATE_DIR/worker.pid")"; fi
  preflight
  rm -f "$STATE_DIR/PIPELINE.FAILED" "$STATE_DIR/PIPELINE.SUCCESS"
  say "starting detached pipeline; log=$SUPERVISOR_LOG"
  nohup setsid bash "$SCRIPT_DIR/run_all.sh" __worker >> "$SUPERVISOR_LOG" 2>&1 < /dev/null &
  local launcher_pid=$!
  printf '%s\n' "$launcher_pid" > "$STATE_DIR/launcher.pid"
  sleep 2
  if is_running; then
    say "started PID $(cat "$STATE_DIR/worker.pid"); closing SSH is safe"
  elif kill -0 "$launcher_pid" 2>/dev/null; then
    say "launcher PID $launcher_pid is starting; check status in a few seconds"
  else
    tail -n 80 "$SUPERVISOR_LOG" || true
    die "detached worker exited during startup"
  fi
}

show_status() {
  local state="stopped" pid="-" stage="not_started"
  if is_running; then state="running"; pid="$(cat "$STATE_DIR/worker.pid")"; fi
  [[ -s "$STATE_DIR/current_stage" ]] && stage="$(cat "$STATE_DIR/current_stage")"
  [[ -s "$STATE_DIR/PIPELINE.SUCCESS" ]] && state="complete"
  [[ -s "$STATE_DIR/PIPELINE.FAILED" && "$state" != running ]] && state="failed"
  printf 'state: %s\npid: %s\nstage: %s\noutput: %s\nlog: %s\n' "$state" "$pid" "$stage" "$OUTPUT_ROOT" "$SUPERVISOR_LOG"
  if [[ -s "$RESULTS_DIR/summary.json" ]]; then
    "$PY" "$PIPELINE_TOOLS" show-summary --summary "$RESULTS_DIR/summary.json"
  fi
  [[ -s "$STATE_DIR/PIPELINE.FAILED" ]] && { echo "last failure:"; cat "$STATE_DIR/PIPELINE.FAILED"; }
  [[ -s "$SUPERVISOR_LOG" ]] && { echo "--- latest log ---"; tail -n 25 "$SUPERVISOR_LOG"; }
}

stop_worker() {
  if ! is_running; then say "pipeline is not running"; return; fi
  local pid; pid="$(cat "$STATE_DIR/worker.pid")"
  say "sending TERM to detached process group $pid"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
  say "stop requested; use status to confirm, then start/resume later"
}

command="${1:-}"
case "$command" in
  preflight) preflight ;;
  start|resume) start_detached ;;
  foreground) preflight; worker_main ;;
  status) resolve_python; show_status ;;
  logs) touch "$SUPERVISOR_LOG"; tail -n 200 -F "$SUPERVISOR_LOG" ;;
  collect) collect_results ;;
  stop) stop_worker ;;
  __worker) worker_main ;;
  -h|--help|help|"") usage ;;
  *) usage; die "unknown command: $command" ;;
esac
