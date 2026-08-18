#!/usr/bin/env bash
# VERA M3 Diagnostic & Reproduction Battery
# ============================================================
# Purpose: systematically reproduce the old high-water-mark config and
# ablate every architectural change introduced in v2 to isolate which
# change(s) caused the AUC regression observed in the first server run.
#
# Does NOT re-run M2 inference (boxes_det.npy assumed ready and in place).
# Does NOT train M4 (M3 diagnostics only).
#
# Design — 2^3 factorial over three toggle dimensions:
#   G = global_head        ON=old/xwalk_v2   OFF=new/v2
#   D = derive_no_finding  OFF=old/xwalk_v2  ON=new/v2
#   A = aggregation        attention=old     lse=new/v2
#
# Groups:
#   g0  sanity eval of already-finished run_all.sh checkpoints  (eval-only)
#   g1  re-train OLD faithful config in NEW env to validate env/data
#   g2  2^3 factorial cells — one-at-a-time and pairwise isolation
#   g3  best-candidate configs that relax exactly one v2 constraint
#   g4  concept-head sensitivity (faithful vs mlp vs nonneg disease head)
#
# Run from any directory.  See server_hoang/README.md for first-time setup.

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PIPELINE_TOOLS="$SCRIPT_DIR/pipeline_tools.py"
RESULT_COLLECTOR="$SCRIPT_DIR/collect_results.py"

# Optional persistent server config (same file as run_all.sh)
SERVER_ENV="${SERVER_ENV:-$SCRIPT_DIR/server.env}"
if [[ -f "$SERVER_ENV" ]]; then
  set -a; source "$SERVER_ENV"; set +a
fi

# ── paths ────────────────────────────────────────────────────────────────────
INPUT_ROOT="${INPUT_ROOT:-$SCRIPT_DIR/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
FEATURE_ROOT="${FEATURE_ROOT:-$INPUT_ROOT/frozen}"
M3_LABELS_INPUT="${M3_LABELS_INPUT:-$INPUT_ROOT/m3_labels}"

# Diagnostic outputs go to a separate subtree so they never clash with
# the main run_all.sh output tree.
DIAG_ROOT="${DIAG_ROOT:-$OUTPUT_ROOT/diag}"
RUNS_ROOT="${RUNS_ROOT:-$DIAG_ROOT/runs}"
M3_LOGDIR="${M3_LOGDIR:-$DIAG_ROOT/logs/m3}"
M3_DIAGDIR="${M3_DIAGDIR:-$DIAG_ROOT/diagnostics/m3}"
M3_LABELS_WORK="${M3_LABELS_WORK:-$DIAG_ROOT/labels/m3}"
STATE_DIR="${STATE_DIR:-$DIAG_ROOT/state}"
RESULTS_DIR="${RESULTS_DIR:-$DIAG_ROOT/results}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-$DIAG_ROOT/logs/diag.supervisor.log}"

# If main runs are in a different root, set this so g0 eval-only runs can
# find existing checkpoints produced by run_all.sh.
MAIN_RUNS_ROOT="${MAIN_RUNS_ROOT:-$OUTPUT_ROOT/runs}"
MAIN_M3_DIAGDIR="${MAIN_M3_DIAGDIR:-$OUTPUT_ROOT/diagnostics/m3}"

# ── tuning knobs ─────────────────────────────────────────────────────────────
M3_EPOCHS="${M3_EPOCHS:-40}"
M3_BATCH="${M3_BATCH:-64}"
M3_WORKERS="${M3_WORKERS:-8}"
M3_EVAL_WORKERS="${M3_EVAL_WORKERS:-8}"
M3_AMP="${M3_AMP:-1}"
RESULTS_INTERVAL="${RESULTS_INTERVAL:-60}"
GPU_IDS="${GPU_IDS:-}"
REMOTE_ROOT="${REMOTE_ROOT:-}"
M3_SYNC_EVERY="${M3_SYNC_EVERY:-0}"
SEED="${SEED:-42}"

# ============================================================
# Diagnostic run registry
# add_run NAME BOX_SOURCE EVAL_ONLY [train_flags...]
#   EVAL_ONLY=1 → look for checkpoint from MAIN_RUNS_ROOT, skip training
#   EVAL_ONLY=0 → train from scratch (resumable)
# ============================================================
declare -a DIAG_NAMES=()
declare -A DIAG_BOX DIAG_FLAGS DIAG_EVAL_ONLY

add_run() {
  local name="$1" box="$2" eval_only="${3:-0}"; shift 3
  DIAG_NAMES+=("$name")
  DIAG_BOX["$name"]="$box"
  DIAG_FLAGS["$name"]="$*"
  DIAG_EVAL_ONLY["$name"]="$eval_only"
}

# Shared flag blocks (keep explicit — do not rely on config.py defaults)
_B_FAITHFUL="--mode B --head-type mlp --disease-head faithful --detach-concept"
_B_MLP="--mode B --head-type mlp --disease-head mlp --detach-concept"
_B_NONNEG="--mode B --head-type mlp --disease-head nonneg --detach-concept"
_A_DIRECT="--mode A --head-type mlp"

# ── Group 0: sanity eval of run_all.sh completed checkpoints (eval-only) ─────
# Re-evaluate using this script's eval path to confirm eval parity.
# Expected: numbers match server run_all.sh output.
add_run diag_g0_vera_main        detector 1 \
  $_B_FAITHFUL --no-global-head --derive-no-finding --region-agg lse
add_run diag_g0_no_concept       detector 1 \
  $_A_DIRECT   --no-global-head --derive-no-finding --region-agg lse
add_run diag_g0_global_fusion    detector 1 \
  $_B_FAITHFUL --global-head    --derive-no-finding --region-agg lse

# ── Group 1: environment / data validation ────────────────────────────────────
# Re-train the OLD faithful config (global head ON, attention agg, no derive)
# using the NEW env + NEW boxes_det.npy.
#
# Key question:
#   If result ≈ 0.829  → env/data are NOT the cause; gap is architecture.
#   If result ≪ 0.829  → boxes_det.npy or env change introduced the drop.
#
# Reference: m3_B_faithful_xwalk_v2 → Val AUC 0.8313 / Test AUC 0.8293
add_run diag_g1_old_repro        detector 0 \
  $_B_FAITHFUL --global-head --region-agg attention
  # Note: NO --derive-no-finding (was OFF in xwalk_v2 faithful run)

# ── Group 2: 2^3 factorial over (G, D, A) ────────────────────────────────────
# G: global head    (1=ON / 0=OFF)
# D: derive_no_finding (0=OFF / 1=ON)
# A: aggregation    (at=attention / lse=log-sum-exp)
#
# Cell (1,0,at) — old anchor — must reproduce diag_g1_old_repro (sanity)
add_run diag_g2_G1_D0_Aat        detector 0 \
  $_B_FAITHFUL --global-head     --region-agg attention

# Cell (1,0,lse) — isolates aggregation change only (attention→lse)
add_run diag_g2_G1_D0_Alse       detector 0 \
  $_B_FAITHFUL --global-head     --region-agg lse

# Cell (1,1,at) — isolates derive-no-finding only
add_run diag_g2_G1_D1_Aat        detector 0 \
  $_B_FAITHFUL --global-head     --derive-no-finding --region-agg attention

# Cell (1,1,lse) — global ON + derive ON + lse (all changes except no-global)
add_run diag_g2_G1_D1_Alse       detector 0 \
  $_B_FAITHFUL --global-head     --derive-no-finding --region-agg lse

# Cell (0,0,at) — isolates global-head removal only
# Expected: ~0.813 (mirrors old m3_Bf_noglobal_xwalk_v2 result)
add_run diag_g2_G0_D0_Aat        detector 0 \
  $_B_FAITHFUL --no-global-head  --region-agg attention

# Cell (0,0,lse) — global OFF + lse; NO derive-no-finding  ← KEY diagnostic
# This is VERA-v2-main minus derive-no-finding.
# If this is substantially higher than v2-main (0.799), derive-no-finding is the problem.
add_run diag_g2_G0_D0_Alse       detector 0 \
  $_B_FAITHFUL --no-global-head  --region-agg lse

# Cell (0,1,at) — global OFF + derive ON; attention only
add_run diag_g2_G0_D1_Aat        detector 0 \
  $_B_FAITHFUL --no-global-head  --derive-no-finding --region-agg attention

# Cell (0,1,lse) — new v2 anchor  — must reproduce m3v2_vera_graph_lse_det (0.799)
add_run diag_g2_G0_D1_Alse       detector 0 \
  $_B_FAITHFUL --no-global-head  --derive-no-finding --region-agg lse

# ── Group 3: best-candidate configs (relax one v2 constraint at a time) ───────

# 3a: Bring global head back into VERA-v2 (= m3v2_graph_global_fusion_det, eval-only)
add_run diag_g3_global_back      detector 1 \
  $_B_FAITHFUL --global-head  --derive-no-finding --region-agg lse

# 3b: VERA-v2-main but with global head restored (fresh training — different seed path)
#     Compare to 3a (eval-only) to see if training with global head from scratch differs.
add_run diag_g3_global_back_retrain detector 0 \
  $_B_FAITHFUL --global-head  --derive-no-finding --region-agg lse

# 3c: Best-of-old-style faithful (global ON, lse, NO derive) — highest plausible ceiling
#     while keeping faithful bottleneck + no-global bypass is NOT enforced
add_run diag_g3_global_lse_nonderive detector 0 \
  $_B_FAITHFUL --global-head  --region-agg lse

# 3d: GT-box oracle for Group 1 old_repro — upper bound on detector box error cost
add_run diag_g3_old_repro_gt     gt      0 \
  $_B_FAITHFUL --global-head  --region-agg attention

# 3e: GT-box oracle for VERA-v2-main — upper bound on detector box error cost in new arch
add_run diag_g3_vera_main_gt     gt      1 \
  $_B_FAITHFUL --no-global-head --derive-no-finding --region-agg lse

# ── Group 4: concept-head sensitivity ────────────────────────────────────────
# Tests whether the faithful/masked head (vs mlp/nonneg disease head) is itself
# the bottleneck once global head is removed.

# 4a: Mode B + MLP disease head + no-global + derive + lse (= m3v2_concept_mlp_det, eval-only)
#     If this is higher than vera-main, the graph-masked faithful head is the bottleneck.
add_run diag_g4_mlp_G0_D1_Alse  detector 1 \
  $_B_MLP --no-global-head  --derive-no-finding --region-agg lse

# 4b: Mode B + nonneg (unmasked non-negative) + no-global + no-derive + lse
#     Relaxed faithfulness: non-negative but no mask on concept→disease edges
add_run diag_g4_nonneg_G0_D0_Alse detector 0 \
  $_B_NONNEG --no-global-head --region-agg lse

# 4c: Mode B + MLP disease head + no-global + NO derive + lse
#     Free MLP, no global head, no derive → accuracy ceiling for faithful ablations
add_run diag_g4_mlp_G0_D0_Alse  detector 0 \
  $_B_MLP --no-global-head  --region-agg lse

# 4d: Mode A (direct, no concept) + global head + no derive + lse
#     Real accuracy ceiling (no bottleneck at all, global head back)
add_run diag_g4_direct_global    detector 0 \
  $_A_DIRECT --global-head  --region-agg lse

# Total: 21 registered runs
#   g0: 3 eval-only
#   g1: 1 train
#   g2: 8 trains (including 2 anchors that should reproduce known numbers)
#   g3: 2 eval-only + 3 trains
#   g4: 1 eval-only + 3 trains
#   → 16 new training runs, 6 eval-only sanity checks

# ============================================================
# Infrastructure helpers
# ============================================================

usage() {
  cat <<'EOF'
Usage: bash server_hoang/run_diag.sh COMMAND [--scope GROUP] [--run NAME]

Commands:
  preflight   Validate environment and all required inputs.
  start       Preflight then launch detached (safe after SSH disconnect).
  resume      Alias for start; completed runs are skipped automatically.
  foreground  Run in the current terminal (debug mode).
  status      Show PID, run progress, and latest result table.
  logs        Follow the supervisor log (Ctrl-C stops tail only).
  collect     Refresh diag/results/diag_results.md now.
  stop        Gracefully stop the detached process group.

Options:
  --scope GROUP   g0|g1|g2|g3|g4|all (default: all)
  --run NAME      Single named run (overrides --scope)

Environment variables:
  INPUT_ROOT, OUTPUT_ROOT, DIAG_ROOT, GPU_IDS, M3_BATCH, M3_WORKERS,
  MAIN_RUNS_ROOT   root of run_all.sh checkpoints (for eval-only g0 runs)
  REMOTE_ROOT      optional rclone destination for checkpoint sync

Design:
  g0  Sanity eval of existing run_all.sh checkpoints  (no GPU training)
  g1  Re-train OLD faithful config (global+attention+no-derive) in new env
      → if ≈0.829: env/data OK, gap is architecture
      → if ≪0.829: boxes_det or env change caused the drop
  g2  Full 2^3 factorial over (global_head × derive_nf × aggregation)
      → isolates each toggle individually and in all pairwise combinations
  g3  Best-candidate relaxations: restore global head, remove derive-nf
  g4  Concept-head type sensitivity: faithful vs mlp vs nonneg disease head
EOF
}

timestamp()       { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
say()             { printf '[%s] %s\n' "$(timestamp)" "$*"; }
die()             { say "ERROR: $*" >&2; exit 2; }
require_file()    { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir()     { [[ -d "$1" ]] || die "missing directory: $1"; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "command not found: $1"; }

resolve_python() {
  if   [[ -n "${PY:-}" ]];                             then :
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]];         then PY="$REPO_ROOT/.venv/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then PY="$REPO_ROOT/.venv/Scripts/python.exe"
  else PY="python3"; fi
  export PY
}

resolve_gpus() {
  if [[ -z "$GPU_IDS" ]]; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
  fi
  [[ -n "$GPU_IDS" ]] || die "no GPU detected; set GPU_IDS explicitly"
  IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
  ((${#GPU_ARRAY[@]} > 0)) || die "GPU_IDS is empty"
  export GPU_IDS
}

write_stage() {
  mkdir -p "$STATE_DIR"
  printf '%s\n' "$1" >"$STATE_DIR/diag_stage.tmp"
  mv -f "$STATE_DIR/diag_stage.tmp" "$STATE_DIR/diag_stage"
}

mark_run_complete() {
  mkdir -p "$STATE_DIR/runs"
  printf '%s\n' "$(timestamp)" >"$STATE_DIR/runs/$1.DONE"
}

run_complete() { [[ -s "$STATE_DIR/runs/$1.DONE" ]]; }

# ── Result collection ─────────────────────────────────────────────────────────

collect_results() {
  resolve_python
  mkdir -p "$RESULTS_DIR"
  (
    exec 8>"$RESULTS_DIR/collector.lock"
    flock -w 30 8 || exit 0
    "$PY" - "$M3_DIAGDIR" "$RESULTS_DIR" <<'PY'
import json, sys
from pathlib import Path

diagdir = Path(sys.argv[1])
outdir  = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

rows = []
for jf in sorted(diagdir.glob("*.test.json")):
    try:
        d    = json.loads(jf.read_text())
        name = jf.stem.replace(".test", "")
        # val AUC from companion file if present
        vf = diagdir / (name + ".val.json")
        val_auc = "?"
        if vf.exists():
            try: val_auc = json.loads(vf.read_text()).get("image_auc_macro", "?")
            except Exception: pass
        faith = "?"
        ff = diagdir / (name + ".faithfulness.json")
        if ff.exists():
            try:
                fd = json.loads(ff.read_text())
                faith = "PASS" if fd.get("why_faithful_allowed") else "FAIL"
                if "intervention_correct_frac" in fd:
                    faith += f" {fd['intervention_correct_frac']:.2%}"
            except Exception: pass
        rows.append({
            "run":        name,
            "val_auc":    val_auc,
            "test_auc":   d.get("image_auc_macro", "?"),
            "test_f1":    d.get("image_f1_macro",  "?"),
            "region_f1":  d.get("region_f1_macro", "?"),
            "concept_f1": d.get("concept_f1_macro","?"),
            "faith":      faith,
        })
    except Exception:
        pass

hdr = ["| Run | Val AUC | Test AUC | Test F1 | Region F1 | Concept F1 | Faith |",
       "|---|---:|---:|---:|---:|---:|---|"]
body = [
    f"| {r['run']} | {r['val_auc']} | {r['test_auc']} | {r['test_f1']}"
    f" | {r['region_f1']} | {r['concept_f1']} | {r['faith']} |"
    for r in rows
]
md = "\n".join(hdr + body) + "\n"
(outdir / "diag_results.md").write_text(md)
(outdir / "diag_results.json").write_text(json.dumps(rows, indent=2))
print(f"[collect] {len(rows)} runs → {outdir}/diag_results.md")
for r in rows:
    print(f"  {r['run']:52s}  val={r['val_auc']}  test={r['test_auc']}  f1={r['test_f1']}  faith={r['faith']}")
PY
  )
}

collector_loop() {
  while true; do
    collect_results || say "collector warning: refresh failed; will retry"
    sleep "$RESULTS_INTERVAL"
  done
}

# ── Preflight ────────────────────────────────────────────────────────────────

preflight() {
  cd "$REPO_ROOT"
  resolve_python
  require_command nvidia-smi
  require_command nohup
  require_command setsid
  require_command flock
  [[ -n "$REMOTE_ROOT" ]] && require_command rclone

  require_file "$REPO_ROOT/phase_3/scripts/4-train.py"
  require_file "$REPO_ROOT/phase_3/scripts/5-eval.py"
  require_file "$REPO_ROOT/phase_3/scripts/6-faithfulness.py"
  require_file "$REPO_ROOT/data/m3_concept_space.json"
  require_file "$REPO_ROOT/phase_3/src/m3_concept_space.json"

  require_dir  "$FEATURE_ROOT"
  require_dir  "$M3_LABELS_INPUT"

  for name in manifest.jsonl region_concepts.npy region_chexpert.npy \
              image_chexpert.npy boxes.npy present_mask.npy \
              boxes_det.npy present_mask_det.npy; do
    require_file "$M3_LABELS_INPUT/$name"
  done

  [[ -n "$(find "$FEATURE_ROOT" -type f \( -name '*.pt' -o -name '*.npy' \) -print -quit 2>/dev/null)" ]] \
    || die "no .pt/.npy feature file under $FEATURE_ROOT"

  "$PY" - <<'PY'
import json
from pathlib import Path
paths = [Path("data/m3_concept_space.json"),
         Path("phase_3/src/m3_concept_space.json"),
         Path("phase_4/src/m3_concept_space.json")]
objects = [json.loads(p.read_text(encoding="utf-8-sig")) for p in paths]
if not all(x == objects[0] for x in objects[1:]):
    raise SystemExit("[ERROR] concept-space JSON files differ; synchronize them first")
concepts = objects[0].get("concepts", objects[0])
if len(concepts) != 69:
    raise SystemExit(f"[ERROR] expected 69 concepts, found {len(concepts)}")
print(f"[preflight] concept-space: 69 concepts, all copies agree")
PY

  resolve_gpus
  say "preflight OK | GPUs=$GPU_IDS | labels=$M3_LABELS_INPUT | features=$FEATURE_ROOT"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  df -h "$DIAG_ROOT" 2>/dev/null || df -h "$(dirname "$DIAG_ROOT")"
}

# ── Prepare labels ────────────────────────────────────────────────────────────

prepare_labels() {
  local marker="$STATE_DIR/m3_labels_ready.SUCCESS"
  mkdir -p "$M3_LABELS_WORK" "$STATE_DIR"
  if [[ ! -s "$marker" ]]; then
    say "copying M3 base labels into writable work dir"
    cp -a "$M3_LABELS_INPUT/." "$M3_LABELS_WORK/"
    for name in manifest.jsonl region_concepts.npy region_chexpert.npy \
                image_chexpert.npy boxes.npy present_mask.npy \
                boxes_det.npy present_mask_det.npy; do
      require_file "$M3_LABELS_WORK/$name"
    done
    printf '%s\n' "$(timestamp)" >"$marker.tmp"; mv -f "$marker.tmp" "$marker"
    say "labels ready at $M3_LABELS_WORK"
  fi
}

# ── Train / eval one run ─────────────────────────────────────────────────────

is_trained() {
  local last="$RUNS_ROOT/$1/last.pt"
  [[ -f "$last" ]] || return 1
  local completed
  completed=$("$PY" - "$last" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(c.get("epoch", -1)) + 1)
PY
)
  [[ "$completed" -ge "$M3_EPOCHS" ]]
}

train_one() {
  local name="$1" box="${DIAG_BOX[$1]}" flags="${DIAG_FLAGS[$1]}"
  local run="$RUNS_ROOT/$name" log="$M3_LOGDIR/$name.train.log"
  mkdir -p "$run" "$(dirname "$log")"

  # Eval-only: look for checkpoint from main runs root first
  if [[ "${DIAG_EVAL_ONLY[$name]}" == "1" ]]; then
    if [[ ! -f "$run/best.pt" ]]; then
      local src="$MAIN_RUNS_ROOT/${name#diag_g?_}"   # strip diag_gN_ prefix for lookup
      # Also try exact name match in main runs root
      local src_exact="$MAIN_RUNS_ROOT/$name"
      if   [[ -f "$src_exact/best.pt" ]]; then
        say "[g0/eval-only] $name: linking from $src_exact"
        ln -sfn "$src_exact" "$run" 2>/dev/null || cp -r "$src_exact" "$run"
      elif [[ -f "$src/best.pt" ]]; then
        say "[g0/eval-only] $name: linking from $src"
        ln -sfn "$src" "$run" 2>/dev/null || cp -r "$src" "$run"
      else
        say "[g0/eval-only] $name: no checkpoint found in MAIN_RUNS_ROOT — skipping"
        return
      fi
    fi
    say "[g0/eval-only] $name: using existing checkpoint at $run/best.pt"
    return
  fi

  # Training run
  if is_trained "$name"; then
    say "[train] $name: already completed $M3_EPOCHS epochs — skipping"
    return
  fi

  if [[ -f "$run/best.pt" && ! -f "$run/last.pt" ]]; then
    die "$run has best.pt but no last.pt; refusing to overwrite (clean up manually)"
  fi

  local resume_flag=()
  [[ -f "$run/last.pt" ]] && resume_flag=(--resume)

  local sync_flags=()
  [[ -n "$REMOTE_ROOT" ]] && sync_flags=(--sync-remote "${REMOTE_ROOT%/}/diag_runs" --sync-every "$M3_SYNC_EVERY")

  local amp_flags=()
  [[ "$M3_AMP" == "1" ]] && amp_flags=(--amp)

  say "[train] $name (box=$box)"
  # shellcheck disable=SC2086
  "$PY" phase_3/scripts/4-train.py \
    --labels-dir "$M3_LABELS_WORK" --features-root "$FEATURE_ROOT" \
    --out "$RUNS_ROOT" --name "$name" \
    --epochs "$M3_EPOCHS" --batch "$M3_BATCH" --workers "$M3_WORKERS" \
    --device "${DEVICE:-cuda:0}" \
    --box-source "$box" --select-by auc --seed "$SEED" \
    $flags "${resume_flag[@]}" "${sync_flags[@]}" "${amp_flags[@]}" \
    --log-every 100 \
    2>&1 | tee -a "$log"
}

pred_dump_ok() {
  local pred="$1"
  [[ -s "$pred" ]] || return 1
  "$PY" - "$pred" <<'PY' >/dev/null 2>&1
import sys, numpy as np
with np.load(sys.argv[1], allow_pickle=False) as z:
    assert "image_id" in z.files or len(z.files) > 0
PY
}

eval_one() {
  local name="$1" box="${DIAG_BOX[$1]}" ck="$RUNS_ROOT/$name/best.pt"
  if [[ ! -f "$ck" ]]; then
    say "[eval] $name: no checkpoint — skipping"
    return
  fi

  for split in val test; do
    local diag="$M3_DIAGDIR/$name.$split.json"
    local pred="$M3_DIAGDIR/$name.$split.pred.npz"
    if [[ -s "$diag" ]] && pred_dump_ok "$pred"; then
      say "[eval] $name/$split: artifacts exist — skipping"
    else
      say "[eval] $name/$split"
      "$PY" phase_3/scripts/5-eval.py \
        --ckpt "$ck" --labels-dir "$M3_LABELS_WORK" --features-root "$FEATURE_ROOT" \
        --box-source "$box" --split "$split" \
        --batch "$M3_BATCH" --workers "$M3_EVAL_WORKERS" \
        --device "${DEVICE:-cuda:0}" \
        --diagnostics-json "$diag" --pred-dump "$pred" \
        --min-region-pos 30 --min-region-neg 30 \
        2>&1 | tee "$M3_LOGDIR/$name.$split.eval.log"
    fi
  done

  # Faithfulness audit for mode B runs only
  if [[ "${DIAG_FLAGS[$name]}" == *"--mode B"* ]]; then
    local faith="$M3_DIAGDIR/$name.faithfulness.json"
    if [[ ! -s "$faith" ]]; then
      say "[faithfulness] $name"
      "$PY" phase_3/scripts/6-faithfulness.py \
        --ckpt "$ck" --labels-dir "$M3_LABELS_WORK" --features-root "$FEATURE_ROOT" \
        --box-source "$box" --split test \
        --batch "$M3_BATCH" --workers "$M3_EVAL_WORKERS" \
        --device "${DEVICE:-cuda:0}" \
        --diagnostics-json "$faith" \
        2>&1 | tee "$M3_LOGDIR/$name.faithfulness.log"
    else
      say "[faithfulness] $name: already exists — skipping"
    fi
  fi
}

# ── Scope filter ─────────────────────────────────────────────────────────────

filter_scope() {
  local scope="$1" single_run="${2:-}"
  local -a filtered=()
  if [[ -n "$single_run" ]]; then
    local found=0
    for n in "${DIAG_NAMES[@]}"; do [[ "$n" == "$single_run" ]] && found=1; done
    ((found)) || die "--run '$single_run' is not registered (check spelling)"
    filtered=("$single_run")
  elif [[ "$scope" == "all" ]]; then
    filtered=("${DIAG_NAMES[@]}")
  else
    for n in "${DIAG_NAMES[@]}"; do
      [[ "$n" == diag_${scope}_* ]] && filtered+=("$n")
    done
    ((${#filtered[@]} > 0)) || \
      die "scope '$scope' matched no runs (valid: g0|g1|g2|g3|g4|all)"
  fi
  DIAG_NAMES=("${filtered[@]}")
}

# ── GPU worker ────────────────────────────────────────────────────────────────

run_diag_gpu_worker() {
  local worker="$1" gpu="$2" i name
  for ((i=worker; i<${#DIAG_NAMES[@]}; i+=GPU_COUNT)); do
    name="${DIAG_NAMES[$i]}"
    if run_complete "$name"; then
      say "[worker $worker / GPU $gpu] $name: already complete — skipping"
      continue
    fi
    say "[worker $worker / GPU $gpu] → $name"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export DEVICE="cuda:0"
      cd "$REPO_ROOT"
      train_one "$name"
      eval_one  "$name"
      mark_run_complete "$name"
      collect_results
    )
  done
}

# ── Main pipeline ─────────────────────────────────────────────────────────────

run_diagnostics() {
  cd "$REPO_ROOT"
  resolve_python
  resolve_gpus
  GPU_COUNT="${#GPU_ARRAY[@]}"; export GPU_COUNT
  mkdir -p "$RUNS_ROOT" "$M3_LOGDIR" "$M3_DIAGDIR" "$RESULTS_DIR" "$STATE_DIR/runs"

  exec 9>"$STATE_DIR/diag.lock"
  flock -n 9 || die "another run_diag.sh worker already holds the lock; run stop first"
  echo "$$" >"$STATE_DIR/diag.pid.tmp"; mv -f "$STATE_DIR/diag.pid.tmp" "$STATE_DIR/diag.pid"

  cleanup() {
    local code=$?
    if [[ -n "${collector_pid:-}" ]]; then
      kill "$collector_pid" 2>/dev/null || true
      wait "$collector_pid" 2>/dev/null || true
    fi
    collect_results || true
    if ((code == 0)); then
      write_stage complete
      printf '%s\n' "$(timestamp)" >"$STATE_DIR/DIAG.SUCCESS"
      rm -f "$STATE_DIR/DIAG.FAILED"
    else
      printf '%s exit=%s stage=%s\n' "$(timestamp)" "$code" \
        "$(cat "$STATE_DIR/diag_stage" 2>/dev/null || echo unknown)" \
        >"$STATE_DIR/DIAG.FAILED"
    fi
    rm -f "$STATE_DIR/diag.pid"
  }
  trap cleanup EXIT
  trap 'say "termination requested; stopping children"; exit 143' TERM INT

  collector_loop & collector_pid=$!

  say "VERA M3 diagnostic battery started"
  say "  scope:  $DIAG_SCOPE"
  say "  runs:   ${#DIAG_NAMES[@]}"
  say "  GPUs:   $GPU_IDS  (${#GPU_ARRAY[@]} workers)"
  say "  output: $DIAG_ROOT"
  echo ""
  say "Run plan:"
  for n in "${DIAG_NAMES[@]}"; do
    local eo="train"
    [[ "${DIAG_EVAL_ONLY[$n]}" == "1" ]] && eo="eval-only"
    printf '  %-52s box=%-8s %s\n' "$n" "${DIAG_BOX[$n]}" "$eo"
  done

  prepare_labels
  write_stage m3_diagnostic_matrix

  local pids=() worker failed=0
  for ((worker=0; worker<GPU_COUNT; worker++)); do
    run_diag_gpu_worker "$worker" "${GPU_ARRAY[$worker]}" &
    pids+=("$!")
  done
  for worker in "${pids[@]}"; do
    if ! wait "$worker"; then failed=1; fi
  done
  ((failed == 0)) || die "one or more GPU workers failed; run resume to continue"

  collect_results
  say "Diagnostic battery complete."
  echo ""
  cat "$RESULTS_DIR/diag_results.md" 2>/dev/null || true
}

# ── Status / control ──────────────────────────────────────────────────────────

is_running() {
  local pid=""
  [[ -s "$STATE_DIR/diag.pid" ]] && pid="$(cat "$STATE_DIR/diag.pid")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start_detached() {
  mkdir -p "$DIAG_ROOT/logs" "$STATE_DIR"
  is_running && die "diagnostic worker already running (PID $(cat "$STATE_DIR/diag.pid"))"
  preflight
  rm -f "$STATE_DIR/DIAG.FAILED" "$STATE_DIR/DIAG.SUCCESS"
  say "starting detached diagnostic; log=$SUPERVISOR_LOG"
  nohup setsid bash "$SCRIPT_DIR/run_diag.sh" __worker \
    --scope "$DIAG_SCOPE" ${_SINGLE_RUN_ARG} \
    >>"$SUPERVISOR_LOG" 2>&1 </dev/null &
  local launcher_pid=$!
  printf '%s\n' "$launcher_pid" >"$STATE_DIR/diag_launcher.pid"
  sleep 2
  if is_running; then
    say "started PID $(cat "$STATE_DIR/diag.pid"); closing SSH is safe"
  elif kill -0 "$launcher_pid" 2>/dev/null; then
    say "launcher PID $launcher_pid still starting; check status in a few seconds"
  else
    tail -n 80 "$SUPERVISOR_LOG" || true
    die "detached worker exited during startup"
  fi
}

show_status() {
  local state="stopped" pid="-" stage="not_started"
  is_running && { state="running"; pid="$(cat "$STATE_DIR/diag.pid")"; }
  [[ -s "$STATE_DIR/diag_stage" ]]   && stage="$(cat "$STATE_DIR/diag_stage")"
  [[ -s "$STATE_DIR/DIAG.SUCCESS" ]] && state="complete"
  [[ -s "$STATE_DIR/DIAG.FAILED" && "$state" != "running" ]] && state="failed"
  printf 'state:  %s\npid:    %s\nstage:  %s\noutput: %s\nlog:    %s\n\n' \
    "$state" "$pid" "$stage" "$DIAG_ROOT" "$SUPERVISOR_LOG"

  local done_count=0
  for n in "${DIAG_NAMES[@]}"; do
    if run_complete "$n"; then
      printf "  [DONE]        %s\n" "$n"; ((done_count++))
    elif [[ -f "$RUNS_ROOT/$n/last.pt" ]]; then
      printf "  [IN PROGRESS] %s\n" "$n"
    else
      printf "  [pending]     %s\n" "$n"
    fi
  done
  printf '\n%d / %d runs complete\n' "$done_count" "${#DIAG_NAMES[@]}"

  if [[ -s "$RESULTS_DIR/diag_results.md" ]]; then
    echo ""; echo "--- latest results ---"
    cat "$RESULTS_DIR/diag_results.md"
  fi
  [[ -s "$STATE_DIR/DIAG.FAILED" ]] && { echo "--- last failure ---"; cat "$STATE_DIR/DIAG.FAILED"; }
  [[ -s "$SUPERVISOR_LOG" ]]        && { echo "--- latest log ---"; tail -n 25 "$SUPERVISOR_LOG"; }
}

stop_worker() {
  is_running || { say "diagnostic worker is not running"; return; }
  local pid; pid="$(cat "$STATE_DIR/diag.pid")"
  say "sending TERM to process group $pid"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
  say "stop requested; use 'status' to confirm, then 'start'/'resume' later"
}

# ============================================================
# Argument parsing
# ============================================================

DIAG_SCOPE="all"
DIAG_SINGLE_RUN=""
_SINGLE_RUN_ARG=""
COMMAND="${1:-}"
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope) DIAG_SCOPE="${2:?}"; shift 2 ;;
    --run)   DIAG_SINGLE_RUN="${2:?}"; _SINGLE_RUN_ARG="--run $DIAG_SINGLE_RUN"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

filter_scope "$DIAG_SCOPE" "$DIAG_SINGLE_RUN"

case "$COMMAND" in
  preflight)          preflight ;;
  start|resume)       start_detached ;;
  foreground)         preflight; run_diagnostics ;;
  status)             resolve_python; show_status ;;
  logs)               touch "$SUPERVISOR_LOG"; tail -n 200 -F "$SUPERVISOR_LOG" ;;
  collect)            resolve_python; collect_results ;;
  stop)               stop_worker ;;
  __worker)           run_diagnostics ;;
  -h|--help|help|"")  usage ;;
  *)                  usage; die "unknown command: $COMMAND" ;;
esac
