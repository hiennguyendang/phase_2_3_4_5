#!/usr/bin/env bash
# MS-CXR-T Evaluation Runner
# ============================================================
# Sweeps every M4 checkpoint found under RUNS_ROOT matching m4v2_*
# and runs 5-mscxrt_audit.py on each.  Results are written as JSON
# into DIAGDIR and a summary markdown table is emitted to stdout
# and saved to DIAGDIR/mscxrt_summary.md.
#
# Skips checkpoints whose .mscxrt.json already exists (resumable).
# Safe to run while run_all.sh or run_diag.sh are still active
# (read-only for the checkpoints, writes only to DIAGDIR).
#
# Usage:
#   bash server_hoang/run_mscxrt.sh              # foreground, all m4v2_* runs
#   bash server_hoang/run_mscxrt.sh start        # detached
#   bash server_hoang/run_mscxrt.sh status       # show progress
#   bash server_hoang/run_mscxrt.sh logs         # follow log
#   bash server_hoang/run_mscxrt.sh --pattern "m4v2_grid_kl0050*"  # filter
#
# Run from any directory.

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

SERVER_ENV="${SERVER_ENV:-$SCRIPT_DIR/server.env}"
if [[ -f "$SERVER_ENV" ]]; then
  set -a; source "$SERVER_ENV"; set +a
fi

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
RUNS_ROOT="${RUNS_ROOT:-$OUTPUT_ROOT/runs}"
# MS-CXR-T results go alongside the main M4 diag JSONs so collect_results.py
# can pick them up automatically.
DIAGDIR="${DIAGDIR:-$OUTPUT_ROOT/diagnostics/m4}"
LOGDIR="${LOGDIR:-$OUTPUT_ROOT/logs/mscxrt}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-$LOGDIR/mscxrt.supervisor.log}"
STATE_DIR="${STATE_DIR:-$OUTPUT_ROOT/state}"

# M4 region cache built by run_all.sh (needed by the dataset)
CACHE_DET="${CACHE_DET:-$OUTPUT_ROOT/cache/m4_region_detector}"
CACHE_GT="${CACHE_GT:-$OUTPUT_ROOT/cache/m4_region_gt_oracle}"

# Labels and features
FEATURE_ROOT="${FEATURE_ROOT:-${INPUT_ROOT:-$SCRIPT_DIR/input}/frozen}"
M3_LABELS_WORK="${M3_LABELS_WORK:-$OUTPUT_ROOT/labels/m3}"
M4_LABELS="${M4_LABELS:-${INPUT_ROOT:-$SCRIPT_DIR/input}/m4_labels}"

# MS-CXR-T CSV — server absolute path
MS_CSV="${MS_CSV:-/home/jovyan/phase_2_3_4_5/data/MS_CXR_T_temporal_image_classification_v1.0.0.csv}"

# ── tuning knobs ──────────────────────────────────────────────────────────────
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
# Glob pattern to select runs (matched against basename of run dir)
RUN_PATTERN="${RUN_PATTERN:-m4v2_*}"
# aggregation strategies to report (all → mean, max, lse)
AGG="${AGG:-all}"
# MS-CXR-T split (use "all" for full external benchmark)
SPLIT="${SPLIT:-all}"

# ── helpers ───────────────────────────────────────────────────────────────────
timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
say()       { printf '[%s] %s\n' "$(timestamp)" "$*"; }
die()       { say "ERROR: $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

resolve_python() {
  if   [[ -n "${PY:-}" ]];                             then :
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]];         then PY="$REPO_ROOT/.venv/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then PY="$REPO_ROOT/.venv/Scripts/python.exe"
  else PY="python3"; fi
  export PY
}

# ── collect_runs: find all matching m4 checkpoints ────────────────────────────
collect_runs() {
  # Returns a newline-separated list of run directories that have best.pt
  local pattern="$1"
  find "$RUNS_ROOT" -maxdepth 1 -type d -name "$pattern" \
    | sort \
    | while read -r d; do
        [[ -f "$d/best.pt" ]] && echo "$d"
      done
}

# ── out_json: path for a run's mscxrt result ──────────────────────────────────
out_json() {
  local run_dir="$1"
  local name; name="$(basename "$run_dir")"
  echo "$DIAGDIR/$name.mscxrt.json"
}

# ── eval one checkpoint ───────────────────────────────────────────────────────
eval_one() {
  local run_dir="$1"
  local name; name="$(basename "$run_dir")"
  local ckpt="$run_dir/best.pt"
  local out="$(out_json "$run_dir")"
  local log="$LOGDIR/$name.mscxrt.log"

  if [[ -s "$out" ]]; then
    say "[skip] $name — result already exists at $out"
    return
  fi

  say "[eval] $name"
  mkdir -p "$(dirname "$log")"

  # Choose region cache: detector or gt based on checkpoint box_source
  local box_src
  box_src="$("$PY" - "$ckpt" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(c.get("box_source", "detector"))
PY
)"
  local region_cache
  if [[ "$box_src" == "gt" ]]; then
    region_cache="$CACHE_GT"
  else
    region_cache="$CACHE_DET"
  fi

  say "[eval] $name  box_source=$box_src  cache=$region_cache"

  "$PY" phase_4/scripts/5-mscxrt_audit.py \
    --ckpt "$ckpt" \
    --csv  "$MS_CSV" \
    --region-cache   "$region_cache" \
    --features-root  "$FEATURE_ROOT" \
    --m3-labels-dir  "$M3_LABELS_WORK" \
    --split  "$SPLIT" \
    --agg    "$AGG" \
    --batch  "$BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --out-json "$out" \
    2>&1 | tee "$log"

  say "[done] $name → $out"
}

# ── build summary table ───────────────────────────────────────────────────────
build_summary() {
  resolve_python
  mkdir -p "$DIAGDIR"
  "$PY" - "$DIAGDIR" "$RUN_PATTERN" <<'PY'
import json, sys, re
from pathlib import Path

diagdir  = Path(sys.argv[1])
pattern  = sys.argv[2]  # unused for filtering here; we scan all .mscxrt.json

rows = []
for jf in sorted(diagdir.glob("*.mscxrt.json")):
    try:
        d = json.loads(jf.read_text())
    except Exception:
        continue
    name = jf.stem.replace(".mscxrt", "")
    aggs = d.get("aggregations", {})
    # prefer lse aggregation as the headline number; fall back to mean
    for preferred in ("lse", "mean", "max"):
        if preferred in aggs:
            headline = aggs[preferred]
            headline_agg = preferred
            break
    else:
        continue

    all_agg_str = "  ".join(
        f"{a}: bal={v.get('avg_finding_balanced_accuracy', float('nan')):.4f}"
        for a, v in aggs.items()
    )

    rows.append({
        "run":           name,
        "agg":           headline_agg,
        "prog_f1":       headline.get("prog_f1_macro", float("nan")),
        "change_f1":     headline.get("change_f1_macro", float("nan")),
        "accuracy":      headline.get("accuracy", float("nan")),
        "bal_accuracy":  headline.get("balanced_accuracy", float("nan")),
        "avg_find_acc":  headline.get("avg_finding_accuracy", float("nan")),
        "avg_find_bal":  headline.get("avg_finding_balanced_accuracy", float("nan")),
        "n":             headline.get("n_valid", 0),
        "all_aggs":      all_agg_str,
        "per_finding":   aggs.get(headline_agg, {}).get("per_finding", {}),
    })

# Sort by avg_find_bal descending (the SOTA comparison metric)
rows.sort(key=lambda r: r["avg_find_bal"], reverse=True)

# ── header ────────────────────────────────────────────────────────────────────
lines = [
    "# MS-CXR-T Evaluation Summary",
    "",
    "> **Metric:** avg-finding balanced accuracy (= mean per-class recall averaged across",
    "> findings) — this matches BioViL-T and CoCa-CXR reporting.",
    "> BioViL-T baseline: **0.602** | CoCa-CXR fine-tune: **0.650**",
    "",
    "## Overall Table",
    "",
    "| Run | Agg | Prog F1 | Change F1 | Accuracy | Bal Acc | **Avg-Find Bal Acc** | n |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| `{r['run']}` | {r['agg']} "
        f"| {r['prog_f1']:.4f} | {r['change_f1']:.4f} "
        f"| {r['accuracy']:.4f} | {r['bal_accuracy']:.4f} "
        f"| **{r['avg_find_bal']:.4f}** | {r['n']:,} |"
    )

# ── per-finding breakdown for top-3 runs ──────────────────────────────────────
lines += ["", "## Per-Finding Detail (top 3 runs by avg-find-bal-acc)", ""]
for r in rows[:3]:
    lines.append(f"### {r['run']} (agg={r['agg']})")
    lines.append("")
    lines.append("| Finding | Acc | Bal Acc | Change F1 | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for finding, fv in r["per_finding"].items():
        lines.append(
            f"| {finding} "
            f"| {fv.get('accuracy', float('nan')):.4f} "
            f"| {fv.get('balanced_accuracy', float('nan')):.4f} "
            f"| {fv.get('change_f1', float('nan')):.4f} "
            f"| {fv.get('n', 0):,} |"
        )
    lines.append("")

# ── all aggregations for top run ──────────────────────────────────────────────
if rows:
    top = rows[0]
    jf = diagdir / f"{top['run']}.mscxrt.json"
    try:
        d = json.loads(jf.read_text())
        lines += [f"## All Aggregations — Best Run ({top['run']})", ""]
        lines.append("| Agg | Prog F1 | Change F1 | Accuracy | Bal Acc | Avg-Find Acc | Avg-Find Bal Acc |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for a, v in d.get("aggregations", {}).items():
            lines.append(
                f"| {a} | {v.get('prog_f1_macro', float('nan')):.4f} "
                f"| {v.get('change_f1_macro', float('nan')):.4f} "
                f"| {v.get('accuracy', float('nan')):.4f} "
                f"| {v.get('balanced_accuracy', float('nan')):.4f} "
                f"| {v.get('avg_finding_accuracy', float('nan')):.4f} "
                f"| {v.get('avg_finding_balanced_accuracy', float('nan')):.4f} |"
            )
    except Exception:
        pass

md = "\n".join(lines) + "\n"
out = diagdir / "mscxrt_summary.md"
out.write_text(md, encoding="utf-8")
print(f"[summary] {len(rows)} runs → {out}")
for r in rows:
    print(f"  {r['run']:50s}  avg-find-bal={r['avg_find_bal']:.4f}  acc={r['accuracy']:.4f}")
PY
}

# ── main evaluation loop ──────────────────────────────────────────────────────
run_all_evals() {
  cd "$REPO_ROOT"
  resolve_python
  mkdir -p "$DIAGDIR" "$LOGDIR"

  say "MS-CXR-T sweep started"
  say "  runs root: $RUNS_ROOT"
  say "  pattern:   $RUN_PATTERN"
  say "  csv:       $MS_CSV"
  say "  split:     $SPLIT"
  say "  agg:       $AGG"
  say "  diagdir:   $DIAGDIR"

  require_file "$MS_CSV"

  local runs
  mapfile -t runs < <(collect_runs "$RUN_PATTERN")
  if ((${#runs[@]} == 0)); then
    die "no run directories found under $RUNS_ROOT matching '$RUN_PATTERN'"
  fi

  say "found ${#runs[@]} run(s):"
  for d in "${runs[@]}"; do
    local out; out="$(out_json "$d")"
    local mark="pending"
    [[ -s "$out" ]] && mark="done"
    printf '  [%-7s] %s\n' "$mark" "$(basename "$d")"
  done

  local failed=0
  for d in "${runs[@]}"; do
    eval_one "$d" || { say "WARN: $d failed — continuing"; failed=1; }
  done

  say "building summary table"
  build_summary
  echo ""
  cat "$DIAGDIR/mscxrt_summary.md" 2>/dev/null || true

  ((failed == 0)) || say "WARNING: one or more runs failed; check logs in $LOGDIR"
  say "MS-CXR-T sweep complete"
}

# ── status / detached control ─────────────────────────────────────────────────
is_running() {
  local pid=""
  [[ -s "$STATE_DIR/mscxrt.pid" ]] && pid="$(cat "$STATE_DIR/mscxrt.pid")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start_detached() {
  mkdir -p "$LOGDIR" "$STATE_DIR"
  is_running && die "mscxrt worker already running (PID $(cat "$STATE_DIR/mscxrt.pid"))"
  say "starting detached MS-CXR-T sweep; log=$SUPERVISOR_LOG"
  nohup setsid bash "$SCRIPT_DIR/run_mscxrt.sh" __worker \
    >>"$SUPERVISOR_LOG" 2>&1 </dev/null &
  local launcher_pid=$!
  sleep 1
  say "launcher PID=$launcher_pid; closing SSH is safe"
  say "follow with: bash server_hoang/run_mscxrt.sh logs"
}

show_status() {
  resolve_python
  local state="stopped"
  is_running && state="running (PID $(cat "$STATE_DIR/mscxrt.pid" 2>/dev/null))"

  printf 'state:   %s\ndiagdir: %s\nlog:     %s\n\n' "$state" "$DIAGDIR" "$SUPERVISOR_LOG"

  local runs
  mapfile -t runs < <(collect_runs "$RUN_PATTERN" 2>/dev/null || true)
  local done_count=0
  for d in "${runs[@]}"; do
    local out; out="$(out_json "$d")"
    if [[ -s "$out" ]]; then
      printf '  [DONE]   %s\n' "$(basename "$d")"; ((done_count++))
    else
      printf '  [pending] %s\n' "$(basename "$d")"
    fi
  done
  printf '\n%d / %d runs complete\n' "$done_count" "${#runs[@]}"

  if [[ -s "$DIAGDIR/mscxrt_summary.md" ]]; then
    echo ""; echo "--- latest results ---"
    cat "$DIAGDIR/mscxrt_summary.md"
  fi
}

# ── argument parsing ──────────────────────────────────────────────────────────
COMMAND="${1:-foreground}"
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pattern) RUN_PATTERN="${2:?}"; shift 2 ;;
    --csv)     MS_CSV="${2:?}"; shift 2 ;;
    --split)   SPLIT="${2:?}"; shift 2 ;;
    --agg)     AGG="${2:?}"; shift 2 ;;
    --batch)   BATCH="${2:?}"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$COMMAND" in
  foreground|run)   run_all_evals ;;
  start)            start_detached ;;
  status)           show_status ;;
  logs)             touch "$SUPERVISOR_LOG"; tail -n 200 -F "$SUPERVISOR_LOG" ;;
  summary)          resolve_python; build_summary; cat "$DIAGDIR/mscxrt_summary.md" ;;
  __worker)
    mkdir -p "$STATE_DIR"
    echo "$$" >"$STATE_DIR/mscxrt.pid.tmp"
    mv -f "$STATE_DIR/mscxrt.pid.tmp" "$STATE_DIR/mscxrt.pid"
    cleanup() { rm -f "$STATE_DIR/mscxrt.pid"; }
    trap cleanup EXIT
    run_all_evals
    ;;
  -h|--help|help|"")
    cat <<'EOF'
Usage: bash server_hoang/run_mscxrt.sh [COMMAND] [OPTIONS]

Commands:
  foreground   Run in current terminal (default)
  start        Launch detached (safe after SSH disconnect)
  status       Show per-run completion and latest results
  logs         Follow the supervisor log
  summary      Regenerate and print mscxrt_summary.md from existing JSONs

Options:
  --pattern GLOB   Run name glob (default: m4v2_*)
  --csv PATH       MS-CXR-T CSV path (default: from server.env or hardcoded)
  --split SPLIT    all|train|val|test (default: all)
  --agg AGG        all|mean|max|lse (default: all — reports all three)
  --batch N        batch size (default: 64)

Results:
  Each run: DIAGDIR/<run>.mscxrt.json
  Summary:  DIAGDIR/mscxrt_summary.md

Key metric: avg_finding_balanced_accuracy
  BioViL-T baseline:  0.602
  CoCa-CXR fine-tune: 0.650
EOF
    ;;
  *) die "unknown command: $COMMAND" ;;
esac
