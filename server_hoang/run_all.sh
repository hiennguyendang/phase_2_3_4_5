#!/usr/bin/env bash
# VERA server supervisor: resumable M2 inference -> M3 paper matrix -> M4 paper matrix.
# Run from any directory. See server_hoang/README.md before the first launch.

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

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
  "$PY" - "$1" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for block in iter(lambda: f.read(8 << 20), b""):
        h.update(block)
print(h.hexdigest())
PY
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

  "$PY" - "$GPU_IDS" "$REPO_ROOT" <<'PY'
import importlib, json, sys
from pathlib import Path
for module in ("torch", "numpy", "ultralytics", "sklearn", "matplotlib", "pandas"):
    importlib.import_module(module)
import torch
ids = [x for x in sys.argv[1].split(",") if x]
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] torch cannot access CUDA")
print("torch:", torch.__version__, "visible CUDA devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  cuda", i, torch.cuda.get_device_name(i))
root = Path(sys.argv[2])
copies = [root / "data/m3_concept_space.json", root / "phase_3/src/m3_concept_space.json",
          root / "phase_4/src/m3_concept_space.json"]
objects = [json.loads(p.read_text(encoding="utf-8-sig")) for p in copies]
assert all(x == objects[0] for x in objects[1:]), "concept-space copies differ"
PY

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
  "$PY" - "$STATE_DIR/stages/$stage.SUCCESS.json" "$stage" "$(timestamp)" <<'PY'
import json, os, sys
from pathlib import Path
path, stage, completed = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps({"stage": stage, "status": "complete", "completed_at": completed}, indent=2) + "\n")
os.replace(tmp, path)
PY
}

stage_complete() { [[ -s "$STATE_DIR/stages/$1.SUCCESS.json" ]]; }

collect_results() {
  resolve_python
  mkdir -p "$RESULTS_DIR"
  (
  exec 8>"$RESULTS_DIR/collector.lock"
  flock -w 30 8 || exit 0
  OUTPUT_ROOT="$OUTPUT_ROOT" RUNS_ROOT="$RUNS_ROOT" M2_ROOT="$M2_ROOT" \
    M3_DIAGDIR="$M3_DIAGDIR" M4_DIAGDIR="$M4_DIAGDIR" RESULTS_DIR="$RESULTS_DIR" \
    TARGET_M3_EPOCHS="$M3_EPOCHS" TARGET_M4_EPOCHS="$M4_EPOCHS" \
    "$PY" - <<'PY'
import csv, datetime as dt, json, math, os
from pathlib import Path

import torch

runs = Path(os.environ["RUNS_ROOT"])
m2 = Path(os.environ["M2_ROOT"])
m3diag = Path(os.environ["M3_DIAGDIR"])
m4diag = Path(os.environ["M4_DIAGDIR"])
out = Path(os.environ["RESULTS_DIR"])
targets = {"M3": int(os.environ["TARGET_M3_EPOCHS"]), "M4": int(os.environ["TARGET_M4_EPOCHS"])}
now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
m3_names = [
    "m3v2_vera_graph_lse_det", "m3v2_no_concept_det", "m3v2_concept_mlp_det",
    "m3v2_graph_global_fusion_det", "m3v2_global_only_det",
    "m3v2_graph_attention_det", "m3v2_graph_mean_det", "m3v2_graph_max_det",
    "m3v2_vera_graph_lse_gt",
]

def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}

def number(value):
    try:
        value = float(value)
        return "" if not math.isfinite(value) else f"{value:.6f}"
    except Exception:
        return ""

rows = []
m2_success = load_json(m2 / "M2.SUCCESS.json")
rows.append({
    "stage": "M2", "run": "yolo29_full_inference", "status": "complete" if m2_success else "pending",
    "epochs": "", "target_epochs": "", "box_source": "detector",
    "val_primary": "", "test_primary": "", "val_aux": "", "test_aux": "",
    "best_checkpoint": str(m2 / "predictions.jsonl") if m2_success else "", "updated_at": now,
})

run_dirs = {p.name: p for p in runs.glob("*") if p.is_dir()} if runs.exists() else {}
names = list(m3_names) + sorted(n for n in run_dirs if n.startswith("m4v2_"))
for name in names:
    stage = "M3" if name.startswith("m3v2_") else "M4"
    run = run_dirs.get(name, runs / name)
    last = load_ckpt(run / "last.pt")
    best_path = run / "best.pt"
    best = load_ckpt(best_path)
    epochs = int(last.get("epoch", -1)) + 1 if last else 0
    target = targets[stage]
    status = "complete" if epochs >= target and best_path.is_file() else ("running" if last or run.exists() else "pending")
    val = load_json((m3diag if stage == "M3" else m4diag) / f"{name}.val.json")
    test = load_json((m3diag if stage == "M3" else m4diag) / f"{name}.test.json")
    if stage == "M3":
        val_primary = val.get("image_auc_macro", best.get("val_auc"))
        test_primary = test.get("image_auc_macro")
        val_aux = val.get("image_f1_macro", best.get("val_f1"))
        test_aux = test.get("image_f1_macro")
        box = best.get("box_source", last.get("box_source", ""))
    else:
        val_primary = val.get("change_f1_macro", best.get("val_change_f1"))
        test_primary = test.get("change_f1_macro")
        val_aux = val.get("prog_f1_macro", best.get("val_f1"))
        test_aux = test.get("prog_f1_macro")
        box = best.get("box_source", last.get("box_source", ""))
    rows.append({
        "stage": stage, "run": name, "status": status, "epochs": epochs,
        "target_epochs": target, "box_source": box, "val_primary": number(val_primary),
        "test_primary": number(test_primary), "val_aux": number(val_aux),
        "test_aux": number(test_aux), "best_checkpoint": str(best_path) if best_path.is_file() else "",
        "updated_at": now,
    })

fields = ["stage", "run", "status", "epochs", "target_epochs", "box_source",
          "val_primary", "test_primary", "val_aux", "test_aux", "best_checkpoint", "updated_at"]
csv_tmp = out / "runs.csv.tmp"
with csv_tmp.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
os.replace(csv_tmp, out / "runs.csv")

md = ["# VERA server result collector", "", f"Updated: `{now}`", "",
      "For M3, primary = image macro-AUC and auxiliary = image macro-F1. "
      "For M4, primary = change-only macro-F1 and auxiliary = progression macro-F1.", "",
      "| Stage | Run | Status | Epoch | Box | Val primary | Test primary | Val aux | Test aux |",
      "|---|---|---:|---:|---|---:|---:|---:|---:|"]
for r in rows:
    epoch = f"{r['epochs']}/{r['target_epochs']}" if r["target_epochs"] else "-"
    md.append(f"| {r['stage']} | `{r['run']}` | {r['status']} | {epoch} | {r['box_source']} | "
              f"{r['val_primary'] or '-'} | {r['test_primary'] or '-'} | "
              f"{r['val_aux'] or '-'} | {r['test_aux'] or '-'} |")
md_tmp = out / "runs.md.tmp"
md_tmp.write_text("\n".join(md) + "\n", encoding="utf-8")
os.replace(md_tmp, out / "runs.md")

summary = {
    "updated_at": now,
    "counts": {s: sum(r["status"] == s for r in rows) for s in ("complete", "running", "pending")},
    "rows": rows,
}
js_tmp = out / "summary.json.tmp"
js_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
os.replace(js_tmp, out / "summary.json")
print(f"[collector] {now}: {summary['counts']} -> {out / 'runs.md'}")
PY
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
  "$PY" - "$contract" "$YOLO_WEIGHTS" "$M3_LABELS_WORK/manifest.jsonl" "$IMAGE_ROOT" \
    "$YOLO_IMGSZ" "$YOLO_CONF" "$YOLO_IOU" "$M2_BATCH" "$M2_NUM_SHARDS" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
dest, weights, manifest, images = map(Path, sys.argv[1:5])
want = {
    "schema_version": 1, "weights": str(weights.resolve()), "weights_sha256": sha(weights),
    "manifest": str(manifest.resolve()), "manifest_sha256": sha(manifest),
    "image_root": str(images.resolve()), "imgsz": int(sys.argv[5]), "conf": float(sys.argv[6]),
    "iou": float(sys.argv[7]), "batch": int(sys.argv[8]), "num_shards": int(sys.argv[9]),
}
if dest.exists():
    got=json.loads(dest.read_text(encoding="utf-8"))
    if got != want:
        raise SystemExit(f"[ERROR] M2 contract changed. Preserve the old OUTPUT_ROOT and choose a new one.\nold={got}\nnew={want}")
else:
    tmp=dest.with_suffix(".json.tmp"); tmp.write_text(json.dumps(want,indent=2)+"\n"); os.replace(tmp,dest)
print("[M2 contract]", json.dumps(want, indent=2))
PY
}

m2_shard_complete() {
  local shard="$1" marker="$M2_ROOT/shards/shard_$(printf '%04d' "$shard")/SUCCESS.json"
  [[ -s "$marker" ]] || return 1
  "$PY" - "$marker" "$M2_ROOT/shards/shard_$(printf '%04d' "$shard")/predictions.jsonl" \
    "$M3_LABELS_WORK/manifest.jsonl" "$shard" "$M2_NUM_SHARDS" <<'PY' >/dev/null 2>&1
import json, sys
from pathlib import Path
marker, pred, manifest = map(Path, sys.argv[1:4]); shard, nshards = map(int, sys.argv[4:6])
n=len({str(json.loads(x)["image_id"]) for x in manifest.read_text(encoding="utf-8-sig").splitlines() if x.strip()})
expected=max(0,(n-shard+nshards-1)//nshards)
got=sum(1 for x in pred.open(encoding="utf-8") if x.strip())
m=json.loads(marker.read_text(encoding="utf-8"))
assert m.get("status")=="complete" and m.get("rows")==got==expected
PY
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
    "$PY" - "$out/predictions.jsonl" "$M3_LABELS_WORK/manifest.jsonl" "$shard" \
      "$M2_NUM_SHARDS" "$out/SUCCESS.json" <<'PY'
import json, os, sys
from pathlib import Path
pred, manifest, dest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[5])
shard, nshards = int(sys.argv[3]), int(sys.argv[4])
n=len({str(json.loads(x)["image_id"]) for x in manifest.read_text(encoding="utf-8-sig").splitlines() if x.strip()})
expected=max(0,(n-shard+nshards-1)//nshards)
rows=sum(1 for x in pred.open(encoding="utf-8") if x.strip())
if rows != expected: raise SystemExit(f"[ERROR] shard {shard}: rows={rows}, expected={expected}")
tmp=dest.with_suffix(".json.tmp")
tmp.write_text(json.dumps({"status":"complete","shard":shard,"num_shards":nshards,"rows":rows},indent=2)+"\n")
os.replace(tmp,dest)
PY
  done
}

merge_m2_shards() {
  "$PY" - "$M2_ROOT" "$M3_LABELS_WORK/manifest.jsonl" "$M2_NUM_SHARDS" <<'PY'
import json, os, sys
from pathlib import Path
root, manifest, nshards = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
expected={str(json.loads(x)["image_id"]) for x in manifest.read_text(encoding="utf-8-sig").splitlines() if x.strip()}
seen=set(); tmp=root/"predictions.jsonl.tmp"
with tmp.open("w",encoding="utf-8") as dst:
    for shard in range(nshards):
        src=root/"shards"/f"shard_{shard:04d}"/"predictions.jsonl"
        for line in src.open(encoding="utf-8"):
            if not line.strip(): continue
            rec=json.loads(line); iid=str(rec["image_id"])
            if iid in seen: raise SystemExit(f"[ERROR] duplicate M2 image_id during merge: {iid}")
            seen.add(iid); dst.write(line if line.endswith("\n") else line+"\n")
missing=expected-seen; extra=seen-expected
if missing or extra:
    raise SystemExit(f"[ERROR] M2 merge coverage mismatch: missing={len(missing)} extra={len(extra)} first_missing={sorted(missing)[:5]}")
os.replace(tmp,root/"predictions.jsonl")
print(f"[M2 merge] {len(seen):,} unique predictions -> {root/'predictions.jsonl'}")
PY
}

write_detector_provenance() {
  "$PY" - "$YOLO_WEIGHTS" "$M2_ROOT/predictions.jsonl" "$M3_LABELS_WORK" \
    "$YOLO_IMGSZ" "$YOLO_CONF" "$YOLO_IOU" "$M2_BATCH" "$M2_NUM_SHARDS" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
weights, pred, labels = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
boxes, mask = labels/"boxes_det.npy", labels/"present_mask_det.npy"
m=np.load(mask,mmap_mode="r")
payload={"schema_version":2,"detector_checkpoint":str(weights),"detector_checkpoint_sha256":sha(weights),
 "prediction_jsonl":str(pred),"prediction_jsonl_sha256":sha(pred),"manifest_rows":int(m.shape[0]),
 "imgsz":int(sys.argv[4]),"conf":float(sys.argv[5]),"iou":float(sys.argv[6]),"batch":int(sys.argv[7]),
 "num_shards":int(sys.argv[8]),"coordinate_frame":448,"boxes_det_sha256":sha(boxes),
 "present_mask_det_sha256":sha(mask),"shape":list(m.shape),"mean_regions_per_image":float(m.sum(axis=1).mean())}
dest=labels/"detector_provenance.json"; tmp=dest.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload,indent=2)+"\n"); os.replace(tmp,dest)
print(json.dumps(payload,indent=2))
PY
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
    M2_NUM_SHARDS="$($PY - "$M2_ROOT/contract.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1],encoding="utf-8"))["num_shards"]))
PY
)"
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
  "$PY" - "$M2_ROOT/M2.SUCCESS.json" "$M2_ROOT/predictions.jsonl" "$M3_LABELS_WORK" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
dest,pred,labels=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])
payload={"status":"complete","predictions_sha256":sha(pred),"boxes_det_sha256":sha(labels/'boxes_det.npy'),
 "present_mask_det_sha256":sha(labels/'present_mask_det.npy')}
tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');os.replace(tmp,dest)
PY
  mark_stage_complete m2
  collect_results
}

m3_remote() { [[ -n "$REMOTE_ROOT" ]] && printf '%s/m3_runs' "${REMOTE_ROOT%/}" || true; }
m4_remote() { [[ -n "$REMOTE_ROOT" ]] && printf '%s/m4_runs' "${REMOTE_ROOT%/}" || true; }

write_run_manifest() {
  local git_commit="unavailable"
  git_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unavailable)"
  "$PY" - "$STATE_DIR/run_manifest.json" "$git_commit" "$REPO_ROOT" "$INPUT_ROOT" "$OUTPUT_ROOT" \
    "$GPU_IDS" "$M3_EPOCHS" "$M3_BATCH" "$M4_EPOCHS" "$M4_BATCH" "$(timestamp)" <<'PY'
import json, os, sys
from pathlib import Path
dest=Path(sys.argv[1])
payload={"schema_version":1,"git_commit":sys.argv[2],"repo_root":sys.argv[3],
 "input_root":sys.argv[4],"output_root":sys.argv[5],"gpu_ids":sys.argv[6],
 "m3_epochs":int(sys.argv[7]),"m3_batch_per_run":int(sys.argv[8]),
 "m4_epochs":int(sys.argv[9]),"m4_batch":int(sys.argv[10]),"started_at":sys.argv[11]}
if dest.exists():
    previous=json.loads(dest.read_text(encoding="utf-8"))
    immutable=("git_commit","input_root","output_root","m3_epochs","m4_epochs")
    bad={k:(previous.get(k),payload.get(k)) for k in immutable if previous.get(k)!=payload.get(k)}
    if bad:
        raise SystemExit(f"[ERROR] run manifest changed for an existing output tree: {bad}. Use a new OUTPUT_ROOT.")
    payload["first_started_at"]=previous.get("first_started_at",previous.get("started_at"))
else:
    payload["first_started_at"]=payload["started_at"]
tmp=dest.with_suffix(".json.tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,dest)
print("[run manifest]",dest)
PY
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
    "$PY" - "$RESULTS_DIR/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8")); print("results updated:",x.get("updated_at")); print("counts:",x.get("counts"))
PY
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
