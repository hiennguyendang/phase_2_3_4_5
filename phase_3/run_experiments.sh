#!/usr/bin/env bash
# M3 experiment grid — runs SEQUENTIALLY (one job at a time: MIG slice shares SMs, parallel = slower).
# Idempotent: a run whose best.pt already exists is skipped (delete it to re-run). Each run also
# eval (test) + faithfulness (val); logs go to logs/<name>.*.log; a summary prints at the end.
#
#   bash phase_3/run_experiments.sh
#   EP=20 BATCH=512 W=16 bash phase_3/run_experiments.sh      # override defaults
#   nohup bash phase_3/run_experiments.sh > logs/grid.log 2>&1 &   # background, survive logout

cd "$(dirname "$0")/.." || exit 1          # repo root
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

LABELS=data/m3_labels
FEAT=data/features/frozen
RUNS=data/run
EP=${EP:-30}; BATCH=${BATCH:-512}; W=${W:-16}
mkdir -p logs

run () {                                    # run <name> <box-source> <train flags...>
  name="$1"; box="$2"; shift 2
  ck="$RUNS/$name/best.pt"
  echo ""; echo "===== $name (box=$box) $* ====="
  if [ -f "$ck" ]; then
    echo "  [skip train] $ck exists"
  else
    python3 phase_3/scripts/4-train.py --labels-dir "$LABELS" --features-root "$FEAT" \
      --epochs "$EP" --batch "$BATCH" --workers "$W" --device cuda \
      --box-source "$box" "$@" --name "$name" 2>&1 | tee "logs/$name.train.log"
  fi
  python3 phase_3/scripts/5-eval.py --ckpt "$ck" --labels-dir "$LABELS" --features-root "$FEAT" \
    --split test --box-source "$box" --workers "$W" --device cuda 2>&1 | tee "logs/$name.eval.log"
  python3 phase_3/scripts/6-faithfulness.py --ckpt "$ck" --labels-dir "$LABELS" --features-root "$FEAT" \
    --split val --box-source "$box" --workers "$W" --device cuda 2>&1 | tee "logs/$name.faith.log"
}

# ---- Tier 0/1: the 3 spec directions + mode-B disease-head variants (cfg is saved in each ckpt) ----
run m3_B_faithful  detector  --mode B --disease-head faithful   # non-neg + masked -> intervention PASS
run m3_A           detector  --mode A
run m3_B           detector  --mode B --disease-head mlp        # accuracy CBM (intervention FAILs)
# run m3_B_linear    detector  --mode B --disease-head linear     # learned dense linear (no sign)
run m3_B_nonneg    detector  --mode B --disease-head nonneg     # non-neg, no mask
# run m3_C           detector  --mode C                           # hybrid (run leakage test)

# ---- Tier 2: trunk ablations, one toggle off from the faithful-B baseline ----
# run m3_Bf_nomask   detector  --mode B --disease-head faithful --no-mask-bbox
# run m3_Bf_neck128  detector  --mode B --disease-head faithful --neck-dim 128
run m3_Bf_aggmax   detector  --mode B --disease-head faithful --region-agg max
run m3_Bf_noglobal detector  --mode B --disease-head faithful --no-global-head

# ---- Tier 3: gold-box oracle (how much detector-box error costs M3) ----
run m3_Bf_gtbox    gt        --mode B --disease-head faithful

# ---------------- summary ----------------
echo ""; echo "================= SUMMARY ================="
printf "%-18s %-28s %s\n" "run" "eval(test)" "why-faithful"
for d in "$RUNS"/*/; do
  n=$(basename "$d")
  e=$(grep -h "image  F1 macro" "logs/$n.eval.log" 2>/dev/null | tail -1 | sed 's/^ *//')
  f=$(grep -h "faithful.*claim allowed" "logs/$n.faith.log" 2>/dev/null | tail -1)
  printf "%-18s %-28s %s\n" "$n" "${e:-—}" "${f:-—}"
done
echo ""
echo "Curves/compare: python3 phase_3/scripts/plot_metrics.py $RUNS/*"
