#!/usr/bin/env bash
# M4 grid v2 — the (a)(b)(c) improvements, each a SEPARATE OPTION on top of the v1 baseline.
# Nothing from v1 is removed; run_experiments.sh (v1) still reproduces the old runs. This layers:
#   (a) selection: pick best.pt by val CHANGE-only F1 + early-stop (val peaks by epoch ~1-10)
#   (b) signal:    --same-view  -> keep only same-ViewPosition prior pairs (clean Siamese diff)
#   (c) structure: --head-mode twostage -> factorized P(change) x P(direction), targets `improved`
# All v2 runs adopt (a) (it's strictly a selection/stopping change), then add (b)/(c) one at a time.
# v1 `m4_mlp` (macro-select, all views, flat head) stays the reference to compare against.
#
#   bash phase_4/run_experiments_v2.sh
#   EP=40 PAT=6 BATCH=64 W=8 bash phase_4/run_experiments_v2.sh
#   nohup bash phase_4/run_experiments_v2.sh > logs/m4v2_grid.log 2>&1 &
#
# Headline = CHANGE-only F1. ⚠ still SILVER test (README B2) — numbers provisional.

cd "$(dirname "$0")/.." || exit 1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

M3_CKPT=${M3_CKPT:-data/run/m3_B_faithful/best.pt}
FEAT=${FEAT:-data/features/frozen}
M3LAB=${M3LAB:-data/m3_labels}
M4LAB=${M4LAB:-data/m4_labels}
PAIRS=${PAIRS:-data/m4_labels/m3_pairs.jsonl}
CACHE=${CACHE:-data/m4_region_cache}
RUNS=${RUNS:-data/run}
EP=${EP:-30}; BATCH=${BATCH:-256}; W=${W:-16}; PAT=${PAT:-6}
mkdir -p logs

# ---- bridge (resumable; reuses the SAME B-faithful cache all runs share) ----
echo "===== BRIDGE (resumable, workers=$W) ====="
python3 phase_3/scripts/8-precompute_regions.py --ckpt "$M3_CKPT" \
  --labels-dir "$M3LAB" --features-root "$FEAT" --out-dir "$CACHE" \
  --batch "$BATCH" --workers "$W" --device cuda 2>&1 | tee logs/m4_bridge.log

# (a) is common to every v2 run: select on change-F1, early-stop after PAT stale evals
COMMON="--select-metric change --patience $PAT"

run () {                                     # run <name> <extra train flags...>
  name="$1"; shift
  ck="$RUNS/$name/best.pt"
  echo ""; echo "===== $name $* ====="
  if [ -f "$ck" ]; then
    echo "  [skip train] $ck exists"
  else
    # pass BOTH sources; the dataset picks region-cache (regiondiff) or features-root (tempfuse) by --arch
    python3 phase_4/scripts/2-train.py --region-cache "$CACHE" --features-root "$FEAT" \
      --m3-labels-dir "$M3LAB" --m4-labels-dir "$M4LAB" --pairs "$PAIRS" --out "$RUNS" \
      --epochs "$EP" --batch "$BATCH" --workers "$W" --device cuda \
      "$@" --name "$name" 2>&1 | tee "logs/$name.train.log"
  fi
  # eval reads arch / same_view / head_mode / box_source from the ckpt -> test set + model match training
  python3 phase_4/scripts/3-eval.py --ckpt "$ck" --region-cache "$CACHE" --features-root "$FEAT" \
    --m3-labels-dir "$M3LAB" --m4-labels-dir "$M4LAB" --pairs "$PAIRS" --split test --device cuda \
    2>&1 | tee "logs/$name.eval.log"
}

# ============================ v2 GRID ============================
# --- regiondiff arch (v1 cache): the (a)(b)(c) selection/signal/structure improvements ---
run m4v2_base       $COMMON                                   # (a) only  = change-select + early-stop
run m4v2_sameview   $COMMON --same-view                       # (a)+(b)
run m4v2_twostage   $COMMON --head-mode twostage              # (a)+(c)
run m4v2_sv2stage   $COMMON --same-view --head-mode twostage  # (a)+(b)+(c)

# --- tempfuse arch (nấc 3): read M1 patch grids, cross-attn(curr<-prior) -> M4 bbox pool -> head ---
# Reads data/features/frozen directly (NO bridge/region-cache). Head after the pool is the "mlp branch"
# we ablate (mlp/kan/linear) alongside the same (a)(b)(c) options + fusion depth.
TF="$COMMON --arch tempfuse"
run m4v3_tf            $TF                                    # tempfuse baseline (mlp, flat, 1 block)
run m4v3_tf_sameview   $TF --same-view                        # + (b) clean Siamese
run m4v3_tf_twostage   $TF --head-mode twostage               # + (c) change x direction
run m4v3_tf_sv2stage   $TF --same-view --head-mode twostage   # + (b)+(c)  ← full proposal
# run m4v3_tf_kan        $TF --head-type kan                    # pool-head ablation: KAN
# run m4v3_tf_linear     $TF --head-type linear                 # pool-head ablation: linear floor
run m4v3_tf_2blocks    $TF --fuse-blocks 2                    # deeper fusion (watch for overfit)
run m4v3_tf_detbox     $TF --box-source detector              # detector boxes instead of GT (deployable)

# ============================ SUMMARY ============================
echo ""; echo "================= SUMMARY (test) ================="
printf "%-18s %s\n" "run" "macro-F1 / change-only F1  (+improved-F1)"
for d in "$RUNS"/m4v2_*/ "$RUNS"/m4v3_*/ "$RUNS"/m4_mlp/; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  e=$(grep -h "change-only F1" "logs/$n.eval.log" 2>/dev/null | tail -1 | sed 's/^ *//')
  imp=$(grep -h "^  improved" "logs/$n.eval.log" 2>/dev/null | tail -1 | sed 's/^ *//')
  printf "%-18s %s   [%s]\n" "$n" "${e:-—}" "${imp:-—}"
done
echo ""
echo "Reference = m4_mlp (v1). regiondiff=m4v2_* (cache) | tempfuse=m4v3_* (patch cross-attn)."
echo "Ledger for M5:  python3 phase_4/scripts/4-infer.py --ckpt $RUNS/m4v3_tf_sv2stage/best.pt \\"
echo "  --features-root $FEAT --m3-labels-dir $M3LAB --m4-labels-dir $M4LAB --pairs $PAIRS --split test --out m4_pred.jsonl"
