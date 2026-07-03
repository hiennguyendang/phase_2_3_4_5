#!/usr/bin/env bash
# M4 (T-KAN) experiment grid — runs SEQUENTIALLY (one job at a time: MIG slice shares SMs).
# Idempotent: a run whose best.pt already exists is skipped (delete it to re-run). The frozen-M3
# region cache is built ONCE up front (all ablations consume the SAME B-faithful cache). Each run
# trains + evals (test); logs go to logs/<name>.*.log; a summary prints at the end.
#
#   bash phase_4/run_experiments.sh
#   EP=40 BATCH=64 W=8 bash phase_4/run_experiments.sh            # override defaults
#   nohup bash phase_4/run_experiments.sh > logs/m4_grid.log 2>&1 &   # background, survive logout
#
# Read CHANGE-ONLY F1 as the headline number (macro-F1 alongside). accuracy ~= "stable" is a RED FLAG.
# ⚠ These numbers are on the SILVER (NLP comparison_cues) test set — provisional until the
#   human-annotated temporal eval set (MS-CXR-T) lands (README note B2).

cd "$(dirname "$0")/.." || exit 1          # repo root
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ---- paths (override via env if your server layout differs) ------------------
M3_CKPT=${M3_CKPT:-data/m3_B_faithful/best.pt}     # shipping frozen M3 (mode B-faithful)
FEAT=${FEAT:-data/features}                        # BioViL-T .pt feature cache
M3LAB=${M3LAB:-data/m3_labels}
M4LAB=${M4LAB:-data/m4_labels}
PAIRS=${PAIRS:-data/m4_labels/m3_pairs.jsonl}
CACHE=${CACHE:-data/m3_region_cache}
RUNS=${RUNS:-data/run}
EP=${EP:-40}; BATCH=${BATCH:-64}; W=${W:-8}
mkdir -p logs

# ---- bridge (GPU, once): freeze B-faithful M3 -> region_feat‖logit for every image ----
if ls "$CACHE"/*.npy >/dev/null 2>&1; then
  echo "[skip bridge] region cache already populated at $CACHE"
else
  echo "===== BRIDGE: precompute region cache from $M3_CKPT ====="
  python3 phase_3/scripts/8-precompute_regions.py --ckpt "$M3_CKPT" \
    --labels-dir "$M3LAB" --features-root "$FEAT" --out-dir "$CACHE" \
    2>&1 | tee logs/m4_bridge.log
fi

run () {                                    # run <name> <train flags...>
  name="$1"; shift
  ck="$RUNS/$name/best.pt"
  echo ""; echo "===== $name $* ====="
  if [ -f "$ck" ]; then
    echo "  [skip train] $ck exists"
  else
    python3 phase_4/train.py --region-cache "$CACHE" --m3-labels-dir "$M3LAB" \
      --m4-labels-dir "$M4LAB" --pairs "$PAIRS" --out "$RUNS" \
      --epochs "$EP" --batch "$BATCH" --workers "$W" --device cuda \
      "$@" --name "$name" 2>&1 | tee "logs/$name.train.log"
  fi
  python3 phase_4/eval.py --ckpt "$ck" --region-cache "$CACHE" --m3-labels-dir "$M3LAB" \
    --m4-labels-dir "$M4LAB" --pairs "$PAIRS" --split test --device cuda \
    2>&1 | tee "logs/$name.eval.log"
}

# ============================ THE GRID ============================
# Baseline = full input, mlp head, class-weighted CE, time-flip augment, require-prior-present.

# ---- Tier 0: head architecture (the "T-KAN" claim) — everything else at baseline ----
run m4_mlp        --head-type mlp                       # baseline
run m4_kan        --head-type kan                       # FastKAN (RBF-spline)
run m4_linear     --head-type linear                    # linear probe = the floor

# ---- Tier 1: input signal — where does the progression signal live? ----
run m4_in_concat  --input-mode concat                   # keep both sides, drop explicit difference
run m4_in_diff    --input-mode diff                     # pure Siamese: [c-p ; lc-lp]
run m4_in_logits  --input-mode logits                   # M3 disease logits ONLY (cheap-signal probe)
run m4_in_feat    --input-mode feat                     # region features ONLY (no logits)

# ---- Tier 2: class imbalance — fight the "predict stable only" collapse ----
run m4_noweight   --no-class-weight                     # ablate inverse-freq weighting
run m4_focal      --loss focal --focal-gamma 2.0        # focal instead of weighted CE
run m4_noaug      --no-augment                          # ablate time-flip augmentation

# ---- Tier 3: supervision masking ----
run m4_currpresent --no-require-prior                   # supervise cells present in CURRENT only

# ---- Tier 4 (OPTIONAL): M3 logit-source — needs OTHER M3 ckpts, rebuild cache per source ----
# Baseline feeds B-faithful logits. To test A/C logits into the SAME M4, build a separate cache
# from each ckpt (CACHE=data/m3_region_cache_A M3_CKPT=data/run/m3_A/best.pt bash ...), then:
#   CACHE=data/m3_region_cache_A run m4_srcA --head-type mlp
#   CACHE=data/m3_region_cache_C run m4_srcC --head-type mlp
# (Left commented: you only have the B-faithful ckpt local; the M2-prior branch is out of scope.)

# ============================ SUMMARY ============================
echo ""; echo "================= SUMMARY (test) ================="
printf "%-16s %s\n" "run" "macro-F1 / change-only F1"
for d in "$RUNS"/m4_*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  e=$(grep -h "change-only F1" "logs/$n.eval.log" 2>/dev/null | tail -1 | sed 's/^ *//')
  printf "%-16s %s\n" "$n" "${e:-—}"
done
echo ""
echo "Change-ledger for M5 (best run): "
echo "  python3 phase_4/infer.py --ckpt $RUNS/m4_kan/best.pt --region-cache $CACHE \\"
echo "    --m3-labels-dir $M3LAB --m4-labels-dir $M4LAB --pairs $PAIRS --split test --out m4_pred.jsonl"
