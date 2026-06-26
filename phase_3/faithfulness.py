"""Faithfulness tests for M3 — spec 3.4. These pick which direction VERA can claim "why" for,
using *faithfulness numbers* (not accuracy) as the deciding rule.

  1. go/no-go concept-from-image  — concept macro-F1 on silver split. Too low -> "why-by-concept"
     is off the table for BOTH B and C; ship direction A (where-faithful) and demote concepts.
  2. concept-intervention test (mode B) — force each concept on/off; the disease it feeds must move
     in the right direction. No movement -> the bottleneck is fake -> concepts not faithful.
  3. leakage test (mode C) — zero / randomize the concept channel into the disease head while keeping
     the image channel. If disease F1 barely drops, concepts are decorative (CBM leakage) -> the
     model decides *around* them -> NOT allowed to be presented as "why".

    python phase_3/faithfulness.py --ckpt <run>/best.pt --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import M3Dataset, collate
from eval import _f1_table, evaluate


# ---- shared: recompute region disease logits under a concept override --------
def _region_disease(model, grid, glob, present, boxes, concept_override=None):
    """Reproduce model's region_disease_logits, optionally replacing the concept tensor.
    concept_override(concept_logits)->tensor lets a test intervene on the bottleneck."""
    pooled, _ = model.pool(grid, glob, boxes)
    feats29 = model.neck(pooled[:, :C.NUM_REGIONS, :])
    if model.mode == "A":
        return model.disease_head(feats29), None
    concept_logits = model.concept_head(feats29)              # [B,29,69]
    if model.mode == "B":
        act = torch.sigmoid(concept_logits)
        if concept_override is not None:
            act = concept_override(act)
        return model.disease_head(act), concept_logits
    # mode C / hybrid
    cl = concept_logits if concept_override is None else concept_override(concept_logits)
    feat_in = model.feat_leak(feats29)                        # eval mode -> dropout off
    return model.disease_head(torch.cat([feat_in, cl], dim=-1)), concept_logits


# ---- test 2: concept-intervention (mode B) -----------------------------------
@torch.no_grad()
def intervention_test(model, loader, device, max_batches=20):
    """Force concept c -> 1 vs 0; measure signed change on the disease it feeds.
    A real bottleneck: turning a concept ON raises its disease (delta > 0)."""
    pairs = [(ci, C.CONCEPT_TO_CHEX[ci]) for ci in range(C.NUM_CONCEPTS) if C.CONCEPT_TO_CHEX[ci] >= 0]
    sums = np.zeros(len(pairs)); cnt = 0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        grid, glob = b["grid"].to(device), b["global"].to(device)
        present, boxes = b["present_mask"].to(device), b["boxes"].to(device)
        pooled, _ = model.pool(grid, glob, boxes)
        feats29 = model.neck(pooled[:, :C.NUM_REGIONS, :])
        act = torch.sigmoid(model.concept_head(feats29))     # [B,29,69]
        m = present.bool().unsqueeze(-1)                      # [B,29,1]
        for k, (ci, di) in enumerate(pairs):
            on, off = act.clone(), act.clone()
            on[..., ci] = 1.0; off[..., ci] = 0.0
            d_on = torch.sigmoid(model.disease_head(on))[..., di]
            d_off = torch.sigmoid(model.disease_head(off))[..., di]
            diff = (d_on - d_off).unsqueeze(-1)[m]            # present regions only
            sums[k] += float(diff.sum())
        cnt += int(m.sum())
    mean_delta = sums / max(cnt, 1)
    frac_correct = float((mean_delta > 0).mean())
    return {"median_delta": float(np.median(mean_delta)),
            "mean_delta": float(mean_delta.mean()),
            "frac_correct_direction": frac_correct,
            "n_mapped_concepts": len(pairs)}


# ---- test 3: leakage (mode C) ------------------------------------------------
@torch.no_grad()
def leakage_test(model, loader, device):
    """Region disease F1 with the concept channel: (a) intact, (b) zeroed, (c) randomized.
    Small drop from intact->zeroed => concepts decorative => leakage."""
    def collect(override):
        P, T, M = [], [], []
        for b in loader:
            grid, glob = b["grid"].to(device), b["global"].to(device)
            present, boxes = b["present_mask"].to(device), b["boxes"].to(device)
            rd, _ = _region_disease(model, grid, glob, present, boxes, override)
            P.append(torch.sigmoid(rd).cpu().numpy())
            T.append(b["region_chexpert"].numpy()); M.append(b["present_mask"].numpy())
        P, T, M = np.concatenate(P), np.concatenate(T), np.concatenate(M).astype(bool)
        f1, _ = _f1_table(P[M], T[M], C.CHEX_NAMES)
        return f1

    intact = collect(None)
    zeroed = collect(lambda cl: torch.zeros_like(cl))
    randomized = collect(lambda cl: cl[torch.randperm(cl.shape[0])])  # shuffle concepts across batch
    return {"region_f1_intact": intact, "region_f1_zeroed": zeroed,
            "region_f1_randomized": randomized,
            "drop_zeroed": intact - zeroed, "drop_randomized": intact - randomized}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3 faithfulness tests (spec 3.4)")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, default=config.DEFAULT_LABELS_DIR)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--split", default="val")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--device", default="cuda")
    p.add_argument("--concept-f1-thresh", type=float, default=0.30,
                   help="go/no-go bar for concept-from-image macro-F1 (judgment call)")
    p.add_argument("--max-batches", type=int, default=20, help="cap for the intervention sweep")
    return p.parse_args()


def main() -> int:
    import model as M
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    config.USE_GLOBAL_TOKEN = ck.get("use_global", config.USE_GLOBAL_TOKEN)
    m = M.build_model(ck["feat_dim"], ck["mode"]).to(args.device).eval()
    m.load_state_dict(ck["model"])
    mode = ck["mode"]

    ds = M3Dataset(args.labels_dir, args.features_root, args.split)
    loader = DataLoader(ds, batch_size=args.batch, collate_fn=collate)
    print(f"[faithfulness] mode={mode} split={args.split} n={len(ds):,}\n")

    if mode == "A":
        print("Mode A (Direct): no concept channel -> 'why-by-concept' N/A. VERA = where-faithful.")
        print("  Faithfulness here = region grounding (softmax_r, alpha), reported by infer.py.")
        return 0

    # 1) go/no-go concept-from-image
    res = evaluate(m, loader, args.device)
    cf1 = res.get("concept_f1_macro", float("nan"))
    go = cf1 >= args.concept_f1_thresh
    print(f"[1] go/no-go concept-from-image: macro-F1 = {cf1:.4f}  (bar {args.concept_f1_thresh}) "
          f"-> {'PASS' if go else 'FAIL — demote concepts to ablation'}")

    if mode == "B":
        iv = intervention_test(m, loader, args.device, args.max_batches)
        ok = iv["frac_correct_direction"] >= 0.7 and iv["median_delta"] > 0.01
        print(f"[2] concept-intervention (B): median Δ={iv['median_delta']:+.4f} "
              f"mean Δ={iv['mean_delta']:+.4f} correct-direction={iv['frac_correct_direction']:.0%} "
              f"over {iv['n_mapped_concepts']} concepts -> {'PASS (bottleneck real)' if ok else 'FAIL (fake bottleneck)'}")
        print(f"\n=> 'why'-faithful claim allowed: {bool(go and ok)}")
    else:  # C
        lk = leakage_test(m, loader, args.device)
        leak = lk["drop_zeroed"] < 0.02
        print(f"[3] leakage (C): region-F1 intact={lk['region_f1_intact']:.4f} "
              f"zeroed={lk['region_f1_zeroed']:.4f} (drop {lk['drop_zeroed']:+.4f}) "
              f"randomized={lk['region_f1_randomized']:.4f} (drop {lk['drop_randomized']:+.4f})")
        print(f"    {'LEAKAGE: concepts decorative -> do NOT present as why' if leak else 'concepts drive disease (low leakage)'}")
        print(f"\n=> 'why'-faithful claim allowed: {bool(go and not leak)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
