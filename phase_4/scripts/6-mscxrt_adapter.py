"""Optional MS-CXR-T adapter fine-tune for M4.

This is a development experiment, not the final external audit: MS-CXR-T has image-level labels, so
we train on a deterministic subject-hash train split and select on the hash-val split.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_4/src

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import collate, move_batch
from eval import build_from_ckpt, multiclass_f1
from losses import class_weight_from_counts
from mscxrt import FINDINGS, MSCXRTDataset, aggregate_mscxrt_probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a small M4 adapter on MS-CXR-T")
    p.add_argument("--base-ckpt", type=Path, required=True)
    p.add_argument("--csv", type=Path,
                   default=Path("data/MS_CXR_T_temporal_image_classification_v1.0.0.csv"))
    p.add_argument("--region-cache", type=Path, default=config.DEFAULT_REGION_CACHE)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--m3-labels-dir", type=Path, default=config.DEFAULT_M3_LABELS_DIR)
    p.add_argument("--out", type=Path, default=config.DEFAULT_RUNS_DIR)
    p.add_argument("--name", default="m4_mscxrt_adapter")
    p.add_argument("--train-scope", default="head", choices=["head", "pool-head", "all"],
                   help="which parameters to unfreeze")
    p.add_argument("--agg", default="mean", choices=["mean", "max", "lse"],
                   help="differentiable region-to-image aggregation used by the adapter loss")
    p.add_argument("--select-metric", default="change", choices=["macro", "change"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-class-weight", action="store_true")
    return p.parse_args()


def _set_train_scope(model, scope: str) -> int:
    for p in model.parameters():
        p.requires_grad = scope == "all"
    if scope in ("head", "pool-head"):
        for p in model.head.parameters():
            p.requires_grad = True
    if scope == "pool-head" and hasattr(model, "pool"):
        for p in model.pool.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _loss(logits, target, region_mask, agg, weight):
    probs = aggregate_mscxrt_probs(logits, region_mask, agg).clamp_min(1e-7)
    valid = target != C.UNKNOWN
    if valid.sum() == 0:
        return logits.sum() * 0.0, 0
    return F.nll_loss(probs.log()[valid], target[valid], weight=weight), int(valid.sum())


@torch.no_grad()
def _eval(model, loader, device, agg: str) -> dict:
    model.eval()
    preds, tgts = [], []
    for b in loader:
        bb = move_batch(b, device)
        probs = aggregate_mscxrt_probs(model(bb), bb["region_mask"], agg)
        preds.append(probs.argmax(-1).cpu())
        tgts.append(b["target_mscxrt"])
    pred = torch.cat(preds).numpy() if preds else torch.zeros((0, len(FINDINGS)), dtype=torch.long).numpy()
    tgt = torch.cat(tgts).numpy() if tgts else torch.zeros((0, len(FINDINGS)), dtype=torch.long).numpy()
    valid = tgt != C.UNKNOWN
    if not valid.any():
        return {"macro": float("nan"), "change": float("nan"), "per_class": {}}
    macro, per, change = multiclass_f1(pred[valid], tgt[valid])
    return {"macro": macro, "change": change, "per_class": per}


def main() -> int:
    args = parse_args()
    ck = torch.load(args.base_ckpt, map_location=args.device)
    model = build_from_ckpt(ck, args.device)
    arch = ck.get("arch", "regiondiff")
    tf_input = ck.get("tempfuse_input_mode", "feat")
    ds_kw = dict(arch=arch, m3_labels_dir=args.m3_labels_dir, region_cache=args.region_cache,
                 features_root=args.features_root, box_source=ck.get("box_source", config.BOX_SOURCE),
                 tempfuse_input_mode=tf_input)
    train_ds = MSCXRTDataset(args.csv, split="train", **ds_kw)
    val_ds = MSCXRTDataset(args.csv, split="val", **ds_kw)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"[ERROR] empty MS-CXR-T split train={len(train_ds)} val={len(val_ds)}")
    trainable = _set_train_scope(model, args.train_scope)
    print(f"arch={arch} tf_input={tf_input} train={len(train_ds):,} val={len(val_ds):,} "
          f"scope={args.train_scope} trainable={trainable:,} agg={args.agg}")
    print(f"skipped(train)={train_ds.skipped} skipped(val)={val_ds.skipped}")

    weight = None
    if not args.no_class_weight:
        weight = class_weight_from_counts(train_ds.class_counts()).to(args.device)
        print("[class_weight]", {n: f"{float(w):.2f}" for n, w in zip(C.PROG_NAMES, weight)})

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, drop_last=False)
    vl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=config.WEIGHT_DECAY)
    run_dir = args.out / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        run_loss, run_n = 0.0, 0
        for b in tl:
            bb = move_batch(b, args.device)
            loss, n = _loss(model(bb), bb["target_mscxrt"], bb["region_mask"], args.agg, weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss) * max(n, 1)
            run_n += max(n, 1)
        res = _eval(model, vl, args.device, args.agg)
        sel = res[args.select_metric]
        print(f"epoch {epoch + 1:3}/{args.epochs} | loss {run_loss/max(run_n,1):.4f} | "
              f"val macro {res['macro']:.4f} change {res['change']:.4f} "
              f"(per {{ {', '.join(f'{k}:{v:.2f}' for k, v in res['per_class'].items())} }})")
        save = {**ck, "model": model.state_dict(), "epoch": epoch, "best": max(best, sel),
                "mscxrt_adapter": True, "mscxrt_base_ckpt": str(args.base_ckpt),
                "mscxrt_train_scope": args.train_scope, "mscxrt_agg": args.agg,
                "mscxrt_val_macro": res["macro"], "mscxrt_val_change": res["change"]}
        torch.save(save, run_dir / "last.pt")
        if sel > best:
            best = sel
            torch.save(save, run_dir / "best.pt")
    print(f"[DONE] best val {args.select_metric}={best:.4f} -> {run_dir/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
