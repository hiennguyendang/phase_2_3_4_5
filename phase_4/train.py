"""Train M4 (T-KAN) on cached frozen-M3 region tensors.

    python phase_4/train.py --region-cache data/m3_region_cache --m3-labels-dir data/m3_labels \
        --m4-labels-dir data/m4_labels --pairs data/m3_pairs.jsonl --device cuda

Drive-resumable (same pattern as phase_3): --resume + --sync-remote.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import M4Dataset, RegionCache, collate
from eval import evaluate
from losses import class_weight_from_counts, progression_loss


def _move(batch: dict, device) -> dict:
    out = dict(batch)
    for k in ("feat_curr", "logit_curr", "feat_prior", "logit_prior", "region_mask", "progression"):
        out[k] = batch[k].to(device)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M4 T-KAN")
    p.add_argument("--region-cache", type=Path, default=config.DEFAULT_REGION_CACHE)
    p.add_argument("--m3-labels-dir", type=Path, default=config.DEFAULT_M3_LABELS_DIR)
    p.add_argument("--m4-labels-dir", type=Path, default=config.DEFAULT_M4_LABELS_DIR)
    p.add_argument("--pairs", type=Path, default=config.DEFAULT_PAIRS_PATH)
    p.add_argument("--out", type=Path, default=config.DEFAULT_RUNS_DIR)
    p.add_argument("--name", default="m4")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--head-type", default=config.HEAD_TYPE)
    p.add_argument("--no-augment", action="store_true", help="disable train-time time-flip augmentation")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sync-remote", default=None, help="rclone remote, e.g. dhint:CHEX-DATA/m4_runs")
    p.add_argument("--sync-every", type=int, default=0)
    return p.parse_args()


def _rclone(*a) -> None:
    import shutil
    import subprocess
    if not shutil.which("rclone"):
        print("[sync] rclone not on PATH; skipping"); return
    try:
        subprocess.run(["rclone", *a], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        print(f"[sync] rclone failed (continuing): {e}")


def main() -> int:
    import model as M
    args = parse_args()
    config.HEAD_TYPE = args.head_type
    run_dir = args.out / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    cache = RegionCache(args.region_cache)
    train_ds = M4Dataset(cache, args.m3_labels_dir, args.m4_labels_dir, args.pairs, "train",
                         augment=config.AUGMENT_TIME_FLIP and not args.no_augment)
    val_ds = M4Dataset(cache, args.m3_labels_dir, args.m4_labels_dir, args.pairs, "val")  # never augmented
    n_base = len(train_ds.rows)
    print(f"train={len(train_ds):,} (base {n_base:,}, augment={train_ds.augment}) "
          f"val={len(val_ds):,} | skipped(train)={train_ds.skipped}")
    if len(train_ds) == 0:
        raise SystemExit("[ERROR] no training pairs (cache/prior/labels missing?)")
    feat_dim = train_ds.feat_dim
    print(f"feat_dim={feat_dim} | region_in_dim={M.region_in_dim(feat_dim)} | head={args.head_type}")

    weight = None
    if config.USE_CLASS_WEIGHT:
        counts = train_ds.class_counts()             # reflects flips when augmenting
        weight = class_weight_from_counts(counts).to(args.device)
        print("[class_weight]", {n: f"{int(c)}->{float(w):.2f}"
                                  for n, c, w in zip(C.PROG_NAMES, counts, weight)})

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)

    model = M.build_model(feat_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    remote = f"{args.sync_remote.rstrip('/')}/{args.name}" if args.sync_remote else None

    def push():
        if remote:
            _rclone("copy", str(run_dir), remote, "--transfers", "4", "--quiet")

    best, start_epoch = -1.0, 0
    if args.resume:
        if remote:
            _rclone("copy", remote, str(run_dir), "--quiet")
        last = run_dir / "last.pt"
        if last.exists():
            ck = torch.load(last, map_location=args.device)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); start_epoch = ck["epoch"] + 1
            best = ck.get("best", -1.0)
            print(f"[resume] from epoch {start_epoch} (best {best:.4f})")
        else:
            print("[resume] no last.pt -> fresh start")

    step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss, run_n = 0.0, 0
        for batch in tl:
            b = _move(batch, args.device)
            logits = model(b["feat_curr"], b["logit_curr"], b["feat_prior"], b["logit_prior"])
            loss, nval = progression_loss(logits, b["progression"], b["region_mask"], weight)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss) * max(nval, 1); run_n += max(nval, 1)
            step += 1
            if args.sync_every and step % args.sync_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "feat_dim": feat_dim,
                            "head_type": args.head_type, "epoch": epoch, "best": best}, run_dir / "last.pt")
                push()
        sched.step()
        res = evaluate(model, vl, args.device)
        f1 = res["prog_f1_macro"]
        print(f"epoch {epoch + 1:3}/{args.epochs} | loss {run_loss/max(run_n,1):.4f} | "
              f"val prog-F1 {f1:.4f} change-F1 {res['change_f1_macro']:.4f} "
              f"(per {{ {', '.join(f'{k}:{v:.2f}' for k,v in res['per_class'].items())} }})")

        is_best = f1 > best
        if is_best:
            best = f1
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "feat_dim": feat_dim, "head_type": args.head_type, "epoch": epoch,
                "val_f1": f1, "best": best}
        torch.save(ckpt, run_dir / "last.pt")
        if is_best:
            torch.save(ckpt, run_dir / "best.pt")
        push()

    print(f"\n[DONE] best val prog-F1 = {best:.4f} -> {run_dir/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
