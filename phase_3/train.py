"""Train M3 (C-KAN). Pick direction with --mode {A,B,C}; everything else from config.

    python phase_3/train.py --mode B --labels-dir data/m3_labels \
        --features-root <feature cache> --epochs 40 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from dataset import M3Dataset, collate
from eval import evaluate
from losses import compute_losses

_LABEL_KEYS = ("region_concepts", "region_chexpert", "image_chexpert", "present_mask")


def to_device(batch: dict, device) -> dict:
    out = dict(batch)
    for k in ("grid", "global", *_LABEL_KEYS):
        out[k] = batch[k].to(device)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M3 C-KAN")
    p.add_argument("--mode", choices=["A", "B", "C"], default=config.HEAD_MODE)
    p.add_argument("--labels-dir", type=Path, default=config.DEFAULT_LABELS_DIR)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--out", type=Path, default=config.DEFAULT_RUNS_DIR)
    p.add_argument("--name", default=None, help="run name (default m3_<mode>)")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--head-type", default=config.HEAD_TYPE)
    p.add_argument("--use-global", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true", help="continue from last.pt (pulled from Drive if --sync-remote)")
    p.add_argument("--sync-remote", default=None, help="rclone remote, e.g. dhint:CHEX-DATA/m3_runs")
    p.add_argument("--sync-every", type=int, default=0, help="also push every N steps (0 = each epoch only)")
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
    config.USE_GLOBAL_TOKEN = args.use_global
    name = args.name or f"m3_{args.mode}"
    run_dir = args.out / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = M3Dataset(args.labels_dir, args.features_root, "train")
    val_ds = M3Dataset(args.labels_dir, args.features_root, "val")
    print(f"train={len(train_ds):,} val={len(val_ds):,}")
    if len(train_ds) == 0:
        raise SystemExit("[ERROR] no training samples (features missing or split empty)")
    feat_dim = train_ds.feat_dim()
    print(f"feat_dim={feat_dim} | mode={args.mode} | head={args.head_type} | global={args.use_global}")

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)

    model = M.build_model(feat_dim, args.mode).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    remote = f"{args.sync_remote.rstrip('/')}/{name}" if args.sync_remote else None

    def push():
        if remote:
            _rclone("copy", str(run_dir), remote, "--transfers", "4", "--quiet")

    best, start_epoch = -1.0, 0
    if args.resume:
        if remote:                              # pull the run dir back from Drive first
            _rclone("copy", remote, str(run_dir), "--quiet")
        last = run_dir / "last.pt"
        if last.exists():
            ck = torch.load(last, map_location=args.device)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); start_epoch = ck["epoch"] + 1
            best = ck.get("best", -1.0)
            print(f"[resume] from epoch {start_epoch} (best so far {best:.4f})")
        else:
            print("[resume] no last.pt found -> fresh start")

    step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {}
        for batch in tl:
            b = to_device(batch, args.device)
            out = model(b["grid"], b["global"], b["present_mask"])
            loss, parts = compute_losses(out, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            step += 1
            if args.sync_every and step % args.sync_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "feat_dim": feat_dim, "mode": args.mode,
                            "head_type": args.head_type, "use_global": args.use_global,
                            "epoch": epoch, "best": best}, run_dir / "last.pt")
                push()
        sched.step()
        n = max(1, len(tl))
        res = evaluate(model, vl, args.device)
        auc = res["image_auc_macro"]
        print(f"epoch {epoch + 1:3}/{args.epochs} | loss {running.get('total', 0)/n:.4f} "
              f"(c {running.get('concept', 0)/n:.3f} r {running.get('region_chex', 0)/n:.3f} "
              f"i {running.get('image_chex', 0)/n:.3f}) | val img-AUC {auc:.4f} "
              f"region {res['region_auc_macro']:.4f}"
              + (f" concept {res.get('concept_auc_macro', float('nan')):.4f}" if "concept_auc_macro" in res else ""))

        if auc > best:
            best = auc
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "feat_dim": feat_dim, "mode": args.mode, "head_type": args.head_type,
                "use_global": args.use_global, "epoch": epoch, "val_auc": auc, "best": best}
        torch.save(ckpt, run_dir / "last.pt")
        if auc >= best:
            torch.save(ckpt, run_dir / "best.pt")
        push()                                  # Drive checkpoint each epoch

    print(f"\n[DONE] best val image-AUC = {best:.4f} -> {run_dir/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
