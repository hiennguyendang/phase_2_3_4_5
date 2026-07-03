"""Train M4 (T-KAN) on cached frozen-M3 region tensors.

    python phase_4/scripts/2-train.py --region-cache data/m3_region_cache --m3-labels-dir data/m3_labels \
        --m4-labels-dir data/m4_labels --pairs data/m4_labels/m3_pairs.jsonl --device cuda

Drive-resumable (same pattern as phase_3): --resume + --sync-remote.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_4/src

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
    p.add_argument("--head-type", default=config.HEAD_TYPE, choices=["mlp", "linear", "kan"])
    p.add_argument("--input-mode", default=config.INPUT_MODE,
                   choices=["full", "concat", "diff", "logits", "feat"])
    p.add_argument("--hidden", type=int, default=config.HEAD_HIDDEN)
    p.add_argument("--dropout", type=float, default=config.HEAD_DROPOUT)
    p.add_argument("--loss", default=config.LOSS_TYPE, choices=["ce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=config.FOCAL_GAMMA)
    p.add_argument("--head-mode", default=config.HEAD_MODE, choices=["flat", "twostage"],
                   help="(c) flat 3-way softmax, or factorized change x direction")
    p.add_argument("--same-view", action="store_true",
                   help="(b) keep only same-ViewPosition prior pairs (eval inherits this from the ckpt)")
    p.add_argument("--select-metric", default=config.SELECT_METRIC, choices=["macro", "change"],
                   help="(a) which val metric selects best.pt")
    p.add_argument("--patience", type=int, default=config.PATIENCE,
                   help="(a) early-stop after N evals with no val improvement (0 = off)")
    p.add_argument("--no-class-weight", action="store_true", help="disable inverse-freq class weighting")
    p.add_argument("--no-require-prior", action="store_true",
                   help="supervise cells present in CURRENT only (default: require prior present too)")
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
    # config toggles read globally downstream (heads defaults, dataset masking) — set before use.
    config.HEAD_TYPE = args.head_type
    config.INPUT_MODE = args.input_mode
    config.HEAD_MODE = args.head_mode
    config.REQUIRE_PRIOR_PRESENT = not args.no_require_prior
    run_dir = args.out / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    cache = RegionCache(args.region_cache)
    train_ds = M4Dataset(cache, args.m3_labels_dir, args.m4_labels_dir, args.pairs, "train",
                         augment=config.AUGMENT_TIME_FLIP and not args.no_augment,
                         same_view_only=args.same_view)
    val_ds = M4Dataset(cache, args.m3_labels_dir, args.m4_labels_dir, args.pairs, "val",
                       same_view_only=args.same_view)  # never augmented; same view filter as train
    n_base = len(train_ds.rows)
    print(f"train={len(train_ds):,} (base {n_base:,}, augment={train_ds.augment}, "
          f"same_view={args.same_view}) val={len(val_ds):,} | skipped(train)={train_ds.skipped}")
    if len(train_ds) == 0:
        raise SystemExit("[ERROR] no training pairs (cache/prior/labels missing?)")
    feat_dim = train_ds.feat_dim
    print(f"feat_dim={feat_dim} | input={args.input_mode} | "
          f"region_in_dim={M.region_in_dim(feat_dim, args.input_mode)} | head={args.head_type}/{args.head_mode} | "
          f"loss={args.loss} | require_prior={config.REQUIRE_PRIOR_PRESENT} | "
          f"select={args.select_metric} patience={args.patience}")

    weight = None
    if config.USE_CLASS_WEIGHT and not args.no_class_weight:
        counts = train_ds.class_counts()             # reflects flips when augmenting
        weight = class_weight_from_counts(counts).to(args.device)
        print("[class_weight]", {n: f"{int(c)}->{float(w):.2f}"
                                  for n, c, w in zip(C.PROG_NAMES, counts, weight)})

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)

    model = M.build_model(feat_dim, args.head_type, args.input_mode, args.hidden, args.dropout,
                          args.head_mode).to(args.device)
    # model-rebuild cfg — persisted in every ckpt so eval.py/infer.py reconstruct the exact head + data gates.
    mcfg = {"feat_dim": feat_dim, "head_type": args.head_type, "input_mode": args.input_mode,
            "hidden": args.hidden, "dropout": args.dropout, "loss": args.loss,
            "head_mode": args.head_mode, "same_view": args.same_view,
            "require_prior_present": config.REQUIRE_PRIOR_PRESENT}
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

    step, stale = 0, 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss, run_n = 0.0, 0
        for batch in tl:
            b = _move(batch, args.device)
            logits = model(b["feat_curr"], b["logit_curr"], b["feat_prior"], b["logit_prior"])
            loss, nval = progression_loss(logits, b["progression"], b["region_mask"], weight,
                                          loss_type=args.loss, gamma=args.focal_gamma)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss) * max(nval, 1); run_n += max(nval, 1)
            step += 1
            if args.sync_every and step % args.sync_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "epoch": epoch, "best": best, **mcfg},
                           run_dir / "last.pt")
                push()
        sched.step()
        res = evaluate(model, vl, args.device)
        f1 = res["prog_f1_macro"]
        # (a) select best.pt by the chosen val metric (macro = old behaviour, change = headline)
        sel = res["change_f1_macro"] if args.select_metric == "change" else f1
        print(f"epoch {epoch + 1:3}/{args.epochs} | loss {run_loss/max(run_n,1):.4f} | "
              f"val prog-F1 {f1:.4f} change-F1 {res['change_f1_macro']:.4f} "
              f"(per {{ {', '.join(f'{k}:{v:.2f}' for k,v in res['per_class'].items())} }})")

        is_best = sel > best
        if is_best:
            best = sel; stale = 0
        else:
            stale += 1
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "val_f1": f1, "val_change_f1": res["change_f1_macro"],
                "select_metric": args.select_metric, "best": best, **mcfg}
        torch.save(ckpt, run_dir / "last.pt")
        if is_best:
            torch.save(ckpt, run_dir / "best.pt")
        push()
        if args.patience and stale >= args.patience:      # (a) early stop
            print(f"[early-stop] no val-{args.select_metric} gain in {args.patience} evals "
                  f"(best {best:.4f})")
            break

    print(f"\n[DONE] best val {args.select_metric}-F1 = {best:.4f} -> {run_dir/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
