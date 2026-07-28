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
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import collate, make_dataset, move_batch
from eval import evaluate
from losses import class_weight_from_counts, flip_consistency_loss, progression_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M4 T-KAN")
    p.add_argument("--concept-cache", type=Path, default=getattr(config, "DEFAULT_CONCEPT_CACHE", None),
                   help="ftcb arch: [29,69] M3 concept-activation cache (from 8-precompute --concept-cache-out)")
    p.add_argument("--arch", default=config.ARCH, choices=["regiondiff", "tempfuse", "ftcb"],
                   help="regiondiff = frozen-M3 region cache; tempfuse = M1 patch grids + cross-attn + M4 pool")
    p.add_argument("--region-cache", type=Path, default=config.DEFAULT_REGION_CACHE)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT,
                   help="M1 patch grids (tempfuse only)")
    p.add_argument("--box-source", default=config.BOX_SOURCE, choices=["gt", "detector"],
                   help="bbox source for the tempfuse pool masks")
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
    p.add_argument("--loss", default=config.LOSS_TYPE, choices=["ce", "focal", "cdw"])
    p.add_argument("--focal-gamma", type=float, default=config.FOCAL_GAMMA)
    p.add_argument("--cdw-alpha", type=float, default=config.CDW_ALPHA,
                   help="cdw exponent: |i-c|^alpha ordinal distance penalty (Polat 2022/24); used by --loss cdw and --cdw-weight")
    p.add_argument("--cdw-weight", type=float, default=0.0,
                   help="hybrid: add lambda*CDW-CE on top of the base loss (safety-sensitivity frontier); 0 = off")
    p.add_argument("--label-smoothing", type=float, default=config.LABEL_SMOOTHING,
                   help="CE-only smoothing for noisy silver comparison_cues")
    p.add_argument("--opposite-penalty-weight", type=float, default=0.0,
                   help="extra penalty for assigning improved mass to worsened, or worsened mass to improved")
    p.add_argument("--distance-penalty-weight", type=float, default=0.0,
                   help="expected ordinal distance penalty on improved < stable < worsened")
    p.add_argument("--flip-consistency-weight", type=float, default=0.0,
                   help="symmetric KL weight for (curr,prior) vs flipped (prior,curr)")
    p.add_argument("--flip-consistency-temperature", type=float, default=1.0,
                   help="temperature for flip-consistency KL")
    p.add_argument("--head-mode", default=config.HEAD_MODE, choices=["flat", "twostage"],
                   help="(c) flat 3-way softmax, or factorized change x direction")
    p.add_argument("--fuse-blocks", type=int, default=config.FUSE_BLOCKS,
                   help="tempfuse: #cross-attn blocks (keep shallow — overfits fast)")
    p.add_argument("--tempfuse-input-mode", default=config.TEMPFUSE_INPUT_MODE,
                   choices=["feat", "feat_logits"],
                   help="tempfuse head input: fused region feature only, or + M3 curr/prior/delta logits")
    p.add_argument("--same-view", action="store_true",
                   help="(b) keep only same-ViewPosition prior pairs (eval inherits this from the ckpt)")
    p.add_argument("--curriculum-same-view-epochs", type=int, default=0,
                   help="train first N epochs on same-view pairs, then continue on all pairs")
    p.add_argument("--select-metric", default=config.SELECT_METRIC, choices=["macro", "prog", "change", "acc"],
                   help="(a) which val metric selects best.pt; all runs also save best_acc/prog/change.pt")
    p.add_argument("--patience", type=int, default=config.PATIENCE,
                   help="(a) early-stop after N evals with no val improvement (0 = off)")
    p.add_argument("--no-class-weight", action="store_true", help="disable inverse-freq class weighting")
    p.add_argument("--no-require-prior", action="store_true",
                   help="supervise cells present in CURRENT only (default: require prior present too)")
    p.add_argument("--no-augment", action="store_true", help="disable train-time time-flip augmentation")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
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


def _make_flipped_batch(batch: dict) -> dict:
    """Create the reversed temporal pair in-memory for flip-consistency regularization."""
    fb = dict(batch)
    if "concept_curr" in batch:
        fb["concept_curr"], fb["concept_prior"] = batch["concept_prior"], batch["concept_curr"]
        if "logit_curr" in batch:
            fb["logit_curr"], fb["logit_prior"] = batch["logit_prior"], batch["logit_curr"]
    if "feat_curr" in batch:
        fb["feat_curr"], fb["feat_prior"] = batch["feat_prior"], batch["feat_curr"]
        fb["logit_curr"], fb["logit_prior"] = batch["logit_prior"], batch["logit_curr"]
    if "patch_curr" in batch:
        fb["patch_curr"], fb["patch_prior"] = batch["patch_prior"], batch["patch_curr"]
        if "box_prior" in batch:
            fb["box_curr"] = batch["box_prior"]
        if "logit_curr" in batch:
            fb["logit_curr"], fb["logit_prior"] = batch["logit_prior"], batch["logit_curr"]
    if "region_mask_flip" in batch:
        fb["region_mask"] = batch["region_mask_flip"]
    return fb


def main() -> int:
    import model as M
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # config toggles read globally downstream (heads defaults, dataset masking) — set before use.
    config.ARCH = args.arch
    config.HEAD_TYPE = args.head_type
    config.INPUT_MODE = args.input_mode
    config.HEAD_MODE = args.head_mode
    config.FUSE_BLOCKS = args.fuse_blocks
    config.TEMPFUSE_INPUT_MODE = args.tempfuse_input_mode
    config.REQUIRE_PRIOR_PRESENT = not args.no_require_prior
    run_dir = args.out / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    ds_kw = dict(region_cache=args.region_cache, features_root=args.features_root,
                 same_view_only=args.same_view, box_source=args.box_source,
                 tempfuse_input_mode=args.tempfuse_input_mode, concept_cache=args.concept_cache)
    train_ds = make_dataset(args.arch, args.m3_labels_dir, args.m4_labels_dir, args.pairs, "train",
                            augment=config.AUGMENT_TIME_FLIP and not args.no_augment, **ds_kw)
    curriculum_ds = None
    if args.curriculum_same_view_epochs > 0 and not args.same_view:
        cur_kw = dict(ds_kw)
        cur_kw["same_view_only"] = True
        curriculum_ds = make_dataset(args.arch, args.m3_labels_dir, args.m4_labels_dir, args.pairs,
                                     "train",
                                     augment=config.AUGMENT_TIME_FLIP and not args.no_augment,
                                     **cur_kw)
    val_ds = make_dataset(args.arch, args.m3_labels_dir, args.m4_labels_dir, args.pairs, "val", **ds_kw)
    n_base = len(train_ds.rows)
    print(f"arch={args.arch} train={len(train_ds):,} (base {n_base:,}, augment={train_ds.augment}, "
          f"same_view={args.same_view}) val={len(val_ds):,} | skipped(train)={train_ds.skipped}")
    if len(train_ds) == 0:
        raise SystemExit("[ERROR] no training pairs (cache/features/prior/labels missing?)")
    if curriculum_ds is not None:
        print(f"curriculum_same_view_epochs={args.curriculum_same_view_epochs} "
              f"curriculum_train={len(curriculum_ds):,} | skipped(curriculum)={curriculum_ds.skipped}")
    feat_dim = train_ds.feat_dim
    extra = (f"region_in_dim={M.region_in_dim(feat_dim, args.input_mode)}" if args.arch == "regiondiff"
             else f"fuse_blocks={config.FUSE_BLOCKS} box={args.box_source} "
                  f"tf_input={args.tempfuse_input_mode} "
                  f"tempfuse_in_dim={M.tempfuse_in_dim(feat_dim, args.tempfuse_input_mode)}")
    print(f"feat_dim={feat_dim} | input={args.input_mode} | {extra} | head={args.head_type}/{args.head_mode} | "
          f"loss={args.loss}{f' alpha={args.cdw_alpha:g}' if args.loss == 'cdw' else ''}"
          f"{f' +cdw{args.cdw_weight:g}@a{args.cdw_alpha:g}' if args.cdw_weight > 0 else ''} "
          f"label_smoothing={args.label_smoothing:g} | "
          f"opp_pen={args.opposite_penalty_weight:g} dist_pen={args.distance_penalty_weight:g} | "
          f"flip_kl={args.flip_consistency_weight:g}@T{args.flip_consistency_temperature:g} | "
          f"require_prior={config.REQUIRE_PRIOR_PRESENT} | "
          f"select={args.select_metric} patience={args.patience} "
          f"curriculum_same_view_epochs={args.curriculum_same_view_epochs}")

    weight = None
    if config.USE_CLASS_WEIGHT and not args.no_class_weight:
        counts = train_ds.class_counts()             # reflects flips when augmenting
        weight = class_weight_from_counts(counts).to(args.device)
        print("[class_weight]", {n: f"{int(c)}->{float(w):.2f}"
                                  for n, c, w in zip(C.PROG_NAMES, counts, weight)})

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True)
    cur_tl = None
    if curriculum_ds is not None:
        cur_tl = DataLoader(curriculum_ds, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)

    model = M.build_model(feat_dim, args.head_type, args.input_mode, args.hidden, args.dropout,
                          args.head_mode, args.arch, args.fuse_blocks,
                          args.tempfuse_input_mode).to(args.device)
    # model-rebuild cfg — persisted in every ckpt so eval.py/infer.py reconstruct the exact head + data gates.
    mcfg = {"feat_dim": feat_dim, "arch": args.arch, "head_type": args.head_type,
            "input_mode": args.input_mode, "hidden": args.hidden, "dropout": args.dropout,
            "loss": args.loss, "head_mode": args.head_mode, "same_view": args.same_view,
            "box_source": args.box_source, "fuse_blocks": args.fuse_blocks,
            "tempfuse_input_mode": args.tempfuse_input_mode,
            "label_smoothing": args.label_smoothing,
            "cdw_alpha": args.cdw_alpha,
            "cdw_weight": args.cdw_weight,
            "opposite_penalty_weight": args.opposite_penalty_weight,
            "distance_penalty_weight": args.distance_penalty_weight,
            "flip_consistency_weight": args.flip_consistency_weight,
            "flip_consistency_temperature": args.flip_consistency_temperature,
            "seed": args.seed,
            "require_prior_present": config.REQUIRE_PRIOR_PRESENT,
            "curriculum_same_view_epochs": args.curriculum_same_view_epochs}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    remote = f"{args.sync_remote.rstrip('/')}/{args.name}" if args.sync_remote else None

    def push():
        if remote:
            _rclone("copy", str(run_dir), remote, "--transfers", "4", "--quiet")

    best, start_epoch = -1.0, 0
    bests = {"acc": -1.0, "prog": -1.0, "change": -1.0}
    if args.resume:
        if remote:
            _rclone("copy", remote, str(run_dir), "--quiet")
        last = run_dir / "last.pt"
        if last.exists():
            ck = torch.load(last, map_location=args.device)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); start_epoch = ck["epoch"] + 1
            best = ck.get("best", -1.0)
            bests.update(ck.get("bests", {}))
            if bests["prog"] < 0 and "val_f1" in ck:
                bests["prog"] = float(ck["val_f1"])
            if bests["change"] < 0 and "val_change_f1" in ck:
                bests["change"] = float(ck["val_change_f1"])
            if bests["acc"] < 0 and "val_acc" in ck:
                bests["acc"] = float(ck["val_acc"])
            print(f"[resume] from epoch {start_epoch} (best {best:.4f})")
        else:
            print("[resume] no last.pt -> fresh start")

    step, stale = 0, 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss, run_n = 0.0, 0
        epoch_loader = cur_tl if (cur_tl is not None and
                                  epoch < args.curriculum_same_view_epochs) else tl
        for batch in epoch_loader:
            b = move_batch(batch, args.device)
            logits = model(b)
            loss, nval = progression_loss(logits, b["progression"], b["region_mask"], weight,
                                          loss_type=args.loss, gamma=args.focal_gamma,
                                          label_smoothing=args.label_smoothing,
                                          cdw_alpha=args.cdw_alpha,
                                          cdw_weight=args.cdw_weight,
                                          opposite_penalty_weight=args.opposite_penalty_weight,
                                          distance_penalty_weight=args.distance_penalty_weight)
            if args.flip_consistency_weight > 0:
                flipped_logits = model(_make_flipped_batch(b))
                kl, _ = flip_consistency_loss(
                    logits, flipped_logits, b["region_mask"], b.get("region_mask_flip"),
                    temperature=args.flip_consistency_temperature)
                loss = loss + args.flip_consistency_weight * kl
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss) * max(nval, 1); run_n += max(nval, 1)
            step += 1
            if args.sync_every and step % args.sync_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "epoch": epoch, "best": best,
                            "bests": bests, **mcfg},
                           run_dir / "last.pt")
                push()
        sched.step()
        res = evaluate(model, vl, args.device)
        f1 = res["prog_f1_macro"]
        metrics = {"acc": res["acc"], "prog": f1, "change": res["change_f1_macro"]}
        # (a) select best.pt by the chosen val metric (macro/prog = old behaviour, change = headline)
        sel_key = "prog" if args.select_metric == "macro" else args.select_metric
        sel = metrics[sel_key]
        print(f"epoch {epoch + 1:3}/{args.epochs} | loss {run_loss/max(run_n,1):.4f} | "
              f"val acc {res['acc']:.4f} prog-F1 {f1:.4f} change-F1 {res['change_f1_macro']:.4f} "
              f"(per {{ {', '.join(f'{k}:{v:.2f}' for k,v in res['per_class'].items())} }})")

        is_best = sel > best
        if is_best:
            best = sel; stale = 0
        else:
            stale += 1
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "val_f1": f1, "val_change_f1": res["change_f1_macro"],
                "val_acc": res["acc"], "select_metric": args.select_metric, "best": best,
                "bests": bests, **mcfg}
        torch.save(ckpt, run_dir / "last.pt")
        if is_best:
            torch.save(ckpt, run_dir / "best.pt")
        for key, value in metrics.items():
            if value > bests[key]:
                bests[key] = value
                ckpt["bests"] = dict(bests)
                ckpt["best_variant"] = key
                torch.save(ckpt, run_dir / f"best_{key}.pt")
        push()
        if args.patience and stale >= args.patience:      # (a) early stop
            print(f"[early-stop] no val-{args.select_metric} gain in {args.patience} evals "
                  f"(best {best:.4f})")
            break

    print(f"\n[DONE] best selected val {args.select_metric} = {best:.4f} -> {run_dir/'best.pt'}")
    print("[DONE] best variants -> "
          f"acc={bests['acc']:.4f} ({run_dir/'best_acc.pt'}), "
          f"prog={bests['prog']:.4f} ({run_dir/'best_prog.pt'}), "
          f"change={bests['change']:.4f} ({run_dir/'best_change.pt'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
