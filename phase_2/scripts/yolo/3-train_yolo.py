"""Step 1 — train the 29-region detector (YOLOv8l, ultralytics).

    pip install ultralytics
    python train_yolo.py                       # fresh run
    python train_yolo.py --resume              # continue last run (new session)
    python train_yolo.py --imgsz 640 --batch 8 --epochs 50

Multi-session note (Kaggle caps GPU sessions at ~9-12h): checkpoints land in
<runs>/detect/<name>/weights/ every SAVE_PERIOD epochs. To continue next session,
keep the same --runs/--name and pass --resume (loads last.pt with its optimizer).
Make the run dir part of your notebook's persisted /kaggle/working output.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
from pathlib import Path

import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 detector for 29 regions")
    p.add_argument("--data", type=Path, default=config.DEFAULT_DATASET_DIR / "dataset.yaml")
    p.add_argument("--model", default=config.MODEL_WEIGHTS)
    p.add_argument("--runs", type=Path, default=config.DEFAULT_RUNS_DIR)
    p.add_argument("--name", default="det29")
    p.add_argument("--imgsz", type=int, default=config.IMGSZ)
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="train on this fraction of the dataset (1.0 = all). 29 anatomical "
                        "regions are 'deceptively easy' (see docs/critical_yolo.md), so a "
                        "subset trains a solid box detector far faster. Eval/val unaffected.")
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--save-period", type=int, default=config.SAVE_PERIOD)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="False", help="False | ram | disk")
    p.add_argument("--resume", action="store_true")
    # durable checkpointing to an rclone remote (Kaggle: survives session death)
    p.add_argument("--sync-remote", default=None,
                   help="rclone remote path, e.g. dhint:CHEX-DATA/phase2_runs "
                        "(pushes run dir every --sync-every steps + each epoch)")
    p.add_argument("--sync-every", type=int, default=300,
                   help="push a checkpoint every N optimizer steps (default 300)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    # Defensive: strip stray surrounding quotes from the remote. IPython `!cmd $VAR`
    # can leak literal quotes into argv (e.g. '"dhint:phase2_runs"'), which makes
    # rclone treat it as a local path and silently NOT write to Drive.
    if args.sync_remote:
        args.sync_remote = args.sync_remote.strip().strip('"').strip("'")

    cache = args.cache if args.cache in ("ram", "disk") else False

    def _attach_sync(model) -> None:
        if args.sync_remote:
            from kaggle_sync import attach_rclone_sync
            attach_rclone_sync(model, args.sync_remote, every=args.sync_every)

    if args.resume:
        # pull the latest run dir from the remote first so last.pt exists locally
        if args.sync_remote:
            from kaggle_sync import pull_run
            pull_run(args.sync_remote, args.runs, args.name)
        last = args.runs / args.name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"[ERROR] --resume but no checkpoint at {last}")
        print(f"Resuming from {last}")
        model = YOLO(str(last))
        _attach_sync(model)
        model.train(resume=True)
        return 0

    if not args.data.exists():
        raise SystemExit(f"[ERROR] dataset.yaml not found: {args.data}\n"
                         f"Run build_yolo_dataset.py first.")

    model = YOLO(args.model)
    _attach_sync(model)
    model.train(
        data=str(args.data),
        project=str(args.runs),
        name=args.name,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        fraction=args.fraction,
        patience=args.patience,
        save_period=args.save_period,
        device=args.device,
        workers=args.workers,
        cache=cache,
        amp=True,
        cos_lr=True,
        # anatomy-safe augmentation
        mosaic=config.AUG["mosaic"],
        mixup=config.AUG["mixup"],
        degrees=config.AUG["degrees"],
        perspective=config.AUG["perspective"],
    )
    print(f"\nWeights: {args.runs / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
