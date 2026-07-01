"""Step 2 — extract FROZEN BioViL-T features for every image in the worklist (resumable).

Loads BioViL-T once (eval/no-grad), streams the worklist through a DataLoader, and writes one
<image_id>.pt ([197, C] float16) per image. Every --flush-every new files it rclone-copies the
staging dir to Drive and deletes the local copies (so /kaggle/working never fills). On restart it
skips image_ids already on Drive or staged locally -> safe to run repeatedly until coverage is full.

    python 2-extract_features.py --worklist /kaggle/working/worklist.jsonl \
        --out-dir /kaggle/working/features --remote dhint:CHEX-DATA/biovilt_features --device cuda
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_1/src

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import biovilt
import config
import io_features as io


class WorklistDataset(Dataset):
    """Yields (image_id, [3,res,res] tensor) for the not-yet-done rows of the worklist."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        img = biovilt.load_image(row["path"])
        return row["image_id"], img


def _collate(batch):
    """Keep the batch as a list of (image_id, tensor) — module-level so DataLoader workers
    can pickle it (a lambda can't be pickled under spawn)."""
    return batch


def _load_worklist(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frozen BioViL-T features (resumable)")
    p.add_argument("--worklist", type=Path, default=config.DEFAULT_WORKLIST)
    p.add_argument("--out-dir", type=Path, default=config.DEFAULT_FEATURES_OUT)
    p.add_argument("--remote", default=None,
                   help="rclone remote for the feature cache, e.g. dhint:CHEX-DATA/biovilt_features")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--flush-every", type=int, default=config.FLUSH_EVERY)
    p.add_argument("--limit", type=int, default=0, help="stop after N new images (0 = all; smoke test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.worklist.exists():
        raise SystemExit(f"[ERROR] worklist not found: {args.worklist} (run 1-build_worklist.py)")

    rows = _load_worklist(args.worklist)
    done = io.done_ids(args.out_dir, args.remote)
    todo = [r for r in rows if r["image_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"worklist {len(rows):,} | already done {len(done):,} | to extract {len(todo):,}"
          + (f" (limited to {args.limit})" if args.limit else ""))
    if not todo:
        print("[DONE] nothing to extract — cache already complete.")
        return 0

    print(f"loading FROZEN BioViL-T encoder on {args.device} ...")
    model = biovilt.load_encoder(args.device)
    feat_dim = biovilt.detect_feat_dim(model, args.device)
    print(f"feature dim C = {feat_dim}  (expected {config.FEAT_DIM}; transform={config.TRANSFORM_MODE})")
    if feat_dim != config.FEAT_DIM:
        print(f"[WARN] C={feat_dim} != config.FEAT_DIM={config.FEAT_DIM}. The phase_3 loader "
              f"auto-detects C, but the WHOLE cache must share one C — verify against the reference.")

    loader = DataLoader(WorklistDataset(todo), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=_collate)  # batch = list[(id, img)]

    written = since_flush = 0
    total = len(todo)
    for batch in loader:
        ids = [iid for iid, _ in batch]
        imgs = torch.stack([img for _, img in batch], dim=0)         # [B,3,res,res]
        feats = biovilt.encode_batch(model, imgs, args.device)       # [B,197,C] f16 cpu
        for iid, feat in zip(ids, feats):
            io.save_feature(args.out_dir, iid, feat)
            written += 1
            since_flush += 1
        if since_flush >= args.flush_every:
            removed = io.flush_to_drive(args.out_dir, args.remote)
            print(f"  [{written:,}/{total:,}] flushed {since_flush} -> {args.remote} "
                  f"(freed {removed} local)")
            since_flush = 0
        elif written % (args.batch * 20) == 0:
            print(f"  [{written:,}/{total:,}] extracted")

    removed = io.flush_to_drive(args.out_dir, args.remote)            # final flush
    print(f"[DONE] extracted {written:,} new features; final flush freed {removed} local files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
