"""Run a trained M4 model -> per-(image, prior) progression readout for M5 (the change-ledger).

One JSON line per current image that has a prior:
  image_id, prior_image_id,
  regions: { region: { disease: {prog: "worsened"|"improved"|"stable", prob: p, probs:[s,i,w]} } }
By default only CHANGE cells (argmax != stable) are emitted (the rest are "stable" by readout);
pass --include-stable to dump every present (region, disease) cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import M4Dataset, collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M4 inference -> per-image progression JSON")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--region-cache", type=Path, default=config.DEFAULT_REGION_CACHE)
    p.add_argument("--m3-labels-dir", type=Path, default=config.DEFAULT_M3_LABELS_DIR)
    p.add_argument("--m4-labels-dir", type=Path, default=config.DEFAULT_M4_LABELS_DIR)
    p.add_argument("--pairs", type=Path, default=config.DEFAULT_PAIRS_PATH)
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "m4_pred.jsonl")
    p.add_argument("--include-stable", action="store_true")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    import model as M
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    m = M.build_model(ck["feat_dim"]).to(args.device).eval()
    m.load_state_dict(ck["model"])

    ds = M4Dataset(args.region_cache, args.m3_labels_dir, args.m4_labels_dir, args.pairs, args.split)
    loader = DataLoader(ds, batch_size=args.batch, collate_fn=collate)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for b in loader:
            logits = m(b["feat_curr"].to(args.device), b["logit_curr"].to(args.device),
                       b["feat_prior"].to(args.device), b["logit_prior"].to(args.device))  # [B,29,14,3]
            probs = F.softmax(logits, dim=-1).cpu()
            pred = probs.argmax(-1)
            mask = b["region_mask"]
            for j in range(len(b["image_id"])):
                rec = {"image_id": b["image_id"][j], "prior_image_id": b["prior_image_id"][j], "regions": {}}
                for r in range(C.NUM_REGIONS):
                    if mask[j, r] < 0.5:
                        continue
                    cells = {}
                    for d in range(C.NUM_CHEX):
                        cls = int(pred[j, r, d])
                        if cls == 0 and not args.include_stable:
                            continue
                        pv = probs[j, r, d]
                        cells[C.CHEX_NAMES[d]] = {"prog": C.PROG_NAMES[cls],
                                                  "prob": round(float(pv[cls]), 3),
                                                  "probs": [round(float(x), 3) for x in pv]}
                    if cells:
                        rec["regions"][C.REGION_NAMES[r]] = cells
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    print(f"[DONE] {written:,} progression readouts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
