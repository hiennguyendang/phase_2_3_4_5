"""[M4 PREP] Freeze M3 and dump per-image region features + disease logits.

M3 is frozen after Phase-3 training, so its region outputs are deterministic. We cache them
ONCE here so Phase-4 (T-KAN) never has to run the pool/backbone — the Siamese "shared frozen
branch" of spec 4.1 becomes a cheap cache lookup, and phase_4 stays import-independent of phase_3.

Runs over EVERY image that has features (current AND prior images alike — priors are needed for
the Siamese). For each image writes  <image_id>.npy  float16  [29, feat_dim + 14]:
    [:, :feat_dim]      region_feats   (post-pool, post-neck)
    [:, feat_dim:]      region_disease_logits  (soft 14 logits — M4 needs the magnitude)

    python phase_3/precompute_regions.py --ckpt <run>/m3_A/best.pt \
        --labels-dir data/m3_labels --features-root <feat> --out-dir data/m3_region_cache
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache frozen-M3 region features + logits for M4")
    p.add_argument("--ckpt", type=Path, required=True, help="trained M3 checkpoint (e.g. m3_A/best.pt)")
    p.add_argument("--labels-dir", type=Path, default=config.DEFAULT_LABELS_DIR)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--out-dir", type=Path, default=config.REPO_ROOT / "data" / "m3_region_cache"
                   if hasattr(config, "REPO_ROOT") else Path("data/m3_region_cache"))
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    import model as M
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=args.device)
    config.USE_GLOBAL_TOKEN = ck.get("use_global", config.USE_GLOBAL_TOKEN)
    m = M.build_model(ck["feat_dim"], ck["mode"]).to(args.device).eval()
    m.load_state_dict(ck["model"])
    print(f"[precompute] M3 mode={ck['mode']} feat_dim={ck['feat_dim']} -> region cache")

    ds = M3Dataset(args.labels_dir, args.features_root, split=None)   # ALL images (curr + prior)
    loader = DataLoader(ds, batch_size=args.batch, collate_fn=collate)
    print(f"[precompute] {len(ds):,} images with features")

    written = 0
    for b in loader:
        out = m(b["grid"].to(args.device), b["global"].to(args.device),
                b["present_mask"].to(args.device), b["boxes"].to(args.device))
        feat = out["region_feats"].cpu().numpy()                     # [B,29,feat]
        logit = out["region_disease_logits"].cpu().numpy()           # [B,29,14]
        arr = np.concatenate([feat, logit], axis=-1).astype(np.float16)  # [B,29,feat+14]
        for j, iid in enumerate(b["image_id"]):
            np.save(args.out_dir / f"{iid}.npy", arr[j])
            written += 1
        if written % 5000 < args.batch:
            print(f"  {written:,} cached ...")
    print(f"[DONE] {written:,} region caches -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
