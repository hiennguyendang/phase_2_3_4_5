"""Run a trained M3 model -> per-image predictions for M4/M5.

Emits one JSON line per image:
  image_id, image_disease[14] (prob), region_disease[29][14] (prob),
  region_concepts[29] -> {concept: prob for present regions, top-k},
  region_feats are NOT dumped (large) — M4 should re-run the model or read a feature dump.

For now boxes/regions come from the M3 label arrays (MIMIC). A detector-box source
for CheXplus/NIH can be plugged into M3Dataset later (same shapes).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_3/src

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import M3Dataset, collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3 inference -> per-image JSON")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, default=config.DEFAULT_LABELS_DIR)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "m3_pred.jsonl")
    p.add_argument("--topk-concepts", type=int, default=8)
    p.add_argument("--topk-cells", type=int, default=0,
                   help="dump top-k attention-pool grid cells per region (M5 'where'); 0 = off")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    import model as M
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    config.USE_GLOBAL_TOKEN = ck.get("use_global", config.USE_GLOBAL_TOKEN)
    m = M.build_model(ck["feat_dim"], ck["mode"]).to(args.device).eval()
    m.load_state_dict(ck["model"])

    ds = M3Dataset(args.labels_dir, args.features_root, args.split)
    loader = DataLoader(ds, batch_size=args.batch, collate_fn=collate)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for b in loader:
            out = m(b["grid"].to(args.device), b["global"].to(args.device),
                    b["present_mask"].to(args.device), b["boxes"].to(args.device))
            img = torch.sigmoid(out["image_disease_logits"]).cpu()
            rd = torch.sigmoid(out["region_disease_logits"]).cpu()
            cc = (torch.sigmoid(out["concept_logits"]).cpu()
                  if out["concept_logits"] is not None else None)
            attn = out["region_attn"].cpu() if args.topk_cells else None  # [B,29,196]
            mask = b["present_mask"]
            for j, iid in enumerate(b["image_id"]):
                rec = {
                    "image_id": iid,
                    "image_disease": {C.CHEX_NAMES[c]: round(float(img[j, c]), 4) for c in range(C.NUM_CHEX)},
                    "regions": {},
                }
                for r in range(C.NUM_REGIONS):
                    if mask[j, r] < 0.5:
                        continue
                    entry = {"disease": {C.CHEX_NAMES[c]: round(float(rd[j, r, c]), 3)
                                         for c in range(C.NUM_CHEX) if rd[j, r, c] > 0.5}}
                    if cc is not None:
                        top = torch.topk(cc[j, r], args.topk_concepts)
                        entry["concepts"] = {C.CONCEPT_NAMES[int(i)]: round(float(p), 3)
                                             for p, i in zip(top.values, top.indices) if p > 0.5}
                    if attn is not None:                 # faithful "where" cells -> (row, col, weight)
                        tc = torch.topk(attn[j, r], args.topk_cells)
                        entry["cells"] = [[int(i) // config.GRID_W, int(i) % config.GRID_W, round(float(w), 3)]
                                          for w, i in zip(tc.values, tc.indices)]
                    rec["regions"][C.REGION_NAMES[r]] = entry
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    print(f"[DONE] {written:,} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
