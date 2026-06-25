"""Metrics for M3: image-level CheXpert AUC (macro), region + concept AUC.

AUC is computed with a dependency-free rank formula (ignores the -100 sentinel),
so no sklearn needed.
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


def auc_binary(scores: np.ndarray, targets: np.ndarray) -> float:
    pos, neg = targets == 1, targets == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _auc_table(prob: np.ndarray, tgt: np.ndarray, names: list[str]) -> tuple[float, dict]:
    aucs = {}
    for c in range(prob.shape[-1]):
        m = (tgt[..., c] == 0) | (tgt[..., c] == 1)
        aucs[names[c]] = auc_binary(prob[..., c][m], tgt[..., c][m]) if m.any() else float("nan")
    macro = float(np.nanmean(list(aucs.values())))
    return macro, aucs


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    img_p, img_t = [], []
    rd_p, rd_t, rd_m = [], [], []
    cc_p, cc_t, cc_m = [], [], []
    for b in loader:
        out = model(b["grid"].to(device), b["global"].to(device), b["present_mask"].to(device))
        img_p.append(torch.sigmoid(out["image_disease_logits"]).cpu().numpy())
        img_t.append(b["image_chexpert"].numpy())
        rd_p.append(torch.sigmoid(out["region_disease_logits"]).cpu().numpy())
        rd_t.append(b["region_chexpert"].numpy())
        rd_m.append(b["present_mask"].numpy())
        if out["concept_logits"] is not None:
            cc_p.append(torch.sigmoid(out["concept_logits"]).cpu().numpy())
            cc_t.append(b["region_concepts"].numpy())
            cc_m.append(b["present_mask"].numpy())

    res = {}
    P, T = np.concatenate(img_p), np.concatenate(img_t)
    res["image_auc_macro"], res["image_per_class"] = _auc_table(P, T, C.CHEX_NAMES)

    rp, rt, rm = np.concatenate(rd_p), np.concatenate(rd_t), np.concatenate(rd_m).astype(bool)
    rp, rt = rp[rm], rt[rm]                      # [n_present, 14]
    res["region_auc_macro"], _ = _auc_table(rp, rt, C.CHEX_NAMES)

    if cc_p:
        cp, ct, cm = np.concatenate(cc_p), np.concatenate(cc_t), np.concatenate(cc_m).astype(bool)
        cp, ct = cp[cm], ct[cm]                  # [n_present, 69]
        res["concept_auc_macro"], _ = _auc_table(cp, ct, C.CONCEPT_NAMES)
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an M3 checkpoint")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, default=config.DEFAULT_LABELS_DIR)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    import model as M
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    ds = M3Dataset(args.labels_dir, args.features_root, args.split)
    loader = DataLoader(ds, batch_size=args.batch, collate_fn=collate)
    config.USE_GLOBAL_TOKEN = ck.get("use_global", config.USE_GLOBAL_TOKEN)
    m = M.build_model(ck["feat_dim"], ck["mode"]).to(args.device)
    m.load_state_dict(ck["model"])
    res = evaluate(m, loader, args.device)
    print(f"[{args.split}] image AUC macro = {res['image_auc_macro']:.4f} | "
          f"region {res['region_auc_macro']:.4f}"
          + (f" | concept {res.get('concept_auc_macro', float('nan')):.4f}" if "concept_auc_macro" in res else ""))
    for c, a in res["image_per_class"].items():
        print(f"  {c:<26} {a:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
