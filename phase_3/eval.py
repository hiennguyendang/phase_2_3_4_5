"""Metrics for M3: macro-F1 (headline, spec 3.6) + AUC, for image / region / concept.

Both are computed dependency-free (ignore the -100 sentinel), so no sklearn needed.
F1 is the metric that drives checkpoint selection; AUC is reported alongside.
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


def _f1_table(prob: np.ndarray, tgt: np.ndarray, names: list[str],
              thr: float = 0.5) -> tuple[float, dict]:
    """Binary F1 per class at a fixed threshold (ignores -100). Macro = mean over classes."""
    f1s = {}
    for c in range(prob.shape[-1]):
        m = (tgt[..., c] == 0) | (tgt[..., c] == 1)
        if not m.any():
            f1s[names[c]] = float("nan"); continue
        p = (prob[..., c][m] >= thr).astype(np.int64)
        t = tgt[..., c][m].astype(np.int64)
        tp = int(((p == 1) & (t == 1)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        denom = 2 * tp + fp + fn
        f1s[names[c]] = (2.0 * tp / denom) if denom > 0 else float("nan")
    macro = float(np.nanmean(list(f1s.values())))
    return macro, f1s


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    img_p, img_t = [], []
    rd_p, rd_t, rd_m = [], [], []
    cc_p, cc_t, cc_m = [], [], []
    for b in loader:
        out = model(b["grid"].to(device), b["global"].to(device),
                    b["present_mask"].to(device), b["boxes"].to(device))
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
    res["image_f1_macro"], res["image_f1_per_class"] = _f1_table(P, T, C.CHEX_NAMES)

    rp, rt, rm = np.concatenate(rd_p), np.concatenate(rd_t), np.concatenate(rd_m).astype(bool)
    rp, rt = rp[rm], rt[rm]                      # [n_present, 14]
    res["region_auc_macro"], _ = _auc_table(rp, rt, C.CHEX_NAMES)
    res["region_f1_macro"], _ = _f1_table(rp, rt, C.CHEX_NAMES)

    if cc_p:
        cp, ct, cm = np.concatenate(cc_p), np.concatenate(cc_t), np.concatenate(cc_m).astype(bool)
        cp, ct = cp[cm], ct[cm]                  # [n_present, 69]
        res["concept_auc_macro"], _ = _auc_table(cp, ct, C.CONCEPT_NAMES)
        res["concept_f1_macro"], _ = _f1_table(cp, ct, C.CONCEPT_NAMES)
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
    print(f"[{args.split}] image  F1 macro = {res['image_f1_macro']:.4f}  AUC macro = {res['image_auc_macro']:.4f}")
    print(f"          region F1 {res['region_f1_macro']:.4f}  AUC {res['region_auc_macro']:.4f}"
          + (f"  | concept F1 {res.get('concept_f1_macro', float('nan')):.4f}"
             f"  AUC {res.get('concept_auc_macro', float('nan')):.4f}" if "concept_auc_macro" in res else ""))
    print(f"  {'class':<26} {'F1':>7} {'AUC':>7}")
    for c in res["image_per_class"]:
        print(f"  {c:<26} {res['image_f1_per_class'][c]:>7.4f} {res['image_per_class'][c]:>7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
