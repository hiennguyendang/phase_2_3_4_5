"""M4 dataset: pair each current image with its prior and serve cached region tensors.

For a (current, prior) pair it returns, per region:
  feat_curr / feat_prior   [29, feat_dim]   (frozen-M3 region features, from the cache)
  logit_curr / logit_prior [29, 14]         (frozen-M3 disease logits, soft)
  region_mask              [29]             present in current (AND prior, if REQUIRE_PRIOR_PRESENT)
  progression              [29, 14]         class {0,1,2} or -100 (target)

No backbone is run here — everything is a cache lookup (see phase_3/precompute_regions.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config
import constants as C


class RegionCache:
    """Maps image_id -> <root>/<image_id>.npy  (float16 [29, feat_dim+14])."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._index: dict[str, Path] | None = None
        self.feat_dim: int | None = None

    @property
    def index(self) -> dict[str, Path]:
        if self._index is None:
            idx: dict[str, Path] = {}
            for p in self.root.rglob("*.npy"):
                idx.setdefault(p.stem, p)
            if not idx:
                raise FileNotFoundError(f"no '*.npy' region caches under {self.root}")
            self._index = idx
        return self._index

    def has(self, image_id: str) -> bool:
        return image_id in self.index

    def load(self, image_id: str) -> tuple[np.ndarray, np.ndarray]:
        arr = np.load(self.index[image_id]).astype(np.float32)     # [29, feat+14]
        feat, logit = arr[:, : -C.NUM_CHEX], arr[:, -C.NUM_CHEX:]
        if self.feat_dim is None:
            self.feat_dim = feat.shape[1]
        return feat, logit

    def detect_dim(self) -> int:
        feat, _ = self.load(next(iter(self.index)))
        return feat.shape[1]


def _present_by_image(m3_labels_dir: Path) -> dict[str, np.ndarray]:
    pm = np.load(Path(m3_labels_dir) / "present_mask.npy", mmap_mode="r")
    out: dict[str, np.ndarray] = {}
    with open(Path(m3_labels_dir) / "manifest.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            if m.get("ok", True):
                out[m["image_id"]] = np.asarray(pm[i], dtype=np.float32)
    return out


def _prior_by_image(pairs_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["image_id"]] = r["prior_image_id"]
    return out


class M4Dataset(Dataset):
    def __init__(self, region_cache, m3_labels_dir, m4_labels_dir, pairs_path,
                 split: str | None = None, augment: bool = False):
        # time-flip augmentation is TRAIN ONLY (constraint 2). The flip stays within the train pairs
        # (constraint 1): we only generate (C,P) from a (P,C) that is already in this split.
        self.augment = augment
        self.flip_map = torch.tensor(C.FLIP_CLASS_MAP, dtype=torch.int64)
        self.flip_exclude_idx = [C.CHEX_INDEX[n] for n in config.FLIP_EXCLUDE_DISEASES
                                 if n in C.CHEX_INDEX]
        self.cache = region_cache if isinstance(region_cache, RegionCache) else RegionCache(region_cache)
        self.prog = np.load(Path(m4_labels_dir) / "progression.npy", mmap_mode="r")
        present = _present_by_image(m3_labels_dir)
        prior = _prior_by_image(pairs_path)

        manifest = [json.loads(l) for l in open(Path(m4_labels_dir) / "manifest.jsonl", encoding="utf-8")]
        self.rows: list[tuple[int, str, str]] = []        # (prog_row, curr_id, prior_id)
        skipped = {"no_cue": 0, "no_prior": 0, "no_cache": 0, "no_present": 0, "split": 0}
        for i, m in enumerate(manifest):
            if not m.get("ok", True):
                continue
            if split is not None and str(m.get("split", "")).lower() != split:
                skipped["split"] += 1; continue
            if m.get("n_cued", 0) <= 0:
                skipped["no_cue"] += 1; continue
            cid = m["image_id"]
            pid = prior.get(cid)
            if pid is None:
                skipped["no_prior"] += 1; continue
            if not (self.cache.has(cid) and self.cache.has(pid)):
                skipped["no_cache"] += 1; continue
            if cid not in present or pid not in present:
                skipped["no_present"] += 1; continue
            self.rows.append((i, cid, pid))
        self.present = present
        self.feat_dim = self.cache.detect_dim()
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.rows) * (2 if self.augment else 1)

    def class_counts(self) -> np.ndarray:
        """Per-class cell counts over THIS split's rows (including flips if augmenting), for
        class-weighting. Exact: counts the same labels the loss will actually see."""
        if not self.rows:
            return np.zeros(C.NUM_PROG, dtype=np.int64)
        idx = [i for i, _, _ in self.rows]
        sub = np.asarray(self.prog[idx]).astype(np.int64)        # [n,29,14]
        counts = np.array([(sub == k).sum() for k in range(C.NUM_PROG)], dtype=np.int64)
        if self.augment:
            flip = sub.copy()
            m = flip != C.UNKNOWN
            flip[m] = np.asarray(C.FLIP_CLASS_MAP)[flip[m]]
            if self.flip_exclude_idx:
                flip[:, :, self.flip_exclude_idx] = C.UNKNOWN
            counts += np.array([(flip == k).sum() for k in range(C.NUM_PROG)], dtype=np.int64)
        return counts

    def _progression(self, i: int, flipped: bool) -> torch.Tensor:
        tgt = torch.from_numpy(self.prog[i].astype(np.int64))                   # [29,14]
        if not flipped:
            return tgt
        valid = tgt != C.UNKNOWN
        tgt = torch.where(valid, self.flip_map[tgt.clamp_min(0)], tgt)          # improved<->worsened
        if self.flip_exclude_idx:                                              # non-antisymmetric -> mask
            tgt[:, self.flip_exclude_idx] = C.UNKNOWN
        return tgt

    def __getitem__(self, k: int) -> dict:
        n = len(self.rows)
        flipped = self.augment and k >= n
        i, cid, pid = self.rows[k - n] if flipped else self.rows[k]
        # flipped sample = swap roles: current<->prior on BOTH feat and logit (constraint 4),
        # so the model's diff = feat_curr-feat_prior auto-negates. Labels flip in _progression.
        a, b = (pid, cid) if flipped else (cid, pid)
        fc, lc = self.cache.load(a)
        fp, lp = self.cache.load(b)
        region_mask = self.present[a].copy()
        if config.REQUIRE_PRIOR_PRESENT:
            region_mask = region_mask * self.present[b]
        return {
            "image_id": (cid + "~flip") if flipped else cid,
            "prior_image_id": b,
            "feat_curr": torch.from_numpy(fc), "logit_curr": torch.from_numpy(lc),
            "feat_prior": torch.from_numpy(fp), "logit_prior": torch.from_numpy(lp),
            "region_mask": torch.from_numpy(region_mask),                       # [29]
            "progression": self._progression(i, flipped),                      # [29,14]
        }


def collate(batch: list[dict]) -> dict:
    out = {"image_id": [b["image_id"] for b in batch],
           "prior_image_id": [b["prior_image_id"] for b in batch]}
    for kk in ("feat_curr", "logit_curr", "feat_prior", "logit_prior", "region_mask", "progression"):
        out[kk] = torch.stack([b[kk] for b in batch])
    return out
