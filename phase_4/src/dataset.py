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


class PatchStore:
    """Maps image_id -> frozen M1 BioViL-T patch grid  [196, dim]  (.pt or .npy, [1+196,C] or [196,C]).
    The `tempfuse` arch reads these directly — no bridge/region-cache. Mirrors phase_3 FeatureStore."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._index: dict[str, Path] | None = None
        self.feat_dim: int | None = None

    @property
    def index(self) -> dict[str, Path]:
        if self._index is None:
            idx: dict[str, Path] = {}
            for suf in (".npy", ".pt"):                     # .npy preferred on stem clash
                for p in self.root.rglob(f"*{suf}"):
                    idx.setdefault(p.stem, p)
            if not idx:
                raise FileNotFoundError(f"no '*.npy'/'*.pt' features under {self.root}")
            self._index = idx
        return self._index

    def has(self, image_id: str) -> bool:
        return image_id in self.index

    def load(self, image_id: str) -> np.ndarray:
        p = self.index[image_id]
        if p.suffix == ".pt":
            t = torch.load(p, map_location="cpu")
            if isinstance(t, dict):
                t = next((v for v in t.values() if torch.is_tensor(v)), None)
            arr = t.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.load(p).astype(np.float32)
        if arr.shape[0] == config.GRID_TOKENS + 1:         # drop the global row
            arr = arr[1:]
        elif arr.shape[0] != config.GRID_TOKENS:
            raise ValueError(f"{p}: rows={arr.shape[0]}, expected {config.GRID_TOKENS} or {config.GRID_TOKENS+1}")
        if self.feat_dim is None:
            self.feat_dim = int(arr.shape[1])
        return np.ascontiguousarray(arr, dtype=np.float32)  # [196, dim]

    def detect_dim(self) -> int:
        return self.load(next(iter(self.index))).shape[1]


def _present_by_image(m3_labels_dir: Path) -> dict[str, np.ndarray]:
    pm = np.load(Path(m3_labels_dir) / "present_mask.npy", mmap_mode="r")
    out: dict[str, np.ndarray] = {}
    with open(Path(m3_labels_dir) / "manifest.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            if m.get("ok", True):
                out[m["image_id"]] = np.asarray(pm[i], dtype=np.float32)
    return out


def _prior_by_image(pairs_path: Path) -> dict[str, tuple[str, bool]]:
    """image_id -> (prior_image_id, same_view). `same_view` from m3_pairs.jsonl (default True if absent)."""
    out: dict[str, tuple[str, bool]] = {}
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["image_id"]] = (r["prior_image_id"], bool(r.get("same_view", True)))
    return out


def _boxes_by_image(m3_labels_dir: Path, box_source: str = "gt"):
    """-> (boxes_mmap [N,29,4] int, image_id -> row). Boxes live in the 448 frame (phase_2/3)."""
    fname = "boxes_det.npy" if box_source == "detector" else "boxes.npy"
    boxes = np.load(Path(m3_labels_dir) / fname, mmap_mode="r")
    id2row: dict[str, int] = {}
    with open(Path(m3_labels_dir) / "manifest.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            if m.get("ok", True):
                id2row[m["image_id"]] = i
    return boxes, id2row


def _filter_rows(manifest, split, prior, present, has_fn, same_view_only):
    """Shared (curr,prior) row selection for both archs. has_fn(image_id) checks feature availability
    (region cache OR patch store). Returns (rows=[(prog_row,curr,prior)], skipped counters)."""
    rows: list[tuple[int, str, str]] = []
    skipped = {"no_cue": 0, "no_prior": 0, "no_feat": 0, "no_present": 0, "split": 0, "cross_view": 0}
    for i, m in enumerate(manifest):
        if not m.get("ok", True):
            continue
        if split is not None and str(m.get("split", "")).lower() != split:
            skipped["split"] += 1; continue
        if m.get("n_cued", 0) <= 0:
            skipped["no_cue"] += 1; continue
        cid = m["image_id"]
        pr = prior.get(cid)
        if pr is None:
            skipped["no_prior"] += 1; continue
        pid, same_view = pr
        if same_view_only and not same_view:
            skipped["cross_view"] += 1; continue
        if not (has_fn(cid) and has_fn(pid)):
            skipped["no_feat"] += 1; continue
        if cid not in present or pid not in present:
            skipped["no_present"] += 1; continue
        rows.append((i, cid, pid))
    return rows, skipped


class _M4Base(Dataset):
    """Shared M4 machinery: row filtering, time-flip augmentation, class weighting, target flip.
    Subclasses only implement _item_tensors(a, b) -> extra feature tensors for the (curr,prior) pair."""

    def __init__(self, m3_labels_dir, m4_labels_dir, pairs_path, split, augment,
                 same_view_only, has_fn):
        self.augment = augment
        self.same_view_only = same_view_only
        self.flip_map = torch.tensor(C.FLIP_CLASS_MAP, dtype=torch.int64)
        self.flip_exclude_idx = [C.CHEX_INDEX[n] for n in config.FLIP_EXCLUDE_DISEASES
                                 if n in C.CHEX_INDEX]
        self.prog = np.load(Path(m4_labels_dir) / "progression.npy", mmap_mode="r")
        self.present = _present_by_image(m3_labels_dir)
        prior = _prior_by_image(pairs_path)
        manifest = [json.loads(l) for l in open(Path(m4_labels_dir) / "manifest.jsonl", encoding="utf-8")]
        self.rows, self.skipped = _filter_rows(manifest, split, prior, self.present,
                                               has_fn, same_view_only)

    def __len__(self) -> int:
        return len(self.rows) * (2 if self.augment else 1)

    def class_counts(self) -> np.ndarray:
        """Per-class cell counts over THIS split's rows (incl. flips if augmenting) — exactly the
        labels the loss will see, for inverse-frequency class weighting."""
        if not self.rows:
            return np.zeros(C.NUM_PROG, dtype=np.int64)
        idx = [i for i, _, _ in self.rows]
        sub = np.asarray(self.prog[idx]).astype(np.int64)
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
        if self.flip_exclude_idx:
            tgt[:, self.flip_exclude_idx] = C.UNKNOWN
        return tgt

    def _region_mask(self, a: str, b: str) -> torch.Tensor:
        rm = self.present[a].copy()
        if config.REQUIRE_PRIOR_PRESENT:
            rm = rm * self.present[b]
        return torch.from_numpy(rm)                                            # [29]

    def _item_tensors(self, a: str, b: str) -> dict:                            # a=current slot, b=prior
        raise NotImplementedError

    def __getitem__(self, k: int) -> dict:
        n = len(self.rows)
        flipped = self.augment and k >= n
        i, cid, pid = self.rows[k - n] if flipped else self.rows[k]
        a, b = (pid, cid) if flipped else (cid, pid)          # flip swaps current<->prior roles
        item = {"image_id": (cid + "~flip") if flipped else cid, "prior_image_id": b,
                "region_mask": self._region_mask(a, b),
                "progression": self._progression(i, flipped)}
        item.update(self._item_tensors(a, b))
        return item


class M4Dataset(_M4Base):
    """regiondiff arch — serves frozen-M3 region features + disease logits from the region cache."""

    def __init__(self, region_cache, m3_labels_dir, m4_labels_dir, pairs_path,
                 split: str | None = None, augment: bool = False, same_view_only: bool = False):
        self.cache = region_cache if isinstance(region_cache, RegionCache) else RegionCache(region_cache)
        super().__init__(m3_labels_dir, m4_labels_dir, pairs_path, split, augment,
                         same_view_only, self.cache.has)
        self.feat_dim = self.cache.detect_dim()

    def _item_tensors(self, a: str, b: str) -> dict:
        fc, lc = self.cache.load(a)
        fp, lp = self.cache.load(b)
        return {"feat_curr": torch.from_numpy(fc), "logit_curr": torch.from_numpy(lc),
                "feat_prior": torch.from_numpy(fp), "logit_prior": torch.from_numpy(lp)}


class M4PatchDataset(_M4Base):
    """tempfuse arch — serves frozen M1 patch grids (curr, prior) + the current image's boxes."""

    def __init__(self, features_root, m3_labels_dir, m4_labels_dir, pairs_path,
                 split: str | None = None, augment: bool = False, same_view_only: bool = False,
                 box_source: str = config.BOX_SOURCE):
        self.store = features_root if isinstance(features_root, PatchStore) else PatchStore(features_root)
        self.boxes, self.box_row = _boxes_by_image(m3_labels_dir, box_source)
        super().__init__(m3_labels_dir, m4_labels_dir, pairs_path, split, augment,
                         same_view_only, self.store.has)
        self.feat_dim = self.store.detect_dim()

    def _item_tensors(self, a: str, b: str) -> dict:
        box = np.asarray(self.boxes[self.box_row[a]], dtype=np.float32)         # current-slot boxes [29,4]
        return {"patch_curr": torch.from_numpy(self.store.load(a)),             # [196,dim]
                "patch_prior": torch.from_numpy(self.store.load(b)),
                "box_curr": torch.from_numpy(box)}


def collate(batch: list[dict]) -> dict:
    """Generic: stack every tensor-valued key, keep string/id keys as lists (works for both archs)."""
    out: dict = {}
    for kk in batch[0]:
        out[kk] = (torch.stack([b[kk] for b in batch]) if torch.is_tensor(batch[0][kk])
                   else [b[kk] for b in batch])
    return out


def move_batch(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_dataset(arch, m3_labels_dir, m4_labels_dir, pairs, split, *, region_cache=None,
                 features_root=None, augment=False, same_view_only=False, box_source=config.BOX_SOURCE):
    """Build the dataset matching `arch` (regiondiff -> region cache; tempfuse -> M1 patch grids)."""
    if arch == "tempfuse":
        return M4PatchDataset(features_root, m3_labels_dir, m4_labels_dir, pairs, split,
                              augment, same_view_only, box_source)
    return M4Dataset(region_cache, m3_labels_dir, m4_labels_dir, pairs, split, augment, same_view_only)
