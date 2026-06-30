"""Loader for precomputed BioViL-T feature grids (the M3 input).

>>> M1 (BioViL-T feature extraction) is implemented & run SEPARATELY (collaborator).
    This file is the *contract*: M1 must write features in the format below, or adjust
    this loader to match. phase_3 never runs the encoder — it only loads the cache.

Expected cache format:
  one  <image_id>.npy  OR  <image_id>.pt  per image, float16, shape [1 + 196, C]
    row 0      = projected_global_embedding   (BioViL-T's own global vector)
    rows 1..196 = projected_patch_embeddings flattened from [C,14,14] -> [196, C]
  C is auto-detected (BioViL-T joint_feature_size, typically 512).

Accepts BOTH numpy (.npy via np.load) and torch (.pt via torch.load) dumps — the BioViL-T
collaborator's extractor writes .pt tensors, so we load those natively rather than forcing a
re-export. A .pt holding a dict (e.g. {"features": tensor}) is tolerated (first tensor value used).
Tolerant of [196, C] files too (no global row) -> global = mean(grid).
phase_3 model code only ever calls this loader, so the on-disk format can change
in ONE place without touching the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import config

GRID_TOKENS = config.GRID_TOKENS  # 196


class FeatureStore:
    """Maps image_id -> feature tensors. Looks for <root>/<image_id>.{npy,pt} (recursively
    indexed once), so the cache can be flat or mirrored and either on-disk format."""

    def __init__(self, root: Path, suffixes: tuple[str, ...] | str = (".npy", ".pt")):
        self.root = Path(root)
        self.suffixes = (suffixes,) if isinstance(suffixes, str) else tuple(suffixes)
        self._index: dict[str, Path] | None = None
        self.feat_dim: int | None = None

    def _build_index(self) -> dict[str, Path]:
        idx: dict[str, Path] = {}
        for suf in self.suffixes:                      # .npy preferred (listed first) on stem clash
            for p in self.root.rglob(f"*{suf}"):
                idx.setdefault(p.stem, p)
        if not idx:
            raise FileNotFoundError(f"no '{', '.join(self.suffixes)}' features under {self.root}")
        return idx

    @property
    def index(self) -> dict[str, Path]:
        if self._index is None:
            self._index = self._build_index()
        return self._index

    def has(self, image_id: str) -> bool:
        return image_id in self.index

    def load(self, image_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (grid [196, C] float32, global [C] float32)."""
        path = self.index.get(image_id)
        if path is None:
            raise KeyError(f"no features for image_id={image_id}")
        if path.suffix == ".pt":                  # torch dump (collaborator's BioViL-T extractor)
            t = torch.load(path, map_location="cpu")
            if isinstance(t, dict):               # tolerate {"features": tensor} style dumps
                t = next((v for v in t.values() if torch.is_tensor(v)), None)
                if t is None:
                    raise ValueError(f"{path}: .pt dict has no tensor value")
            arr = t.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.load(path)                   # [197, C] or [196, C]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected 2-D feature array, got shape {arr.shape}")
        if arr.shape[0] == GRID_TOKENS + 1:       # has a global row
            glob, grid = arr[0], arr[1:]
        elif arr.shape[0] == GRID_TOKENS:         # grid only
            grid = arr
            glob = grid.mean(axis=0)
        else:
            raise ValueError(f"{path}: rows={arr.shape[0]}, expected {GRID_TOKENS} or {GRID_TOKENS + 1}")
        if self.feat_dim is None:
            self.feat_dim = int(grid.shape[1])
        return torch.from_numpy(grid), torch.from_numpy(glob)

    def detect_dim(self) -> int:
        """Peek one file to learn C (call before building the model)."""
        any_id = next(iter(self.index))
        grid, _ = self.load(any_id)
        return int(grid.shape[1])


if __name__ == "__main__":  # tiny self-test on synthetic .npy and .pt feature files
    import tempfile

    d = Path(tempfile.mkdtemp())
    np.save(d / "MIMIC_p1_s1_abc.npy", np.random.randn(197, 512).astype(np.float16))
    torch.save(torch.randn(197, 512, dtype=torch.float16), d / "MIMIC_p2_s2_def.pt")
    fs = FeatureStore(d)
    for iid in ("MIMIC_p1_s1_abc", "MIMIC_p2_s2_def"):
        g, gl = fs.load(iid)
        print(iid, "grid", tuple(g.shape), "global", tuple(gl.shape))
    print("dim", fs.detect_dim())
