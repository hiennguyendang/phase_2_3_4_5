"""Loader for precomputed BioViL-T feature grids (the M3 input).

>>> M1 (BioViL-T feature extraction) is implemented & run SEPARATELY (collaborator).
    This file is the *contract*: M1 must write features in the format below, or adjust
    this loader to match. phase_3 never runs the encoder — it only loads the cache.

Expected cache format:
  one  <image_id>.npy  per image, float16, shape [1 + 196, C]
    row 0      = projected_global_embedding   (BioViL-T's own global vector)
    rows 1..196 = projected_patch_embeddings flattened from [C,14,14] -> [196, C]
  C is auto-detected (BioViL-T joint_feature_size, typically 512).

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
    """Maps image_id -> feature tensors. Looks for <root>/<image_id>.npy (recursively
    indexed once), so the cache can be flat or mirrored."""

    def __init__(self, root: Path, suffix: str = ".npy"):
        self.root = Path(root)
        self.suffix = suffix
        self._index: dict[str, Path] | None = None
        self.feat_dim: int | None = None

    def _build_index(self) -> dict[str, Path]:
        idx: dict[str, Path] = {}
        for p in self.root.rglob(f"*{self.suffix}"):
            idx.setdefault(p.stem, p)
        if not idx:
            raise FileNotFoundError(f"no '*{self.suffix}' features under {self.root}")
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
        arr = np.load(path)                       # [197, C] or [196, C]
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


if __name__ == "__main__":  # tiny self-test on a synthetic feature file
    import tempfile

    d = Path(tempfile.mkdtemp())
    np.save(d / "MIMIC_p1_s1_abc.npy", np.random.randn(197, 512).astype(np.float16))
    fs = FeatureStore(d)
    g, gl = fs.load("MIMIC_p1_s1_abc")
    print("grid", tuple(g.shape), "global", tuple(gl.shape), "dim", fs.detect_dim())
