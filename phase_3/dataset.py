"""M3 dataset: join cached features + per-region labels (from labels.py) by image_id.

Keeps only rows whose split matches AND whose features exist in the FeatureStore.
Arrays are memory-mapped; manifest row i lines up with array row i.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from features import FeatureStore


class M3Dataset(Dataset):
    def __init__(self, labels_dir: Path, features_root: Path, split: str | None = None,
                 box_source: str | None = None):
        labels_dir = Path(labels_dir)
        self.rc = np.load(labels_dir / "region_concepts.npy", mmap_mode="r")
        self.rx = np.load(labels_dir / "region_chexpert.npy", mmap_mode="r")
        self.ic = np.load(labels_dir / "image_chexpert.npy", mmap_mode="r")

        # bbox source: "detector" (YOLO, default — same source train & launch) | "gt" (silver oracle)
        src = (box_source or config.BOX_SOURCE).lower()
        if src == "detector":
            bx_f, pm_f = labels_dir / "boxes_det.npy", labels_dir / "present_mask_det.npy"
            if not bx_f.exists():
                raise FileNotFoundError(
                    f"BOX_SOURCE='detector' but {bx_f.name} is missing in {labels_dir}. "
                    "Run phase_2/infer_yolo.py then phase_3/boxes_from_pred.py, or set "
                    "config.BOX_SOURCE='gt' / pass box_source='gt'.")
        else:
            bx_f, pm_f = labels_dir / "boxes.npy", labels_dir / "present_mask.npy"
        self.box_source = src
        self.bx = np.load(bx_f, mmap_mode="r")
        self.pm = np.load(pm_f, mmap_mode="r")
        self.store = FeatureStore(features_root)

        manifest = [json.loads(l) for l in open(labels_dir / "manifest.jsonl", encoding="utf-8")]
        self.rows: list[tuple[int, str]] = []
        for i, m in enumerate(manifest):
            if not m.get("ok", True):
                continue
            if split is not None and str(m.get("split", "")).lower() != split:
                continue
            iid = m["image_id"]
            if self.store.has(iid):
                self.rows.append((i, iid))

    def __len__(self) -> int:
        return len(self.rows)

    def feat_dim(self) -> int:
        return self.store.detect_dim()

    def __getitem__(self, k: int) -> dict:
        i, iid = self.rows[k]
        grid, glob = self.store.load(iid)                       # [196,C], [C]
        return {
            "image_id": iid,
            "grid": grid,
            "global": glob,
            "region_concepts": torch.from_numpy(self.rc[i].astype(np.int64)),   # [29,69]
            "region_chexpert": torch.from_numpy(self.rx[i].astype(np.int64)),   # [29,14]
            "image_chexpert": torch.from_numpy(self.ic[i].astype(np.int64)),    # [14]
            "present_mask": torch.from_numpy(self.pm[i].astype(np.float32)),    # [29]
            "boxes": torch.from_numpy(self.bx[i].astype(np.int64)),             # [29,4]
        }


def collate(batch: list[dict]) -> dict:
    out = {"image_id": [b["image_id"] for b in batch]}
    for k in ("grid", "global", "region_concepts", "region_chexpert",
              "image_chexpert", "present_mask", "boxes"):
        out[k] = torch.stack([b[k] for b in batch])
    return out
