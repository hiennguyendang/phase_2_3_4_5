"""MS-CXR-T helpers for external temporal evaluation and small adapter experiments."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config
import constants as C
from dataset import PatchStore, RegionCache, _boxes_by_image, _present_by_image


FINDINGS = ("consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax")
FINDING_TO_CHEX = {
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pleural_effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
}
FINDING_CHEX_INDICES = [C.CHEX_INDEX[FINDING_TO_CHEX[f]] for f in FINDINGS]
LABEL_TO_PROG = {
    "stable": 0,
    "improving": 1,
    "improved": 1,
    "worsening": 2,
    "worsened": 2,
}


def mimic_image_id(dicom_path: str, *, subject_id: str | int | None = None,
                   study_id: str | int | None = None) -> str:
    """Convert MS-CXR-T's pXX/pSUBJECT/sSTUDY/dicom path into this repo's image_id."""
    parts = [p.strip() for p in str(dicom_path).replace("\\", "/").split("/") if p.strip()]
    if not parts:
        raise ValueError("empty dicom path")
    dicom = parts[-1]
    subjects = [p for p in parts if p.startswith("p") and p[1:].isdigit()]
    studies = [p for p in parts if p.startswith("s") and p[1:].isdigit()]
    subject = max(subjects, key=len) if subjects else f"p{subject_id}"
    study = max(studies, key=len) if studies else f"s{study_id}"
    return f"MIMIC_{subject}_{study}_{dicom}"


def split_for_subject(subject_id: str | int, train_pct: int = 80, val_pct: int = 10) -> str:
    h = hashlib.md5(str(subject_id).encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def read_mscxrt_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            target = np.full(len(FINDINGS), C.UNKNOWN, dtype=np.int64)
            qualities: dict[str, str] = {}
            for j, finding in enumerate(FINDINGS):
                val = str(raw.get(f"{finding}_progression", "")).strip().lower()
                if val in LABEL_TO_PROG:
                    target[j] = LABEL_TO_PROG[val]
                qualities[finding] = str(raw.get(f"{finding}_label_quality", "")).strip()
            sid = raw.get("subject_id", "")
            row = {
                "image_id": mimic_image_id(raw.get("dicom_id", ""),
                                           subject_id=sid, study_id=raw.get("study_id")),
                "prior_image_id": mimic_image_id(raw.get("previous_dicom_id", ""),
                                                 subject_id=sid,
                                                 study_id=raw.get("previous_study_id")),
                "subject_id": str(sid),
                "study_id": str(raw.get("study_id", "")),
                "previous_study_id": str(raw.get("previous_study_id", "")),
                "target": target,
                "qualities": qualities,
            }
            row["split"] = split_for_subject(row["subject_id"])
            rows.append(row)
    return rows


class MSCXRTDataset(Dataset):
    """Serves current/prior tensors plus five image-level temporal labels from MS-CXR-T."""

    def __init__(self, csv_path, *, arch: str, m3_labels_dir, region_cache=None, features_root=None,
                 split: str = "all", box_source: str = config.BOX_SOURCE,
                 tempfuse_input_mode: str = config.TEMPFUSE_INPUT_MODE):
        self.arch = arch
        self.tempfuse_input_mode = tempfuse_input_mode
        self.present = _present_by_image(Path(m3_labels_dir))
        self.skipped = {
            "split": 0, "no_label": 0, "no_feat": 0, "no_present": 0, "no_box": 0,
        }
        self.store = None
        self.cache = None
        self.logit_cache = None
        self.boxes = None
        self.box_row = None

        if arch == "tempfuse":
            self.store = features_root if isinstance(features_root, PatchStore) else PatchStore(features_root)
            self.boxes, self.box_row = _boxes_by_image(Path(m3_labels_dir), box_source)
            if tempfuse_input_mode != "feat":
                if region_cache is None:
                    raise ValueError("tempfuse_input_mode != 'feat' requires --region-cache")
                self.logit_cache = region_cache if isinstance(region_cache, RegionCache) else RegionCache(region_cache)
        elif arch == "regiondiff":
            self.cache = region_cache if isinstance(region_cache, RegionCache) else RegionCache(region_cache)
        else:
            raise ValueError(f"unknown arch: {arch}")

        self.rows = []
        for row in read_mscxrt_rows(Path(csv_path)):
            if split != "all" and row["split"] != split:
                self.skipped["split"] += 1
                continue
            if not (row["target"] != C.UNKNOWN).any():
                self.skipped["no_label"] += 1
                continue
            cid, pid = row["image_id"], row["prior_image_id"]
            if not self._has_features(cid, pid):
                self.skipped["no_feat"] += 1
                continue
            if cid not in self.present or pid not in self.present:
                self.skipped["no_present"] += 1
                continue
            if self.arch == "tempfuse" and cid not in self.box_row:
                self.skipped["no_box"] += 1
                continue
            self.rows.append(row)

        if self.arch == "tempfuse":
            self.feat_dim = self.store.detect_dim()
        else:
            self.feat_dim = self.cache.detect_dim()

    def _has_features(self, cid: str, pid: str) -> bool:
        if self.arch == "regiondiff":
            return self.cache.has(cid) and self.cache.has(pid)
        ok = self.store.has(cid) and self.store.has(pid)
        if self.logit_cache is not None:
            ok = ok and self.logit_cache.has(cid) and self.logit_cache.has(pid)
        return ok

    def __len__(self) -> int:
        return len(self.rows)

    def class_counts(self) -> np.ndarray:
        if not self.rows:
            return np.zeros(C.NUM_PROG, dtype=np.int64)
        tgt = np.stack([r["target"] for r in self.rows])
        return np.array([(tgt == k).sum() for k in range(C.NUM_PROG)], dtype=np.int64)

    def _region_mask(self, cid: str, pid: str) -> torch.Tensor:
        rm = self.present[cid].copy()
        if config.REQUIRE_PRIOR_PRESENT:
            rm = rm * self.present[pid]
        return torch.from_numpy(rm.astype(np.float32))

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        cid, pid = row["image_id"], row["prior_image_id"]
        item = {
            "image_id": cid,
            "prior_image_id": pid,
            "subject_id": row["subject_id"],
            "target_mscxrt": torch.from_numpy(row["target"].copy()),
            "region_mask": self._region_mask(cid, pid),
        }
        if self.arch == "regiondiff":
            fc, lc = self.cache.load(cid)
            fp, lp = self.cache.load(pid)
            item.update({"feat_curr": torch.from_numpy(fc), "logit_curr": torch.from_numpy(lc),
                         "feat_prior": torch.from_numpy(fp), "logit_prior": torch.from_numpy(lp)})
        else:
            box = np.asarray(self.boxes[self.box_row[cid]], dtype=np.float32)
            item.update({"patch_curr": torch.from_numpy(self.store.load(cid)),
                         "patch_prior": torch.from_numpy(self.store.load(pid)),
                         "box_curr": torch.from_numpy(box)})
            if self.logit_cache is not None:
                _, lc = self.logit_cache.load(cid)
                _, lp = self.logit_cache.load(pid)
                item.update({"logit_curr": torch.from_numpy(lc),
                             "logit_prior": torch.from_numpy(lp)})
        return item


def aggregate_mscxrt_probs(logits: torch.Tensor, region_mask: torch.Tensor,
                           agg: str = "mean") -> torch.Tensor:
    """Collapse [B,R,14,3] region logits to [B,5,3] image-level finding probabilities."""
    sub = logits[:, :, FINDING_CHEX_INDICES, :]             # [B,R,5,3]
    mask = region_mask.bool()
    valid = mask[:, :, None, None]
    denom = mask.sum(dim=1).clamp_min(1).to(sub.dtype).view(-1, 1, 1)
    if agg == "mean":
        probs = F.softmax(sub, dim=-1)
        return (probs * valid.to(probs.dtype)).sum(dim=1) / denom
    if agg == "max":
        probs = F.softmax(sub, dim=-1).masked_fill(~valid, -1.0)
        return probs.amax(dim=1).clamp_min(0.0)
    if agg == "lse":
        masked = sub.masked_fill(~valid, float("-inf"))
        pooled = torch.logsumexp(masked, dim=1) - denom.log()
        return F.softmax(pooled, dim=-1)
    raise ValueError(f"unknown agg: {agg} (choose mean, max, or lse)")
