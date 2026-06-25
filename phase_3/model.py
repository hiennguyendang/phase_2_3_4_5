"""M3 C-KAN: BioViL-T grid -> attention-pool 29 regions -> concept (69) + disease (14).

Three configurable directions (config.HEAD_MODE):
  "C"  region_feat ---------------------------------> 14         (direct baseline / ceiling)
  "A"  region_feat -> 69 concept -> (concept only) -> 14         (pure concept bottleneck, faithful)
  "B"  region_feat -> 69 concept; [feat (+leak-dropout) ⊕ 69] -> 14   (hybrid: accuracy + partial explain)

Outputs per forward:
  concept_logits        [B, 29, 69]   per-region concept (None for mode C)
  region_disease_logits [B, 29, 14]   per-region CheXpert
  image_disease_logits  [B, 14]       image-level CheXpert (masked attention over regions)
M4 hook: `region_feats` is returned so the temporal head can consume it later.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
import constants as C
from heads import make_head
from pooling import RegionAttentionPool


class CKAN(nn.Module):
    def __init__(self, feat_dim: int, mode: str = config.HEAD_MODE):
        super().__init__()
        self.mode = mode
        self.pool = RegionAttentionPool(feat_dim)

        if mode != "C":
            self.concept_head = make_head(feat_dim, C.NUM_CONCEPTS)
        if mode == "C":
            self.disease_head = make_head(feat_dim, C.NUM_CHEX)
        elif mode == "A":
            self.disease_head = make_head(C.NUM_CONCEPTS, C.NUM_CHEX)
        elif mode == "B":
            self.disease_head = make_head(feat_dim + C.NUM_CONCEPTS, C.NUM_CHEX)
            self.feat_leak = nn.Dropout(config.FEATURE_LEAK_DROPOUT)
        else:
            raise ValueError(f"unknown HEAD_MODE: {mode}")

        # region -> image attention aggregation
        self.agg_score = nn.Linear(feat_dim, 1)
        self.region_agg = config.REGION_AGG

    def forward(self, grid: torch.Tensor, global_vec: torch.Tensor | None = None,
                present_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        region_feats = self.pool(grid, global_vec)        # [B, R, C]  R = 29 (+1 if global)
        feats29 = region_feats[:, :C.NUM_REGIONS, :]      # supervised region slots

        concept_logits = None
        if self.mode == "C":
            region_disease = self.disease_head(feats29)
        else:
            concept_logits = self.concept_head(feats29)   # [B,29,69]
            if self.mode == "A":
                region_disease = self.disease_head(torch.sigmoid(concept_logits))
            else:  # B
                feat_in = self.feat_leak(feats29)
                region_disease = self.disease_head(
                    torch.cat([feat_in, concept_logits], dim=-1))

        image_disease = self._aggregate(feats29, region_disease, present_mask)
        return {
            "concept_logits": concept_logits,             # [B,29,69] or None
            "region_disease_logits": region_disease,      # [B,29,14]
            "image_disease_logits": image_disease,        # [B,14]
            "region_feats": feats29,                      # [B,29,C]  (M4 hook)
        }

    def _aggregate(self, feats: torch.Tensor, region_disease: torch.Tensor,
                   present_mask: torch.Tensor | None) -> torch.Tensor:
        b, r, _ = region_disease.shape
        if present_mask is None:
            present_mask = torch.ones(b, r, device=region_disease.device)
        m = present_mask.bool()
        if self.region_agg == "max":
            masked = region_disease.masked_fill(~m.unsqueeze(-1), float("-inf"))
            out = masked.max(dim=1).values
            return torch.nan_to_num(out, neginf=0.0)
        if self.region_agg == "mean":
            w = m.float().unsqueeze(-1)
            return (region_disease * w).sum(1) / w.sum(1).clamp_min(1.0)
        # attention (default): learn a weight per present region, share across diseases
        score = self.agg_score(feats).squeeze(-1)         # [B,R]
        score = score.masked_fill(~m, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(-1)     # [B,R,1]
        w = torch.nan_to_num(w)                           # rows with no present region -> 0
        return (w * region_disease).sum(dim=1)            # [B,14]


def build_model(feat_dim: int, mode: str = config.HEAD_MODE) -> CKAN:
    return CKAN(feat_dim, mode)
