"""M3 C-KAN: BioViL-T grid -> attention-pool 29 regions -> concept (69) + disease (14).

Three configurable directions (config.HEAD_MODE) — spec 3.3 letters:
  "A" Direct:  region_feat -------------------------------> 14   (faithful "where", accuracy ceiling)
  "B" CBM:     region_feat -> 69 concept -> (concept only) -> 14   (pure bottleneck, "why"-faithful path)
  "C" Hybrid:  region_feat -> 69; [feat (+leak-dropout) ⊕ 69] -> 14   (accuracy + CBM-leakage risk)

Pipeline: pool 196->29 (bbox-masked) -> neck (OFF by default -> keep 512) -> heads.
Image-level 14 = region-aggregate fused with a GlobalHead via a learned per-disease gate (spec 3.5).

Outputs per forward:
  concept_logits        [B, 29, 69]   per-region concept (None for mode A)
  region_disease_logits [B, 29, 14]   per-region CheXpert
  image_disease_logits  [B, 14]       image-level CheXpert (region<->global gate fusion)
  region_feats          [B, 29, 512]  region features (128 if NECK_DIM set)  (M4 hook)
  region_attn           [B, 29, 196]  attention-pool weights      (M5 grounding "where")
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

        # neck 512 -> 128 (spec 3.2): compact, normalized region feature shared with M4
        if config.NECK_DIM:
            self.neck = nn.Sequential(
                nn.Linear(feat_dim, config.NECK_DIM), nn.LayerNorm(config.NECK_DIM), nn.GELU())
            rdim = config.NECK_DIM
        else:
            self.neck = nn.Identity()
            rdim = feat_dim
        self.region_dim = rdim

        if mode != "A":
            self.concept_head = make_head(rdim, C.NUM_CONCEPTS)
        if mode == "A":
            self.disease_head = make_head(rdim, C.NUM_CHEX)
        elif mode == "B":
            self.disease_head = make_head(C.NUM_CONCEPTS, C.NUM_CHEX)
        elif mode == "C":
            self.disease_head = make_head(rdim + C.NUM_CONCEPTS, C.NUM_CHEX)
            self.feat_leak = nn.Dropout(config.FEATURE_LEAK_DROPOUT)
        else:
            raise ValueError(f"unknown HEAD_MODE: {mode}")

        # region -> image attention aggregation
        self.agg_score = nn.Linear(rdim, 1)
        self.region_agg = config.REGION_AGG

        # global branch (spec 3.5): relational findings + learned per-disease gate
        self.use_global_head = config.USE_GLOBAL_HEAD
        if self.use_global_head:
            self.global_head = make_head(feat_dim, C.NUM_CHEX)
            self.gate = nn.Linear(feat_dim, C.NUM_CHEX)

    def forward(self, grid: torch.Tensor, global_vec: torch.Tensor | None = None,
                present_mask: torch.Tensor | None = None,
                boxes: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pooled, alpha = self.pool(grid, global_vec, boxes)   # [B,R,C], [B,29,196]
        feats29 = self.neck(pooled[:, :C.NUM_REGIONS, :])    # [B,29,rdim]  supervised slots

        concept_logits = None
        if self.mode == "A":
            region_disease = self.disease_head(feats29)
        else:
            concept_logits = self.concept_head(feats29)      # [B,29,69]
            if self.mode == "B":
                region_disease = self.disease_head(torch.sigmoid(concept_logits))
            else:  # C / hybrid
                feat_in = self.feat_leak(feats29)
                region_disease = self.disease_head(
                    torch.cat([feat_in, concept_logits], dim=-1))

        image_local = self._aggregate(feats29, region_disease, present_mask)
        if self.use_global_head and global_vec is not None:
            image_global = self.global_head(global_vec)      # [B,14]
            g = torch.sigmoid(self.gate(global_vec))         # [B,14] per-disease gate
            image_disease = g * image_global + (1.0 - g) * image_local
        else:
            image_disease = image_local

        return {
            "concept_logits": concept_logits,                # [B,29,69] or None
            "region_disease_logits": region_disease,         # [B,29,14]
            "image_disease_logits": image_disease,           # [B,14]
            "region_feats": feats29,                         # [B,29,128]  (M4 hook)
            "region_attn": alpha,                            # [B,29,196]  (M5 grounding)
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
        score = self.agg_score(feats).squeeze(-1)            # [B,R]
        score = score.masked_fill(~m, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(-1)        # [B,R,1]
        w = torch.nan_to_num(w)                              # rows with no present region -> 0
        return (w * region_disease).sum(dim=1)               # [B,14]


def build_model(feat_dim: int, mode: str = config.HEAD_MODE) -> CKAN:
    return CKAN(feat_dim, mode)
