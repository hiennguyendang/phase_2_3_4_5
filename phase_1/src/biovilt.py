"""FROZEN BioViL-T image encoder: CXR jpg -> [197, C] float16 (the M1 cache row format).

Loads the BioViL-T image encoder ONCE (eval, no grad) and exposes:
  - load_image(): photometric-only PIL->tensor (ToTensor + expand-to-3-channels). BioViL-T uses
    NO ImageNet normalization, so we don't add one. By default we do NO geometric resize/crop —
    the 448x448 input frame (which the m3 boxes were rescaled into) is preserved exactly.
  - encode_batch(): forward [B,3,448,448] -> [B, 197, C] float16, with
        row 0       = projected_global_embedding
        rows 1..196 = projected_patch_embeddings [C,14,14] flattened to [196,C], index = y*14+x
        (`patch.flatten(2).transpose(1,2)` — matches pooling.py's reshape(b, r, gh*gw)).

ALIGNMENT (risk #1): BioViL-T's *default* inference transform is Resize(512)+CenterCrop(448),
which RE-FRAMES the image and would desync it from the boxes. We skip it by default
(TRANSFORM_MODE="stretch448"): the mimic-cxr-448 jpgs are already a straight stretch to
448x448, so feeding them as-is keeps the box<->grid mapping exact (cell = 448/14 = 32 px).
scripts/3-verify_features.py reproduces a known reference .pt to confirm the choice empirically.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

import config


# ---- model -------------------------------------------------------------------
def load_encoder(device: str | torch.device = "cpu"):
    """Load the FROZEN BioViL-T image encoder (eval, requires_grad=False)."""
    try:
        from health_multimodal.image import get_biovil_t_image_encoder
    except ImportError:  # older/newer layout keeps it under the pretrained submodule
        from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder

    model = get_biovil_t_image_encoder()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---- preprocessing -----------------------------------------------------------
def load_image(path: str | Path, res: int = config.INPUT_RES,
               mode: str = config.TRANSFORM_MODE) -> torch.Tensor:
    """Load one CXR jpg -> [3, res, res] float32 in [0,1] (BioViL-T's expected input).

    "stretch448" (default): feed the image at `res` as-is — the mimic-cxr-448 jpgs are already
        a stretch to 448x448, the exact frame the boxes live in (no geometric re-framing).
    "resize_crop": reproduce BioViL-T's default Resize(res*512/448) + CenterCrop(res). Only for
        matching a cache that was built that way; it does NOT align with the stretched boxes.
    """
    img = Image.open(path).convert("L")               # grayscale single channel
    if mode == "resize_crop":
        resize = round(res * 512 / 448)               # BioViL-T's 512 for a 448 crop
        img = TF.resize(img, [resize], antialias=True)
        img = TF.center_crop(img, [res, res])
    elif img.size != (res, res):                      # "stretch448": only resize if not already res
        img = img.resize((res, res), Image.BICUBIC)
    t = TF.to_tensor(img)                             # [1,res,res] float in [0,1]
    return t.expand(3, -1, -1).contiguous()           # [3,res,res] (== BioViL-T's ExpandChannels)


# ---- forward -----------------------------------------------------------------
def _pick_features(out) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull (global [B,C], patch [B,C,gh,gw]) from an ImageModelOutput per config.FEATURE_SOURCE.

    "backbone" (default) uses the 512-d pre-projection features (img_embedding + patch_embeddings)
    to match the existing reference cache; "projected" uses the 128-d joint-space head."""
    if config.FEATURE_SOURCE == "projected":
        glob = getattr(out, "projected_global_embedding", None)
        patch = getattr(out, "projected_patch_embeddings", None)
        names = "projected_global_embedding/projected_patch_embeddings"
    else:  # "backbone" — 512-d, what the collaborator reference was built from
        glob = getattr(out, "img_embedding", None)
        patch = getattr(out, "patch_embeddings", None)
        names = "img_embedding/patch_embeddings"
    if glob is None or patch is None:
        raise RuntimeError(
            f"BioViL-T output is missing {names} (FEATURE_SOURCE={config.FEATURE_SOURCE}) "
            "— check the installed health_multimodal version."
        )
    return glob, patch


@torch.no_grad()
def encode_batch(model, batch: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """[B,3,res,res] -> [B, 1+gh*gw, C] float16 (CPU). Row 0 global, rows 1.. patch grid."""
    out = model(batch.to(device, non_blocking=True))
    glob, patch = _pick_features(out)                 # [B,C], [B,C,gh,gw]
    if patch.dim() != 4:
        raise RuntimeError(f"expected 4-D patch embeddings [B,C,gh,gw], got {tuple(patch.shape)}")
    grid = patch.flatten(2).transpose(1, 2)           # [B, gh*gw, C], token index = y*gw+x
    feats = torch.cat([glob.unsqueeze(1), grid], dim=1)   # [B, 1+gh*gw, C]
    return feats.to(torch.float16).cpu()


def detect_feat_dim(model, device: str | torch.device) -> int:
    """Forward a single zero image to learn C (call once before the big loop)."""
    dummy = torch.zeros(1, 3, config.INPUT_RES, config.INPUT_RES)
    return int(encode_batch(model, dummy, device).shape[-1])
