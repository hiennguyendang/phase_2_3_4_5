"""M5 core: assemble a faithful report from M3/M4 prediction rows (spec 5, tiers 1-5).

NO diagnosis is generated here — every sentence is a readout of an M3/M4 table cell, and each
finding carries a provenance pointer back to that cell. The temporal guard (tier 4) is structural:
a progression clause can ONLY be emitted when an M4 cell exists for this image; with no prior there
is no code path that produces temporal language.
"""

from __future__ import annotations

import math

import config
import constants as C


# ---- tier 3: temperature scaling + thresholds --------------------------------
def apply_temperature(p: float, t: float = None) -> float:
    t = config.TEMPERATURE if t is None else t
    if t == 1.0:
        return p
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p)) / t
    return 1.0 / (1.0 + math.exp(-logit))


def calibrate_prob(p: float, disease: str, temps: dict | None) -> float:
    """Per-class temperature if available (from calibrate.py), else the global TEMPERATURE."""
    t = (temps or {}).get(disease, config.TEMPERATURE)
    return apply_temperature(p, t)


def status_of(p: float) -> str:
    if p >= config.TAU_ASSERT:
        return "assert"
    if p >= config.TAU_UNCERTAIN:
        return "hedge"
    if p >= config.TAU_ABSTAIN:
        return "abstain"           # "cannot be excluded" — defer to the radiologist
    return "omit"


# ---- tier 2: grounding "where" ----------------------------------------------
def ground(m3rec: dict, disease: str) -> tuple[str | None, list[dict]]:
    """-> (lead_region, [{region, prob} sorted desc]). Reads M3's per-region disease probs."""
    hits = []
    for region, entry in (m3rec.get("regions") or {}).items():
        prob = (entry.get("disease") or {}).get(disease)
        if prob is not None:
            hits.append({"region": region, "prob": float(prob)})
    hits.sort(key=lambda h: h["prob"], reverse=True)
    lead = hits[0]["region"] if hits and hits[0]["prob"] >= config.TAU_REGION else None
    return lead, hits


def region_cells(m3rec: dict, region: str | None) -> list:
    """The attention-pool 'where' cells [[row,col,weight],...] for a region (M3 infer --topk-cells).
    Faithful intra-region grounding (spec 5.2, 'tín hiệu lấy từ đâu'); [] if not dumped."""
    if region is None:
        return []
    return ((m3rec.get("regions") or {}).get(region) or {}).get("cells", []) or []


# ---- coverage map: a status for every one of the 29 regions (spec 5.x) -------
def coverage_map(m3rec: dict) -> dict:
    """region -> 'abnormal' | 'normal' | 'not_assessable'. Turns silence into a verifiable claim:
    a present region with no finding is asserted normal; an absent region is flagged not-assessable.
    (M3 infer only dumps region diseases >0.5, so 'uncertain' is not separable here.)"""
    present = m3rec.get("regions") or {}
    out = {}
    for region in C.REGION_NAMES:
        entry = present.get(region)
        if entry is None:
            out[region] = "not_assessable"
        elif entry.get("disease"):
            out[region] = "abnormal"
        else:
            out[region] = "normal"
    return out


# ---- tier 4: temporal guard (structural) ------------------------------------
def temporal_of(m4rec: dict | None, disease: str, lead_region: str | None) -> dict | None:
    """A progression clause exists ONLY if there is an M4 cell. No M4 row (no prior) -> None."""
    if m4rec is None:
        return None
    regions = m4rec.get("regions") or {}
    # prefer the lead region; else take the highest-prob change cell for this disease anywhere
    candidates = []
    for region, cells in regions.items():
        cell = (cells or {}).get(disease)
        if cell:
            candidates.append((region, cell))
    if not candidates:
        return None
    if lead_region is not None:
        for region, cell in candidates:
            if region == lead_region:
                best = (region, cell); break
        else:
            best = max(candidates, key=lambda rc: rc[1].get("prob", 0))
    else:
        best = max(candidates, key=lambda rc: rc[1].get("prob", 0))
    region, cell = best
    prog, prob = cell.get("prog", "stable"), float(cell.get("prob", 0))
    if prog == "stable" or prob < config.TAU_PROG:
        return None                                  # don't speak weak/stable change
    return {"prog": prog, "prob": prob, "region": region}


# ---- tier 1: structured core -------------------------------------------------
_PLURAL = {"Support Devices"}


def _finding_text(disease: str, status: str, lead_region: str | None, temporal: dict | None) -> str:
    phrase = C.DISEASE_PHRASE.get(disease, disease.lower())
    verb = "are" if disease in _PLURAL else "is"
    loc = f" in the {lead_region}" if lead_region else ""
    # abstain is too uncertain to attach a temporal claim
    tmp = f", {C.PROG_PHRASE[temporal['prog']]} compared to the prior" if (temporal and status != "abstain") else ""
    if status == "assert":
        s = f"{phrase} {verb} present{loc}{tmp}."
    elif status == "hedge":
        s = f"there may be {phrase}{loc}{tmp}."
    else:  # abstain
        s = f"{phrase} cannot be excluded{loc}."
    return s[0].upper() + s[1:]


def assemble_image(m3rec: dict, m4rec: dict | None, temps: dict | None = None) -> dict:
    image_disease = m3rec.get("image_disease") or {}
    findings = []
    for disease in C.CHEX_NAMES:
        if disease == C.NO_FINDING:
            continue
        p = calibrate_prob(float(image_disease.get(disease, 0.0)), disease, temps)
        st = status_of(p)
        if st == "omit":
            continue
        lead, region_hits = ground(m3rec, disease)
        temporal = temporal_of(m4rec, disease, lead) if st != "abstain" else None
        findings.append({
            "disease": disease,
            "status": st,
            "prob": round(p, 4),
            "lead_region": lead,
            "regions": region_hits,
            "temporal": temporal,
            "text": _finding_text(disease, st, lead, temporal),
            "provenance": {                          # pointer back to the source cells
                "m3_image_prob": round(p, 4),
                "m3_lead_region": lead,
                "m3_region_probs": {h["region"]: round(h["prob"], 4) for h in region_hits},
                "m3_cells": region_cells(m3rec, lead),   # attention-pool "where" (tier 2)
                "m4": temporal,
            },
        })
    has_prior = m4rec is not None
    normal = len(findings) == 0
    return {
        "image_id": m3rec.get("image_id"),
        "prior_image_id": (m4rec or {}).get("prior_image_id"),
        "has_prior": has_prior,
        "normal": normal,
        "findings": findings,
        "coverage_map": coverage_map(m3rec),         # 29-region status (spec 5.x)
    }


# ---- tier 5: realize (template, faithful default) ---------------------------
def realize_template(report: dict) -> str:
    if report["normal"]:
        return "No acute cardiopulmonary abnormality."
    # asserts first, then hedges, then abstains; stable order within by disease index
    order = {"assert": 0, "hedge": 1, "abstain": 2}
    findings = sorted(report["findings"], key=lambda f: (order[f["status"]], C.CHEX_NAMES.index(f["disease"])))
    return " ".join(f["text"] for f in findings)
