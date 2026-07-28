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


def calibrate_threshold(threshold: float, disease: str, temps: dict | None) -> float:
    """Move a raw-M3 decision threshold onto the same calibrated scale as its probability."""
    t = (temps or {}).get(disease, config.TEMPERATURE)
    return apply_temperature(threshold, t)


def disease_thresholds(disease: str, thresholds: dict | None,
                       level: str = "image",
                       region: str | None = None) -> tuple[float | None, float | None]:
    """Return (absent, present) thresholds for image or region predictions."""
    if thresholds is None:
        return None, None
    if level == "region" and region is not None and "region_by_name" in thresholds:
        scope = (thresholds.get("region_by_name") or {}).get(region, {})
    else:
        scope = thresholds.get(level, thresholds)
    item = scope.get(disease) if isinstance(scope, dict) else None
    if item is None:
        return None, None
    if not isinstance(item, dict):
        return None, float(item)
    absent = item.get("absent_threshold")
    present = item.get("present_threshold", item.get("threshold", item.get("positive_threshold")))
    return (None if absent is None else float(absent),
            None if present is None else float(present))


def threshold_state(p: float, absent_threshold: float | None,
                    present_threshold: float | None) -> str | None:
    """Dual-threshold decision. The middle band is internal unknown/abstain."""
    if present_threshold is not None and p >= present_threshold:
        return "present"
    if absent_threshold is not None and p <= absent_threshold:
        return "absent"
    return "unknown" if absent_threshold is not None or present_threshold is not None else None


def status_of(p: float) -> str:
    if p >= config.TAU_ASSERT:
        return "assert"
    if p >= config.TAU_UNCERTAIN:
        return "hedge"
    if p >= config.TAU_ABSTAIN:
        return "abstain"           # "cannot be excluded" — defer to the radiologist
    return "omit"


# ---- tier 2: region-disease cells -------------------------------------------
def disease_regions(m3rec: dict | None, disease: str, state: str,
                    thresholds: dict | None,
                    concept_gate: dict | None = None) -> list[dict]:
    """Return every confident (region, disease) cell matching the requested state."""
    if m3rec is None:
        return []
    rows = []
    for region, entry in (m3rec.get("regions") or {}).items():
        absent_tau, present_tau = disease_thresholds(
            disease, thresholds, "region", region=region)
        if thresholds is None and absent_tau is None:
            absent_tau = config.TAU_ABSTAIN
        if thresholds is None and present_tau is None:
            present_tau = config.TAU_REGION
        prob = (entry.get("disease") or {}).get(disease)
        if prob is None:
            continue
        prob = float(prob)
        cell_state = threshold_state(prob, absent_tau, present_tau)
        if cell_state != state:
            continue
        rows.append({
            "region": region,
            "state": cell_state,
            "confidence": round(prob if cell_state == "present" else 1.0 - prob, 4),
            "concepts": concept_evidence(m3rec, region, disease, concept_gate=concept_gate)
            if cell_state == "present" else [],
            "cells": region_cells(m3rec, region),
            "bbox": entry.get("bbox"),
        })
    rows.sort(key=lambda x: x["confidence"], reverse=True)
    return rows


def grounding_type(disease: str) -> str:
    """GlobalHead-driven relational findings are whole-image grounded, not fake region-grounded."""
    return "global" if disease in config.GLOBAL_GROUNDING_DISEASES else "regional"


def region_cells(m3rec: dict, region: str | None) -> list:
    """The attention-pool 'where' cells [[row,col,weight],...] for a region (M3 infer --topk-cells).
    Faithful intra-region grounding (spec 5.2, 'tín hiệu lấy từ đâu'); [] if not dumped."""
    if region is None:
        return []
    return ((m3rec.get("regions") or {}).get(region) or {}).get("cells", []) or []


def concept_evidence(m3rec: dict, region: str | None, disease: str,
                     topk: int = 3,
                     concept_gate: dict | None = None) -> list[dict]:
    """Graph-valid concept evidence for one disease-region prediction.

    New M3 output contains disease-conditioned concept lists. Older JSONL remains readable by
    filtering its generic regional concepts through the same concept-disease crosswalk.
    """
    if region is None:
        return []
    # No gate means no report-visible concept evidence. Disease inference is
    # unaffected because M3 already consumed the continuous concept bottleneck.
    if concept_gate is None:
        return []
    entry = ((m3rec.get("regions") or {}).get(region) or {})
    concepts = (entry.get("disease_concepts") or {}).get(disease)
    if concepts is None:
        allowed = C.CONCEPTS_BY_DISEASE.get(disease, set())
        concepts = {name: prob for name, prob in (entry.get("concepts") or {}).items()
                    if name in allowed}
    items = []
    graph_allowed = C.CONCEPTS_BY_DISEASE.get(disease, set())
    for name, value in concepts.items():
        if name not in graph_allowed:
            continue
        if isinstance(value, dict):
            probability = float(value.get("prob", value.get("probability", 0.0)))
            contribution = float(value.get("contribution", probability))
            edge_weight = value.get("edge_weight")
        else:
            probability = float(value)
            contribution = probability
            edge_weight = None
        if "region_by_name" in concept_gate:
            gate_item = (((concept_gate.get("region_by_name") or {}).get(region) or {}).get(name))
            if not isinstance(gate_item, dict) or not gate_item.get("allowed_for_why"):
                continue
            threshold = gate_item.get("present_threshold")
        else:
            threshold = (concept_gate.get("global") or {}).get(name)
        if threshold is None or probability < float(threshold):
            continue
        item = {"concept": name, "prob": probability, "contribution": contribution}
        if edge_weight is not None:
            item["edge_weight"] = float(edge_weight)
        items.append(item)
    items.sort(key=lambda x: (x["contribution"], x["prob"]), reverse=True)
    return items[:topk]


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
        elif any(float(p) >= config.TAU_REGION for p in (entry.get("disease") or {}).values()):
            out[region] = "abnormal"
        else:
            out[region] = "normal"
    return out


# ---- tier 4: temporal guard (structural) ------------------------------------
def _calibrate_temporal_probs(probs: list[float], disease: str,
                              temporal_temps: dict | None) -> list[float]:
    if not temporal_temps or disease not in temporal_temps:
        return [float(x) for x in probs]
    t = max(float(temporal_temps[disease]), 1e-6)
    logits = [math.log(max(float(x), 1e-8)) / t for x in probs]
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps) or 1.0
    return [x / total for x in exps]


def temporal_of(m4rec: dict | None, disease: str,
                temporal_temps: dict | None = None,
                temporal_gates: dict | None = None) -> dict | None:
    """Return a confident three-class disease-level temporal decision."""
    if m4rec is None:
        return None
    disease_readout = (m4rec.get("diseases") or {}).get(disease)
    if disease_readout is not None:
        change = disease_readout.get("change", "stable")
        probs = _calibrate_temporal_probs(disease_readout.get("probs") or [], disease, temporal_temps)
        cls = int(max(range(len(probs)), key=lambda i: probs[i])) if probs else 0
        change = C.PROG_NAMES[cls]
        confidence = float(probs[cls]) if probs else 0.0
        gate = config.TAU_PROG
        if temporal_gates is not None and disease in temporal_gates:
            gate = (temporal_gates[disease].get(change) or {}).get("threshold")
            if gate is None:
                return None
        if confidence < float(gate):
            return None
        return {
            "prog": change,
            "prob": confidence,
            "probs": probs,
            "region": disease_readout.get("lead_region") if change != "stable" else None,
            "aggregation": m4rec.get("aggregation", "logsumexp_region_logits"),
        }

    # Backward-compatible rollup for older inference JSONL. Summing regional
    # class probabilities preserves all three classes instead of treating an
    # omitted change cell as stable.
    regions = m4rec.get("regions") or {}
    candidates: list[tuple[str, dict]] = []
    for region, cells in regions.items():
        cell = (cells or {}).get(disease)
        if cell:
            candidates.append((region, cell))
    if not candidates:
        return None
    scores = [0.0, 0.0, 0.0]
    for _, cell in candidates:
        probs = cell.get("probs")
        if probs is None:
            probs = [0.0, 0.0, 0.0]
            probs[C.PROG_NAMES.index(cell.get("prog", "stable"))] = float(cell.get("prob", 0.0))
        for k in range(3):
            scores[k] += float(probs[k])
    total = sum(scores) or 1.0
    cls = max(range(3), key=lambda k: scores[k])
    change = C.PROG_NAMES[cls]
    lead_region = None
    if change != "stable":
        lead_region = max(
            candidates,
            key=lambda rc: float((rc[1].get("probs") or [0.0, 0.0, 0.0])[cls]),
        )[0]
    probs = _calibrate_temporal_probs([score / total for score in scores], disease, temporal_temps)
    cls = int(max(range(3), key=lambda k: probs[k]))
    change = C.PROG_NAMES[cls]
    confidence = probs[cls]
    gate = config.TAU_PROG
    if temporal_gates is not None and disease in temporal_gates:
        gate = (temporal_gates[disease].get(change) or {}).get("threshold")
        if gate is None:
            return None
    if confidence < float(gate):
        return None
    return {
        "prog": change,
        "prob": confidence,
        "probs": probs,
        "region": lead_region,
        "aggregation": "mean_region_probabilities_legacy",
    }


# ---- tier 1: structured core -------------------------------------------------
_PLURAL = {"Support Devices"}


def _compact_names(names: list[str], limit: int = 3) -> str:
    if len(names) <= limit:
        return ", ".join(names)
    remainder = len(names) - limit
    return f"{', '.join(names[:limit])}, and {remainder} other regions"


def _finding_text(disease: str, status: str, regions: list[dict] | None = None,
                  concepts: list[dict] | None = None) -> str:
    phrase = C.DISEASE_PHRASE.get(disease, disease.lower())
    if status == "absent":
        negative_phrase = phrase
        for article in ("a ", "an ", "another "):
            if negative_phrase.startswith(article):
                negative_phrase = negative_phrase[len(article):]
                break
        return f"No {negative_phrase}."
    verb = "are" if disease in _PLURAL else "is"
    region_names = []
    for row in regions or []:
        name = row.get("region")
        if name and name not in region_names:
            region_names.append(name)
    loc = f" in {_compact_names(region_names)}" if region_names else ""
    evidence_names = []
    for item in concepts or []:
        name = item.get("concept")
        if name and name not in evidence_names:
            evidence_names.append(name)
    evidence = f"; supported by {', '.join(evidence_names[:3])}" if evidence_names else ""
    if status == "assert":
        s = f"{phrase} {verb} present{loc}{evidence}."
    elif status == "hedge":
        s = f"there may be {phrase}{loc}{evidence}."
    else:  # abstain
        s = f"{phrase} cannot be excluded{loc}."
    return s[0].upper() + s[1:]


def _change_text(disease: str, change: str, lead_region: str | None) -> str:
    phrase = C.DISEASE_PHRASE.get(disease, disease.lower())
    loc = f", greatest in the {lead_region}" if lead_region else ""
    be = "are" if disease in _PLURAL else "is"
    have = "have" if disease in _PLURAL else "has"
    if change == "new":
        text = f"new {phrase} since the prior{loc}."
    elif change == "resolved":
        text = f"{phrase} {have} resolved since the prior{loc}."
    elif change == "stable":
        text = f"{phrase} {be} unchanged from the prior."
    else:
        text = f"{phrase} {have} {change} compared to the prior{loc}."
    return text[0].upper() + text[1:]


def assemble_image(m3rec: dict, m4rec: dict | None, temps: dict | None = None,
                   prior_m3rec: dict | None = None,
                   thresholds: dict | None = None,
                   temporal_temps: dict | None = None,
                   temporal_gates: dict | None = None,
                   concept_gate: dict | None = None,
                   include_absent: bool = False) -> dict:
    del include_absent  # binary present/absent output is now the report contract
    image_disease = m3rec.get("image_disease") or {}
    current_findings: list[dict] = []
    classification_rows: list[dict] = []
    interval_changes: list[dict] = []
    disease_states: list[dict] = []
    state_cache: dict[str, dict] = {}

    for disease in C.CHEX_NAMES:
        if disease == C.NO_FINDING:
            continue
        p = calibrate_prob(float(image_disease.get(disease, 0.0)), disease, temps)
        raw_abs_tau, raw_present_tau = disease_thresholds(disease, thresholds, "image")
        if thresholds is None:
            raw_abs_tau = config.TAU_ABSTAIN if raw_abs_tau is None else raw_abs_tau
            raw_present_tau = config.TAU_ASSERT if raw_present_tau is None else raw_present_tau
        abs_tau = calibrate_threshold(raw_abs_tau, disease, temps) if raw_abs_tau is not None else None
        present_tau = calibrate_threshold(raw_present_tau, disease, temps) \
            if raw_present_tau is not None else None
        state = threshold_state(p, abs_tau, present_tau)
        confidence = p if state == "present" else (1.0 - p if state == "absent" else None)
        current_regions = disease_regions(m3rec, disease, state, thresholds, concept_gate) \
            if state in {"present", "absent"} else []

        prior_state = None
        prior_confidence = None
        prior_present_regions: list[dict] = []
        if prior_m3rec is not None:
            prior_raw = float((prior_m3rec.get("image_disease") or {}).get(disease, 0.0))
            prior_p = calibrate_prob(prior_raw, disease, temps)
            prior_state = threshold_state(prior_p, abs_tau, present_tau)
            if prior_state in {"present", "absent"}:
                prior_confidence = prior_p if prior_state == "present" else 1.0 - prior_p
            prior_present_regions = disease_regions(
                prior_m3rec, disease, "present", thresholds, concept_gate)

        current_present_regions = disease_regions(
            m3rec, disease, "present", thresholds, concept_gate)
        state_cache[disease] = {
            "state": state,
            "confidence": confidence,
            "regions": current_regions,
            "present_regions": current_present_regions,
            "prior_state": prior_state,
            "prior_confidence": prior_confidence,
            "prior_present_regions": prior_present_regions,
            "thresholds": {
                "raw_absent": raw_abs_tau,
                "raw_present": raw_present_tau,
                "calibrated_absent": abs_tau,
                "calibrated_present": present_tau,
            },
        }

        if state not in {"present", "absent"} or not current_regions:
            continue
        disease_states.append({"disease": disease, "state": state, "confidence": round(confidence, 4)})
        status = "assert" if state == "present" else "absent"
        concepts = [{"region": row["region"], **concept}
                    for row in current_regions for concept in row["concepts"]]
        finding = {
            "disease": disease,
            "status": status,
            "state": state,
            "confidence": round(confidence, 4),
            "regions": current_regions,
            "concepts": concepts,
            "text": _finding_text(disease, status, current_regions, concepts),
            "provenance": {
                "m3_image_prob": round(p, 4),
                "m3_thresholds": state_cache[disease]["thresholds"],
                "m3_region_cells": current_regions,
                "m3_concepts": concepts,
            },
        }
        current_findings.append(finding)
        for region_row in current_regions:
            classification_rows.append({
                "disease": disease,
                "region": region_row["region"],
                "state": state,
                "evidence": region_row["concepts"] if state == "present" else [],
                "evidence_source": ("predicted_graph_valid_concepts" if state == "present"
                                     else "no_concept_for_absent_state"),
                "confidence": round(confidence, 4),
                "region_score": region_row["confidence"],
                "bbox": region_row.get("bbox"),
                "image_confidence": round(confidence, 4),
            })

    if m4rec is not None and prior_m3rec is not None:
        for disease in C.CHEX_NAMES:
            if disease in C.PROGRESSION_EXCLUDED:
                continue
            cached = state_cache[disease]
            state = cached["state"]
            prior_state = cached["prior_state"]
            current_present = cached["present_regions"]
            prior_present = cached["prior_present_regions"]
            if state != "present" and prior_state != "present":
                continue

            temporal = temporal_of(m4rec, disease, temporal_temps, temporal_gates)
            change = None
            lead_region = None
            lead_source = None
            if state == "present" and prior_state == "absent":
                change = "new"
                lead = max(current_present, key=lambda row: row["confidence"], default=None)
                lead_region = lead["region"] if lead else None
                lead_source = "current_presence_crossing"
                regions = current_present
                confidence = min(cached["confidence"], cached["prior_confidence"])
            elif state == "absent" and prior_state == "present":
                change = "resolved"
                lead = max(prior_present, key=lambda row: row["confidence"], default=None)
                lead_region = lead["region"] if lead else None
                lead_source = "prior_presence_crossing"
                regions = prior_present
                confidence = min(cached["confidence"], cached["prior_confidence"])
            elif temporal is not None:
                change = temporal["prog"]
                lead_region = temporal.get("region") if change != "stable" else None
                lead_source = temporal.get("aggregation") if lead_region else None
                regions = current_present or prior_present
                confidence = float(temporal["prob"])
            else:
                continue

            current_by_name = {row["region"]: row for row in current_present}
            prior_by_name = {row["region"]: row for row in prior_present}
            region_names = []
            for row in regions:
                if row["region"] not in region_names:
                    region_names.append(row["region"])
            region_payload = [{
                "region": name,
                "current_bbox": (current_by_name.get(name) or {}).get("bbox"),
                "prior_bbox": (prior_by_name.get(name) or {}).get("bbox"),
            } for name in region_names]
            lead_current_bbox = (((m3rec.get("regions") or {}).get(lead_region) or {}).get("bbox")
                                 if lead_region else None)
            lead_prior_bbox = (((prior_m3rec.get("regions") or {}).get(lead_region) or {}).get("bbox")
                               if lead_region else None)
            interval_changes.append({
                "disease": disease,
                "change": change,
                "regions": region_payload,
                "lead_region": lead_region,
                "lead_bbox": {
                    "current": lead_current_bbox,
                    "prior": lead_prior_bbox,
                } if lead_region else None,
                "confidence": round(float(confidence), 4),
                "temporal": temporal,
                "text": _change_text(disease, change, lead_region),
                "provenance": {
                    "lead_source": lead_source,
                    "current_state": state,
                    "prior_state": prior_state,
                    "m4": temporal,
                },
            })

    classification_box_map: dict[str, dict] = {}
    for row in classification_rows:
        if row["state"] != "present" or not row.get("bbox"):
            continue
        box = classification_box_map.setdefault(row["region"], {
            "region": row["region"], "bbox": row["bbox"], "diseases": [],
        })
        box["diseases"].append(row["disease"])
    classification_boxes = list(classification_box_map.values())
    progression_current_boxes = [{
        "disease": row["disease"], "change": row["change"],
        "region": row["lead_region"], "bbox": (row.get("lead_bbox") or {}).get("current"),
    } for row in interval_changes if row.get("lead_region") and (row.get("lead_bbox") or {}).get("current")]
    progression_prior_boxes = [{
        "disease": row["disease"], "change": row["change"],
        "region": row["lead_region"], "bbox": (row.get("lead_bbox") or {}).get("prior"),
    } for row in interval_changes if row.get("lead_region") and (row.get("lead_bbox") or {}).get("prior")]
    normal = (len(disease_states) == len(C.CHEX_NAMES) - 1 and
              all(row["state"] == "absent" for row in disease_states))
    report = {
        "schema_version": "vera_report_v2",
        "threshold_policy": ("validation_per_disease_dual_threshold"
                             if thresholds is not None else "fixed_provisional_fallback"),
        "image_id": m3rec.get("image_id"),
        "prior_image_id": (m4rec or {}).get("prior_image_id"),
        "has_prior": m4rec is not None,
        "normal": normal,
        "classification": {
            "image": {"image_id": m3rec.get("image_id"), "path": None, "boxes": classification_boxes},
            "table": classification_rows,
            "text": "",
        },
        "progression": {
            "images": {
                "prior": {"image_id": (m4rec or {}).get("prior_image_id"), "path": None,
                          "boxes": progression_prior_boxes},
                "current": {"image_id": m3rec.get("image_id"), "path": None,
                            "boxes": progression_current_boxes},
            },
            "table": interval_changes,
            "text": "",
        } if m4rec is not None else None,
        "current_findings": current_findings,
        "interval_changes": interval_changes,
        "disease_states": disease_states,
        "findings": current_findings,
        "tables": {"classification": classification_rows, "progression": interval_changes},
        "coverage_map": coverage_map(m3rec),
        "box_source": m3rec.get("box_source"),
        "calibration_provenance": {
            "disease_thresholds": (thresholds or {}).get("_artifact"),
            "concept_gate": (concept_gate or {}).get("_artifact"),
            "m3_checkpoint_sha256": m3rec.get("m3_checkpoint_sha256"),
        },
    }
    return report


# ---- tier 5: realize (template, faithful default) ---------------------------
def realize_template(report: dict) -> str:
    if report["normal"]:
        return "No acute cardiopulmonary abnormality."
    # asserts first, then hedges, then abstains; stable order within by disease index
    order = {"assert": 0, "hedge": 1, "abstain": 2, "absent": 3}
    findings = report.get("current_findings") or report.get("findings") or []
    findings = sorted(findings, key=lambda f: (order[f["status"]], C.CHEX_NAMES.index(f["disease"])))
    return " ".join(f["text"] for f in findings)


def realize_interval_template(report: dict) -> str:
    """Render only the prior-current disease-change table."""
    changes = report.get("interval_changes") or []
    changes = sorted(changes, key=lambda f: C.CHEX_NAMES.index(f["disease"]))
    return " ".join(f["text"] for f in changes)
