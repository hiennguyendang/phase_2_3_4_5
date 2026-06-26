"""Tier 6 verify (spec 5.6) — DETERMINISTIC round-trip + coverage.

CRITICAL: the verifier must be a deterministic label extractor (CheXbert/RadGraph) or hard string
match — NEVER an LLM (an LLM verifier hallucinates, defeating the whole point). This module ships a
hard-match interim verifier; `extract_labels` is the single seam where CheXbert/RadGraph plugs in.

Catches:
  - out_of_table : a finding spoken in the text that is NOT in the M3/M4 table (added/hallucinated).
  - coverage_miss: an asserted-positive table cell NOT spoken in the text (under-reporting).
  - temporal_halluc: temporal language with no prior / no M4 cell backing it.
"""

from __future__ import annotations

import re

import constants as C

_NEG = ("no ", "without ", "not ", "negative for ")
_TEMPORAL_MARKERS = ("compared to the prior", "improved", "worsened", "unchanged",
                     "increased", "decreased", "interval")


def _search_terms(disease: str) -> list[str]:
    return list({C.DISEASE_PHRASE.get(disease, disease.lower()), disease.lower()})


def extract_labels(text: str) -> set[str]:
    """Hard-match label extractor (interim stand-in for CheXbert/RadGraph).
    Returns the set of CheXpert findings asserted-positive in `text` (negations excluded)."""
    low = text.lower()
    found = set()
    for disease in C.CHEX_NAMES:
        if disease == C.NO_FINDING:
            continue
        for term in _search_terms(disease):
            idx = low.find(term)
            if idx < 0:
                continue
            window = low[max(0, idx - 18):idx]            # short look-back for a negation
            if any(neg in window for neg in _NEG):
                break                                     # negated -> not positive
            found.add(disease)
            break
    return found


def verify(report: dict, text: str) -> dict:
    table = {f["disease"] for f in report["findings"]}                 # assert + hedge
    asserts = {f["disease"] for f in report["findings"] if f["status"] == "assert"}
    spoken = extract_labels(text)

    out_of_table = sorted(spoken - table)                              # spoke something not in table
    coverage_miss = sorted(asserts - spoken)                          # asserted but not spoken

    # temporal: any temporal marker in text must be backed by a finding with an M4 cell
    has_temporal_text = any(m in text.lower() for m in _TEMPORAL_MARKERS)
    backed = report["has_prior"] and any(f.get("temporal") for f in report["findings"])
    temporal_halluc = bool(has_temporal_text and not backed)

    ok = not out_of_table and not coverage_miss and not temporal_halluc
    return {
        "ok": ok,
        "out_of_table": out_of_table,
        "coverage_miss": coverage_miss,
        "temporal_halluc": temporal_halluc,
        "spoken": sorted(spoken),
        "extractor": "hardmatch",          # <- becomes "chexbert"/"radgraph" when plugged in
    }
