"""Shared scoring helpers for the scene-graph evaluators (LLM + rule parser + gold).

Extracted here so the entry-point scripts import from one library instead of each other (they are
renamed with run-order prefixes and live in different scripts/ subfolders). Pure functions over the
flat schema produced by sg_schema / rule_parser.
"""

from __future__ import annotations

import json
import re

from sg_schema import PROG_NAMES


def pos_cells(flat: dict) -> set[tuple[str, str]]:
    """(region, finding) pairs asserted present ("yes")."""
    return {(r, f["finding"]) for r, fs in flat.items() for f in fs
            if f.get("presence", "yes") == "yes"}


def unc_cells(flat: dict) -> set[tuple[str, str]]:
    """(region, finding) pairs flagged uncertain (hedged), regardless of polarity."""
    return {(r, f["finding"]) for r, fs in flat.items() for f in fs if f.get("uncertain")}


def prog_cells(flat: dict) -> dict[tuple[str, str], str]:
    """(region, finding) -> progression word, for present findings that carry one."""
    out = {}
    for r, fs in flat.items():
        for f in fs:
            if f.get("presence", "yes") == "yes" and f.get("progression") in PROG_NAMES:
                out[(r, f["finding"])] = f["progression"]
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def raw_parse_ok(text: str) -> bool:
    """Did the generation contain a parseable top-level JSON object (any content)?"""
    s = text.find("{")
    if s == -1:
        return False
    depth = 0
    for i in range(s, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    json.loads(text[s: i + 1])
                    return True
                except json.JSONDecodeError:
                    return False
    return False


_USER_RE = re.compile(r"Available regions:\s*(?P<regions>.*?)\n\nReport:\n(?P<report>.*?)\n\nFindings JSON:",
                      re.DOTALL)


def report_and_regions(user_msg: str) -> tuple[str, list[str]]:
    """Recover (report, regions) from an SFT user message."""
    m = _USER_RE.search(user_msg)
    if not m:
        return "", []
    regions = [r.strip() for r in m.group("regions").split(",") if r.strip()]
    return m.group("report").strip(), regions
