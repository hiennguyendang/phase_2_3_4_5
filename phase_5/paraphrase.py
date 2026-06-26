"""Tier 5 constrained paraphraser (spec 5.5) — OPTIONAL, pluggable, prose-from-table.

The template realizer is the faithful default and stands alone. This module is the seam for an LLM
that makes the prose smoother — but it is PROSE-FROM-TABLE, not prose-from-image: it may only
rephrase the findings already in the report, never add/remove one. The caller MUST re-run verify on
the output and fall back to the template if it fails (see run.py). Default backend = None = identity
(returns the template text), so nothing here can hallucinate until a real backend is wired in.
"""

from __future__ import annotations

import constants as C


def build_prompt(report: dict, template_text: str) -> str:
    """A locked-down instruction: rephrase ONLY these findings, add nothing."""
    rows = []
    for f in report["findings"]:
        loc = f["lead_region"] or "unspecified location"
        tmp = f" [{f['temporal']['prog']} vs prior]" if f.get("temporal") else ""
        rows.append(f"- {f['disease']} ({f['status']}, conf {f['prob']}) @ {loc}{tmp}")
    table = "\n".join(rows) if rows else "- (no positive findings)"
    return (
        "Rewrite the following chest X-ray findings as a fluent radiology report. "
        "Use ONLY these findings — do not add, remove, infer, or speculate any finding, "
        "location, device, or temporal change beyond what is listed.\n\n"
        f"FINDINGS TABLE:\n{table}\n\n"
        f"DRAFT (template):\n{template_text}\n\nREPORT:"
    )


def paraphrase(report: dict, template_text: str, backend=None) -> str:
    """backend: Optional[Callable[[str], str]] (e.g. a Claude call). None -> identity (template)."""
    if backend is None:
        return template_text
    try:
        out = backend(build_prompt(report, template_text))
    except Exception as e:  # noqa: BLE001
        print(f"[paraphrase] backend failed -> template ({e})")
        return template_text
    return (out or "").strip() or template_text
