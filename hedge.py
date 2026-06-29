"""Single source of truth for radiology HEDGE detection (uncertainty in a report sentence).

Shared verbatim by the LLM branch (phase_2/sg_schema.py) AND the M3/M4 label builders
(phase_3/labels.py, phase_4/labels.py) so "what counts as uncertain" can NEVER diverge between
train (silver scene graphs) and launch (LLM-parsed reports). Each of those files imports
`is_hedged` from here via a one-line repo-root path insert; this module imports nothing project-
specific, so it never collides with the per-phase `constants.py` / `config.py`.

A finding asserted in a hedged sentence keeps its polarity (yes/no) but is flagged uncertain:
  "possible pneumonia"      -> presence yes, uncertain   ("uncertain yes")
  "no definite pneumonia"   -> presence no,  uncertain   ("uncertain no")
Curated for high precision: leaning-positive words that radiologists use assertively
("consistent with", "compatible with", "likely", "probable") are intentionally EXCLUDED so we
do not over-mask real positives.
"""

from __future__ import annotations

import re

# Positive-leaning hedges ("possible X") + negative-leaning hedges ("no definite X").
HEDGE_PATTERN = (
    r"\b(?:possible|possibly|may|might|could|questionable|equivocal|indeterminate|"
    r"presumed|presumably|suspected)\b"
    r"|cannot\s+(?:be\s+)?exclude|can\s*not\s+exclude|not\s+excluded|cannot\s+rule\s+out|"
    r"\brule[-\s]?out\b|suspicious\s+for|suspicion\s+for|concerning\s+for|concern\s+for|"
    r"worrisome\s+for|question\s+of|suggestive\s+of|\bdifferential\b|\bversus\b|\bvs\.?\b|"
    r"\bno\s+definite\b|\bno\s+definitive\b|\bnot\s+definitely\b|\bwithout\s+definite\b|"
    r"\bno\s+clear\b|\bnot\s+clearly\b"
)
_HEDGE_RE = re.compile(HEDGE_PATTERN, re.IGNORECASE)


def is_hedged(text) -> bool:
    """True if the sentence carries an uncertainty/hedge cue."""
    return bool(_HEDGE_RE.search(str(text or "")))


if __name__ == "__main__":   # quick self-check
    for s, want in [
        ("possible pneumonia", True),
        ("no definite consolidation", True),
        ("cannot exclude effusion", True),
        ("findings suspicious for malignancy", True),
        ("bibasilar atelectasis", False),
        ("consistent with edema", False),       # assertive, NOT hedged
        ("likely pneumonia", False),            # leaning-positive, NOT masked
    ]:
        got = is_hedged(s)
        print(("ok " if got == want else "FAIL"), repr(s), "->", got)
