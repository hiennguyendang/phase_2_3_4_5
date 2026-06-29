"""Flat scene-graph schema + deterministic mapper — the SIMPLIFIED LLM branch of M2.

Instead of making the LLM emit ImaGenome relation strings ("anatomicalfinding|yes|atelectasis")
the LLM emits a PLAIN, structured object per region:

    { "<region>": [ {"finding": "atelectasis", "presence": "yes", "progression": "worsened"}, ... ] }

- "finding"     : one of the 69 canonical concept names (data/m3_concept_space.json)
- "presence"    : "yes" (report asserts it) | "no" (report explicitly denies it)
- "progression" : "improved" | "stable" | "worsened"   (omit when the report makes no comparison)

A *deterministic* mapper (no model) turns this back into the relation/cue strings that phase_3
(M3) and phase_4 (M4) already consume; then `assemble_scene_graph` (sg_lib) builds the
*_SceneGraph.json from the flat findings + detector boxes.

Why flat (vs the old compact relation-string target):
  * a 3B model only has to copy a finding NAME, not memorize the pipe-delimited vocab;
  * `parse_flat` validates every field against the closed vocab, so hallucinations (bad region,
    unknown finding, illegal presence/progression) are dropped — no separate snap_to_vocab;
  * the schema is a tiny fixed grammar -> trivial to constrain with structured decoding.

Round-trip used across the pipeline:
    silver scene graph --flat_from_scene_graph-->  flat target          (build_sft_dataset, SFT label)
    LLM(report, regions) --parse_flat-->           flat                 (inference)
    flat --compact_from_flat--> compact --assemble_scene_graph--> *_SceneGraph.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from constants import CLASS_NAMES, canonical_name

# canonical hedge detector (hedge.py at repo root) shared by phase_2/3/4. Search ancestor dirs so
# it resolves both in-place (repo root = parents[1]) AND when phase_2 is copied next to hedge.py
# on Kaggle (e.g. /kaggle/working/{phase_2,hedge.py}).
for _cand in Path(__file__).resolve().parents:
    if (_cand / "hedge.py").exists():
        sys.path.insert(0, str(_cand))
        break
from hedge import is_hedged  # noqa: E402

# ---------------------------------------------------------------------------
# Finding vocabulary — the 69-concept M3 label space is the single source of truth.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent


def _find_concept_space() -> Path:
    for cand in (_REPO / "data" / "m3_concept_space.json", _HERE / "m3_concept_space.json",
                 _REPO / "phase_3" / "m3_concept_space.json", Path("m3_concept_space.json")):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "m3_concept_space.json not found (looked in data/, phase_2/, phase_3/). "
        "It defines the 69 allowed findings.")


_cs = json.loads(_find_concept_space().read_text(encoding="utf-8"))
_CONCEPTS: list[dict] = _cs["concepts"]

# finding name (lower) -> ImaGenome category ("anatomicalfinding"/"disease"/"tubesandlines"/"device")
FINDING_TO_CATEGORY: dict[str, str] = {c["name"].strip().lower(): c["category"] for c in _CONCEPTS}
VALID_FINDINGS: frozenset[str] = frozenset(FINDING_TO_CATEGORY)
# findings listed in concept-space order (frequency desc within category) — for the prompt menu.
FINDINGS_IN_ORDER: list[str] = [c["name"].strip().lower() for c in _CONCEPTS]

# ---------------------------------------------------------------------------
# Presence + progression vocab (must stay in lockstep with phase_4/constants.py).
# ---------------------------------------------------------------------------
# presence keeps the ImaGenome polarity (yes/no). Uncertainty is a SEPARATE boolean flag, so
# the scene graph stays structurally identical to ImaGenome (the finding is still present with
# its polarity) and the hedge is just an annotation -> resolved to a soft logit at M4, masked at
# M3. We never change a finding's polarity because of a hedge.
PRESENCE_VALUES: frozenset[str] = frozenset({"yes", "no"})
_PRES_PRIORITY: dict[str, int] = {"no": 0, "yes": 1}   # a positive assertion wins on merge
# Tolerant aliases parse_flat maps onto the canonical presence words.
_PRESENCE_ALIASES: dict[str, str] = {
    "present": "yes", "positive": "yes", "true": "yes",
    "absent": "no", "negative": "no", "false": "no",
}
# If the model expresses uncertainty THROUGH the presence field, recover (polarity, uncertain).
_PRESENCE_HEDGE: dict[str, tuple[str, bool]] = {
    "possible": ("yes", True), "probable": ("yes", True), "likely": ("yes", True),
    "suspected": ("yes", True), "maybe": ("yes", True), "uncertain": ("yes", True),
    "equivocal": ("yes", True), "indeterminate": ("yes", True), "questionable": ("yes", True),
    "unlikely": ("no", True), "doubtful": ("no", True),
}

PROG_NAMES: tuple[str, ...] = ("stable", "improved", "worsened")
PROG_VALUES: frozenset[str] = frozenset(PROG_NAMES)
_PROG_PRIORITY = {"stable": 0, "improved": 1, "worsened": 2}   # worsened wins on conflict
# scene-graph cue label  <->  our progression word
CUE_TO_PROG: dict[str, str] = {"no change": "stable", "improved": "improved", "worsened": "worsened"}
PROG_TO_CUE: dict[str, str] = {"stable": "no change", "improved": "improved", "worsened": "worsened"}
_COMPARISON_CATEGORY = "comparison"


def parse_triplet(s: str) -> tuple[str, str, str] | None:
    """`anatomicalfinding|yes|lung opacity` -> ('anatomicalfinding','yes','lung opacity')."""
    parts = str(s).split("|", 2)
    if len(parts) != 3:
        return None
    cat, pol, label = (p.strip() for p in parts)
    return (cat.lower(), pol.lower(), label.lower()) if cat and label else None


# ---------------------------------------------------------------------------
# silver scene graph  ->  flat target  (build the SFT label)
# ---------------------------------------------------------------------------
def _comparison_prog(cue_list: Any) -> str | None:
    """A phrase's comparison_cues entry (list of 'comparison|yes|<label>') -> prog word or None."""
    best = None
    for s in (cue_list or []):
        t = parse_triplet(str(s))
        if t is None:
            continue
        cat, pol, label = t
        if cat != _COMPARISON_CATEGORY or pol != "yes":
            continue
        prog = CUE_TO_PROG.get(label)
        if prog is None:
            continue
        if best is None or _PROG_PRIORITY[prog] > _PROG_PRIORITY[best]:
            best = prog
    return best


def flat_from_scene_graph(scene: dict[str, Any]) -> dict[str, list[dict]]:
    """Distill an ImaGenome scene graph into the flat per-region findings target.

    Mirrors phase_4/labels.py: a finding is kept iff its name is one of the 69 concepts; a
    phrase's comparison cue (if present and the finding is a CERTAIN yes) sets the progression.
    Negatives ("no") are kept too — M3's negative signal. Polarity is NEVER changed by a hedge;
    instead a mention in a HEDGED sentence (per the source `phrases`) sets the uncertain flag,
    and a finding is uncertain iff it has NO certain mention of its winning polarity.
    """
    # region -> finding -> flags
    acc: dict[str, dict[str, dict]] = {}
    for entry in scene.get("attributes", []) or []:
        region = canonical_name(str(entry.get("bbox_name", "")))
        if region is None:
            continue
        phrase_attrs = entry.get("attributes", []) or []        # list[ list[str] ]
        cues = entry.get("comparison_cues", []) or []           # parallel list[ list[str] ]
        phrases = entry.get("phrases", []) or []                # parallel list[str] (source text)
        for i, attrs in enumerate(phrase_attrs):
            prog = _comparison_prog(cues[i] if i < len(cues) else None)
            hedged = is_hedged(phrases[i] if i < len(phrases) else "")
            for s in (attrs or []):
                t = parse_triplet(str(s))
                if t is None:
                    continue
                _cat, pol, label = t
                if label not in VALID_FINDINGS or pol not in PRESENCE_VALUES:
                    continue
                slot = acc.setdefault(region, {}).setdefault(
                    label, {"yes_c": False, "yes_h": False, "no_c": False, "no_h": False,
                            "prog": None})
                slot[f"{pol}_{'h' if hedged else 'c'}"] = True
                if pol == "yes" and not hedged and prog is not None:   # cue from a certain yes
                    cur = slot["prog"]
                    if cur is None or _PROG_PRIORITY[prog] > _PROG_PRIORITY[cur]:
                        slot["prog"] = prog

    out: dict[str, list[dict]] = {}
    for region in CLASS_NAMES:                                  # stable, deterministic order
        fmap = acc.get(region)
        if not fmap:
            continue
        items: list[dict] = []
        for finding in FINDINGS_IN_ORDER:                      # stable order within a region
            v = fmap.get(finding)
            if v is None:
                continue
            if v["yes_c"] or v["yes_h"]:                       # a positive mention wins
                presence, uncertain = "yes", (not v["yes_c"])
            else:
                presence, uncertain = "no", (not v["no_c"])
            item: dict[str, Any] = {"finding": finding, "presence": presence}
            if uncertain:
                item["uncertain"] = True
            elif presence == "yes" and v["prog"]:              # progression only for a certain yes
                item["progression"] = v["prog"]
            items.append(item)
        if items:
            out[region] = items
    return out


# ---------------------------------------------------------------------------
# flat  <->  text  (for the LLM)
# ---------------------------------------------------------------------------
def dump_flat(flat: dict[str, list[dict]]) -> str:
    """Canonical JSON string the LLM should emit."""
    return json.dumps(flat, ensure_ascii=False)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_flat(text: str) -> dict[str, list[dict]]:
    """Parse LLM output into a validated flat target, tolerant of code fences / prose.

    Drops anything outside the closed vocab (unknown region / finding, illegal presence or
    progression) — this IS the anti-hallucination filter. Returns {} if nothing valid is found.
    """
    if not text:
        return {}
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    return {}
                return _clean_flat(obj)
    return {}


def _clean_flat(obj: Any) -> dict[str, list[dict]]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, list[dict]] = {}
    for region, findings in obj.items():
        cname = canonical_name(str(region))
        if cname is None or not isinstance(findings, list):
            continue
        items: list[dict] = []
        seen: set[str] = set()
        for f in findings:
            if not isinstance(f, dict):
                continue
            name = str(f.get("finding", "")).strip().lower()
            if name not in VALID_FINDINGS or name in seen:
                continue
            raw = str(f.get("presence", "yes")).strip().lower()
            uncertain = bool(f.get("uncertain")) or str(f.get("uncertain", "")).lower() == "true"
            if raw in _PRESENCE_HEDGE:                     # uncertainty expressed via presence word
                pol, unc2 = _PRESENCE_HEDGE[raw]
                uncertain = uncertain or unc2
            else:
                pol = _PRESENCE_ALIASES.get(raw, raw)
                if pol not in PRESENCE_VALUES:
                    pol = "yes"
            item: dict[str, Any] = {"finding": name, "presence": pol}
            if uncertain:
                item["uncertain"] = True
            else:
                prog = str(f.get("progression", "")).strip().lower()
                if pol == "yes" and prog in PROG_VALUES:   # progression only for a certain yes
                    item["progression"] = prog
            items.append(item)
            seen.add(name)
        if items:
            out[cname] = items
    return out


# ---------------------------------------------------------------------------
# flat  ->  compact  (the deterministic mapper feeding assemble_scene_graph)
# ---------------------------------------------------------------------------
def compact_from_flat(flat: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """flat findings -> the compact relation/cue structure `assemble_scene_graph` expects:

        { region: [ {"rel": ["<cat>|<pol>|<finding>"],
                     "comparison": ["comparison|yes|<cue>"],     # certain yes only
                     "uncertainty": ["uncertainty|yes|hedged"]} ] }   # when uncertain

    Polarity is ALWAYS kept (never dropped) so the scene graph matches ImaGenome; uncertainty is
    carried as its own cue so the M3/M4 label builders can mask it (the soft-logit decision is
    deferred to M4). Category + cue spelling are looked up deterministically; no hallucination.
    """
    out: dict[str, list[dict]] = {}
    for region, findings in flat.items():
        cname = canonical_name(str(region))
        if cname is None:
            continue
        items: list[dict] = []
        for f in findings:
            name = str(f.get("finding", "")).strip().lower()
            cat = FINDING_TO_CATEGORY.get(name)
            if cat is None:
                continue
            pol = str(f.get("presence", "yes")).strip().lower()
            pol = _PRESENCE_ALIASES.get(pol, pol)
            if pol not in PRESENCE_VALUES:
                pol = "yes"
            item: dict[str, list[str]] = {"rel": [f"{cat}|{pol}|{name}"]}
            if f.get("uncertain"):
                item["uncertainty"] = ["uncertainty|yes|hedged"]
            else:
                prog = str(f.get("progression", "")).strip().lower()
                if pol == "yes" and prog in PROG_TO_CUE:
                    item["comparison"] = [f"comparison|yes|{PROG_TO_CUE[prog]}"]
            items.append(item)
        if items:
            out.setdefault(cname, []).extend(items)
    return out


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def _findings_menu() -> str:
    """The 69 allowed findings, grouped by category, for the system prompt."""
    by_cat: dict[str, list[str]] = {}
    for c in _CONCEPTS:
        by_cat.setdefault(c["category"], []).append(c["name"].strip().lower())
    order = ("anatomicalfinding", "disease", "tubesandlines", "device")
    lines = []
    for cat in order:
        if by_cat.get(cat):
            lines.append(f"  [{cat}] " + ", ".join(by_cat[cat]))
    return "\n".join(lines)


ALLOWED_FINDINGS_MENU: str = _findings_menu()

SYSTEM_PROMPT: str = (
    "You are a radiology report parser. Given a chest X-ray report and the list of "
    "anatomical regions present in the image, extract the findings into a compact JSON "
    "object. Map each region that has a finding to a list of objects, each with:\n"
    '  "finding"     : EXACTLY one of the allowed finding names below.\n'
    '  "presence"    : "yes" if the report mentions the finding as present, "no" if the '
    "report denies it (keep the polarity the report states).\n"
    '  "uncertain"   : true ONLY when the report hedges the finding (e.g. possible / may '
    "represent / suspicious for / concerning for / cannot exclude / differential / no "
    "definite). Keep the polarity in \"presence\" and just add this flag; OMIT the key "
    "when the report is confident.\n"
    '  "progression" : "improved" | "stable" | "worsened" if the report compares the '
    "finding to a prior study; OMIT when no comparison is made or the finding is uncertain.\n"
    "Rules: only use regions from the provided list; use ONLY the allowed finding names "
    "(verbatim); attach each finding to the most specific region the report implies; do "
    "NOT include findings the report does not mention; never invent findings. Output the "
    "JSON object only, no prose, no code fences.\n\n"
    "Allowed findings:\n" + ALLOWED_FINDINGS_MENU
)


def build_user_prompt(report: str, regions: list[str]) -> str:
    region_menu = ", ".join(regions)
    report = (report or "").strip()
    return (
        f"Available regions: {region_menu}\n\n"
        f"Report:\n{report}\n\n"
        "Findings JSON:"
    )


# A STRICTER, self-contained prompt for ZERO-SHOT (un-finetuned) models: hard rules + an inline
# worked example so a base model emits clean schema-valid JSON. eval_sg_llm.py --prompt strict
# swaps this in. (The finetuned model uses the plain SYSTEM_PROMPT it was trained on.)
SYSTEM_PROMPT_STRICT: str = (
    "You are a precise radiology report parser. Output ONLY one JSON object and NOTHING else — "
    "no prose, no markdown, no code fences.\n\n"
    "For each anatomical region that has a finding, map the region name to a list of objects:\n"
    '  {"finding": <one allowed finding name, exact spelling>,\n'
    '   "presence": "yes" or "no",\n'
    '   "uncertain": true        (INCLUDE ONLY if the report hedges it; otherwise omit the key),\n'
    '   "progression": "improved"|"stable"|"worsened"  (INCLUDE ONLY if compared to a prior study; '
    "else omit)}\n\n"
    "Hard rules:\n"
    '- Use ONLY region names from the user\'s "Available regions" list.\n'
    "- Use ONLY finding names from the allowed list below (verbatim spelling).\n"
    "- Include a finding ONLY if the report explicitly states it. NEVER invent findings.\n"
    '- presence="no" for explicit denials ("no effusion"). Set "uncertain": true for hedges '
    '("possible", "may represent", "cannot exclude", "suspicious for", "no definite").\n'
    "- Attach each finding to the most specific region the report implies.\n"
    "- If nothing mappable is mentioned, output exactly {}.\n\n"
    "Example:\n"
    "Available regions: left lower lung zone, cardiac silhouette\n"
    "Report: Stable bibasilar atelectasis. Possible early pneumonia at the left base. "
    "Heart size is normal.\n"
    'Output: {"left lower lung zone": [{"finding": "atelectasis", "presence": "yes", '
    '"progression": "stable"}, {"finding": "pneumonia", "presence": "yes", "uncertain": true}]}\n\n'
    "Allowed findings:\n" + ALLOWED_FINDINGS_MENU
)
