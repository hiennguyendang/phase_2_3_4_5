"""Non-LLM report parser — report text -> the SAME flat findings schema sg_schema produces.

Replicates (a lightweight version of) how ImaGenome built its silver scene graphs WITHOUT any
model: a curated finding lexicon + NegEx-lite negation + the shared hedge detector + an anatomy
lexicon that maps location words to the 29 bbox regions (with containment) + a comparison-cue
lexicon for progression. Deterministic, fast, no GPU, no hallucination, fully auditable.

    from rule_parser import parse_report
    flat = parse_report("Stable bibasilar atelectasis. No pleural effusion.", available_regions)
    # -> {"left lower lung zone": [{"finding":"atelectasis","presence":"yes","progression":"stable"}], ...}

Output is the flat schema of sg_schema (region -> [{finding, presence, uncertain?, progression?}]),
so it feeds compact_from_flat / assemble_scene_graph and is scored by eval_rule_parser.py with the
exact metrics used for the LLM. This is a measurable, GPU-free alternative to the LLM parser.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# shared infra (constants, sg_schema, config) lives beside this module in phase_2/src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import CLASS_NAMES
from sg_schema import FINDINGS_IN_ORDER, FINDING_TO_CATEGORY, VALID_FINDINGS, is_hedged

_REGIONS = set(CLASS_NAMES)

# ---------------------------------------------------------------------------
# 1. Finding lexicon: concept name -> regex triggers (lowercase). Curated for the high-frequency
#    concepts; every other concept falls back to a regex auto-built from its own name.
# ---------------------------------------------------------------------------
_CURATED: dict[str, list[str]] = {
    "lung opacity": [r"opacit", r"opacification", r"airspace disease"],
    "airspace opacity": [r"airspace opacit", r"air space opacit"],
    "pleural effusion": [r"pleural effusion", r"effusion", r"pleural fluid"],
    "pulmonary edema/hazy opacity": [r"pulmonary edema", r"\bedema\b", r"hazy"],
    "vascular congestion": [r"congesti", r"vascular prominence", r"vascular plethora"],
    "vascular redistribution": [r"cephalization", r"vascular redistribution"],
    "atelectasis": [r"atelecta", r"volume loss"],
    "linear/patchy atelectasis": [r"(linear|patchy|discoid|plate[- ]?like) atelecta"],
    "lobar/segmental collapse": [r"\bcollapse\b", r"lobar collapse"],
    "consolidation": [r"consolidat"],
    "pneumonia": [r"pneumonia", r"pneumonic", r"\binfection\b", r"infectious"],
    "pneumothorax": [r"pneumothorax", r"pneumothoraces", r"\bptx\b"],
    "hydropneumothorax": [r"hydropneumothorax"],
    # flexible: "heart is mildly enlarged", "enlargement of the cardiac silhouette", "cardiac
    # silhouette ... enlarged" — the heart/cardiac anchor and the enlarge word can be a few words
    # apart (within a clause, no sentence stop between). Was too rigid -> missed 1162 cells.
    "enlarged cardiac silhouette": [
        r"cardiomegaly",
        r"\benlarge\w*\b[^.,;]{0,18}\b(cardiac|heart|cardio)",          # enlargement of the heart
        r"\b(heart|cardiac|cardio\w*)\b[^.,;]{0,22}?\benlarg",          # heart is mildly enlarged
        r"\bcardiac silhouette\b[^.,;]{0,45}?\benlarg",                 # cardiac silhouette ... enlarged
        r"cardiac (silhouette )?enlargement"],
    "mediastinal widening": [r"mediastin\w* widen", r"widen\w* (of the )?mediastin", r"wide mediastin"],
    "mediastinal displacement": [r"mediastinal shift", r"mediastinal displace", r"shift of the mediastin"],
    "superior mediastinal mass/enlargement": [r"superior mediastinal"],
    "enlarged hilum": [r"hilar (enlarge|prominence|fullness|mass)", r"enlarged hil", r"prominent hil"],
    "tortuous aorta": [r"tortuous", r"ectatic aorta", r"unfolded aorta", r"aortic ectasia"],
    # bidirectional within a clause window: "calcifications ... in the aorta" / "aortic knob
    # calcification" — the calcif word and the aorta anchor need not be adjacent.
    "vascular calcification": [r"atherosclerot", r"vascular calcif",
                               r"calcif\w*[^.,;]{0,30}\b(aort|arch|knob)",
                               r"\b(aort\w*|arch|knob)\b[^.,;]{0,30}calcif"],
    "vascular redistribution": [r"cephalization", r"redistribution", r"upper zone redistribution"],
    "pleural/parenchymal scarring": [r"scarring", r"scar\b", r"fibrosis", r"fibrotic"],
    "lung lesion": [r"\blesion\b"],
    "mass/nodule (not otherwise specified)": [r"\bmass\b", r"\bnodule\b", r"nodular"],
    "multiple masses/nodules": [r"(multiple|several|numerous) (pulmonary )?(masses|nodules|opacit)",
                                r"innumerable"],
    "calcified nodule": [r"calcified (nodule|granuloma)"],
    "lung cancer": [r"carcinoma", r"malignan", r"neoplas", r"lung cancer", r"metasta"],
    "granulomatous disease": [r"granulomatous", r"\bgranuloma\b"],
    "alveolar hemorrhage": [r"alveolar hemorrhage", r"pulmonary hemorrhage"],
    "interstitial lung disease": [r"interstitial lung disease", r"\bild\b", r"pulmonary fibrosis",
                                  r"honeycomb"],
    "costophrenic angle blunting": [r"blunt", r"costophrenic angle blunt"],
    "rib fracture": [r"rib fracture", r"fracture.*rib"],
    "spinal fracture": [r"(vertebral|spinal|compression) fracture"],
    "clavicle fracture": [r"clavic\w* fracture", r"fracture.*clavic"],
    "infiltration": [r"infiltrat"],
    # high-precision only: bare "reticular"/"interstitial markings" mostly map (in silver) to edema/
    # lung opacity, not this rare ILD pattern — they wrecked precision (0.13-0.17). Keep the
    # unambiguous ILD terms.
    "increased reticular markings/ild pattern": [r"\bild\b", r"honeycomb", r"reticulation",
                                                 r"(pulmonary|interstitial) fibrosis",
                                                 r"interstitial lung disease"],
    "elevated hemidiaphragm": [r"elevat\w* (of the )?(hemi)?diaphragm", r"diaphragm.*elevat"],
    "spinal degenerative changes": [r"degenerative (change|disease)", r"spondylosis", r"osteophyt"],
    "scoliosis": [r"scoliosis", r"scoliotic"],
    "hyperaeration": [r"hyperinflat", r"hyperaerat", r"hyperexpan"],
    "sub-diaphragmatic air": [r"free (intraperitoneal |sub.?diaphragmatic )?air", r"pneumoperitoneum"],
    "subcutaneous air": [r"subcutaneous (air|emphysema)"],
    "pneumomediastinum": [r"pneumomediastinum"],
    "copd/emphysema": [r"\bcopd\b", r"emphysema"],
    "interstitial lung disease": [r"interstitial lung disease", r"\bild\b"],
    "fluid overload/heart failure": [r"fluid overload", r"heart failure", r"\bchf\b", r"volume overload"],
    "aspiration": [r"aspiration"],
    # tubes / lines / devices
    "endotracheal tube": [r"endotracheal tube", r"\bet tube\b", r"\bett\b"],
    "enteric tube": [r"enteric tube", r"\bng tube\b", r"nasogastric", r"\bog tube\b", r"orogastric",
                     r"feeding tube", r"dobh?off"],
    "picc": [r"\bpicc\b"],
    "ij line": [r"internal jugular", r"\bij (line|catheter)\b"],
    "subclavian line": [r"subclavian (line|catheter)"],
    "chest tube": [r"chest tube"],
    "pigtail catheter": [r"pigtail"],
    "tracheostomy tube": [r"tracheostomy", r"\btrach\b"],
    "swan-ganz catheter": [r"swan-?ganz", r"pulmonary artery catheter"],
    "cardiac pacer and wires": [r"pacemaker", r"\bpacer\b", r"\baicd\b", r"defibrillator",
                                r"pacing (lead|wire)", r"\bicd\b"],
    "prosthetic valve": [r"prosthetic valve", r"valve replacement", r"mechanical valve"],
    "cabg grafts": [r"\bcabg\b", r"bypass graft", r"sternal wire"],
    "chest port": [r"port-?a-?cath", r"chest port", r"mediport"],
}


# generic single words that must NOT become a standalone trigger via the name fallback
_GENERIC_STOP = {"pleural", "lung", "lobar", "segmental", "bone", "vascular", "cardiac",
                 "mass", "nodule", "spinal", "superior", "left", "right", "multiple"}


def _name_to_regex(name: str) -> list[str]:
    """Fallback trigger from a concept name (used ONLY when not curated): drop parentheticals,
    split on '/', keep parts that are specific enough (>=5 chars, not a generic single word)."""
    out = []
    base = re.sub(r"\(.*?\)", "", name).strip()
    for part in re.split(r"[/]", base):
        part = part.strip()
        if len(part) >= 5 and part not in _GENERIC_STOP:
            out.append(re.escape(part).replace(r"\ ", r"\s+"))
    return out


# MINED triggers: high-precision surface phrases learned from silver `phrases` per concept
# (mine_finding_lexicon.py -> rule_finding_triggers.json). Literal phrases; SUPPLEMENT the curated
# lexicon to widen long-tail coverage toward ImaGenome's 271-entity lexicon. Each was assigned to a
# single most-specific concept, so they don't cross-fire.
def _load_mined() -> dict:
    for c in (Path(__file__).resolve().parent / "rule_finding_triggers.json",
              Path("rule_finding_triggers.json")):
        if c.exists():
            return json.loads(c.read_text(encoding="utf-8"))
    return {}


_MINED = _load_mined()


def _literal_regex(phrase: str) -> str:
    """A mined phrase -> word-boundary regex (spaces tolerate MIMIC whitespace runs)."""
    return r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"


# concept -> compiled trigger regex. Curated triggers are AUTHORITATIVE (override name fallback);
# only non-curated concepts use the name-based fallback. Mined triggers are appended to BOTH.
_TRIGGERS: list[tuple[str, re.Pattern]] = []
for _c in FINDINGS_IN_ORDER:
    pats = list(_CURATED.get(_c) or ([] if _c in _CURATED else _name_to_regex(_c)))
    pats += [_literal_regex(p) for p in _MINED.get(_c, [])]
    if pats:
        _TRIGGERS.append((_c, re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)))

# ---------------------------------------------------------------------------
# 2. Negation (NegEx-lite) + 3. progression cues
# ---------------------------------------------------------------------------
# "no" / "without" are NOT negations in "no (significant interval) change in X" / "without change" —
# those assert X is PRESENT and stable. A negative lookahead skips that construct so the finding
# survives (its stability is captured separately by the progression cues).
_NO = r"no(?!\s+(?:\w+\s+){0,3}change)"
_NEG = re.compile(
    rf"\b({_NO}|not|without(?!\s+(?:\w+\s+){{0,3}}change)|negative for|free of|absence of|absent|"
    r"resolv\w*|clear(?:s|ed| of)?|no evidence of|no sign\w* of|nor|unremarkable|ruled out|"
    r"rather than|instead of)\b", re.IGNORECASE)
# a clause carrying one of these is a fresh POSITIVE assertion -> it does NOT inherit a prior
# negation. Only CLEAR positive-state words; ambiguous PREDICATES (is/are/seen/noted/present/
# demonstrated) are EXCLUDED because "no X is seen" / "no A, B or C is present/demonstrated" are
# negated predicates, not resets — including them flipped whole negation lists positive (the #1
# detection-FP cause, e.g. effusion).
_POS_MARKER = re.compile(
    r"\b(stable|unchanged|new|increas\w*|worsen\w*|improv\w*|persist\w*|develop\w*|enlarg\w*)\b",
    re.IGNORECASE)

_PROG = {
    "worsened": re.compile(r"\b(worsen\w*|increas\w*|enlarg\w*|larger|more |progress\w*|"
                           r"develop\w*|new\b|interval (development|increase|worsening))", re.IGNORECASE),
    "improved": re.compile(r"\b(improv\w*|decreas\w*|smaller|less |resolv\w*|resolution|clearing|"
                           r"reduced|interval (improvement|decrease|resolution|clearing))", re.IGNORECASE),
    "stable": re.compile(r"\b(unchanged|stable|no (significant )?(interval )?change|similar|"
                         r"persist\w*|redemonstrat\w*)", re.IGNORECASE),
}
_PROG_PRIORITY = {"stable": 0, "improved": 1, "worsened": 2}


def _progression(text: str) -> str | None:
    best = None
    for name, rx in _PROG.items():
        if rx.search(text):
            if best is None or _PROG_PRIORITY[name] > _PROG_PRIORITY[best]:
                best = name
    return best


# ---------------------------------------------------------------------------
# 4. Anatomy lexicon: location words -> 29 bbox regions (with containment), + per-finding defaults
# ---------------------------------------------------------------------------
def _locate(text: str) -> set[str] | None:
    """Map location words in a clause to bbox regions. None if no location is stated."""
    bilat = re.search(r"\b(bilateral|bibasilar|biapical|both|diffuse|throughout)\b", text)
    sides = []
    if bilat or re.search(r"\bleft\b", text):
        sides.append("left")
    if bilat or re.search(r"\bright\b", text):
        sides.append("right")
    zoned_sides = sides or ["left", "right"]   # a zone word with no side -> assume bilateral

    r: set[str] = set()
    lower = re.search(r"\b(base|bases|basal|basilar|bibasilar|lower)\b", text)
    mid = re.search(r"\b(mid|middle)\b", text)
    upper = re.search(r"\b(upper|apex|apical|apices|apico)\b", text)
    cp = re.search(r"costophrenic|\bcp angle\b", text)
    hilar = re.search(r"\b(perihilar|hilar|hilum|hila)\b", text)
    lung = re.search(r"\b(lung|lungs|pulmonary|parenchym)\b", text)
    for s in zoned_sides:
        if lower:
            r |= {f"{s} lung", f"{s} lower lung zone"}
        if mid:
            r |= {f"{s} lung", f"{s} mid lung zone"}
        if upper:
            r |= {f"{s} lung", f"{s} upper lung zone", f"{s} apical zone"}
        if cp:
            r.add(f"{s} costophrenic angle")
        if hilar:
            r.add(f"{s} hilar structures")
        if lung and not (lower or mid or upper):
            r.add(f"{s} lung")

    if re.search(r"\b(cardiomegaly|cardiac|heart|cardio)\b", text):
        r.add("cardiac silhouette")
    if re.search(r"retrocardiac", text):
        r |= {"left lower lung zone", "left lung"}
    if re.search(r"\bmediastin", text):
        r |= {"mediastinum", "upper mediastinum"}
    if re.search(r"\baort", text):
        r.add("aortic arch")
    if re.search(r"\btrachea|tracheal\b", text):
        r.add("trachea")
    if re.search(r"\bcarina|carinal\b", text):
        r.add("carina")
    if re.search(r"\b(spine|spinal|vertebr|thoracic spine)\b", text):
        r.add("spine")
    if re.search(r"diaphragm", text):
        r |= {f"{s} hemidiaphragm" for s in zoned_sides}
    if re.search(r"clavic", text):
        r |= {f"{s} clavicle" for s in zoned_sides}
    if re.search(r"\b(abdom|bowel|stomach|sub.?diaphragm|gastric)\b", text):
        r.add("abdomen")
    if re.search(r"\b(svc|superior vena cava)\b", text):
        r.add("svc")
    return r & _REGIONS or None


# region set a finding gets when the report states NO location (ImaGenome-style defaults)
_DEFAULT = {
    "pleural effusion": {"left costophrenic angle", "right costophrenic angle",
                         "left lower lung zone", "right lower lung zone", "left lung", "right lung"},
    "costophrenic angle blunting": {"left costophrenic angle", "right costophrenic angle"},
    "enlarged cardiac silhouette": {"cardiac silhouette"},
    "pneumothorax": {"left apical zone", "right apical zone", "left lung", "right lung"},
    "hydropneumothorax": {"left lung", "right lung", "left costophrenic angle", "right costophrenic angle"},
    "atelectasis": {"left lower lung zone", "right lower lung zone", "left lung", "right lung"},
    "linear/patchy atelectasis": {"left lower lung zone", "right lower lung zone"},
    "vascular congestion": {"left hilar structures", "right hilar structures", "left lung", "right lung"},
    "pulmonary edema/hazy opacity": {"left lung", "right lung"},
    "mediastinal widening": {"mediastinum", "upper mediastinum"},
    "mediastinal displacement": {"mediastinum"},
    "tortuous aorta": {"aortic arch", "mediastinum"},
    "vascular calcification": {"aortic arch"},
    "enlarged hilum": {"left hilar structures", "right hilar structures"},
    "endotracheal tube": {"trachea", "carina"},
    "tracheostomy tube": {"trachea"},
    "enteric tube": {"mediastinum", "abdomen"},
    "cardiac pacer and wires": {"cardiac silhouette", "left lung"},
    "cabg grafts": {"mediastinum", "cardiac silhouette"},
    "prosthetic valve": {"cardiac silhouette"},
    "sub-diaphragmatic air": {"abdomen"},
    "elevated hemidiaphragm": {"left hemidiaphragm", "right hemidiaphragm"},
    "scoliosis": {"spine"},
    "spinal degenerative changes": {"spine"},
}
_GENERIC_DEFAULT = {"left lung", "right lung"}

# DATA-DRIVEN defaults: per finding, the regions silver occupies in >= DEFAULT_THRESH of that
# finding's positive images (mined from train by mine in eval; rule_region_priors.json). This
# replaces the hand defaults above and matches silver's actual region breadth -> higher recall.
DEFAULT_THRESH = float(os.environ.get("RULE_DEFAULT_THRESH", "0.55"))


def _load_priors() -> dict:
    for c in (Path(__file__).resolve().parent / "rule_region_priors.json",
              Path("rule_region_priors.json")):
        if c.exists():
            return json.loads(c.read_text(encoding="utf-8"))
    return {}


_PRIORS = _load_priors()
_DATA_DEFAULT = {f: {r for r, frac in pr.items() if frac >= DEFAULT_THRESH and r in _REGIONS}
                 for f, pr in _PRIORS.items()}

# PLAUSIBLE regions per finding = every region silver EVER tags it in (>= ALLOW_THRESH of its
# positive images). Used to FILTER `loc`: a clause's location words can belong to a NEIGHBOURING
# finding ("cardiomegaly with bibasilar atelectasis" -> the heart finding must not inherit the
# basilar zones, nor a lung finding the cardiac region). loc is kept only where the finding is
# anatomically plausible. Low threshold (it's a sanity mask, not a default).
ALLOW_THRESH = float(os.environ.get("RULE_ALLOW_THRESH", "0.05"))
_ALLOWED = {f: {r for r, frac in pr.items() if frac >= ALLOW_THRESH and r in _REGIONS}
            for f, pr in _PRIORS.items()}


# concept PARENT-CHILD propagation (ImaGenome ontology, Table 2): a child finding implies its
# parent in the SAME region (consolidation -> lung opacity). Mined from silver
# (rule_concept_parents.json); transitive closure so collapse -> atelectasis -> lung opacity.
def _load_parents() -> dict:
    for c in (Path(__file__).resolve().parent / "rule_concept_parents.json",
              Path("rule_concept_parents.json")):
        if c.exists():
            return json.loads(c.read_text(encoding="utf-8"))
    return {}


def _closure(parents: dict) -> dict[str, set]:
    out: dict[str, set] = {}
    for child in parents:
        seen: set[str] = set()
        stack = list(parents[child])
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            stack.extend(parents.get(a, []))
        out[child] = {a for a in seen if a in VALID_FINDINGS}
    return out


_ANCESTORS = _closure(_load_parents())

# region CONTAINMENT (ImaGenome Table 2): a finding in a sub-region implies the encompassing region
# (sub-zone/cp/hilar/apical -> that side's lung; mediastinal substructures -> mediastinum). Gated by
# RULE_REGION_CONTAINMENT because the data-driven footprint defaults already tag the parent regions
# (mined from silver) — see the empirical NOTE at the bottom of this block.
_REGION_PARENTS = {}
for _s in ("left", "right"):
    for _sub in ("apical zone", "upper lung zone", "mid lung zone", "lower lung zone",
                 "costophrenic angle", "hilar structures"):
        _REGION_PARENTS[f"{_s} {_sub}"] = f"{_s} lung"
for _m in ("aortic arch", "upper mediastinum", "svc", "carina", "cavoatrial junction",
           "right atrium", "trachea"):
    _REGION_PARENTS[_m] = "mediastinum"
_REGION_PARENTS = {k: v for k, v in _REGION_PARENTS.items() if k in _REGIONS and v in _REGIONS}
_USE_CONTAINMENT = os.environ.get("RULE_REGION_CONTAINMENT", "0") == "1"
# When the report explicitly localizes a finding, use ONLY those regions (don't union the broad
# marginal footprint). Attacks footprint over-spray = 48% of all FPs. Tuned/decided empirically.
_LOC_ONLY = os.environ.get("RULE_LOC_ONLY", "0") == "1"
# Localize each comma-clause independently so findings in one segment don't bleed each other's
# regions ("cardiomegaly, bibasilar atelectasis" -> heart finding must not get basilar zones).
_CLAUSE_LOC = os.environ.get("RULE_CLAUSE_LOC", "1") == "1"
# Forward-scope negation: negate only triggers AFTER a negation cue in the clause ("X with no Y" ->
# X positive). Off = the old whole-clause negation (any cue negates everything in the clause).
_FWD_NEG = os.environ.get("RULE_FWD_NEG", "1") == "1"
# Silver rolls each report up to the LAST mention per (region, finding) (ImaGenome postprocessing).
# _LAST_WINS replicates that rollup, but tested SLIGHTLY WORSE (F1 0.857 vs 0.860) — yes>no priority
# agrees with silver better in practice (our mention extraction differs from theirs). Default OFF.
_LAST_WINS = os.environ.get("RULE_LAST_WINS", "0") == "1"
# NOTE: explicit region CONTAINMENT propagation was tried (both before and AFTER the mined-lexicon
# recall boost) and gives NO gain: the data-driven footprint defaults already include the parent
# regions (they are mined from silver, which tags them), so containment is already baked in. Default
# OFF to stay lean; flip RULE_REGION_CONTAINMENT=1 to reproduce the null result.


def _segment_side(text: str) -> str | None:
    """The single laterality stated in a segment, else None (bilateral / unstated -> None)."""
    if re.search(r"\b(bilateral|bibasilar|biapical|both)\b", text):
        return None
    left, right = re.search(r"\bleft\b", text), re.search(r"\bright\b", text)
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def _lateralize(regions: set[str], side: str | None) -> set[str]:
    """Drop the opposite side's regions when a single side is stated (keep side-neutral ones)."""
    if side is None:
        return regions
    other = "right " if side == "left" else "left "
    return {r for r in regions if not r.startswith(other)}


# ZONE CLIP: the 12 zone-specific lung regions (clipped to the zones the report actually names).
# Lungs / mediastinum / cardiac / bones stay untouched — only cross-zone spray is removed.
_ZONE_REGIONS = {f"{s} {z}" for s in ("left", "right")
                 for z in ("upper lung zone", "mid lung zone", "lower lung zone", "apical zone",
                           "costophrenic angle", "hilar structures")}
# NOTE: zone-clip (restrict footprint to the named zone) tested NULL/slightly negative
# (F1 0.784 off vs 0.781 on) — clipping costs more recall than the over-spray it removes. Default
# OFF; flip RULE_ZONE_CLIP=1 to reproduce. (loc-only was worse still: F1 0.674.)
_USE_ZONE_CLIP = os.environ.get("RULE_ZONE_CLIP", "0") == "1"


def _zone_allowed(text: str, sides: list[str]) -> set[str] | None:
    """Regions compatible with the zone words in this segment (per side). None if no zone stated, in
    which case nothing is clipped. Keeps the footprint's recall but only within the named zone."""
    lower = re.search(r"\b(base|bases|basal|basilar|bibasilar|lower|infrahilar)\b", text)
    mid = re.search(r"\b(mid|middle)\b", text)
    upper = re.search(r"\b(upper|apex|apical|apices|apico)\b", text)
    cp = re.search(r"costophrenic|\bcp angle\b", text)
    hilar = re.search(r"\b(perihilar|hilar|hilum|hila)\b", text)
    if not (lower or mid or upper or cp or hilar):
        return None
    allowed: set[str] = set()
    for s in sides:
        if lower:
            allowed |= {f"{s} lower lung zone", f"{s} costophrenic angle"}
        if mid:
            allowed |= {f"{s} mid lung zone", f"{s} hilar structures"}
        if upper:
            allowed |= {f"{s} upper lung zone", f"{s} apical zone"}
        if cp:
            allowed |= {f"{s} costophrenic angle", f"{s} lower lung zone"}
        if hilar:
            allowed |= {f"{s} hilar structures"}
    return allowed


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
# segment boundaries: sentence stops + contrastive conjunctions. NOT newlines — MIMIC reports
# line-wrap mid-sentence, so a "\n" would wrongly cut a negation list ("No a,\nb or c") and leak
# the tail as positive. parse_report collapses whitespace first.
_SEG_SPLIT = re.compile(r"[.;:]| but | however | otherwise | aside from | except ", re.IGNORECASE)
_PRESENCE_PRIORITY = {"no": 0, "yes": 1}

# SECTION FILTER: only FINDINGS / IMPRESSION carry actual observations. INDICATION / HISTORY /
# COMPARISON etc. carry the clinical QUESTION ("assess for pneumonia") — parsing them fires findings
# that silver never labels (the #1 detection-FP source). Drop those sections; keep the rest.
_SECTION_HDR = re.compile(r"(?:^|\n)\s*([A-Z][A-Z0-9 /()_'.-]{2,40}?):", re.MULTILINE)
_DROP_SECTION = re.compile(
    r"\b(INDICATION|HISTORY|CLINICAL|COMPARISON|TECHNIQUE|EXAMINATION|REASON|PROCEDURE|"
    r"NOTIFICATION|REQUEST|INFORMATION|DOSE|REFERENCE|SIGNED|DICTAT|ATTENDING|PHYSICIAN)\b")
_USE_SECTION_FILTER = os.environ.get("RULE_SECTION_FILTER", "1") == "1"


def _strip_nonfinding(report: str) -> str:
    """Keep only finding-bearing sections. Headers are ALLCAPS '...:'; drop the clinical-context
    ones (INDICATION/HISTORY/COMPARISON/...). Preamble before the first header is dropped too
    (usually 'FINAL REPORT' / wet-read boilerplate). Falls back to the whole report if no headers."""
    hdrs = list(_SECTION_HDR.finditer(report))
    if not hdrs:
        return report
    kept = []
    for i, m in enumerate(hdrs):
        header = m.group(1)
        body_start = m.end()
        body_end = hdrs[i + 1].start() if i + 1 < len(hdrs) else len(report)
        if _DROP_SECTION.search(header):
            continue
        kept.append(report[body_start:body_end])
    return " ".join(kept) if kept else report


def _affirm(acc: dict, region: str, finding: str, presence: str, hedged: bool, prog: str | None):
    """Update acc[region][finding]: presence, certain (any non-hedged yes), progression.
    Presence: silver rolls a report up to the LAST mention per (region,finding) — see ImaGenome
    paper §Methods. _LAST_WINS replicates that; otherwise yes>no priority."""
    slot = acc.setdefault(region, {}).setdefault(
        finding, {"presence": presence, "certain": False, "prog": None})
    if _LAST_WINS:
        slot["presence"] = presence                       # last mention wins (silver rollup)
    elif _PRESENCE_PRIORITY[presence] > _PRESENCE_PRIORITY[slot["presence"]]:
        slot["presence"] = presence
    if presence == "yes":
        if not hedged:
            slot["certain"] = True
        if not hedged and prog is not None:
            cur = slot["prog"]
            if cur is None or _PROG_PRIORITY[prog] > _PROG_PRIORITY[cur]:
                slot["prog"] = prog



def parse_report(report: str, available_regions=None) -> dict[str, list[dict]]:
    """report text -> flat {region: [{finding, presence, uncertain?, progression?}]}."""
    avail = set(available_regions) if available_regions else set(CLASS_NAMES)
    report = report or ""
    if _USE_SECTION_FILTER:                        # keep only FINDINGS/IMPRESSION, drop INDICATION/etc
        report = _strip_nonfinding(report)
    report = re.sub(r"\s+", " ", report)          # collapse MIMIC line-wrap newlines -> spaces
    # region -> finding -> flags
    acc: dict[str, dict[str, dict]] = {}

    for segment in _SEG_SPLIT.split(report):
        seg_low = segment.lower()
        if not seg_low.strip():
            continue
        hedged = is_hedged(segment)
        prog = _progression(seg_low)
        seg_loc = _locate(seg_low)
        seg_side = _segment_side(seg_low)
        inherited_neg = False                          # negation carried across a comma-list
        for clause in re.split(r",", seg_low):
            neg_m = _NEG.search(clause)
            neg_pos = neg_m.start() if neg_m else None
            pos_m = _POS_MARKER.search(clause)
            # localize PER CLAUSE, not per segment: in "cardiomegaly, bibasilar atelectasis" the heart
            # finding must not inherit the basilar zones (and vice-versa). Fall back to the segment's
            # location only when the clause itself states none (e.g. a location named before the comma).
            if _CLAUSE_LOC:
                cl_loc = _locate(clause)
                loc = cl_loc if cl_loc is not None else seg_loc
                cl_side = _segment_side(clause)
                side = cl_side if cl_side is not None else seg_side
            else:
                loc, side = seg_loc, seg_side
            za = _zone_allowed(clause if _CLAUSE_LOC else seg_low,
                               [side] if side else ["left", "right"]) if _USE_ZONE_CLIP else None
            for concept, rx in _TRIGGERS:
                m = rx.search(clause)
                if not m:
                    continue
                # FORWARD-SCOPE NegEx: a trigger is negated only if a negation cue appears BEFORE it
                # in the clause ("X with no Y" -> X positive, Y negated), else it inherits the list
                # state ("No A, B" -> both negated). _FWD_NEG gates the old whole-clause behaviour.
                if _FWD_NEG:
                    negated = True if (neg_pos is not None and m.start() > neg_pos) else inherited_neg
                else:
                    negated = inherited_neg or neg_pos is not None
                presence = "no" if negated else "yes"
                # region set. When the report EXPLICITLY localizes the finding (loc), trust that
                # (optionally intersect the footprint to fill in silver's implied sub-regions) instead
                # of unioning the WHOLE footprint — the union is the #1 over-spray source (48% of FPs).
                # When unlocalized, fall back to the data-driven footprint (lateralized to any side).
                base = _DATA_DEFAULT.get(concept) or _DEFAULT.get(concept) or _GENERIC_DEFAULT
                if loc and _LOC_ONLY:
                    regions = set(loc)          # report's call-out, not the broad footprint
                else:
                    regions = _lateralize(base, side)
                    if za is not None:          # clip cross-zone spray to the named zone(s)
                        regions = {r for r in regions if r not in _ZONE_REGIONS or r in za}
                    if loc:                     # add the report's location, but only where this
                        al = _ALLOWED.get(concept)      # finding is anatomically plausible (mask out
                        regions = regions | (loc & al if al is not None else loc)  # neighbour bleed)
                regions = regions & avail
                for rg in regions:
                    _affirm(acc, rg, concept, presence, hedged, prog)
                    if presence == "yes":                       # concept parent propagation:
                        for anc in _ANCESTORS.get(concept, ()):  # child implies parent, same region
                            _affirm(acc, rg, anc, "yes", hedged, None)
            # carry the negation state to the next comma-clause: "No A, B or C" keeps negating;
            # a fresh positive assertion (stable/new/increased/...) ends the negated list.
            if neg_m:
                inherited_neg = True
            elif pos_m:
                inherited_neg = False

    if _USE_CONTAINMENT:            # sub-region positive -> encompassing region positive (gated)
        for child, parent in _REGION_PARENTS.items():
            if parent not in avail:
                continue
            for finding, v in list(acc.get(child, {}).items()):
                if v["presence"] == "yes":
                    _affirm(acc, parent, finding, "yes", not v["certain"], v["prog"])

    out: dict[str, list[dict]] = {}
    for region in CLASS_NAMES:
        fmap = acc.get(region)
        if not fmap:
            continue
        items = []
        for finding in FINDINGS_IN_ORDER:
            v = fmap.get(finding)
            if v is None or finding not in VALID_FINDINGS:
                continue
            item = {"finding": finding, "presence": v["presence"]}
            if v["presence"] == "yes" and not v["certain"]:     # uncertain = no certain mention
                item["uncertain"] = True
            elif v["presence"] == "yes" and v["prog"]:
                item["progression"] = v["prog"]
            items.append(item)
        if items:
            out[region] = items
    return out


if __name__ == "__main__":   # quick demo
    import json
    demo = ("Stable bibasilar atelectasis. Small left pleural effusion, increased. "
            "No pneumothorax. Possible right lower lobe pneumonia. Cardiomegaly unchanged.")
    print(json.dumps(parse_report(demo), ensure_ascii=False, indent=1))
