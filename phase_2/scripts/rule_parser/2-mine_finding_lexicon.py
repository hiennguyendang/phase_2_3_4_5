"""[PREP — local, no GPU] Mine surface-phrase triggers for the rule_parser's finding lexicon.

ImaGenome's silver scene graphs store, for every asserted attribute, the SOURCE SENTENCE that
produced it (`attributes[i]` is a list of `category|polarity|label` strings, parallel to
`phrases[i]`). This script reads those (TRAIN images only — no val/test leakage) and, for each of
the 69 M3 concepts, learns the n-grams that DISCRIMINATIVELY trigger it. Output:

    rule_finding_triggers.json   { "<concept>": ["pleural effusion", "layering", ...], ... }

rule_parser loads it to SUPPLEMENT the hand-curated lexicon, widening coverage toward ImaGenome's
271-entity radiologist lexicon (the main lever left for the long-tail macro-F1).

Method (high-precision, discriminative, exclusive assignment):
  * a trigger IDENTIFIES which finding a phrase is about; POLARITY is handled downstream by the
    parser's NegEx, so we count a concept MENTION (yes OR no), not just positive ones — else
    "effusion" (often negated, "no pleural effusion") would be wrongly rejected.
  * for every source phrase (deduped per image+phrase) record the set of concepts it MENTIONS, and
    count the phrase against each of its n-grams once.
  * P(concept | n-gram) = (#phrases with n-gram that MENTION concept) / (#phrases with n-gram).
    This rewards CONCENTRATION: a finding word ("effusion", "atelectasis") sits almost only on its
    own concept -> P~1; generic anatomy/filler ("lung", "base", "setting") spreads across many
    concepts -> P collapses -> rejected. Only finding-specific surface forms survive.
  * an n-gram is a trigger for a concept iff P(concept | n-gram) >= MIN_PREC and its mention count
    >= MIN_COUNT. Among all qualifying concepts it is assigned to the SINGLE most SPECIFIC one (the
    one with the smallest corpus) — this routes a child term (e.g. "consolidation") to the child, not
    to a co-tagged parent ("lung opacity") that silver propagated it onto. So every trigger fires
    exactly one concept; parent coverage stays with rule_parser's separate concept-parent pass.

    python mine_finding_lexicon.py --scene-root "C:/.../chest-imagenome" \
        --metadata data/mimic_metadata_final.jsonl

Bundled in phase_2/ so it travels with the code on Kaggle (like rule_region_priors.json).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

_PH2 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_PH2), str(_PH2 / "src")]

import config
from scene_to_yolo import dicom_id_from_image_id, iter_jsonl
from sg_schema import VALID_FINDINGS

# n-gram acceptance knobs. These LOCKED values reproduce the committed rule_finding_triggers.json
# (tuned on full val: presence F1 0.728, macro 0.625, recall 0.836, finding-only 0.833 — vs the
# pre-lexicon 0.717/0.546/0.763/0.800). rethreshold_lexicon.py can sweep tighter cutoffs cheaply
# off the rich sidecar; these are the chosen operating point (concentration>=0.90, pos-rate>=0.35).
MIN_TOTAL = 40      # an n-gram must appear in >= this many phrases (any concept)
MIN_COUNT = 40      # ... and MENTION the winning concept in >= this many phrases
MIN_PREC = 0.90     # P(concept | n-gram): concentration of the n-gram on one concept
MIN_POS_RATE = 0.35  # of the n-gram's mentions of the concept, >= this fraction must be POSITIVE
#                     (drops absence/normality words like "clear","lungs are clear" that concentrate
#                      on a concept but only ever NEGATE it)
MAX_PER_CONCEPT = 25
MAX_N = 3           # up to tri-grams

# generic structure / normality words: an n-gram made ENTIRELY of these (or stopwords) is not a
# finding trigger ("lobe","heart","silhouette","clear") — it would fire on normal anatomy.
_ANATOMY = {
    "lung", "lungs", "lobe", "lobes", "base", "bases", "apex", "apices", "apical", "heart",
    "silhouette", "silhouettes", "hila", "hilar", "hilum", "contour", "contours", "cardiac",
    "cardiomediastinal", "mediastinal", "mediastinum", "diaphragm", "hemidiaphragm",
    "hemidiaphragms", "clear", "clears", "cleared", "sinuses", "sinus", "field", "fields",
    "bilaterally", "borders", "border", "outline", "outlines", "structures", "thorax", "thoracic",
    "pleural", "surfaces", "surface",   # "pleural surfaces are clear" -> not an effusion trigger
    "vascular", "vasculature",          # "vascular calcification" must not trigger vascular congestion
}

# tokens that must not stand alone / anchor a trigger (filler, laterality, severity, comparison)
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "there", "no", "not",
    "of", "in", "on", "at", "to", "and", "or", "with", "without", "for", "as", "by", "from",
    "this", "that", "these", "those", "it", "its", "which", "has", "have", "had", "but", "however",
    "unchanged", "stable", "new", "seen", "noted", "note", "evidence", "redemonstrated",
    "demonstrated", "again", "now", "small", "large", "mild", "moderate", "severe", "minimal",
    "trace", "tiny", "left", "right", "bilateral", "both", "upper", "lower", "mid", "middle",
    "patient", "study", "exam", "radiograph", "chest", "film", "view", "images", "image",
    "compared", "comparison", "prior", "since", "previous", "interval", "likely", "probably",
    "probable", "possible", "possibly", "may", "appears", "appear", "consistent", "suggesting",
    "suggestive", "suggest", "findings", "finding", "within", "normal", "unremarkable", "otherwise",
    "overall", "please", "well", "still", "persistent", "grossly", "similar", "redemonstration",
    "remains", "remain", "include", "including", "associated", "due", "presumed", "presumably",
    "concerning", "concern", "could", "would", "if", "than", "more", "less", "slightly", "mildly",
    "near", "approximately", "first", "second", "third", "size", "amount", "degree", "region",
    "area", "areas", "level", "given", "known", "appearance", "demonstrate", "shows", "show",
    "increased", "decreased", "improved", "worsened", "developing", "developed", "redemonstrate",
    # positional / vague words — risky as STANDALONE triggers (would over-fire); the curated
    # multiword device patterns already carry the real signal (e.g. "et tube", "below diaphragm")
    "above", "below", "though", "over", "into", "along", "across", "toward", "towards", "tip",
    "projects", "projecting", "overlying", "overlies", "positioned", "terminating", "terminates",
    "extends", "extending", "reaches", "loss", "crowding", "though", "seem", "seems",
    # generic hedge / qualifier / reasoning filler that concentrates on the dominant parent concept
    # (lung opacity) without naming a finding — would fire on "cannot be excluded", "in the setting of"
    "represent", "represents", "reflect", "reflects", "reflecting", "excluded", "exclude",
    "clinical", "setting", "cannot", "greater", "extent", "combination", "essentially", "versus",
    "process", "possibility", "density", "densities", "change", "changes", "account", "accounts",
    "related", "attributed", "raises", "raise", "favored", "favor", "favors", "volume", "volumes",
    "markings", "marking", "angle", "angles", "correlate", "correlation", "correlated", "context",
    "superimposed", "supervening", "underlying", "adjacent", "subtle", "early", "definite", "diffuse",
    "worrisome", "concerning", "extensive", "component", "complicated",
    "increase", "increases", "increasing", "improvement", "improving", "worsening", "worse",
    "worsen", "progression", "progressed", "progressing", "considered", "accompanied", "severity",
    "minor", "representing", "widespread", "obscured", "obscures", "obscuring", "development",
    # bare "fracture(s)" is ambiguous across rib/clavicle/spine — let the curated rib/clavicle/spine
    # patterns disambiguate; a generic "fractures" trigger mislabels every fracture as a rib fracture.
    "fracture", "fractures", "fractured",
}
# a trigger must contain at least one "content" token: length >= 4 and not a stopword
_TOK = re.compile(r"[a-z][a-z0-9-]+")


def normalise(phrase: str) -> str:
    return re.sub(r"\s+", " ", phrase).strip().lower()


def tokens(phrase: str) -> list[str]:
    return _TOK.findall(phrase)


def trim(ng: tuple[str, ...]) -> tuple[str, ...] | None:
    """Drop leading/trailing stopwords; require >=1 content token (len>=4, non-stop)."""
    lo, hi = 0, len(ng)
    while lo < hi and ng[lo] in _STOP:
        lo += 1
    while hi > lo and ng[hi - 1] in _STOP:
        hi -= 1
    core = ng[lo:hi]
    if not core:
        return None
    if not any(len(t) >= 4 and t not in _STOP for t in core):
        return None
    return core


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine finding-trigger lexicon from silver phrases")
    p.add_argument("--scene-root", type=Path, default=config.DEFAULT_SCENE_ROOT)
    p.add_argument("--metadata", type=Path, default=config.DEFAULT_METADATA)
    src = Path(__file__).resolve().parents[2] / "src"   # bundled data lives beside rule_parser
    p.add_argument("--out", type=Path, default=src / "rule_finding_triggers.json")
    p.add_argument("--limit", type=int, default=0, help="only first N train images (0 = all)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.scene_root.exists():
        raise SystemExit(f"[ERROR] scene-root not found: {args.scene_root}")
    if not args.metadata.exists():
        raise SystemExit(f"[ERROR] metadata not found: {args.metadata}")

    # 1. TRAIN dicom ids only (no val/test/gold leakage), matching build_sft_dataset's split routing
    train = set()
    for row in iter_jsonl(args.metadata):
        if str(row.get("dataset", "")).lower() not in ("mimic", ""):
            continue
        if str(row.get("split", "")).strip().lower() != "train":
            continue
        d = dicom_id_from_image_id(str(row.get("image_id", "")).strip())
        if d:
            train.add(d)
    print(f"train images: {len(train):,}")

    # 2. stream each train scene graph. Per phrase (deduped per image+phrase), record the set of
    #    concepts it MENTIONS (yes OR no). Count the phrase against each n-gram once for the
    #    denominator and against each mentioned concept for the numerator -> concentration = P.
    corpus = Counter()                                   # concept -> #phrases mentioning it
    ngram_men: dict[tuple, Counter] = defaultdict(Counter)   # ngram -> concept -> #phrases mentioning
    ngram_pos: dict[tuple, Counter] = defaultdict(Counter)   # ngram -> concept -> #phrases POSITIVE
    ngram_total: Counter = Counter()                     # ngram -> #phrases containing it
    n = miss = 0
    for d in train:
        path = args.scene_root / f"{d}_SceneGraph.json"
        if not path.exists():
            miss += 1
            continue
        try:
            scene = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            miss += 1
            continue
        # phrase text -> (mentioned concepts, positively-asserted concepts), unioned across the
        # study's region objects
        men: dict[str, set] = defaultdict(set)
        pos: dict[str, set] = defaultdict(set)
        for region_obj in scene.get("attributes", []):
            attrs = region_obj.get("attributes", [])
            phrases = region_obj.get("phrases", [])
            for group, phrase in zip(attrs, phrases):
                ptext = normalise(phrase)
                if not ptext:
                    continue
                men.setdefault(ptext, set())   # register phrase even if no finding
                pos.setdefault(ptext, set())
                for rel in group:
                    parts = rel.split("|")
                    if len(parts) == 3 and parts[2] in VALID_FINDINGS and parts[1] in ("yes", "no"):
                        men[ptext].add(parts[2])
                        if parts[1] == "yes":
                            pos[ptext].add(parts[2])
        for ptext, cs in men.items():
            for c in cs:
                corpus[c] += 1
            toks = tokens(ptext)
            grams = set()
            for k in range(1, MAX_N + 1):
                for i in range(len(toks) - k + 1):
                    g = trim(tuple(toks[i:i + k]))
                    if g:
                        grams.add(g)
            pset = pos[ptext]
            for g in grams:
                ngram_total[g] += 1
                for c in cs:
                    ngram_men[g][c] += 1
                for c in pset:
                    ngram_pos[g][c] += 1
        n += 1
        if args.limit and n >= args.limit:
            break
        if n % 20000 == 0:
            print(f"  ...{n:,} scenes  (missing files: {miss:,})")
    print(f"scenes read: {n:,}  (missing: {miss:,})  | distinct n-grams: {len(ngram_total):,}")

    # 3. assign each qualifying n-gram to its single MOST-SPECIFIC concept (smallest corpus).
    #    Collect (mention_count, precision, pos_rate, phrase) so the rich sidecar can be
    #    re-thresholded cheaply (rethreshold_lexicon) without re-reading the 148k scenes.
    triggers: dict[str, list[tuple]] = defaultdict(list)
    for g, total in ngram_total.items():
        if total < MIN_TOTAL:
            continue
        if all(t in _STOP or t in _ANATOMY for t in g):     # pure anatomy/normality -> not a trigger
            continue
        winners = [(c, men) for c, men in ngram_men[g].items()
                   if men >= MIN_COUNT and men / total >= MIN_PREC
                   and ngram_pos[g].get(c, 0) / men >= MIN_POS_RATE]
        if not winners:
            continue
        # most specific = smallest concept corpus; tie -> higher mention count
        c, men = min(winners, key=lambda kv: (corpus[kv[0]], -kv[1]))
        triggers[c].append((men, round(men / total, 3),
                            round(ngram_pos[g].get(c, 0) / men, 3), " ".join(g)))

    # 4. per concept: sort by count desc, drop a longer phrase if a shorter accepted one is inside it
    out: dict[str, list[str]] = {}
    rich: dict[str, list] = {}
    for c, lst in triggers.items():
        lst.sort(key=lambda t: (-t[0], len(t[3])))
        kept: list[str] = []
        kept_rich: list = []
        for men, prec, pos_rate, phrase in lst:
            if any(re.search(rf"\b{re.escape(s)}\b", phrase) for s in kept):
                continue            # a shorter accepted trigger already covers this phrase
            kept.append(phrase)
            kept_rich.append([phrase, men, prec, pos_rate])
            if len(kept) >= MAX_PER_CONCEPT:
                break
        out[c] = kept
        rich[c] = kept_rich

    out = {c: out[c] for c in sorted(out, key=lambda c: -len(out[c]))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    rich_path = _PH2 / "_work" / "rule_finding_triggers_rich.json"   # dev sidecar (not bundled)
    rich_path.parent.mkdir(parents=True, exist_ok=True)
    rich_path.write_text(json.dumps(rich, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rich sidecar -> {rich_path}")
    n_trig = sum(len(v) for v in out.values())
    print(f"\n[DONE] {n_trig:,} triggers across {len(out)} concepts -> {args.out.name}")
    for c in list(out)[:25]:
        print(f"  {c:<42} {len(out[c]):>2}  e.g. {', '.join(out[c][:4])}")
    missing = [c for c in VALID_FINDINGS if c not in out]
    print(f"\nconcepts with NO mined trigger ({len(missing)}): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
