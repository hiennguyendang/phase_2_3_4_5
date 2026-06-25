"""Core library for the LLM (attribute/relationship) branch of Module 2.

Round-trip between ImaGenome scene graphs and a COMPACT per-region finding target
that the LLM learns to emit from (report + available regions):

    scene graph  --compact_target_from_scene_graph-->  compact target
    LLM(report, regions)  -->  compact text  --parse_compact-->  compact target
    compact target + detector boxes  --assemble_scene_graph-->  *_SceneGraph.json

The compact target is:
    { "<region>": [ {"rel": [...], "comparison": [...], "temporal": [...],
                     "severity": [...], "texture": [...]}, ... ], ... }
Only the 29 canonical regions are kept; empty cue lists are omitted.
"""

from __future__ import annotations

import json
import re
from typing import Any

from constants import CLASS_NAMES, canonical_name

# Cue arrays carried through the pipeline. comparison_cues -> Module 4 (T-KAN).
CUE_FIELDS = ("temporal_cues", "severity_cues", "texture_cues", "comparison_cues")
# key in compact dict  <->  scene-graph cue field
_CUE_COMPACT_TO_SCENE = {
    "temporal": "temporal_cues",
    "severity": "severity_cues",
    "texture": "texture_cues",
    "comparison": "comparison_cues",
}
# Relation prefixes we keep (seen across ImaGenome silver).
RELATION_PREFIXES = (
    "anatomicalfinding", "nlp", "disease", "tubesandlines",
    "technicalassessment", "device",
)


# ---------------------------------------------------------------------------
# scene graph -> compact target
# ---------------------------------------------------------------------------
def compact_target_from_scene_graph(scene: dict[str, Any]) -> dict[str, list[dict]]:
    """Distill scene['attributes'] into the compact per-region target."""
    out: dict[str, list[dict]] = {}
    for entry in scene.get("attributes", []) or []:
        region = canonical_name(str(entry.get("bbox_name", "")))
        if region is None:
            continue
        rel_lists = entry.get("attributes") or []
        n = len(rel_lists)
        cues = {k: (entry.get(v) or []) for k, v in _CUE_COMPACT_TO_SCENE.items()}

        findings: list[dict] = []
        for i in range(n):
            rels = [r for r in (rel_lists[i] or []) if _keep_relation(r)]
            if not rels:
                continue
            finding: dict[str, list[str]] = {"rel": rels}
            for ckey, clist in cues.items():
                vals = clist[i] if i < len(clist) else []
                if vals:
                    finding[ckey] = list(vals)
            findings.append(finding)

        if findings:
            out.setdefault(region, []).extend(findings)
    return out


def _keep_relation(rel: str) -> bool:
    return isinstance(rel, str) and rel.split("|", 1)[0] in RELATION_PREFIXES


# ---------------------------------------------------------------------------
# compact target <-> text (for the LLM)
# ---------------------------------------------------------------------------
def dump_compact(compact: dict[str, list[dict]]) -> str:
    """Serialize compact target to the canonical JSON string the LLM should emit."""
    return json.dumps(compact, ensure_ascii=False)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_compact(text: str) -> dict[str, list[dict]]:
    """Parse the LLM output back into a compact target, tolerant of code fences /
    surrounding prose. Returns {} if nothing valid is found."""
    if not text:
        return {}
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start == -1:
        return {}
    # walk to the matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    obj = json.loads(blob)
                except json.JSONDecodeError:
                    return {}
                return _clean_compact(obj)
    return {}


def _clean_compact(obj: Any) -> dict[str, list[dict]]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, list[dict]] = {}
    for region, findings in obj.items():
        cname = canonical_name(str(region))
        if cname is None or not isinstance(findings, list):
            continue
        clean: list[dict] = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            rels = [r for r in (f.get("rel") or []) if isinstance(r, str)]
            if not rels:
                continue
            item: dict[str, list[str]] = {"rel": rels}
            for ck in ("comparison", "temporal", "severity", "texture"):
                vals = [v for v in (f.get(ck) or []) if isinstance(v, str)]
                if vals:
                    item[ck] = vals
            clean.append(item)
        if clean:
            out[cname] = clean
    return out


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a radiology scene-graph extractor. Given a chest X-ray report and the "
    "list of anatomical regions detected in the image, output ONLY a compact JSON "
    "object mapping each region that has a finding to a list of findings. Each "
    "finding has: \"rel\" (a list of relation strings like "
    "\"anatomicalfinding|yes|atelectasis\"), and optionally \"comparison\", "
    "\"temporal\", \"severity\", \"texture\" (lists of cue strings like "
    "\"comparison|yes|worsened\"). Only use regions from the provided list. Do not "
    "invent findings not supported by the report. Output JSON only, no prose."
)


def build_user_prompt(report: str, regions: list[str]) -> str:
    region_menu = ", ".join(regions)
    report = (report or "").strip()
    return (
        f"Available regions: {region_menu}\n\n"
        f"Report:\n{report}\n\n"
        "Compact scene graph JSON:"
    )


# ---------------------------------------------------------------------------
# bbox helpers
# ---------------------------------------------------------------------------
def bbox_pixel_from_norm(xc: float, yc: float, w: float, h: float,
                         img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """YOLO-normalized (cx,cy,w,h) -> integer pixel (x1,y1,x2,y2)."""
    x1 = round((xc - w / 2) * img_w)
    y1 = round((yc - h / 2) * img_h)
    x2 = round((xc + w / 2) * img_w)
    y2 = round((yc + h / 2) * img_h)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# compact target + detector boxes -> full ImaGenome-style scene graph
# ---------------------------------------------------------------------------
def assemble_scene_graph(
    image_id: str,
    objects: list[dict],
    compact: dict[str, list[dict]],
    *,
    viewpoint: str | None = None,
    patient_id: Any = None,
    study_id: Any = None,
    report: str = "",
) -> dict[str, Any]:
    """Build a *_SceneGraph.json-style dict.

    objects: detector output, each {"bbox_name","x1","y1","x2","y2"[,"conf"]}.
    Findings are attached ONLY to regions the detector actually produced.
    """
    detected = {}
    obj_list = []
    for o in objects:
        region = canonical_name(str(o.get("bbox_name", "")))
        if region is None:
            continue
        oid = f"{image_id}_{region}"
        obj_list.append({
            "object_id": oid,
            "bbox_name": region,
            "x1": o.get("x1"), "y1": o.get("y1"),
            "x2": o.get("x2"), "y2": o.get("y2"),
            "name": region.title(),
            "score": o.get("conf"),
        })
        detected[region] = oid

    attributes = []
    for region, findings in compact.items():
        oid = detected.get(region)
        if oid is None:               # detector didn't find this region -> skip
            continue
        rel_lists, cmp_c, tmp_c, sev_c, tex_c, phrases = [], [], [], [], [], []
        for f in findings:
            rel_lists.append(list(f.get("rel", [])))
            cmp_c.append(list(f.get("comparison", [])))
            tmp_c.append(list(f.get("temporal", [])))
            sev_c.append(list(f.get("severity", [])))
            tex_c.append(list(f.get("texture", [])))
            phrases.append("")
        attributes.append({
            region: True,
            "bbox_name": region,
            "name": region.title(),
            "attributes": rel_lists,
            "phrases": phrases,
            "comparison_cues": cmp_c,
            "temporal_cues": tmp_c,
            "severity_cues": sev_c,
            "texture_cues": tex_c,
            "object_id": oid,
        })

    return {
        "image_id": image_id,
        "viewpoint": viewpoint,
        "patient_id": patient_id,
        "study_id": study_id,
        "objects": obj_list,
        "attributes": attributes,
        "relationships": [],
        "_generated": True,
    }


def snap_to_vocab(compact: dict[str, list[dict]], vocab: dict[str, Any]) -> dict[str, list[dict]]:
    """Drop anything the LLM emitted that is outside the controlled vocab
    (relations not allowed for that region; cue values not in the cue lists).
    Removes hallucinations before assembly."""
    regions_vocab = vocab.get("regions", {})
    cues_vocab = vocab.get("cues", {})
    out: dict[str, list[dict]] = {}
    for region, findings in compact.items():
        allowed = set(regions_vocab.get(region, []))
        if not allowed:
            continue
        kept: list[dict] = []
        for f in findings:
            rels = [r for r in f.get("rel", []) if r in allowed]
            if not rels:
                continue
            item: dict[str, list[str]] = {"rel": rels}
            for ck in ("comparison", "temporal", "severity", "texture"):
                allowed_cue = set(cues_vocab.get(ck, []))
                vals = [v for v in f.get(ck, []) if v in allowed_cue]
                if vals:
                    item[ck] = vals
            kept.append(item)
        if kept:
            out[region] = kept
    return out


def assemble_objects_from_scene(scene: dict[str, Any]) -> list[dict]:
    """Objects list (with bbox_name) from an existing silver scene graph."""
    return scene.get("objects", []) or []


def available_regions(objects: list[dict]) -> list[str]:
    """The region menu offered to the LLM at SFT/inference time = detected regions
    ∩ 29 canonical, in CLASS_NAMES order."""
    have = {canonical_name(str(o.get("bbox_name", ""))) for o in objects}
    have.discard(None)
    return [r for r in CLASS_NAMES if r in have]
