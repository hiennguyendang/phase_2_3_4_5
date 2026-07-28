"""Run a trained M3 model -> per-image predictions for M4/M5.

Emits one JSON line per image:
  image_id, image_disease[14] (prob), region_disease[29][14] (prob),
  region_concepts[29] -> {concept: prob for present regions, top-k},
  region_feats are NOT dumped (large) — M4 should re-run the model or read a feature dump.

For now boxes/regions come from the M3 label arrays (MIMIC). A detector-box source
for CheXplus/NIH can be plugged into M3Dataset later (same shapes).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_3/src

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
import constants as C
from dataset import M3Dataset, collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3 inference -> per-image JSON")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, default=config.DEFAULT_LABELS_DIR)
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "m3_pred.jsonl")
    p.add_argument("--topk-concepts", type=int, default=8)
    p.add_argument("--all-concepts", action="store_true",
                   help="dump all 69 concept probabilities (calibration/audit only; large output)")
    p.add_argument("--concept-gate", type=Path, default=None,
                   help="pair-specific (region,concept) display gate from 11-calibrate_report.py")
    p.add_argument("--topk-cells", type=int, default=0,
                   help="dump top-k attention-pool grid cells per region (M5 'where'); 0 = off")
    p.add_argument("--all-region-diseases", action="store_true",
                   help="dump all 29x14 region-disease probabilities for dual-threshold readout")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--box-source", choices=["detector", "gt"], default=config.BOX_SOURCE,
                   help="bbox source: detector (default) or gt (oracle ablation)")
    p.add_argument("--image-id", action="append", default=None,
                   help="restrict inference to one or more image IDs (repeatable; useful for a demo)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_concept_gate(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    regions = data.get("region_by_name") if isinstance(data, dict) else None
    if not isinstance(regions, dict):
        raise SystemExit("[ERROR] concept gate must contain pair-specific region_by_name entries")
    return data


def _concept_gate_item(gate: dict | None, region: str, concept: str) -> dict | None:
    if gate is None:
        return None
    item = ((gate.get("region_by_name") or {}).get(region) or {}).get(concept)
    return item if isinstance(item, dict) else None


@torch.no_grad()
def main() -> int:
    import model as M
    args = parse_args()
    ck = torch.load(args.ckpt, map_location=args.device)
    config.apply(ck.get("cfg", {}))                     # rebuild the exact trained architecture
    config.USE_GLOBAL_TOKEN = ck.get("use_global", config.USE_GLOBAL_TOKEN)
    m = M.build_model(ck["feat_dim"], ck["mode"]).to(args.device).eval()
    m.load_state_dict(ck["model"])
    concept_gate = _load_concept_gate(args.concept_gate)
    checkpoint_hash = _sha256(args.ckpt)
    manifest_hash = _sha256(args.labels_dir / "manifest.jsonl")
    concept_gate_hash = _sha256(args.concept_gate) if args.concept_gate is not None else None
    if concept_gate is not None:
        gate_provenance = concept_gate.get("provenance") or {}
        expected_checkpoint = gate_provenance.get("checkpoint_sha256")
        expected_box_source = gate_provenance.get("box_source")
        expected_manifest = gate_provenance.get("manifest_sha256")
        if expected_checkpoint and expected_checkpoint != checkpoint_hash:
            raise SystemExit("[ERROR] concept gate was fitted for a different M3 checkpoint")
        if expected_box_source and expected_box_source != args.box_source:
            raise SystemExit("[ERROR] concept gate box_source does not match inference")
        if expected_manifest and expected_manifest != manifest_hash:
            raise SystemExit("[ERROR] concept gate was fitted for a different M3 label manifest")
    effective_edges = None
    if hasattr(m, "disease_head") and hasattr(m.disease_head, "weight") \
            and hasattr(m.disease_head, "cmask"):
        effective_edges = (F.softplus(m.disease_head.weight) * m.disease_head.cmask).detach().cpu()

    ds = M3Dataset(args.labels_dir, args.features_root, args.split, box_source=args.box_source)
    if args.image_id:
        wanted = set(args.image_id)
        ds.rows = [row for row in ds.rows if row[1] in wanted]
        if not ds.rows:
            raise SystemExit("[ERROR] none of --image-id values matched the requested split/features")
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for b in loader:
            out = m(b["grid"].to(args.device), b["global"].to(args.device),
                    b["present_mask"].to(args.device), b["boxes"].to(args.device))
            img = torch.sigmoid(out["image_disease_logits"]).cpu()
            rd = None if out.get("region_disease_logits") is None else torch.sigmoid(out["region_disease_logits"]).cpu()
            cc = (torch.sigmoid(out["concept_logits"]).cpu()
                  if out.get("concept_logits") is not None else None)
            attn = out.get("region_attn").cpu() if args.topk_cells and out.get("region_attn") is not None else None
            mask = b["present_mask"]
            for j, iid in enumerate(b["image_id"]):
                rec = {
                    "image_id": iid,
                    "box_source": args.box_source,
                    "m3_checkpoint_sha256": checkpoint_hash,
                    "m3_manifest_sha256": manifest_hash,
                    "concept_gate_sha256": concept_gate_hash,
                    "concept_evidence_policy": ("pair_specific_gate" if concept_gate is not None
                                                else "legacy_topk_0.5"),
                    "image_disease": {C.CHEX_NAMES[c]: round(float(img[j, c]), 4) for c in range(C.NUM_CHEX)},
                    "regions": {},
                }
                if rd is not None:
                    for r in range(C.NUM_REGIONS):
                        if mask[j, r] < 0.5:
                            continue
                        entry = {
                            "bbox": [int(x) for x in b["boxes"][j, r].tolist()],
                            "disease": {C.CHEX_NAMES[c]: round(float(rd[j, r, c]), 3)
                                        for c in range(C.NUM_CHEX)
                                        if args.all_region_diseases or rd[j, r, c] > 0.5},
                        }
                        if cc is not None:
                            region_name = C.REGION_NAMES[r]
                            if args.all_concepts:
                                entry["concepts"] = {
                                    C.CONCEPT_NAMES[ci]: round(float(cc[j, r, ci]), 6)
                                    for ci in range(C.NUM_CONCEPTS)
                                }
                            elif concept_gate is not None:
                                selected = {}
                                for ci, concept_name in enumerate(C.CONCEPT_NAMES):
                                    gate_item = _concept_gate_item(concept_gate, region_name, concept_name)
                                    threshold = (gate_item or {}).get("present_threshold")
                                    probability = float(cc[j, r, ci])
                                    if ((gate_item or {}).get("allowed_for_why")
                                            and threshold is not None and probability >= float(threshold)):
                                        selected[concept_name] = round(probability, 6)
                                entry["concepts"] = selected
                            else:
                                top = torch.topk(cc[j, r], args.topk_concepts)
                                entry["concepts"] = {
                                    C.CONCEPT_NAMES[int(i)]: round(float(p), 3)
                                    for p, i in zip(top.values, top.indices) if p > 0.5
                                }
                            disease_concepts = {}
                            for disease_idx, concept_ids in C.CHEX_FROM_CONCEPTS.items():
                                if not concept_ids:
                                    continue
                                candidates = []
                                for concept_idx in concept_ids:
                                    concept_name = C.CONCEPT_NAMES[concept_idx]
                                    probability = float(cc[j, r, concept_idx])
                                    if concept_gate is not None:
                                        gate_item = _concept_gate_item(concept_gate, region_name, concept_name)
                                        threshold = (gate_item or {}).get("present_threshold")
                                        if (not (gate_item or {}).get("allowed_for_why")
                                                or threshold is None or probability < float(threshold)):
                                            continue
                                    elif probability <= 0.5:
                                        continue
                                    edge_weight = (float(effective_edges[disease_idx, concept_idx])
                                                   if effective_edges is not None else None)
                                    contribution = (probability * edge_weight
                                                    if edge_weight is not None else probability)
                                    candidates.append((contribution, concept_name, probability, edge_weight))
                                candidates.sort(reverse=True)
                                evidence = {
                                    name: {
                                        "prob": round(probability, 6),
                                        "edge_weight": (round(edge_weight, 6)
                                                        if edge_weight is not None else None),
                                        "contribution": round(contribution, 6),
                                    }
                                    for contribution, name, probability, edge_weight
                                    in candidates[:args.topk_concepts]
                                }
                                if evidence:
                                    disease_concepts[C.CHEX_NAMES[disease_idx]] = evidence
                            entry["disease_concepts"] = disease_concepts
                        if attn is not None:                 # faithful "where" cells -> (row, col, weight)
                            tc = torch.topk(attn[j, r], args.topk_cells)
                            entry["cells"] = [[int(i) // config.GRID_W, int(i) % config.GRID_W, round(float(w), 3)]
                                              for w, i in zip(tc.values, tc.indices)]
                        rec["regions"][C.REGION_NAMES[r]] = entry
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    print(f"[DONE] {written:,} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
