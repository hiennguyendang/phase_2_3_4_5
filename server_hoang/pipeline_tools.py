#!/usr/bin/env python3
"""Small, testable filesystem operations used by run_all.sh."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def manifest_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig") as handle:
        return {str(json.loads(line)["image_id"]) for line in handle if line.strip()}


def expected_shard_rows(manifest: Path, shard: int, num_shards: int) -> int:
    total = len(manifest_ids(manifest))
    return max(0, (total - shard + num_shards - 1) // num_shards)


def command_sha256(args: argparse.Namespace) -> None:
    print(sha256_file(args.path))


def command_preflight(args: argparse.Namespace) -> None:
    for module in ("torch", "numpy", "ultralytics", "sklearn", "matplotlib", "pandas"):
        importlib.import_module(module)
    import torch

    requested = [int(item) for item in args.gpu_ids.split(",") if item]
    if not torch.cuda.is_available():
        raise RuntimeError("torch cannot access CUDA")
    print("torch:", torch.__version__, "visible CUDA devices:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        print("  cuda", index, torch.cuda.get_device_name(index))
    print("requested physical GPU IDs:", requested)

    copies = [
        args.repo_root / "data/m3_concept_space.json",
        args.repo_root / "phase_3/src/m3_concept_space.json",
        args.repo_root / "phase_4/src/m3_concept_space.json",
    ]
    objects = [json.loads(path.read_text(encoding="utf-8-sig")) for path in copies]
    if not all(value == objects[0] for value in objects[1:]):
        raise RuntimeError("concept-space copies differ")
    print("concept-space copies agree")


def command_mark_stage(args: argparse.Namespace) -> None:
    atomic_json(
        args.dest,
        {"stage": args.stage, "status": "complete", "completed_at": args.completed_at},
    )


def command_m2_contract(args: argparse.Namespace) -> None:
    wanted = {
        "schema_version": 1,
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256_file(args.weights),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "batch": args.batch,
        "num_shards": args.num_shards,
    }
    if args.dest.exists():
        current = json.loads(args.dest.read_text(encoding="utf-8"))
        if current != wanted:
            raise RuntimeError(
                "M2 contract changed. Preserve the old OUTPUT_ROOT and choose a new one.\n"
                f"old={current}\nnew={wanted}"
            )
    else:
        atomic_json(args.dest, wanted)
    print("[M2 contract]", json.dumps(wanted, indent=2))


def validate_shard(marker: Path, predictions: Path, manifest: Path, shard: int, num_shards: int) -> int:
    expected = expected_shard_rows(manifest, shard, num_shards)
    rows = sum(1 for line in predictions.open(encoding="utf-8") if line.strip())
    state = json.loads(marker.read_text(encoding="utf-8"))
    if not (state.get("status") == "complete" and state.get("rows") == rows == expected):
        raise RuntimeError(
            f"invalid shard {shard}: marker_rows={state.get('rows')} rows={rows} expected={expected}"
        )
    return rows


def command_shard_check(args: argparse.Namespace) -> None:
    validate_shard(args.marker, args.predictions, args.manifest, args.shard, args.num_shards)


def command_shard_mark(args: argparse.Namespace) -> None:
    expected = expected_shard_rows(args.manifest, args.shard, args.num_shards)
    rows = sum(1 for line in args.predictions.open(encoding="utf-8") if line.strip())
    if rows != expected:
        raise RuntimeError(f"shard {args.shard}: rows={rows}, expected={expected}")
    atomic_json(
        args.dest,
        {"status": "complete", "shard": args.shard, "num_shards": args.num_shards, "rows": rows},
    )


def command_merge_shards(args: argparse.Namespace) -> None:
    expected = manifest_ids(args.manifest)
    seen: set[str] = set()
    destination = args.root / "predictions.jsonl"
    tmp = destination.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as output:
        for shard in range(args.num_shards):
            source = args.root / "shards" / f"shard_{shard:04d}" / "predictions.jsonl"
            for line in source.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                image_id = str(record["image_id"])
                if image_id in seen:
                    raise RuntimeError(f"duplicate M2 image_id during merge: {image_id}")
                seen.add(image_id)
                output.write(line if line.endswith("\n") else line + "\n")
    missing, extra = expected - seen, seen - expected
    if missing or extra:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"M2 merge coverage mismatch: missing={len(missing)} extra={len(extra)} "
            f"first_missing={sorted(missing)[:5]}"
        )
    os.replace(tmp, destination)
    print(f"[M2 merge] {len(seen):,} unique predictions -> {destination}")


def command_detector_provenance(args: argparse.Namespace) -> None:
    import numpy as np

    boxes = args.labels / "boxes_det.npy"
    mask = args.labels / "present_mask_det.npy"
    present = np.load(mask, mmap_mode="r")
    payload = {
        "schema_version": 2,
        "detector_checkpoint": str(args.weights),
        "detector_checkpoint_sha256": sha256_file(args.weights),
        "prediction_jsonl": str(args.predictions),
        "prediction_jsonl_sha256": sha256_file(args.predictions),
        "manifest_rows": int(present.shape[0]),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "batch": args.batch,
        "num_shards": args.num_shards,
        "coordinate_frame": 448,
        "boxes_det_sha256": sha256_file(boxes),
        "present_mask_det_sha256": sha256_file(mask),
        "shape": list(present.shape),
        "mean_regions_per_image": float(present.sum(axis=1).mean()),
    }
    atomic_json(args.labels / "detector_provenance.json", payload)
    print(json.dumps(payload, indent=2))


def command_m2_success(args: argparse.Namespace) -> None:
    payload = {
        "status": "complete",
        "predictions_sha256": sha256_file(args.predictions),
        "boxes_det_sha256": sha256_file(args.labels / "boxes_det.npy"),
        "present_mask_det_sha256": sha256_file(args.labels / "present_mask_det.npy"),
    }
    atomic_json(args.dest, payload)


def command_contract_num_shards(args: argparse.Namespace) -> None:
    print(int(json.loads(args.contract.read_text(encoding="utf-8"))["num_shards"]))


def command_run_manifest(args: argparse.Namespace) -> None:
    payload = {
        "schema_version": 1,
        "git_commit": args.git_commit,
        "repo_root": str(args.repo_root),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "gpu_ids": args.gpu_ids,
        "m3_epochs": args.m3_epochs,
        "m3_batch_per_run": args.m3_batch,
        "m4_epochs": args.m4_epochs,
        "m4_batch": args.m4_batch,
        "started_at": args.started_at,
    }
    if args.dest.exists():
        previous = json.loads(args.dest.read_text(encoding="utf-8"))
        immutable = ("git_commit", "input_root", "output_root", "m3_epochs", "m4_epochs")
        changed = {key: (previous.get(key), payload.get(key)) for key in immutable if previous.get(key) != payload.get(key)}
        if changed:
            raise RuntimeError(
                f"run manifest changed for an existing output tree: {changed}. Use a new OUTPUT_ROOT."
            )
        payload["first_started_at"] = previous.get("first_started_at", previous.get("started_at"))
    else:
        payload["first_started_at"] = payload["started_at"]
    atomic_json(args.dest, payload)
    print("[run manifest]", args.dest)


def command_show_summary(args: argparse.Namespace) -> None:
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    print("results updated:", summary.get("updated_at"))
    print("counts:", summary.get("counts"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    item = sub.add_parser("sha256"); item.add_argument("path", type=Path); item.set_defaults(func=command_sha256)
    item = sub.add_parser("preflight")
    item.add_argument("--gpu-ids", required=True); item.add_argument("--repo-root", type=Path, required=True)
    item.set_defaults(func=command_preflight)
    item = sub.add_parser("mark-stage")
    item.add_argument("--dest", type=Path, required=True); item.add_argument("--stage", required=True)
    item.add_argument("--completed-at", required=True); item.set_defaults(func=command_mark_stage)
    item = sub.add_parser("m2-contract")
    item.add_argument("--dest", type=Path, required=True); item.add_argument("--weights", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True); item.add_argument("--image-root", type=Path, required=True)
    item.add_argument("--imgsz", type=int, required=True); item.add_argument("--conf", type=float, required=True)
    item.add_argument("--iou", type=float, required=True); item.add_argument("--batch", type=int, required=True)
    item.add_argument("--num-shards", type=int, required=True); item.set_defaults(func=command_m2_contract)
    item = sub.add_parser("shard-check")
    item.add_argument("--marker", type=Path, required=True); item.add_argument("--predictions", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True); item.add_argument("--shard", type=int, required=True)
    item.add_argument("--num-shards", type=int, required=True); item.set_defaults(func=command_shard_check)
    item = sub.add_parser("shard-mark")
    item.add_argument("--predictions", type=Path, required=True); item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--shard", type=int, required=True); item.add_argument("--num-shards", type=int, required=True)
    item.add_argument("--dest", type=Path, required=True); item.set_defaults(func=command_shard_mark)
    item = sub.add_parser("merge-shards")
    item.add_argument("--root", type=Path, required=True); item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--num-shards", type=int, required=True); item.set_defaults(func=command_merge_shards)
    item = sub.add_parser("detector-provenance")
    item.add_argument("--weights", type=Path, required=True); item.add_argument("--predictions", type=Path, required=True)
    item.add_argument("--labels", type=Path, required=True); item.add_argument("--imgsz", type=int, required=True)
    item.add_argument("--conf", type=float, required=True); item.add_argument("--iou", type=float, required=True)
    item.add_argument("--batch", type=int, required=True); item.add_argument("--num-shards", type=int, required=True)
    item.set_defaults(func=command_detector_provenance)
    item = sub.add_parser("m2-success")
    item.add_argument("--dest", type=Path, required=True); item.add_argument("--predictions", type=Path, required=True)
    item.add_argument("--labels", type=Path, required=True); item.set_defaults(func=command_m2_success)
    item = sub.add_parser("contract-num-shards")
    item.add_argument("--contract", type=Path, required=True); item.set_defaults(func=command_contract_num_shards)
    item = sub.add_parser("run-manifest")
    item.add_argument("--dest", type=Path, required=True); item.add_argument("--git-commit", required=True)
    item.add_argument("--repo-root", type=Path, required=True); item.add_argument("--input-root", type=Path, required=True)
    item.add_argument("--output-root", type=Path, required=True); item.add_argument("--gpu-ids", required=True)
    item.add_argument("--m3-epochs", type=int, required=True); item.add_argument("--m3-batch", type=int, required=True)
    item.add_argument("--m4-epochs", type=int, required=True); item.add_argument("--m4-batch", type=int, required=True)
    item.add_argument("--started-at", required=True); item.set_defaults(func=command_run_manifest)
    item = sub.add_parser("show-summary")
    item.add_argument("--summary", type=Path, required=True); item.set_defaults(func=command_show_summary)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
