#!/usr/bin/env python3
"""Download the three Kaggle inputs and generate server_hoang/server.env.

Run this once on Hoang's server before invoking run_all.sh.  It is safe to run
again: kagglehub reuses completed downloads, and this script revalidates the
resolved input roots before replacing server.env atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path


DATASETS = (
    ("nguynnghin/vera-v2-inputs", "vera-v2-inputs"),
    ("nguynnghin/frozen", "frozen"),
    ("nguynnghin/mimic-cxr-448", "mimic-cxr-448"),
)
YOLO_SHA256 = "71d4b4e3b173cc046fc45c7120b6cf4489c384ceaaec9f08231182108a40da56"
EXPECTED_FEATURE_COUNT = 252_287
MS_CSV_NAME = "MS_CXR_T_temporal_image_classification_v1.0.0.csv"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download VERA Kaggle inputs, validate them, and write server.env."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("VERA_DATA", "/data/vera/kaggle_downloads")),
        help="persistent directory for Kaggle datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/vera/output_server_hoang"),
        help="persistent output directory used by run_all.sh",
    )
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument("--gpu-ids", default="0,1", help="comma-separated physical GPU IDs")
    parser.add_argument(
        "--login",
        action="store_true",
        help="open kagglehub's login flow before downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force kagglehub to redownload datasets",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="only validate existing files and regenerate server.env",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def first_manifest_id(manifest: Path) -> str:
    with manifest.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                return str(json.loads(line)["image_id"])
    raise RuntimeError(f"empty manifest: {manifest}")


def require(path: Path, kind: str = "file") -> Path:
    ok = path.is_file() if kind == "file" else path.is_dir()
    if not ok:
        raise RuntimeError(f"missing {kind}: {path}")
    return path.resolve()


def find_bundle(dataset_root: Path) -> Path:
    candidates = (dataset_root / "vera_v2", dataset_root)
    for candidate in candidates:
        if (candidate / "m2_detector/last.pt").is_file():
            return candidate.resolve()
    hits = list(dataset_root.glob("*/m2_detector/last.pt"))
    if len(hits) == 1:
        return hits[0].parents[1].resolve()
    raise RuntimeError(
        f"cannot uniquely locate vera_v2/m2_detector/last.pt under {dataset_root}; hits={hits}"
    )


def image_path(root: Path, image_id: str) -> Path | None:
    patient = image_id.split("_", 2)[1]
    parent = root / patient[:3] / patient
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = parent / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def find_image_root(dataset_root: Path, image_id: str) -> Path:
    candidates = (dataset_root / "mimic-cxr-448", dataset_root)
    for candidate in candidates:
        if image_path(candidate, image_id):
            return candidate.resolve()
    for path in dataset_root.rglob(f"{image_id}.*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and len(path.parents) >= 3:
            candidate = path.parents[2]
            if image_path(candidate, image_id):
                return candidate.resolve()
    raise RuntimeError(f"cannot locate image root for {image_id} under {dataset_root}")


def find_feature_root(dataset_root: Path, image_id: str) -> Path:
    candidates = (dataset_root / "frozen", dataset_root)
    for candidate in candidates:
        if (candidate / f"{image_id}.pt").is_file() or (candidate / f"{image_id}.npy").is_file():
            return candidate.resolve()
    for suffix in (".pt", ".npy"):
        hits = list(dataset_root.rglob(f"{image_id}{suffix}"))
        if len(hits) == 1:
            return hits[0].parent.resolve()
    raise RuntimeError(f"cannot locate frozen feature root for {image_id} under {dataset_root}")


def download_all(args: argparse.Namespace) -> dict[str, Path]:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("install first: python3 -m pip install --upgrade kagglehub") from exc

    if args.login:
        kagglehub.login()

    roots: dict[str, Path] = {}
    for handle, dirname in DATASETS:
        destination = (args.data_root / dirname).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        print(f"[download] {handle} -> {destination}", flush=True)
        kwargs = {"output_dir": str(destination)}
        if args.force:
            kwargs["force_download"] = True
        try:
            returned = kagglehub.dataset_download(handle, **kwargs)
        except TypeError:
            # Older kagglehub releases do not expose force_download with output_dir.
            kwargs.pop("force_download", None)
            returned = kagglehub.dataset_download(handle, **kwargs)
        print(f"[download] kagglehub returned: {returned}", flush=True)
        roots[dirname] = destination
    return roots


def existing_roots(data_root: Path) -> dict[str, Path]:
    return {dirname: (data_root / dirname).resolve() for _, dirname in DATASETS}


def write_server_env(args: argparse.Namespace, values: dict[str, Path | str]) -> Path:
    env_path = args.repo_root.resolve() / "server_hoang/server.env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by server_hoang/download_kaggle.py; contains paths, not secrets.",
        "# Re-run the downloader to validate inputs and regenerate this file.",
    ]
    for key, value in values.items():
        lines.append(f"{key}={shlex.quote(str(value))}")
    tmp = env_path.with_suffix(".env.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, env_path)
    return env_path


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    if not (args.repo_root / "server_hoang/run_all.sh").is_file():
        raise RuntimeError(f"not a VERA repository root: {args.repo_root}")
    if not args.gpu_ids or any(not item.strip().isdigit() for item in args.gpu_ids.split(",")):
        raise RuntimeError(f"invalid --gpu-ids: {args.gpu_ids!r}")

    args.data_root.mkdir(parents=True, exist_ok=True)
    roots = existing_roots(args.data_root) if args.skip_download else download_all(args)

    bundle = find_bundle(roots["vera-v2-inputs"])
    m3_labels = require(bundle / "m3_labels_base", "directory")
    m4_labels = require(bundle / "m4_labels", "directory")
    manifest = require(m3_labels / "manifest.jsonl")
    image_id = first_manifest_id(manifest)
    feature_root = find_feature_root(roots["frozen"], image_id)
    image_root = find_image_root(roots["mimic-cxr-448"], image_id)
    weights = require(bundle / "m2_detector/last.pt")
    ms_csv = require(m4_labels / MS_CSV_NAME)

    for name in ("region_concepts.npy", "region_chexpert.npy", "image_chexpert.npy", "boxes.npy", "present_mask.npy"):
        require(m3_labels / name)
    for name in ("manifest.jsonl", "progression.npy", "m3_pairs.jsonl"):
        require(m4_labels / name)
    require(feature_root / f"{image_id}.pt")
    if image_path(image_root, image_id) is None:
        raise RuntimeError(f"sample image is not resolvable from IMAGE_ROOT={image_root}")

    feature_count = sum(
        entry.is_file() and Path(entry.name).suffix.lower() in {".pt", ".npy"}
        for entry in os.scandir(feature_root)
    )
    if feature_count != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"frozen feature count mismatch under {feature_root}: "
            f"got={feature_count}, expected={EXPECTED_FEATURE_COUNT}"
        )

    got_hash = sha256_file(weights)
    if got_hash.lower() != YOLO_SHA256:
        raise RuntimeError(f"YOLO SHA-256 mismatch: got={got_hash}, expected={YOLO_SHA256}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    values: dict[str, Path | str] = {
        "INPUT_ROOT": args.data_root,
        "IMAGE_ROOT": image_root,
        "FEATURE_ROOT": feature_root,
        "YOLO_WEIGHTS": weights,
        "M3_LABELS_INPUT": m3_labels,
        "M4_LABELS": m4_labels,
        "MS_CSV": ms_csv,
        "OUTPUT_ROOT": args.output_root,
        "GPU_IDS": args.gpu_ids,
        "M2_BATCH": "16",
        "M3_BATCH": "64",
        "M3_WORKERS": "8",
        "M4_BATCH": "128",
        "M4_WORKERS": "16",
    }
    env_path = write_server_env(args, values)
    marker = args.data_root / ".server_hoang_download.json"
    marker.write_text(
        json.dumps(
            {
                "datasets": [handle for handle, _ in DATASETS],
                "image_root": str(image_root),
                "feature_root": str(feature_root),
                "feature_count": feature_count,
                "bundle_root": str(bundle),
                "yolo_sha256": got_hash,
                "server_env": str(env_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n[DONE] Kaggle inputs validated and pipeline settings written")
    print(f"  IMAGE_ROOT={image_root}")
    print(f"  FEATURE_ROOT={feature_root}")
    print(f"  FEATURE_COUNT={feature_count}")
    print(f"  BUNDLE_ROOT={bundle}")
    print(f"  OUTPUT_ROOT={args.output_root}")
    print(f"  SERVER_ENV={env_path}")
    print("\nNext commands:")
    print(f"  cd {shlex.quote(str(args.repo_root))}")
    print("  bash server_hoang/run_all.sh preflight")
    print("  bash server_hoang/run_all.sh start")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
