"""Small, dependency-light helpers shared by the v2 Kaggle notebooks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


DRIVE_FOLDER_ID = "1a-a-P5i9lB8iN6t5wP5iNXsvZpDHdARp"
KAGGLE_DATASET_ROOT = Path("/kaggle/input/datasets/nguynnghin")
IMAGE_ROOT = KAGGLE_DATASET_ROOT / "mimic-cxr-448"
FEATURE_ROOT = KAGGLE_DATASET_ROOT / "frozen" / "frozen"
BUNDLE_DATASET_ROOT = KAGGLE_DATASET_ROOT / "vera-v2-inputs"
M2_OUTPUT_DATASET_ROOT = KAGGLE_DATASET_ROOT / "vera-v2-detector-outputs"


def prepare_repo(repo_url: str) -> Path:
    """Clone GitHub when Internet works, otherwise use the uploaded code dataset."""
    target = Path("/kaggle/working/vera_repo")
    if (target / "phase_2" / "scripts" / "yolo" / "5-infer_yolo.py").exists():
        return target
    if target.exists():
        shutil.rmtree(target)
    try:
        subprocess.run(["git", "clone", repo_url, str(target)], check=True)
        commit = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        print("source: GitHub commit", commit)
        return target
    except (OSError, subprocess.CalledProcessError) as exc:
        print("[fallback] GitHub clone unavailable:", exc)

    code_root = KAGGLE_DATASET_ROOT / "vera-v2-code"
    archives = list(code_root.glob("*.zip")) if code_root.exists() else []
    if len(archives) == 1:
        extracted = Path("/kaggle/working/vera_v2_code_extracted")
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archives[0]) as zf:
            zf.extractall(extracted)
        candidates = [extracted] + list(extracted.glob("*/"))
    else:
        candidates = [code_root]
    for candidate in candidates:
        if (candidate / "phase_2" / "scripts" / "yolo" / "5-infer_yolo.py").exists():
            shutil.copytree(candidate, target, dirs_exist_ok=True)
            print("source: Kaggle code dataset", candidate)
            return target
    raise RuntimeError(
        "GitHub clone failed and no valid code dataset was found at "
        "/kaggle/input/datasets/nguynnghin/vera-v2-code"
    )


def find_bundle() -> Path:
    """Find the uploaded vera_v2 bundle without hard-coding Kaggle's dataset slug."""
    roots = [KAGGLE_DATASET_ROOT, Path("/kaggle/working")]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        markers = list(root.glob("m2_detector/last.pt"))
        markers += list(root.glob("*/m2_detector/last.pt"))
        markers += list(root.glob("*/vera_v2/m2_detector/last.pt"))
        for marker in markers:
            candidate = marker.parent.parent
            if (candidate / "m3_labels_base" / "manifest.jsonl").exists():
                hits.append(candidate)
    if not hits:
        dataset = BUNDLE_DATASET_ROOT
        archives = list(dataset.glob("*.zip")) if dataset.exists() else []
        if len(archives) == 1:
            extracted = Path("/kaggle/working/vera_v2_inputs_extracted")
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archives[0]) as zf:
                zf.extractall(extracted)
            for marker in extracted.glob("**/m2_detector/last.pt"):
                candidate = marker.parent.parent
                if (candidate / "m3_labels_base" / "manifest.jsonl").exists():
                    hits.append(candidate)
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one uploaded vera_v2 bundle, found: {hits}")
    return hits[0]


def find_m2_outputs() -> Path:
    """Find the fresh detector-output Kaggle dataset by its provenance marker."""
    root = KAGGLE_DATASET_ROOT
    preferred = M2_OUTPUT_DATASET_ROOT
    candidates = [preferred, preferred / "vera_v2_detector_outputs"]
    candidates += [p.parent.parent for p in root.glob("*/m3_labels_detector_v2/detector_provenance.json")]
    candidates += [p.parent.parent for p in root.glob("*/vera_v2_detector_outputs/m3_labels_detector_v2/detector_provenance.json")]
    hits = []
    for p in candidates:
        if (p / "m3_labels_detector_v2" / "boxes_det.npy").exists() and p not in hits:
            hits.append(p)
    if not hits and preferred.exists():
        archives = list(preferred.glob("*.zip"))
        if len(archives) == 1:
            extracted = Path("/kaggle/working/vera_v2_detector_outputs_extracted")
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archives[0]) as zf:
                zf.extractall(extracted)
            for marker in extracted.glob("**/m3_labels_detector_v2/detector_provenance.json"):
                candidate = marker.parent.parent
                if (candidate / "m3_labels_detector_v2" / "boxes_det.npy").exists():
                    hits.append(candidate)
    if len(hits) != 1:
        raise RuntimeError(
            "attach exactly one detector output dataset named vera-v2-detector-outputs; "
            f"found: {hits}"
        )
    return hits[0]


def configure_drive() -> str:
    """Configure an rclone Drive remote from the Kaggle GDRIVE_TOKEN secret."""
    from kaggle_secrets import UserSecretsClient

    if not shutil.which("rclone"):
        archive = Path("/kaggle/working/rclone-current-linux-amd64.zip")
        urllib.request.urlretrieve(
            "https://downloads.rclone.org/rclone-current-linux-amd64.zip", archive
        )
        extracted = Path("/kaggle/working/rclone_dist")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
        source = next(extracted.glob("rclone-*-linux-amd64/rclone"))
        bindir = Path("/kaggle/working/bin")
        bindir.mkdir(parents=True, exist_ok=True)
        target = bindir / "rclone"
        shutil.copy2(source, target)
        target.chmod(0o755)
        os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")

    token = UserSecretsClient().get_secret("GDRIVE_TOKEN").strip()
    os.environ.update(
        RCLONE_CONFIG_VERA_TYPE="drive",
        RCLONE_CONFIG_VERA_TOKEN=token,
        RCLONE_CONFIG_VERA_SCOPE="drive",
        RCLONE_CONFIG_VERA_ROOT_FOLDER_ID=DRIVE_FOLDER_ID,
    )
    remote = "vera:VERA_KAGGLE_V2"
    subprocess.run(["rclone", "mkdir", remote], check=True)
    probe = subprocess.run(
        ["rclone", "lsd", remote], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )
    if probe.returncode:
        raise RuntimeError(f"Drive remote is not readable: {probe.stdout}")
    print("Drive remote ready:", remote)
    return remote


def copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory using shutil while preserving its relative layout."""
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
