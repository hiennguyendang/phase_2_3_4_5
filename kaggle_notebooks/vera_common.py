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
IMAGE_DATASET_ROOT = KAGGLE_DATASET_ROOT / "mimic-cxr-448"
# Backward-compatible path for already-uploaded notebooks. New notebooks call
# find_image_root() so a differently nested Kaggle dataset is also supported.
IMAGE_ROOT = IMAGE_DATASET_ROOT / "mimic-cxr-448"
FEATURE_ROOT = KAGGLE_DATASET_ROOT / "frozen" / "frozen"
BUNDLE_DATASET_ROOT = KAGGLE_DATASET_ROOT / "vera-v2-inputs"
M2_OUTPUT_DATASET_ROOT = KAGGLE_DATASET_ROOT / "vera-v2-detector-outputs"


def find_image_root() -> Path:
    """Locate the p10..p19 tree without recursively scanning hundreds of thousands of files."""
    candidates = [IMAGE_DATASET_ROOT / "mimic-cxr-448", IMAGE_DATASET_ROOT]
    if IMAGE_DATASET_ROOT.exists():
        candidates += [p for p in IMAGE_DATASET_ROOT.iterdir() if p.is_dir()]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if any((candidate / f"p{i}").is_dir() for i in range(10, 20)):
            return candidate
    raise FileNotFoundError(
        f"could not find p10..p19 image tree under {IMAGE_DATASET_ROOT}; "
        f"top-level entries={list(IMAGE_DATASET_ROOT.iterdir())[:20] if IMAGE_DATASET_ROOT.exists() else []}"
    )


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


def find_bundle(*, minimal: bool = False) -> Path:
    """Resolve the exact attached dataset; avoid recursive scans of feature/image trees."""
    hits = []
    for candidate in (BUNDLE_DATASET_ROOT, BUNDLE_DATASET_ROOT / "vera_v2"):
        if (candidate / "m2_detector" / "last.pt").exists() and (candidate / "m3_labels_base" / "manifest.jsonl").exists():
            hits.append(candidate)
    if not hits:
        archives = list(BUNDLE_DATASET_ROOT.glob("*.zip")) if BUNDLE_DATASET_ROOT.exists() else []
        if len(archives) == 1:
            extracted = Path("/kaggle/working/vera_v2_inputs_extracted")
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archives[0]) as zf:
                if minimal:
                    names = [n for n in zf.namelist() if (
                        n.endswith("m2_detector/last.pt")
                        or n.endswith("m2_detector/results.csv")
                        or n.endswith("m2_detector/audit_report.json")
                        or n.endswith("m3_labels_base/manifest.jsonl")
                    )]
                    zf.extractall(extracted, names=names)
                else:
                    zf.extractall(extracted)
            for candidate in (extracted, extracted / "vera_v2"):
                if (candidate / "m2_detector" / "last.pt").exists() and (candidate / "m3_labels_base" / "manifest.jsonl").exists():
                    hits.append(candidate)
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one uploaded vera_v2 bundle, found: {hits}")
    return hits[0]


def find_m2_outputs() -> Path:
    """Return the unique detector-label directory from any attached Kaggle dataset.

    The M2 export may be attached as a directory or as the convenience ZIP, and
    Kaggle may add one or more wrapper directories.  Resolve by artifact content
    instead of relying on the dataset slug or a fixed nesting depth.
    """
    root = KAGGLE_DATASET_ROOT
    preferred = M2_OUTPUT_DATASET_ROOT
    required = ("boxes_det.npy", "present_mask_det.npy", "detector_provenance.json")

    def valid_label_dir(path: Path) -> bool:
        return path.is_dir() and all((path / name).is_file() for name in required)

    mounted = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    candidates = [
        preferred / "m3_labels_detector_v2",
        preferred / "vera_v2_detector_outputs" / "m3_labels_detector_v2",
        preferred,
    ]
    # Bounded globs avoid walking through the 252k-file feature dataset.
    for pattern in (
        "*/detector_provenance.json",
        "*/*/detector_provenance.json",
        "*/*/*/detector_provenance.json",
        "*/*/*/*/detector_provenance.json",
    ):
        candidates.extend(marker.parent for marker in root.glob(pattern))

    hits: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if valid_label_dir(resolved) and resolved not in hits:
            hits.append(resolved)

    if not hits:
        archives = []
        for dataset_dir in mounted:
            archives.extend(dataset_dir.glob("*.zip"))
            archives.extend(dataset_dir.glob("*/*.zip"))
        matching_archives = []
        for archive in archives:
            try:
                with zipfile.ZipFile(archive) as zf:
                    names = zf.namelist()
                if any(name.endswith("detector_provenance.json") for name in names) \
                        and any(name.endswith("boxes_det.npy") for name in names):
                    matching_archives.append(archive)
            except zipfile.BadZipFile:
                continue
        for index, archive in enumerate(matching_archives):
            extracted = Path(f"/kaggle/working/vera_v2_detector_outputs_extracted_{index}")
            if extracted.exists():
                shutil.rmtree(extracted)
            extracted.mkdir(parents=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extracted)
            for marker in extracted.glob("**/detector_provenance.json"):
                if valid_label_dir(marker.parent) and marker.parent.resolve() not in hits:
                    hits.append(marker.parent.resolve())
    if len(hits) != 1:
        raise RuntimeError(
            "attach exactly one completed M2 detector-output dataset. Expected "
            "m3_labels_detector_v2/{boxes_det.npy,present_mask_det.npy,"
            "detector_provenance.json} or the M2 ZIP. "
            f"Attached dataset dirs={[p.name for p in mounted]}; resolved={hits}"
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
