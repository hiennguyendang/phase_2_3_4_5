"""Write the M1 feature cache (.pt), flush batches to Drive (rclone), and resume.

Kaggle can't hold the whole ~90GB cache and sessions die mid-run, so:
  - features are torch.save'd one <image_id>.pt at a time into a local staging dir;
  - every FLUSH_EVERY new files, `flush_to_drive()` rclone-copies them up then DELETES the
    local copies to free /kaggle/working;
  - at session start, `done_ids()` lists what's already on Drive (rclone lsf) ∪ what's local,
    so finished images are skipped.

`remote` is an rclone path, e.g. dhint:CHEX-DATA/biovilt_features (same OAuth remote as phase_2).
All rclone calls are best-effort: a missing rclone / transient error never crashes extraction.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch


def _rclone(*args: str) -> int:
    """Run `rclone <args>` if rclone is on PATH; return exit code (or -1 if absent/errored)."""
    if not shutil.which("rclone"):
        return -1
    try:
        return subprocess.run(["rclone", *args], check=False).returncode
    except Exception:  # noqa: BLE001 — syncing must never crash extraction
        return -1


def save_feature(out_dir: Path, image_id: str, feat: torch.Tensor, retries: int = 3) -> Path:
    """torch.save a [197, C] float16 tensor to <out_dir>/<image_id>.pt (stem = full image_id).

    Robust for long Kaggle runs: `.clone()` gives the row its OWN storage (a batch-row view would
    otherwise serialize the whole batch); write to a .tmp then atomically rename (a crash never
    leaves a half-written .pt that would corrupt resume); retry a few times on transient FS errors."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{image_id}.pt"
    t = feat.detach().to(torch.float16).clone().contiguous()   # own storage, not a batch view
    tmp = path.with_suffix(".pt.tmp")
    last: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            torch.save(t, tmp)
            tmp.replace(path)                                  # atomic on the same filesystem
            return path
        except Exception as e:  # noqa: BLE001 — transient iostream/ENOSPC etc.
            last = e
            try:
                tmp.unlink()
            except OSError:
                pass
    raise last if last is not None else RuntimeError(f"failed to save {path}")


def local_done_ids(out_dir: Path) -> set[str]:
    """image_ids already staged locally (stem of each .pt)."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return set()
    return {p.stem for p in out_dir.glob("*.pt")}


def remote_done_ids(remote: str | None) -> set[str]:
    """image_ids already on the Drive remote (rclone lsf, .pt stripped). Empty if no remote."""
    if not remote or not shutil.which("rclone"):
        return set()
    try:
        res = subprocess.run(
            ["rclone", "lsf", remote, "--include", "*.pt", "--files-only"],
            check=False, capture_output=True, text=True,
        )
    except Exception:  # noqa: BLE001
        return set()
    done: set[str] = set()
    for line in res.stdout.splitlines():
        name = line.strip().rstrip("/")
        if name.endswith(".pt"):
            done.add(name[:-3])
    return done


def done_ids(out_dir: Path, remote: str | None) -> set[str]:
    """Union of local-staged and remote-uploaded image_ids -> skip these on resume."""
    return local_done_ids(out_dir) | remote_done_ids(remote)


def flush_to_drive(out_dir: Path, remote: str | None, delete_local: bool = True) -> int:
    """rclone copy the staged .pt up to `remote`, then (default) delete the local copies.

    Returns the number of local files removed. No-op (returns 0) without a remote / rclone."""
    out_dir = Path(out_dir)
    files = list(out_dir.glob("*.pt"))
    if not files or not remote or not shutil.which("rclone"):
        return 0
    rc = _rclone("copy", str(out_dir), remote, "--transfers", "8", "--checkers", "8", "--quiet")
    if rc != 0:                       # upload failed -> keep local copies, try again next flush
        print(f"[sync] rclone copy -> {remote} failed (rc={rc}); keeping local staging")
        return 0
    removed = 0
    if delete_local:
        for f in files:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed
