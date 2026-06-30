"""Durable checkpointing for Kaggle (or any ephemeral box) via rclone.

Kaggle does NOT keep /kaggle/working across sessions unless you *commit*, and a
session can die mid-run. So we push the training run dir to a remote (Google
Drive) **every N optimizer steps** and at every epoch end; on the next session we
pull it back and `--resume`.

Wire-up (done for you by train_yolo.py --sync-remote):
    from kaggle_sync import attach_rclone_sync, pull_run
    pull_run(remote, runs_dir, name)            # before training (resume)
    attach_rclone_sync(model, remote, every=300) # registers callbacks

`remote` is an rclone path, e.g.  dhint:CHEX-DATA/phase2_runs
The run lands at  <remote>/<run_name>/  (weights/last.pt, weights/best.pt,
args.yaml, results.csv ...) — exactly what ultralytics needs to resume.

Requires the `rclone` CLI on PATH and a configured remote (see the Kaggle
notebook for installing rclone + writing rclone.conf from a Kaggle Secret).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import shutil
import subprocess
from pathlib import Path


def _rclone() -> str:
    exe = shutil.which("rclone")
    if not exe:
        raise RuntimeError("rclone not found on PATH (install it first)")
    return exe


def _push(local_dir: Path, remote_dir: str, background: bool = True) -> None:
    """rclone copy local_dir -> remote_dir (non-blocking by default)."""
    if not local_dir.exists():
        return
    cmd = [_rclone(), "copy", str(local_dir), remote_dir,
           "--transfers", "8", "--checkers", "8", "--ignore-existing", "--quiet"]
    # weights change every save, so DON'T ignore-existing for them; do a 2nd pass
    cmd_w = [_rclone(), "copy", str(local_dir), remote_dir,
             "--transfers", "8", "--checkers", "8", "--quiet"]
    try:
        if background:
            subprocess.Popen(cmd_w, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd_w, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:  # never let syncing crash training
        print(f"[sync] push failed (continuing): {e}")


def pull_run(remote: str, runs_dir: Path, name: str) -> bool:
    """Before training: pull <remote>/<name> -> <runs_dir>/<name> so --resume works.
    Returns True if something was pulled."""
    remote_dir = f"{remote.rstrip('/')}/{name}"
    local = Path(runs_dir) / name
    local.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([_rclone(), "copy", remote_dir, str(local),
                        "--transfers", "8", "--quiet"], check=False)
        last = local / "weights" / "last.pt"
        if last.exists():
            print(f"[sync] pulled checkpoint -> {last}")
            return True
        print(f"[sync] no checkpoint at {remote_dir} (fresh start)")
    except Exception as e:
        print(f"[sync] pull failed (fresh start): {e}")
    return False


def attach_rclone_sync(model, remote: str, every: int = 300) -> None:
    """Register ultralytics callbacks that checkpoint to `remote`:
       - every `every` optimizer steps (mid-epoch): save_model() + push
       - at each epoch end: push (captures best.pt / results.csv / args.yaml)
       - at train end: final blocking push
    """
    state = {"steps": 0, "remote_dir": None}

    def _remote_dir(trainer):
        # <remote>/<run_name>   (run_name == save_dir.name)
        if state["remote_dir"] is None:
            state["remote_dir"] = f"{remote.rstrip('/')}/{Path(trainer.save_dir).name}"
        return state["remote_dir"]

    def on_train_batch_end(trainer):
        state["steps"] += 1
        if every > 0 and state["steps"] % every == 0:
            try:
                trainer.save_model()  # writes weights/last.pt (+best.pt) mid-epoch
            except Exception as e:
                print(f"[sync] save_model failed (continuing): {e}")
            _push(Path(trainer.save_dir), _remote_dir(trainer), background=True)
            print(f"[sync] step {state['steps']}: pushed -> {_remote_dir(trainer)}")

    def on_fit_epoch_end(trainer):
        _push(Path(trainer.save_dir), _remote_dir(trainer), background=True)

    def on_train_end(trainer):
        _push(Path(trainer.save_dir), _remote_dir(trainer), background=False)
        print(f"[sync] final push -> {_remote_dir(trainer)}")

    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)
    print(f"[sync] rclone sync ON -> {remote}  (every {every} steps + each epoch)")
