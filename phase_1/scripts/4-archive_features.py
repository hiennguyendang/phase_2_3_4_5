"""Step 4 (server-side) — archive the M1 feature cache into N zip volumes for transfer.

Splits <features-root>/*.pt into --num-parts contiguous groups and writes one
<out-dir>/<prefix><i>.zip per group (ZIP_STORED — .pt tensors are ~incompressible, so we skip
compression: far faster, same size). Also writes SHA256SUMS + archive_manifest.json so integrity
can be checked after download (`sha256sum -c SHA256SUMS`).

Each zip is flat (arcname = <image_id>.pt), so unzipping anywhere — or letting Kaggle auto-extract
a zip added to a dataset — reproduces the flat cache phase_3 reads.

    python 4-archive_features.py --features-root data/features --out-dir data/archive --num-parts 5
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # phase_1/src

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import config


def sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Archive M1 features into N zip volumes")
    p.add_argument("--features-root", type=Path, default=config.DEFAULT_FEATURES_OUT)
    p.add_argument("--out-dir", type=Path, default=config.WORK_ROOT / "archive")
    p.add_argument("--num-parts", type=int, default=5)
    p.add_argument("--prefix", default="biovilt-features-frozen-part")
    p.add_argument("--no-checksum", action="store_true", help="skip SHA256 (faster, no verify file)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(Path(args.features_root).rglob("*.pt"))
    if not files:
        raise SystemExit(f"[ERROR] no .pt under {args.features_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = len(files)
    chunk = math.ceil(total / args.num_parts)
    print(f"{total:,} .pt under {args.features_root} -> {args.num_parts} volumes "
          f"(~{chunk:,} files each) in {args.out_dir}")

    rows, grand_bytes = [], 0
    for i in range(args.num_parts):
        group = files[i * chunk:(i + 1) * chunk]
        if not group:
            continue
        zpath = args.out_dir / f"{args.prefix}{i + 1}.zip"
        print(f"[part {i + 1}/{args.num_parts}] zipping {len(group):,} files -> {zpath.name}")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for j, p in enumerate(group):
                zf.write(p, arcname=p.name)          # flat: <image_id>.pt
                if j and j % 5000 == 0:
                    print(f"    {j:,}/{len(group):,}")
        size = zpath.stat().st_size
        grand_bytes += size
        digest = "" if args.no_checksum else sha256(zpath)
        rows.append({"name": zpath.name, "files": len(group),
                     "bytes": size, "sha256": digest})
        print(f"    done: {size / 1e9:.2f} GB" + (f"  sha256={digest[:16]}…" if digest else ""))

    manifest = {"features_root": str(args.features_root), "total_files": total,
                "num_parts": len([r for r in rows]), "volumes": rows,
                "total_bytes": grand_bytes}
    (args.out_dir / "archive_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not args.no_checksum:
        # LF-only so `sha256sum -c` works on Linux even if this ran on Windows
        with open(args.out_dir / "SHA256SUMS", "w", newline="\n") as fh:
            fh.write("".join(f"{r['sha256']}  {r['name']}\n" for r in rows))
    zipped = sum(r["files"] for r in rows)
    print(f"\n[DONE] {zipped:,}/{total:,} files in {len(rows)} volumes, "
          f"{grand_bytes / 1e9:.1f} GB total -> {args.out_dir}")
    if zipped != total:
        print(f"[WARN] archived {zipped} != {total} source files")
    print("verify after transfer:  cd <dest> && sha256sum -c SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
