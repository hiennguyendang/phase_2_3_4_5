"""Post-training detector audit — turns the YOLO concerns in docs/critical_yolo.md
(== B1 in docs/VERA_methodology_concerns.md) into numbers. Run AFTER training; needs
only the trained weights + the built yolo_ds (NO M1/M3).

Concern B1: "does the detector actually LOOK at the image, or just place 29 boxes at
their average anatomical position?" 29 CXR regions are *deceptively easy* (heart mid-left,
apices top, CP angles low) so a lazy detector can score high by ignoring the image — and
it will fail silently exactly on the abnormal cases (large effusion shifting mediastinum,
collapsed lung, huge heart) where localization matters most.

Tests implemented here (the runnable ones):
  1. Per-class mAP (model.val) ................. which regions are weak.
  2. Static-prior baseline .................... a fixed mean-box-per-region template that
       IGNORES the image. YOLO must clearly beat its IoU, else it's "just the template".
  3. Per-region IoU (YOLO vs GT) ............. is the box good enough as an M3 mask.
  4. Stratified by anatomical atypicality .... split images by how far their GT boxes
       deviate from the template; the gap (YOLO IoU - static IoU) in the most-atypical
       stratum = the real value the detector adds where it matters (critical_yolo phep 2).
  5. Perturbation / deletion test ............ black out a region's pixels -> does its box
       move? If not, the box is prior-driven not image-driven (critical_yolo phep 4).

DEFERRED (needs M1 features + M3, not runnable yet): the gold-box vs detector-box ORACLE
ABLATION at M3 (critical_yolo phep 3) — the test that actually decides whether box error
matters downstream. A placeholder section prints what to run once M3 exists.

    python audit_yolo.py --weights runs/det29/weights/best.pt \
        --yolo-ds /kaggle/working/yolo_ds --split val --max-images 2000 \
        --out runs/det29/audit
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import json
import os
from pathlib import Path

import numpy as np

from constants import CLASS_NAMES, NUM_CLASSES


# --------------------------------------------------------------------------- IO
def read_label(path: Path) -> dict[int, tuple[float, float, float, float]]:
    """YOLO label file -> {class_id: (cx,cy,w,h)} normalized. Anatomical regions are
    singletons; if a class repeats, keep the largest-area box."""
    out: dict[int, tuple[float, float, float, float]] = {}
    try:
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c = int(float(p[0]))
            box = tuple(float(x) for x in p[1:])  # cx,cy,w,h
            if c not in out or (box[2] * box[3]) > (out[c][2] * out[c][3]):
                out[c] = box  # type: ignore[assignment]
    except OSError:
        pass
    return out


def load_split_labels(yolo_ds: Path, split: str) -> dict[str, dict[int, tuple]]:
    d = yolo_ds / "labels" / split
    return {p.stem: read_label(p) for p in d.glob("*.txt")} if d.is_dir() else {}


def image_for(yolo_ds: Path, split: str, stem: str) -> Path | None:
    base = yolo_ds / "images" / split
    for ext in (".jpg", ".jpeg", ".png"):
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------------ boxes
def to_xyxy(b):
    cx, cy, w, h = b
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def build_template(train_labels: dict[str, dict], sample: int) -> dict[int, tuple]:
    """Mean (cx,cy,w,h) per class over (a sample of) train labels — the image-blind prior."""
    acc = {c: [] for c in range(NUM_CLASSES)}
    stems = list(train_labels.keys())
    if sample and len(stems) > sample:
        # deterministic stride sample (no RNG -> reproducible)
        step = len(stems) / sample
        stems = [stems[int(i * step)] for i in range(sample)]
    for s in stems:
        for c, box in train_labels[s].items():
            acc[c].append(box)
    return {c: tuple(np.mean(v, axis=0)) for c, v in acc.items() if v}


# ----------------------------------------------------------------- predictions
def predict_per_class(model, img_path: Path, imgsz: int, device: str, conf: float):
    """Return {class_id: (box_xywhn, conf)} keeping the highest-conf box per class."""
    r = model.predict(str(img_path), imgsz=imgsz, device=device, conf=conf,
                      verbose=False, max_det=60)[0]
    best: dict[int, tuple] = {}
    if r.boxes is None or len(r.boxes) == 0:
        return best
    cls = r.boxes.cls.cpu().numpy().astype(int)
    cf = r.boxes.conf.cpu().numpy()
    xywhn = r.boxes.xywhn.cpu().numpy()
    for c, cc, xy in zip(cls, cf, xywhn):
        if c not in best or cc > best[c][1]:
            best[c] = (tuple(float(x) for x in xy), float(cc))
    return best


# ------------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser(description="Audit the 29-region detector (B1 / critical_yolo)")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--yolo-ds", type=Path, required=True, help="dir with images/ labels/ dataset.yaml")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--max-images", type=int, default=2000, help="cap eval images (0=all) for speed")
    p.add_argument("--template-sample", type=int, default=20000)
    p.add_argument("--perturb-images", type=int, default=40)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--imgsz", type=int, default=448)
    p.add_argument("--device", default="0")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skip-map", action="store_true", help="skip model.val mAP (faster)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    yds = args.yolo_ds
    data_yaml = yds / "dataset.yaml"
    print(f"weights : {args.weights}\nyolo-ds : {yds}\nsplit   : {args.split}")
    model = YOLO(str(args.weights))

    # ---- 1. per-class mAP (model.val) ----------------------------------------
    per_class_map = {}
    overall_map = {}
    if not args.skip_map:
        print("\n[1] per-class mAP (model.val) ...")
        m = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz,
                      batch=16, device=args.device, verbose=False)
        overall_map = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                       "precision": float(m.box.mp), "recall": float(m.box.mr)}
        for i, ci in enumerate(m.box.ap_class_index):
            per_class_map[CLASS_NAMES[int(ci)]] = {
                "mAP50": float(m.box.ap50[i]), "mAP50_95": float(m.box.maps[int(ci)])}
        print(f"    overall: {overall_map}")

    # ---- load labels + template ---------------------------------------------
    print("\n[2-4] loading labels + building static-prior template ...")
    train_lab = load_split_labels(yds, "train")
    eval_lab = load_split_labels(yds, args.split)
    template = build_template(train_lab, args.template_sample)
    print(f"    train labels: {len(train_lab):,} | {args.split} labels: {len(eval_lab):,} "
          f"| template classes: {len(template)}")

    stems = list(eval_lab.keys())
    if args.max_images and len(stems) > args.max_images:
        step = len(stems) / args.max_images
        stems = [stems[int(i * step)] for i in range(args.max_images)]
    print(f"    evaluating IoU on {len(stems):,} images ...")

    # ---- 2+3. static-prior vs YOLO IoU, per class + per image ----------------
    cls_static = {c: [] for c in range(NUM_CLASSES)}
    cls_yolo = {c: [] for c in range(NUM_CLASSES)}
    img_rows = []  # (atypicality, mean_yolo_iou, mean_static_iou)
    for n, s in enumerate(stems):
        gt = eval_lab[s]
        if not gt:
            continue
        img = image_for(yds, args.split, s)
        if img is None:
            continue
        pred = predict_per_class(model, img, args.imgsz, args.device, args.conf)
        y_ious, s_ious = [], []
        for c, gbox in gt.items():
            tb = template.get(c)
            siou = iou(gbox, tb) if tb else 0.0
            yiou = iou(gbox, pred[c][0]) if c in pred else 0.0  # missing pred -> 0 (recall)
            cls_static[c].append(siou)
            cls_yolo[c].append(yiou)
            s_ious.append(siou)
            y_ious.append(yiou)
        if y_ious:
            atyp = 1.0 - float(np.mean(s_ious))  # far from template = atypical anatomy
            img_rows.append((atyp, float(np.mean(y_ious)), float(np.mean(s_ious))))
        if (n + 1) % 250 == 0:
            print(f"      {n + 1}/{len(stems)}")

    def mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    per_class_iou = {}
    for c in range(NUM_CLASSES):
        if cls_yolo[c]:
            per_class_iou[CLASS_NAMES[c]] = {
                "n": len(cls_yolo[c]),
                "iou_yolo": round(mean(cls_yolo[c]), 4),
                "iou_static": round(mean(cls_static[c]), 4),
                "gap": round(mean(cls_yolo[c]) - mean(cls_static[c]), 4)}
    all_y = [v for c in cls_yolo.values() for v in c]
    all_s = [v for c in cls_static.values() for v in c]
    overall_iou = {"iou_yolo": round(mean(all_y), 4), "iou_static": round(mean(all_s), 4),
                   "gap": round(mean(all_y) - mean(all_s), 4)}

    # ---- 4. stratify by atypicality (quartiles) ------------------------------
    img_rows.sort(key=lambda r: r[0])
    strata = []
    if img_rows:
        q = len(img_rows) // 4 or 1
        names = ["Q1 typical", "Q2", "Q3", "Q4 atypical"]
        for i in range(4):
            chunk = img_rows[i * q: (i + 1) * q] if i < 3 else img_rows[3 * q:]
            if chunk:
                strata.append({
                    "stratum": names[i], "n": len(chunk),
                    "atypicality": round(float(np.mean([r[0] for r in chunk])), 4),
                    "iou_yolo": round(float(np.mean([r[1] for r in chunk])), 4),
                    "iou_static": round(float(np.mean([r[2] for r in chunk])), 4),
                    "gap": round(float(np.mean([r[1] - r[2] for r in chunk])), 4)})

    # ---- 5. perturbation / deletion test -------------------------------------
    print("\n[5] perturbation (delete region pixels -> does its box move?) ...")
    from PIL import Image
    pert = {"checked": 0, "mean_iou_box_shift": None, "note":
            "high IoU(before,after) = box barely moves when its content is removed = prior-driven"}
    shift_ious = []
    psample = stems[:: max(1, len(stems) // max(1, args.perturb_images))][: args.perturb_images]
    for s in psample:
        gt = eval_lab[s]
        img = image_for(yds, args.split, s)
        if not gt or img is None:
            continue
        try:
            arr = np.array(Image.open(img).convert("RGB"))
        except Exception:
            continue
        H, W = arr.shape[:2]
        before = predict_per_class(model, img, args.imgsz, args.device, args.conf)
        # pick up to 3 classes present in both GT and prediction
        cands = [c for c in gt if c in before][:3]
        for c in cands:
            cx, cy, w, h = gt[c]
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            mod = arr.copy()
            mod[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0  # black out region content
            r = model.predict(mod, imgsz=args.imgsz, device=args.device, conf=args.conf,
                              verbose=False, max_det=60)[0]
            after = None
            if r.boxes is not None and len(r.boxes):
                cl = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                xy = r.boxes.xywhn.cpu().numpy()
                cbest = -1.0
                for cc, ccf, cxy in zip(cl, cf, xy):
                    if cc == c and ccf > cbest:
                        cbest, after = ccf, tuple(float(x) for x in cxy)
            shift_ious.append(iou(before[c][0], after) if after else 0.0)
            pert["checked"] += 1
    if shift_ious:
        pert["mean_iou_box_shift"] = round(float(np.mean(shift_ious)), 4)

    # ---- report --------------------------------------------------------------
    report = {
        "weights": str(args.weights), "split": args.split, "n_images": len(stems),
        "overall_map": overall_map, "overall_iou": overall_iou,
        "per_class": {name: {**per_class_iou.get(name, {}),
                             **({"map": per_class_map[name]} if name in per_class_map else {})}
                      for name in CLASS_NAMES},
        "strata_by_atypicality": strata, "perturbation": pert,
        "oracle_ablation_M3": "DEFERRED — needs M1 features + M3. Run M3 twice (gold-box vs "
                              "detector-box) and compare macro-F1; small gap => box error does "
                              "NOT bottleneck M3 (critical_yolo phep 3, the decisive test).",
    }

    print("\n================= AUDIT SUMMARY =================")
    if overall_map:
        print(f"mAP50={overall_map['mAP50']:.3f}  mAP50-95={overall_map['mAP50_95']:.3f}")
    print(f"IoU  YOLO={overall_iou['iou_yolo']:.3f}  static-prior={overall_iou['iou_static']:.3f}"
          f"  gap=+{overall_iou['gap']:.3f}   <- YOLO must clearly beat static")
    print("\nWORST regions by YOLO IoU (the boxes that may hurt M3 masking):")
    worst = sorted([(v["iou_yolo"], k, v) for k, v in per_class_iou.items()])[:8]
    for v, k, d in worst:
        print(f"  {k:<26} IoU={d['iou_yolo']:.3f} (static={d['iou_static']:.3f}, gap=+{d['gap']:.3f}, n={d['n']})")
    print("\nStratified by anatomical atypicality (the cases that matter most):")
    for st in strata:
        print(f"  {st['stratum']:<12} n={st['n']:<5} YOLO={st['iou_yolo']:.3f} "
              f"static={st['iou_static']:.3f} gap=+{st['gap']:.3f}")
    if pert["mean_iou_box_shift"] is not None:
        print(f"\nPerturbation: mean IoU(box before vs after deleting content) = "
              f"{pert['mean_iou_box_shift']:.3f}  (closer to 1 = more prior-driven)")
    print("\nphep 3 (oracle gold-vs-detector at M3): DEFERRED until M3 exists.")
    print("================================================")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nsaved -> {args.out / 'audit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
