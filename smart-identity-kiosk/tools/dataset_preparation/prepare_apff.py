"""
Project data-preparation utility for the Arab Public Figures Facial Recognition
dataset.

Status: original project code inspired by external dataset requirements.
External reference: Arab Public Figures Facial Recognition dataset.
Student contribution: implemented project-specific detection, alignment,
identity grouping and split-preparation logic rather than copying source code.

This script is local SmartIdentity tooling. It does not copy Kaggle code; it
uses the dataset as an offline evaluation/prototyping source and prepares
aligned images for the project's gallery/test experiments.

Dataset source:
- https://www.kaggle.com/datasets/ashkhalil/arab-public-figures-facial-recognition

See THIRD_PARTY_ATTRIBUTION.md for full source, licence and redistribution
notes.
"""

import os, re, json, shutil

import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import re, json, shutil
import cv2
import numpy as np
from tqdm import tqdm

from models import SCRFD
from utils.helpers import face_alignment

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def base_id(name: str) -> str:
    # turns Abd_al_Rahman_Rafi_7 -> Abd_al_Rahman_Rafi
    return re.sub(r"_\d+$", "", name.strip())

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT

def largest_face(bboxes: np.ndarray):
    # bboxes: Nx5 [x1,y1,x2,y2,conf]
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    return int(np.argmax(areas))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset root (the folder containing images/ or consolidation/)")
    ap.add_argument("--out", default="./datasets/apff_clean")
    ap.add_argument("--det-weight", default="./weights/det_10g.onnx")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--min-face", type=int, default=80, help="min(face_w, face_h) in pixels")
    ap.add_argument("--max-per-id", type=int, default=30, help="cap images per identity")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out_images = out / "images"
    out_aligned = out / "aligned_112"
    out_logs = out / "logs"
    out_splits = out / "splits"
    for d in [out_images, out_aligned, out_logs, out_splits]:
        d.mkdir(parents=True, exist_ok=True)

    # choose which folder actually holds identity folders
    # try images/ first; otherwise consolidation/
    if (src / "images").exists():
        id_root = src / "images"
    elif (src / "consolidation").exists():
        id_root = src / "consolidation"
    else:
        id_root = src

    detector = SCRFD(str(args.det_weight), input_size=(640, 640), conf_thres=args.conf)

    dropped_no_face = []
    dropped_bad = []
    kept = {}  # id -> list of aligned paths

    # iterate identity folders (which may be split into *_1, *_2)
    all_id_dirs = [p for p in id_root.iterdir() if p.is_dir()]
    # group dirs by base id
    groups = {}
    for d in all_id_dirs:
        bid = base_id(d.name)
        groups.setdefault(bid, []).append(d)

    for bid, dirs in tqdm(groups.items(), desc="Identities"):
        img_paths = []
        for d in dirs:
            img_paths.extend([p for p in d.rglob("*") if p.is_file() and is_image(p)])

        if not img_paths:
            continue

        # cap per identity for speed
        img_paths = img_paths[: args.max_per_id]

        (out_images / bid).mkdir(parents=True, exist_ok=True)
        (out_aligned / bid).mkdir(parents=True, exist_ok=True)

        kept_paths = []
        for i, p in enumerate(img_paths):
            img = cv2.imread(str(p))
            if img is None:
                dropped_bad.append(str(p))
                continue

            bboxes, kpss = detector.detect(img, max_num=5)
            if bboxes is None or len(bboxes) == 0 or kpss is None or len(kpss) == 0:
                dropped_no_face.append(str(p))
                continue

            j = largest_face(bboxes)
            x1, y1, x2, y2, conf = bboxes[j]
            w = int(x2 - x1); h = int(y2 - y1)
            if min(w, h) < args.min_face:
                dropped_no_face.append(str(p))
                continue

            kps = kpss[j].astype(np.float32)
            aligned, _ = face_alignment(img, kps, image_size=112)

            # save original copy (optional, useful for debugging)
            dst_img = out_images / bid / f"{i:04d}{p.suffix.lower()}"
            shutil.copyfile(str(p), str(dst_img))

            # save aligned 112x112 (what you’ll actually evaluate on)
            dst_al = out_aligned / bid / f"{i:04d}.jpg"
            cv2.imwrite(str(dst_al), aligned)
            kept_paths.append(str(dst_al))

        if len(kept_paths) >= 2:
            kept[bid] = kept_paths

    # write logs
    (out_logs / "dropped_no_face.txt").write_text("\n".join(dropped_no_face), encoding="utf-8")
    (out_logs / "dropped_bad.txt").write_text("\n".join(dropped_bad), encoding="utf-8")

    # simple split: 1 enroll + rest probe
    rng = np.random.default_rng(args.seed)
    split = {"seed": args.seed, "enroll": {}, "probe": {}}
    for bid, paths in kept.items():
        paths = list(paths)
        rng.shuffle(paths)
        split["enroll"][bid] = [paths[0]]
        split["probe"][bid] = paths[1:]

    (out_splits / f"split_seed{args.seed}.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Done. Identities kept: {len(kept)}. Output: {out.resolve()}")

if __name__ == "__main__":
    main()
