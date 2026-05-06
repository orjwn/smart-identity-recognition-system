import os
import sys
import csv
import cv2
import shutil
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple

# Make project root importable when running from tools/dataset_preparation.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.scrfd import SCRFD
from models.arcface import ArcFace
from utils.helpers import face_alignment


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(IMAGE_EXTS)
    )


def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def avg_embedding(embs: List[np.ndarray]) -> Optional[np.ndarray]:
    if not embs:
        return None
    x = np.mean(np.stack(embs, axis=0), axis=0)
    return l2norm(x.astype(np.float32))


def cosine_scores(embs: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    # both already L2-normalized
    return np.dot(embs, centroid)


def unique_jpg_path(folder: str, stem: str) -> str:
    ensure_dir(folder)
    out = os.path.join(folder, f"{stem}.jpg")
    if not os.path.exists(out):
        return out

    i = 1
    while True:
        cand = os.path.join(folder, f"{stem}_{i}.jpg")
        if not os.path.exists(cand):
            return cand
        i += 1


def copy_for_review(src_path: str, review_folder: str) -> None:
    ensure_dir(review_folder)
    dst = os.path.join(review_folder, os.path.basename(src_path))
    if not os.path.exists(dst):
        shutil.copy2(src_path, dst)


class FaceCleaner:
    def __init__(
        self,
        detector: SCRFD,
        recognizer: ArcFace,
        accept_thresh: float,
        margin: float,
        copy_review: bool = True,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.accept_thresh = accept_thresh
        self.margin = margin
        self.copy_review = copy_review
        self.cache: Dict[str, dict] = {}

    def analyze_image(self, image_path: str) -> dict:
        if image_path in self.cache:
            return self.cache[image_path]

        info = {
            "ok": False,
            "error": "",
            "image_path": image_path,
            "n_faces": 0,
            "det_scores": np.array([], dtype=np.float32),
            "kpss": None,
            "embs": None,
        }

        img = cv2.imread(image_path)
        if img is None:
            info["error"] = "read_failed"
            self.cache[image_path] = info
            return info

        try:
            det, kpss = self.detector.detect(img, max_num=0)
        except Exception as e:
            info["error"] = f"detect_failed:{e}"
            self.cache[image_path] = info
            return info

        if det is None or kpss is None or len(det) == 0 or len(kpss) == 0:
            info["error"] = "no_face"
            self.cache[image_path] = info
            return info

        valid_scores = []
        valid_kpss = []
        valid_embs = []

        for i in range(len(kpss)):
            try:
                emb = self.recognizer.get_embedding(img, kpss[i], normalized=True)
                valid_embs.append(emb.astype(np.float32))
                valid_kpss.append(kpss[i])
                valid_scores.append(float(det[i, 4]))
            except Exception:
                continue

        if not valid_embs:
            info["error"] = "embedding_failed"
            self.cache[image_path] = info
            return info

        info["ok"] = True
        info["n_faces"] = len(valid_embs)
        info["det_scores"] = np.array(valid_scores, dtype=np.float32)
        info["kpss"] = np.array(valid_kpss, dtype=np.float32)
        info["embs"] = np.stack(valid_embs, axis=0)
        self.cache[image_path] = info
        return info

    def choose_face(self, info: dict, centroid: np.ndarray) -> dict:
        if not info["ok"]:
            return {
                "accepted": False,
                "reason": info["error"],
                "best_idx": -1,
                "best_score": -1.0,
                "second_score": -1.0,
                "n_faces": 0,
            }

        sims = cosine_scores(info["embs"], centroid)
        order = np.argsort(sims)[::-1]

        best_idx = int(order[0])
        best_score = float(sims[best_idx])
        second_score = float(sims[order[1]]) if len(order) > 1 else -1.0

        accepted = (
            best_score >= self.accept_thresh and
            (len(order) == 1 or (best_score - second_score) >= self.margin)
        )

        reason = "accepted"
        if not accepted:
            if best_score < self.accept_thresh:
                reason = "low_similarity"
            else:
                reason = "ambiguous_multi_face"

        return {
            "accepted": accepted,
            "reason": reason,
            "best_idx": best_idx,
            "best_score": best_score,
            "second_score": second_score,
            "n_faces": int(info["n_faces"]),
        }


def build_initial_centroids(
    images_root: str,
    cleaner: FaceCleaner,
    min_single_face: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    centroids = {}
    seed_counts = {}

    identities = [
        d for d in sorted(os.listdir(images_root))
        if os.path.isdir(os.path.join(images_root, d))
    ]

    for identity in identities:
        person_dir = os.path.join(images_root, identity)
        image_paths = list_images(person_dir)

        single_face_embs = []
        for image_path in image_paths:
            info = cleaner.analyze_image(image_path)
            if info["ok"] and info["n_faces"] == 1:
                single_face_embs.append(info["embs"][0])

        if len(single_face_embs) >= min_single_face:
            centroids[identity] = avg_embedding(single_face_embs)
            seed_counts[identity] = len(single_face_embs)
            print(f"[seed] {identity}: {len(single_face_embs)} single-face image(s)")
        else:
            print(f"[skip-seed] {identity}: only {len(single_face_embs)} single-face image(s)")

    return centroids, seed_counts


def refine_centroids(
    images_root: str,
    cleaner: FaceCleaner,
    initial_centroids: Dict[str, np.ndarray],
    min_refined: int = 3,
) -> Dict[str, np.ndarray]:
    refined = {}

    for identity, centroid in initial_centroids.items():
        person_dir = os.path.join(images_root, identity)
        image_paths = list_images(person_dir)

        accepted_embs = []
        for image_path in image_paths:
            info = cleaner.analyze_image(image_path)
            choice = cleaner.choose_face(info, centroid)
            if choice["accepted"]:
                accepted_embs.append(info["embs"][choice["best_idx"]])

        if len(accepted_embs) >= min_refined:
            refined[identity] = avg_embedding(accepted_embs)
            print(f"[refine] {identity}: refined from {len(accepted_embs)} accepted image(s)")
        else:
            refined[identity] = centroid
            print(f"[refine] {identity}: kept initial centroid")

    return refined


def save_aligned_face(
    image_path: str,
    landmarks: np.ndarray,
    output_dir: str,
) -> str:
    img = cv2.imread(image_path)
    aligned_face, _ = face_alignment(img, landmarks)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = unique_jpg_path(output_dir, stem)
    ok = cv2.imwrite(out_path, aligned_face)
    if not ok:
        raise RuntimeError(f"Failed to save aligned face: {out_path}")
    return out_path


def final_clean_align(
    images_root: str,
    aligned_root: str,
    review_root: str,
    log_csv: str,
    cleaner: FaceCleaner,
    centroids: Dict[str, np.ndarray],
) -> None:
    ensure_dir(aligned_root)
    ensure_dir(review_root)
    ensure_dir(os.path.dirname(log_csv) or ".")

    rows = []

    identities = [
        d for d in sorted(os.listdir(images_root))
        if os.path.isdir(os.path.join(images_root, d))
    ]

    for identity in identities:
        person_dir = os.path.join(images_root, identity)
        image_paths = list_images(person_dir)

        if identity not in centroids:
            print(f"[skip-final] {identity}: no centroid")
            for image_path in image_paths:
                review_dir = os.path.join(review_root, identity)
                if cleaner.copy_review:
                    copy_for_review(image_path, review_dir)
                rows.append({
                    "identity": identity,
                    "image": os.path.basename(image_path),
                    "faces_found": 0,
                    "best_score": "",
                    "second_score": "",
                    "decision": "review",
                    "reason": "no_centroid",
                    "saved_path": "",
                })
            continue

        identity_out = os.path.join(aligned_root, identity)
        identity_review = os.path.join(review_root, identity)
        centroid = centroids[identity]

        accepted_count = 0
        review_count = 0

        for image_path in image_paths:
            info = cleaner.analyze_image(image_path)
            choice = cleaner.choose_face(info, centroid)

            if choice["accepted"]:
                try:
                    saved_path = save_aligned_face(
                        image_path=image_path,
                        landmarks=info["kpss"][choice["best_idx"]],
                        output_dir=identity_out,
                    )
                    accepted_count += 1
                    rows.append({
                        "identity": identity,
                        "image": os.path.basename(image_path),
                        "faces_found": choice["n_faces"],
                        "best_score": f"{choice['best_score']:.4f}",
                        "second_score": (
                            f"{choice['second_score']:.4f}" if choice["second_score"] >= 0 else ""
                        ),
                        "decision": "saved",
                        "reason": "accepted",
                        "saved_path": saved_path,
                    })
                except Exception as e:
                    if cleaner.copy_review:
                        copy_for_review(image_path, identity_review)
                    review_count += 1
                    rows.append({
                        "identity": identity,
                        "image": os.path.basename(image_path),
                        "faces_found": choice["n_faces"],
                        "best_score": f"{choice['best_score']:.4f}",
                        "second_score": (
                            f"{choice['second_score']:.4f}" if choice["second_score"] >= 0 else ""
                        ),
                        "decision": "review",
                        "reason": f"save_failed:{e}",
                        "saved_path": "",
                    })
            else:
                if cleaner.copy_review:
                    copy_for_review(image_path, identity_review)
                review_count += 1
                rows.append({
                    "identity": identity,
                    "image": os.path.basename(image_path),
                    "faces_found": choice["n_faces"],
                    "best_score": (
                        f"{choice['best_score']:.4f}" if choice["best_score"] >= 0 else ""
                    ),
                    "second_score": (
                        f"{choice['second_score']:.4f}" if choice["second_score"] >= 0 else ""
                    ),
                    "decision": "review",
                    "reason": choice["reason"],
                    "saved_path": "",
                })

        print(f"[done] {identity}: saved={accepted_count} review={review_count}")

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "identity",
                "image",
                "faces_found",
                "best_score",
                "second_score",
                "decision",
                "reason",
                "saved_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nLog written to: {log_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identity-aware cleaning + alignment for foldered raw face images"
    )
    parser.add_argument("--images-root", required=True, help="Root folder with one folder per person")
    parser.add_argument("--aligned-root", required=True, help="Output folder for accepted aligned 112x112 faces")
    parser.add_argument("--review-root", required=True, help="Output folder for raw images that need review")
    parser.add_argument("--log-csv", required=True, help="CSV log path")
    parser.add_argument("--det-weight", required=True, help="Path to SCRFD .onnx")
    parser.add_argument("--rec-weight", required=True, help="Path to ArcFace .onnx")

    parser.add_argument("--det-size", type=int, default=640, help="SCRFD input size")
    parser.add_argument("--conf-thresh", type=float, default=0.5, help="SCRFD confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.4, help="SCRFD NMS threshold")

    parser.add_argument("--min-single-face", type=int, default=2,
                        help="Minimum single-face images needed to build the initial centroid")
    parser.add_argument("--accept-thresh", type=float, default=0.35,
                        help="Minimum cosine similarity needed to accept the chosen face")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Best face must beat second-best by at least this margin")
    parser.add_argument("--no-copy-review", action="store_true",
                        help="Do not copy rejected raw images into review-root")
    return parser.parse_args()


def main():
    args = parse_args()

    detector = SCRFD(
        model_path=args.det_weight,
        input_size=(args.det_size, args.det_size),
        conf_thres=args.conf_thresh,
        iou_thres=args.iou_thresh,
    )
    recognizer = ArcFace(args.rec_weight)

    cleaner = FaceCleaner(
        detector=detector,
        recognizer=recognizer,
        accept_thresh=args.accept_thresh,
        margin=args.margin,
        copy_review=not args.no_copy_review,
    )

    print("\n[1/3] Building initial centroids from single-face images...")
    initial_centroids, seed_counts = build_initial_centroids(
        images_root=args.images_root,
        cleaner=cleaner,
        min_single_face=args.min_single_face,
    )

    if not initial_centroids:
        raise RuntimeError(
            "No centroids were built. Lower --min-single-face or manually inspect the folders."
        )

    print("\n[2/3] Refining centroids from accepted images...")
    refined_centroids = refine_centroids(
        images_root=args.images_root,
        cleaner=cleaner,
        initial_centroids=initial_centroids,
        min_refined=max(3, args.min_single_face),
    )

    print("\n[3/3] Final identity-aware alignment...")
    final_clean_align(
        images_root=args.images_root,
        aligned_root=args.aligned_root,
        review_root=args.review_root,
        log_csv=args.log_csv,
        cleaner=cleaner,
        centroids=refined_centroids,
    )

    print("\nFinished.")
    print(f"Centroids built for: {len(seed_counts)} identities")


if __name__ == "__main__":
    main()
