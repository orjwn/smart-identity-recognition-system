"""Evaluate AdaFace using in-memory gallery embeddings.

This script is for report evidence. It reads aligned evaluation images from
`datasets/gallery` and `datasets/test`, writes a CSV, and does not load or
modify the runtime FAISS databases used by the kiosk.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv
import cv2
import time
import argparse
import logging
import numpy as np
from datetime import datetime

from models import AdaFace
from utils.logging import setup_logging

setup_logging(log_to_file=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AdaFace on aligned 112x112 gallery/test folders without detection or FAISS"
    )

    parser.add_argument(
        "--rec-weight",
        type=str,
        required=True,
        help="Path to AdaFace checkpoint, e.g. ./weights/adaface_ir18_webface4m.ckpt"
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "datasets", "gallery"),
        help="Gallery/enrollment folder: one subfolder per identity"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "datasets", "test"),
        help="Test/query folder: one subfolder per identity"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "evaluation", "adaface_eval_results.csv"),
        help="CSV file to save per-image results"
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for accepting a match"
    )
    parser.add_argument(
        "--gallery-aggregation",
        type=str,
        default="mean",
        choices=["mean", "all"],
        help="Use mean embedding per identity or all gallery embeddings"
    )
    parser.add_argument(
        "--save-misclassified",
        action="store_true",
        help="Save copies of misclassified images with overlay text"
    )
    parser.add_argument(
        "--misclassified-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "evaluation", "misclassified"),
        help="Folder for misclassified examples"
    )

    return parser.parse_args()


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.replace("_", " ").replace("-", " ")
    return " ".join(name.strip().lower().split())


def list_image_files(folder: str):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(valid_exts)
    )


def list_identity_samples(root_dir: str):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    samples = []
    for identity in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, identity)
        if not os.path.isdir(person_dir):
            continue

        for filename in list_image_files(person_dir):
            samples.append({
                "identity": identity,
                "image_path": os.path.join(person_dir, filename),
                "filename": filename,
            })
    return samples


def read_aligned_face(image_path: str, expected_size=(112, 112)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    if (w, h) != expected_size:
        image = cv2.resize(image, expected_size)

    return image


def embedding_from_aligned_image(recognizer: AdaFace, image_bgr: np.ndarray) -> np.ndarray:
    embedding = recognizer(image_bgr)
    return embedding.astype(np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def build_gallery_embeddings(recognizer: AdaFace, gallery_dir: str, aggregation: str):
    gallery_store = {}
    samples = list_identity_samples(gallery_dir)

    if not samples:
        raise ValueError(f"No gallery images found in: {gallery_dir}")

    for sample in samples:
        identity = sample["identity"]
        image_path = sample["image_path"]

        try:
            image = read_aligned_face(image_path, expected_size=(112, 112))
            emb = embedding_from_aligned_image(recognizer, image)
            gallery_store.setdefault(identity, []).append(emb)
        except Exception as e:
            logging.warning(f"Skipping gallery image {image_path}: {e}")

    if not gallery_store:
        raise ValueError("No valid gallery embeddings were created.")

    if aggregation == "mean":
        aggregated = {}
        for identity, embs in gallery_store.items():
            if not embs:
                continue
            stacked = np.stack(embs, axis=0)
            mean_emb = np.mean(stacked, axis=0)
            mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-12)
            aggregated[identity] = mean_emb.astype(np.float32)
        return aggregated

    return gallery_store


def predict_identity(query_emb: np.ndarray, gallery_data, aggregation: str, similarity_thresh: float):
    best_name = "Unknown"
    best_similarity = -1.0

    if aggregation == "mean":
        for identity, gallery_emb in gallery_data.items():
            sim = cosine_similarity(query_emb, gallery_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_name = identity
    else:
        for identity, emb_list in gallery_data.items():
            for gallery_emb in emb_list:
                sim = cosine_similarity(query_emb, gallery_emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_name = identity

    if best_similarity < similarity_thresh:
        return "Unknown", best_similarity

    return best_name, best_similarity


def save_misclassified_image(image, gt_name, pred_name, similarity, out_path):
    canvas = image.copy()
    label = f"GT: {gt_name} | Pred: {pred_name} | Sim: {similarity:.4f}"

    cv2.putText(
        canvas,
        label,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )

    ensure_parent_dir(out_path)
    cv2.imwrite(out_path, canvas)


def main():
    params = parse_args()

    model_name = os.path.splitext(os.path.basename(params.rec_weight))[0]
    logging.info(f"Loading AdaFace recognizer from: {params.rec_weight}")
    logging.info(f"Gallery directory: {params.gallery_dir}")
    logging.info(f"Test directory: {params.test_dir}")
    logging.info(f"Gallery aggregation: {params.gallery_aggregation}")

    try:
        recognizer = AdaFace(params.rec_weight)
    except Exception as e:
        logging.error(f"Failed to load recognizer: {e}")
        return

    try:
        gallery_data = build_gallery_embeddings(
            recognizer=recognizer,
            gallery_dir=params.gallery_dir,
            aggregation=params.gallery_aggregation
        )
    except Exception as e:
        logging.error(f"Failed to build gallery embeddings: {e}")
        return

    try:
        test_samples = list_identity_samples(params.test_dir)
    except Exception as e:
        logging.error(f"Failed to list test images: {e}")
        return

    if not test_samples:
        logging.error(f"No test images found in: {params.test_dir}")
        return

    gallery_identity_count = len(gallery_data)
    logging.info(f"Built in-memory gallery for {gallery_identity_count} identities")
    logging.info(f"Found {len(test_samples)} test images")

    ensure_parent_dir(params.output_csv)

    total = 0
    correct_count = 0
    unknown_count = 0
    error_count = 0
    total_similarity = 0.0
    similarity_count = 0
    total_inference_ms = 0.0

    with open(params.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "model",
            "gallery_aggregation",
            "image_path",
            "ground_truth",
            "predicted_name",
            "similarity",
            "correct",
            "status",
            "inference_ms"
        ])

        for idx, sample in enumerate(test_samples, start=1):
            gt_name = sample["identity"]
            image_path = sample["image_path"]

            start = time.time()
            status = "ok"
            predicted_name = "ERROR"
            similarity = 0.0
            correct = False
            image = None

            try:
                image = read_aligned_face(image_path, expected_size=(112, 112))
                query_emb = embedding_from_aligned_image(recognizer, image)
                predicted_name, similarity = predict_identity(
                    query_emb=query_emb,
                    gallery_data=gallery_data,
                    aggregation=params.gallery_aggregation,
                    similarity_thresh=params.similarity_thresh
                )
                correct = (
                    predicted_name != "Unknown" and
                    normalize_name(predicted_name) == normalize_name(gt_name)
                )
            except Exception as e:
                status = "embedding_failed"
                logging.warning(f"Failed on test image {image_path}: {e}")

            inference_ms = (time.time() - start) * 1000.0
            timestamp = datetime.now().isoformat(timespec="seconds")

            writer.writerow([
                timestamp,
                model_name,
                params.gallery_aggregation,
                image_path,
                gt_name,
                predicted_name,
                f"{similarity:.6f}",
                correct,
                status,
                f"{inference_ms:.3f}"
            ])

            total += 1
            total_inference_ms += inference_ms

            if status != "ok":
                error_count += 1
            else:
                if predicted_name == "Unknown":
                    unknown_count += 1
                else:
                    total_similarity += similarity
                    similarity_count += 1

                if correct:
                    correct_count += 1
                elif params.save_misclassified and image is not None:
                    out_path = os.path.join(
                        params.misclassified_dir,
                        model_name,
                        gt_name,
                        os.path.basename(image_path)
                    )
                    save_misclassified_image(
                        image=image,
                        gt_name=gt_name,
                        pred_name=predicted_name,
                        similarity=similarity,
                        out_path=out_path
                    )

            logging.info(
                f"[{idx}/{len(test_samples)}] "
                f"GT={gt_name} | Pred={predicted_name} | "
                f"Sim={similarity:.4f} | Correct={correct} | Status={status}"
            )

    accuracy = (correct_count / total * 100.0) if total > 0 else 0.0
    avg_similarity = (total_similarity / similarity_count) if similarity_count > 0 else 0.0
    avg_inference_ms = (total_inference_ms / total) if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("AdaFace evaluation summary")
    print("=" * 60)
    print(f"Model:              {model_name}")
    print(f"Gallery identities: {gallery_identity_count}")
    print(f"Test images:        {total}")
    print(f"Correct:            {correct_count}")
    print(f"Accuracy:           {accuracy:.2f}%")
    print(f"Unknown:            {unknown_count}")
    print(f"Errors:             {error_count}")
    print(f"Avg similarity:     {avg_similarity:.4f}")
    print(f"Avg inference time: {avg_inference_ms:.2f} ms")
    print(f"Results CSV:        {params.output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
