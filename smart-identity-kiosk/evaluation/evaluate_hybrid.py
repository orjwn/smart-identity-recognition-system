"""Evaluate the ArcFace plus AdaFace hybrid recognizer.

This script is for report evidence. It builds temporary in-memory galleries
from the evaluation dataset and does not load or modify the runtime FAISS
databases used by the kiosk demo.
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

from models import ArcFace, AdaFace
from utils.logging import setup_logging

setup_logging(log_to_file=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid recognition: ArcFace primary + AdaFace fallback on aligned 112x112 gallery/test folders"
    )

    parser.add_argument(
        "--arcface-weight",
        type=str,
        required=True,
        help="Path to ArcFace ONNX model, e.g. ./weights/w600k_mbf.onnx"
    )
    parser.add_argument(
        "--adaface-weight",
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
        default=os.path.join(PROJECT_ROOT, "evaluation", "results_hybrid.csv"),
        help="CSV file to save per-image results"
    )
    parser.add_argument(
        "--gallery-aggregation",
        type=str,
        default="mean",
        choices=["mean", "all"],
        help="Use mean embedding per identity or all gallery embeddings"
    )
    parser.add_argument(
        "--arcface-high-thresh",
        type=float,
        default=0.5,
        help="ArcFace confident acceptance threshold"
    )
    parser.add_argument(
        "--arcface-low-thresh",
        type=float,
        default=0.40,
        help="ArcFace uncertain-band lower threshold"
    )
    parser.add_argument(
        "--adaface-thresh",
        type=float,
        default=0.50,
        help="AdaFace acceptance threshold when used as fallback"
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


def arcface_embedding_from_aligned_image(recognizer: ArcFace, image_bgr: np.ndarray) -> np.ndarray:
    face_blob = recognizer.preprocess(image_bgr)
    embedding = recognizer.session.run(
        recognizer.output_names,
        {recognizer.input_name: face_blob}
    )[0]

    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    embedding = embedding / norm
    return embedding.flatten().astype(np.float32)


def adaface_embedding_from_aligned_image(recognizer: AdaFace, image_bgr: np.ndarray) -> np.ndarray:
    embedding = recognizer(image_bgr)
    return embedding.astype(np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def build_gallery_embeddings(recognizer, gallery_dir: str, aggregation: str, embedding_fn):
    gallery_store = {}
    samples = list_identity_samples(gallery_dir)

    if not samples:
        raise ValueError(f"No gallery images found in: {gallery_dir}")

    for sample in samples:
        identity = sample["identity"]
        image_path = sample["image_path"]

        try:
            image = read_aligned_face(image_path, expected_size=(112, 112))
            emb = embedding_fn(recognizer, image)
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


def predict_identity(query_emb: np.ndarray, gallery_data, aggregation: str):
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

    return best_name, best_similarity


def hybrid_predict(query_arc_emb: np.ndarray,
                   query_ada_emb: np.ndarray,
                   arc_gallery_data,
                   ada_gallery_data,
                   aggregation: str,
                   arcface_low_thresh: float,
                   arcface_high_thresh: float,
                   adaface_thresh: float):
    arc_name, arc_similarity = predict_identity(query_arc_emb, arc_gallery_data, aggregation)

    if arc_similarity >= arcface_high_thresh:
        return arc_name, arc_similarity, arc_similarity, 0.0, "arcface"

    if arc_similarity >= arcface_low_thresh:
        ada_name, ada_similarity = predict_identity(query_ada_emb, ada_gallery_data, aggregation)

        if ada_similarity >= adaface_thresh:
            return ada_name, ada_similarity, arc_similarity, ada_similarity, "adaface_fallback"

        return "Unknown", max(arc_similarity, ada_similarity), arc_similarity, ada_similarity, "unknown"

    return "Unknown", arc_similarity, arc_similarity, 0.0, "unknown"


def save_misclassified_image(image, gt_name, pred_name, similarity, model_used, out_path):
    canvas = image.copy()
    label = f"GT: {gt_name} | Pred: {pred_name} | Sim: {similarity:.4f} | Via: {model_used}"

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

    arc_model_name = os.path.splitext(os.path.basename(params.arcface_weight))[0]
    ada_model_name = os.path.splitext(os.path.basename(params.adaface_weight))[0]
    hybrid_model_name = f"hybrid_{arc_model_name}_plus_{ada_model_name}"

    logging.info(f"Loading ArcFace recognizer from: {params.arcface_weight}")
    logging.info(f"Loading AdaFace recognizer from: {params.adaface_weight}")
    logging.info(f"Gallery directory: {params.gallery_dir}")
    logging.info(f"Test directory: {params.test_dir}")
    logging.info(f"Gallery aggregation: {params.gallery_aggregation}")
    logging.info(
        f"Hybrid thresholds: arc_high={params.arcface_high_thresh}, "
        f"arc_low={params.arcface_low_thresh}, ada={params.adaface_thresh}"
    )

    try:
        arc_recognizer = ArcFace(params.arcface_weight)
        ada_recognizer = AdaFace(params.adaface_weight)
    except Exception as e:
        logging.error(f"Failed to load recognizers: {e}")
        return

    try:
        arc_gallery_data = build_gallery_embeddings(
            recognizer=arc_recognizer,
            gallery_dir=params.gallery_dir,
            aggregation=params.gallery_aggregation,
            embedding_fn=arcface_embedding_from_aligned_image
        )
        ada_gallery_data = build_gallery_embeddings(
            recognizer=ada_recognizer,
            gallery_dir=params.gallery_dir,
            aggregation=params.gallery_aggregation,
            embedding_fn=adaface_embedding_from_aligned_image
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

    gallery_identity_count = len(arc_gallery_data)
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
    arcface_direct_count = 0
    adaface_fallback_count = 0

    with open(params.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "model",
            "gallery_aggregation",
            "image_path",
            "ground_truth",
            "predicted_name",
            "final_similarity",
            "arcface_similarity",
            "adaface_similarity",
            "model_used",
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
            final_similarity = 0.0
            arcface_similarity = 0.0
            adaface_similarity = 0.0
            model_used = "error"
            correct = False
            image = None

            try:
                image = read_aligned_face(image_path, expected_size=(112, 112))

                query_arc_emb = arcface_embedding_from_aligned_image(arc_recognizer, image)
                query_ada_emb = adaface_embedding_from_aligned_image(ada_recognizer, image)

                predicted_name, final_similarity, arcface_similarity, adaface_similarity, model_used = hybrid_predict(
                    query_arc_emb=query_arc_emb,
                    query_ada_emb=query_ada_emb,
                    arc_gallery_data=arc_gallery_data,
                    ada_gallery_data=ada_gallery_data,
                    aggregation=params.gallery_aggregation,
                    arcface_low_thresh=params.arcface_low_thresh,
                    arcface_high_thresh=params.arcface_high_thresh,
                    adaface_thresh=params.adaface_thresh
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
                hybrid_model_name,
                params.gallery_aggregation,
                image_path,
                gt_name,
                predicted_name,
                f"{final_similarity:.6f}",
                f"{arcface_similarity:.6f}",
                f"{adaface_similarity:.6f}",
                model_used,
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
                    total_similarity += final_similarity
                    similarity_count += 1

                if model_used == "arcface":
                    arcface_direct_count += 1
                elif model_used == "adaface_fallback":
                    adaface_fallback_count += 1

                if correct:
                    correct_count += 1
                elif params.save_misclassified and image is not None:
                    out_path = os.path.join(
                        params.misclassified_dir,
                        hybrid_model_name,
                        gt_name,
                        os.path.basename(image_path)
                    )
                    save_misclassified_image(
                        image=image,
                        gt_name=gt_name,
                        pred_name=predicted_name,
                        similarity=final_similarity,
                        model_used=model_used,
                        out_path=out_path
                    )

            logging.info(
                f"[{idx}/{len(test_samples)}] "
                f"GT={gt_name} | Pred={predicted_name} | "
                f"Final={final_similarity:.4f} | Arc={arcface_similarity:.4f} | "
                f"Ada={adaface_similarity:.4f} | Via={model_used} | Correct={correct} | Status={status}"
            )

    accuracy = (correct_count / total * 100.0) if total > 0 else 0.0
    avg_similarity = (total_similarity / similarity_count) if similarity_count > 0 else 0.0
    avg_inference_ms = (total_inference_ms / total) if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("Hybrid evaluation summary")
    print("=" * 60)
    print(f"Model:                {hybrid_model_name}")
    print(f"Gallery identities:   {gallery_identity_count}")
    print(f"Test images:          {total}")
    print(f"Correct:              {correct_count}")
    print(f"Accuracy:             {accuracy:.2f}%")
    print(f"Unknown:              {unknown_count}")
    print(f"Errors:               {error_count}")
    print(f"ArcFace direct uses:  {arcface_direct_count}")
    print(f"AdaFace fallback uses:{adaface_fallback_count}")
    print(f"Avg similarity:       {avg_similarity:.4f}")
    print(f"Avg inference time:   {avg_inference_ms:.2f} ms")
    print(f"Results CSV:          {params.output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
