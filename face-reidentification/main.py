# main.py
import os
import cv2
import random
import time
import warnings
import argparse
import logging
import numpy as np
import json
import urllib.request
from datetime import datetime
from PIL import Image

from database import FaceDatabase
from models import SCRFD, ArcFace, MaskInv, AdaFace
from models.focusface import FocusFaceEncoder
from utils.logging import setup_logging
from utils.helpers import draw_bbox_info, draw_bbox, face_alignment

warnings.filterwarnings("ignore")
setup_logging(log_to_file=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Live Face Detection-and-Recognition (Webcam) with FAISS")

    parser.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx", help="Path to detection model")

    # ArcFace baseline
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_mbf.onnx", help="Path to ArcFace model")

    # MaskInv ONNX
    parser.add_argument(
        "--maskinv-weight",
        type=str,
        default="./weights/maskinv/maskinv_hg.onnx",
        help="Path to MaskInv student ONNX"
    )

    # AdaFace
    parser.add_argument(
        "--adaface-weight",
        type=str,
        default="./weights/adaface_ir18_webface4m.ckpt",
        help="Path to AdaFace R18 checkpoint"
    )

    # FocusFace
    parser.add_argument(
        "--focusface-weight",
        type=str,
        default="./weights/focus_face_w_pretrained.mdl",
        help="Path to FocusFace checkpoint"
    )

    # mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="arcface",
        choices=["arcface", "maskinv", "adaface", "focusface", "hybrid"],
        help="Embedding model mode"
    )

    parser.add_argument("--similarity-thresh", type=float, default=0.4, help="Similarity threshold between faces")
    parser.add_argument("--confidence-thresh", type=float, default=0.5, help="Confidence threshold for face detection")

    parser.add_argument("--faces-dir", type=str, default="./assets/faces", help="Path to faces stored dir")
    parser.add_argument("--source", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--max-num", type=int, default=0, help="Maximum number of face detections from a frame")

    # DB root
    parser.add_argument(
        "--db-root",
        type=str,
        default="./database/face_database",
        help="DB root folder; will use <db-root>/arcface_mbf, <db-root>/maskinv, <db-root>/adaface, <db-root>/focusface"
    )

    parser.add_argument("--update-db", action="store_true", help="Rebuild the selected mode database then exit")

    # kiosk backend integration
    parser.add_argument(
        "--kiosk-api",
        type=str,
        default="http://127.0.0.1:8000",
        help="FastAPI kiosk backend base URL"
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=10,
        help="Frames required before sending recognition event"
    )
    parser.add_argument(
        "--min-post-interval",
        type=float,
        default=1.5,
        help="Seconds between repeated posts for same identity"
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="laptop-prototype",
        help="Device ID sent to the kiosk backend"
    )
    parser.add_argument(
        "--disable-kiosk-post",
        action="store_true",
        help="Disable POSTing recognition events to kiosk backend"
    )

    parser.add_argument(
        "--arcface-high-thresh",
        type=float,
        default=0.60,
        help="ArcFace confident acceptance threshold in hybrid mode"
    )
    parser.add_argument(
        "--arcface-low-thresh",
        type=float,
        default=0.40,
        help="ArcFace lower bound for uncertain band in hybrid mode"
    )
    parser.add_argument(
        "--adaface-thresh",
        type=float,
        default=0.50,
        help="AdaFace acceptance threshold in hybrid mode"
    )

    return parser.parse_args()


# -------------------------
# Kiosk POST helper
# -------------------------
def post_json(url: str, payload: dict, timeout: float = 0.25):
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=timeout).read()
    except Exception as e:
        logging.debug(f"[kiosk] POST failed: {e}")


class RecognitionDebounce:
    def __init__(self, stable_frames: int = 10, min_post_interval: float = 1.5):
        self.stable_frames = max(1, stable_frames)
        self.min_post_interval = max(0.0, min_post_interval)
        self.last_name = None
        self.count = 0
        self.last_sent_name = None
        self.last_sent_at = 0.0

    def update(self, name: str) -> bool:
        if name == "Unknown":
            self.last_name = None
            self.count = 0
            return False

        if name == self.last_name:
            self.count += 1
        else:
            self.last_name = name
            self.count = 1

        if self.count < self.stable_frames:
            return False

        now = time.time()
        if self.last_sent_name == name and (now - self.last_sent_at) < self.min_post_interval:
            return False

        self.last_sent_name = name
        self.last_sent_at = now
        return True


# -------------------------
# Data loading helpers
# -------------------------
def _load_json_list(filename: str, root_key: str):
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", filename)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(root_key, [])
    except FileNotFoundError:
        logging.error(f"Error: File '{file_path}' not found.")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error: JSON file '{file_path}' is not properly formatted.")
        return []


def read_passports():
    return _load_json_list("passports.json", "passports")


def read_flights():
    return _load_json_list("flights.json", "flights")


def read_boarding_passes():
    return _load_json_list("boarding_passes.json", "boarding_passes")


def normalize_name(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.strip().lower().split())


def fmt_hm(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%H:%M")
    except Exception:
        if len(iso_str) >= 16 and "T" in iso_str:
            return iso_str.split("T", 1)[1][:5]
        return iso_str


def get_traveller_flight_context(recognized_name: str,
                                 passports_by_name: dict,
                                 boarding_by_passport: dict,
                                 flights_by_number: dict):
    key = normalize_name(recognized_name)
    passport = passports_by_name.get(key)
    if not passport:
        return None, None, None, "PASSPORT_NOT_FOUND"

    passport_number = passport.get("passport_number")
    if not passport_number:
        return passport, None, None, "PASSPORT_NUMBER_MISSING"

    boarding = boarding_by_passport.get(passport_number)
    if not boarding:
        return passport, None, None, "BOARDING_PASS_NOT_FOUND"

    if normalize_name(boarding.get("full_name", "")) != normalize_name(passport.get("full_name", "")):
        return passport, boarding, None, "NAME_MISMATCH"

    flight_no = boarding.get("flight_number")
    if not flight_no:
        return passport, boarding, None, "FLIGHT_NUMBER_MISSING"

    flight = flights_by_number.get(flight_no)
    if not flight:
        return passport, boarding, None, "FLIGHT_NOT_FOUND"

    return passport, boarding, flight, None


def build_label(recognized_name: str, passport, boarding, flight, error_code: str):
    if error_code is None and passport and boarding and flight:
        dep = fmt_hm(flight.get("scheduled_departure"))
        gate = flight.get("gate", "?")
        status = flight.get("status", "")
        seat = boarding.get("seat", "")
        flight_no = flight.get("flight_number", "")
        parts = [
            passport.get("full_name", recognized_name),
            flight_no,
            f"Seat {seat}" if seat else None,
            f"Gate {gate}" if gate else None,
            f"Dep {dep}" if dep else None,
            status if status else None
        ]
        return " | ".join([p for p in parts if p])

    if error_code == "PASSPORT_NOT_FOUND":
        return f"{recognized_name} | No passport record"
    if error_code in ("BOARDING_PASS_NOT_FOUND", "FLIGHT_NUMBER_MISSING"):
        return f"{passport.get('full_name', recognized_name)} | No boarding pass"
    if error_code == "NAME_MISMATCH":
        return f"{passport.get('full_name', recognized_name)} | Data mismatch"
    if error_code == "FLIGHT_NOT_FOUND":
        return f"{passport.get('full_name', recognized_name)} | Flight not found"
    if error_code == "PASSPORT_NUMBER_MISSING":
        return f"{passport.get('full_name', recognized_name)} | Missing passport number"

    return recognized_name


def resolve_mode(mode: str) -> str:
    mode = (mode or "arcface").lower()
    if mode not in ("arcface", "maskinv", "adaface", "focusface", "hybrid"):
        mode = "arcface"
    return mode


def get_model_tag(params: argparse.Namespace) -> str:
    mode = resolve_mode(params.mode)

    if mode == "arcface":
        return "arcface_mbf"
    if mode == "maskinv":
        return "maskinv"
    if mode == "adaface":
        return "adaface"
    if mode == "focusface":
        return "focusface"

    return mode


def db_path_for_mode(db_root: str, params: argparse.Namespace) -> str:
    return os.path.join(db_root, get_model_tag(params))


def build_face_database_for_mode(detector: "SCRFD",
                                 arc: "ArcFace",
                                 mki: "MaskInv",
                                 ada: "AdaFace",
                                 focus: "FocusFaceEncoder",
                                 params: argparse.Namespace,
                                 mode: str,
                                 force_update: bool = False) -> "FaceDatabase":
    original_mode = params.mode
    params.mode = mode
    try:
        return build_face_database(detector, arc, mki, ada, focus, params, force_update=force_update)
    finally:
        params.mode = original_mode


def get_embedding(mode: str,
                  arc: ArcFace,
                  mki: MaskInv,
                  ada: AdaFace,
                  focus: FocusFaceEncoder,
                  image_bgr: np.ndarray,
                  kps: np.ndarray) -> np.ndarray:
    mode = resolve_mode(mode)

    if mode == "arcface":
        return arc.get_embedding(image_bgr, kps, normalized=True)

    if mode == "maskinv":
        return mki.get_embedding(image_bgr, kps)

    if mode == "adaface":
        aligned_face, _ = face_alignment(image_bgr, kps)
        return ada(aligned_face)

    if mode == "focusface":
        aligned_face, _ = face_alignment(image_bgr, kps)
        aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(aligned_rgb)
        return focus.get_embedding_from_pil(pil_img)

    raise ValueError(f"Unsupported mode: {mode}")


# -------------------------
# Face DB builder
# -------------------------
def build_face_database(detector: "SCRFD",
                        arc: "ArcFace",
                        mki: "MaskInv",
                        ada: "AdaFace",
                        focus: "FocusFaceEncoder",
                        params: argparse.Namespace,
                        force_update: bool = False) -> "FaceDatabase":
    """
    Builds/loads a FaceDatabase for the selected mode.

    DB stored at:
      <db-root>/arcface_mbf
      <db-root>/maskinv
      <db-root>/adaface
      <db-root>/focusface

    Faces dir:
      ./assets/faces/<PersonName>/<img>.jpg
    """

    mode = resolve_mode(params.mode)
    model_tag = get_model_tag(params)
    db_path = db_path_for_mode(params.db_root, params)

    face_db = FaceDatabase(db_path=db_path, max_workers=4)

    if not force_update and face_db.load():
        logging.info(f"Loaded face database from disk: {db_path} (mode={mode}, model={model_tag})")
        return face_db

    logging.info(f"Building face database into: {db_path} (mode={mode}, model={model_tag})")

    if not os.path.exists(params.faces_dir):
        logging.warning(f"Faces directory {params.faces_dir} does not exist. Creating empty database.")
        face_db.save()
        return face_db

    for item_name in os.listdir(params.faces_dir):
        person_dir_path = os.path.join(params.faces_dir, item_name)
        if not os.path.isdir(person_dir_path):
            continue

        name = item_name

        for filename in os.listdir(person_dir_path):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(person_dir_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Could not read image: {image_path}")
                continue

            try:
                _, kpss = detector.detect(image, max_num=1)
                if kpss is None or len(kpss) == 0:
                    logging.warning(f"No face detected in {image_path} for {name}. Skipping...")
                    continue

                embedding = get_embedding(mode, arc, mki, ada, focus, image, kpss[0])
                face_db.add_face(embedding, name)
                logging.info(f"Added face for: {name} (mode={mode}, model={model_tag}) (Image: {filename})")

            except Exception as e:
                logging.error(f"Error processing {image_path} for {name}: {e}")
                continue

    face_db.save()
    logging.info(f"Face database built and saved. mode={mode}, model={model_tag}, path={db_path}")

    return face_db


def hybrid_search(frame: np.ndarray,
                  kps: np.ndarray,
                  arc: ArcFace,
                  ada: AdaFace,
                  arc_db: FaceDatabase,
                  ada_db: FaceDatabase,
                  arcface_low_thresh: float,
                  arcface_high_thresh: float,
                  adaface_thresh: float) -> tuple[str, float]:
    # First: ArcFace
    arc_emb = arc.get_embedding(frame, kps, normalized=True)
    arc_name, arc_score = arc_db.search(arc_emb, threshold=arcface_high_thresh)

    # Confident ArcFace match
    if arc_name != "Unknown":
        return arc_name, arc_score

    # Check uncertain band
    if arc_score >= arcface_low_thresh:
        aligned_face, _ = face_alignment(frame, kps)
        ada_emb = ada(aligned_face)
        ada_name, ada_score = ada_db.search(ada_emb, threshold=adaface_thresh)

        if ada_name != "Unknown":
            return ada_name, ada_score

        # If AdaFace also fails, keep unknown but preserve the better score for debugging
        return "Unknown", max(arc_score, ada_score)

    # Clearly low ArcFace score → unknown
    return "Unknown", arc_score


# -------------------------
# Frame processing
# -------------------------
def frame_processor(frame: np.ndarray,
                    detector: SCRFD,
                    arc: ArcFace,
                    mki: MaskInv,
                    ada: AdaFace,
                    focus: FocusFaceEncoder,
                    face_db: FaceDatabase | None,
                    colors: dict,
                    params: argparse.Namespace,
                    passports_by_name: dict,
                    boarding_by_passport: dict,
                    flights_by_number: dict,
                    debouncer: RecognitionDebounce,
                    arc_db: FaceDatabase | None = None,
                    ada_db: FaceDatabase | None = None) -> np.ndarray:
    try:
        bboxes, kpss = detector.detect(frame, params.max_num)

        if bboxes is None or len(bboxes) == 0:
            return frame

        mode = resolve_mode(params.mode)
        processed_bboxes = []
        results = []

        for bbox, kps in zip(bboxes, kpss):
            try:
                *bbox_coords, _conf = bbox.astype(np.int32)
                processed_bboxes.append(bbox_coords)

                if mode == "hybrid":
                    if arc_db is None or ada_db is None:
                        raise ValueError("Hybrid mode requires both arc_db and ada_db")

                    name, similarity = hybrid_search(
                        frame=frame,
                        kps=kps,
                        arc=arc,
                        ada=ada,
                        arc_db=arc_db,
                        ada_db=ada_db,
                        arcface_low_thresh=params.arcface_low_thresh,
                        arcface_high_thresh=params.arcface_high_thresh,
                        adaface_thresh=params.adaface_thresh
                    )
                else:
                    if face_db is None:
                        raise ValueError("Selected mode requires a face database.")
                    emb = get_embedding(mode, arc, mki, ada, focus, frame, kps)
                    name, similarity = face_db.search(emb, params.similarity_thresh)

                results.append((name, similarity))

            except Exception as e:
                logging.warning(f"Error processing face embedding: {e}")
                processed_bboxes.append(bbox_coords if 'bbox_coords' in locals() else [0, 0, 0, 0])
                results.append(("Unknown", 0.0))
                continue

        for bbox, (name, similarity) in zip(processed_bboxes, results):
            if name != "Unknown":
                if name not in colors:
                    colors[name] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )

                passport, boarding, flight, err = get_traveller_flight_context(
                    recognized_name=name,
                    passports_by_name=passports_by_name,
                    boarding_by_passport=boarding_by_passport,
                    flights_by_number=flights_by_number
                )

                label = build_label(name, passport, boarding, flight, err)

                if not params.disable_kiosk_post and debouncer.update(name):
                    base = params.kiosk_api.rstrip("/")
                    post_json(
                        f"{base}/kiosk/recognized",
                        {
                            "name": name,
                            "similarity": float(similarity),
                            "device_id": params.device_id,
                            "mode": mode
                        }
                    )

                draw_bbox_info(frame, bbox, similarity=similarity, name=label, color=colors[name])
            else:
                draw_bbox(frame, bbox, (255, 0, 0))

    except Exception as e:
        logging.error(f"Error in frame processing: {e}")

    return frame


def main(params):
    try:
        detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
        arc = ArcFace(params.rec_weight)
        mki = MaskInv(params.maskinv_weight)
        ada = AdaFace(params.adaface_weight)
        focus = FocusFaceEncoder(params.focusface_weight)
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        return

    passports = read_passports()
    flights = read_flights()
    boarding_passes = read_boarding_passes()

    passports_by_name = {
        normalize_name(p.get("full_name", "")): p
        for p in passports
        if p.get("full_name")
    }
    boarding_by_passport = {
        bp.get("passport_number"): bp
        for bp in boarding_passes
        if bp.get("passport_number")
    }
    flights_by_number = {
        f.get("flight_number"): f
        for f in flights
        if f.get("flight_number")
    }

    logging.info(
        f"Loaded passports: {len(passports)} | boarding_passes: {len(boarding_passes)} | flights: {len(flights)}"
    )

    mode = resolve_mode(params.mode)

    if mode == "hybrid":
        arc_db = build_face_database_for_mode(
            detector, arc, mki, ada, focus, params, mode="arcface", force_update=params.update_db
        )
        ada_db = build_face_database_for_mode(
            detector, arc, mki, ada, focus, params, mode="adaface", force_update=params.update_db
        )
        face_db = None
    else:
        face_db = build_face_database(
            detector, arc, mki, ada, focus, params, force_update=params.update_db
        )
        arc_db = None
        ada_db = None

    if params.update_db:
        logging.info("Database updated. Exiting (no live webcam).")
        return

    debouncer = RecognitionDebounce(params.stable_frames, params.min_post_interval)
    colors = {}

    cap = None
    try:
        cap = cv2.VideoCapture(int(params.source), cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise IOError(f"Could not open webcam index: {params.source}")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            frame = frame_processor(
                frame,
                detector,
                arc,
                mki,
                ada,
                focus,
                face_db,
                colors,
                params,
                passports_by_name,
                boarding_by_passport,
                flights_by_number,
                debouncer,
                arc_db=arc_db,
                ada_db=ada_db
            )
            end = time.time()

            cv2.imshow(f"Face Recognition (Live) mode={resolve_mode(params.mode)}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1
            logging.debug(f"Frame {frame_count}, FPS: {1 / (end - start):.2f}")

        logging.info(f"Processed {frame_count} frames (live).")

    except Exception as e:
        logging.error(f"Error during live processing: {e}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)