# server/main.py

import os
import sys
import json
import time
import asyncio
import threading
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO)

# -----------------------------
# Paths / Imports from project root
# -----------------------------
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.append(ROOT)

from database import FaceDatabase
from models import SCRFD, ArcFace, MaskInv, AdaFace
from models.focusface import FocusFaceEncoder
from utils.helpers import face_alignment

# -----------------------------
# Config
# -----------------------------
DATA_DIR = os.getenv("DATA_DIR", os.path.abspath(os.path.join(ROOT, "data")))
PASS_FILE = os.path.join(DATA_DIR, "passports.json")
BOARD_FILE = os.path.join(DATA_DIR, "boarding_passes.json")
FLIGHT_FILE = os.path.join(DATA_DIR, "flights.json")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

DET_WEIGHT = os.getenv("DET_WEIGHT", os.path.join(ROOT, "weights", "det_10g.onnx"))

# Unmasked priority:
# 1) ArcFace MBF
# 2) AdaFace
# 3) ArcFace R50
ARCFACE_MBF_WEIGHT = os.getenv("ARCFACE_MBF_WEIGHT", os.path.join(ROOT, "weights", "w600k_mbf.onnx"))
ADAFACE_WEIGHT = os.getenv("ADAFACE_WEIGHT", os.path.join(ROOT, "weights", "adaface_ir18_webface4m.ckpt"))
ARCFACE_R50_WEIGHT = os.getenv("ARCFACE_R50_WEIGHT", os.path.join(ROOT, "weights", "w600k_r50.onnx"))

# Masked priority:
# 1) FocusFace
# 2) MaskInv
FOCUSFACE_WEIGHT = os.getenv("FOCUSFACE_WEIGHT", os.path.join(ROOT, "weights", "focus_face_w_pretrained.mdl"))
MASKINV_WEIGHT = os.getenv("MASKINV_WEIGHT", os.path.join(ROOT, "weights", "maskinv", "maskinv_hg.onnx"))

# Hardcoded default routing:
# auto = ArcFace MBF -> AdaFace -> ArcFace R50 for unmasked
#        FocusFace -> MaskInv for masked
# You do NOT need to set this every run.
# Optional manual overrides: auto|arcface_mbf|adaface|arcface_r50|focusface|maskinv
ROUTING_MODE = os.getenv("ROUTING_MODE", "auto").lower()

# FocusFace mask score threshold:
# if mask_score >= this value, treat the face as masked
FOCUSFACE_MASK_THRESH = float(os.getenv("FOCUSFACE_MASK_THRESH", "0.50"))

# Per-model recognition thresholds
ARCFACE_MBF_THRESH = float(os.getenv("ARCFACE_MBF_THRESH", "0.40"))
ADAFACE_THRESH = float(os.getenv("ADAFACE_THRESH", "0.50"))
ARCFACE_R50_THRESH = float(os.getenv("ARCFACE_R50_THRESH", "0.40"))
FOCUSFACE_THRESH = float(os.getenv("FOCUSFACE_THRESH", "0.50"))
MASKINV_THRESH = float(os.getenv("MASKINV_THRESH", "0.50"))

# DB root folder that will contain per-model subfolders:
#   DB_ROOT/arcface_mbf
#   DB_ROOT/adaface
#   DB_ROOT/arcface_r50
#   DB_ROOT/focusface
#   DB_ROOT/maskinv
DB_ROOT = os.getenv("DB_ROOT", os.path.join(ROOT, "database", "face_database"))

CONFIDENCE_THRESH = float(os.getenv("CONFIDENCE_THRESH", "0.5"))
MAX_NUM = int(os.getenv("MAX_NUM", "0"))

STABLE_FRAMES = int(os.getenv("STABLE_FRAMES", "5"))
MIN_POST_INTERVAL = float(os.getenv("MIN_POST_INTERVAL", "1.5"))

NO_FACE_RESET_SECONDS = float(os.getenv("NO_FACE_RESET_SECONDS", "3.0"))

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

SERVICE_STATUS: Dict[str, Any] = {
    "camera": {"ready": False, "index": CAMERA_INDEX, "error": None},
    "models": {},
    "databases": {},
    "data": {},
    "recognition_ready": False,
}
"""Runtime readiness snapshot returned by /health and included in kiosk state.

The backend is designed to stay online during a viva/demo even when optional
hardware or large local assets are unavailable. Each startup step records its
own status here instead of crashing the whole API process.
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.strip().lower().split())


def load_json_list(path: str, root_key: str) -> Tuple[list, Optional[str]]:
    if not os.path.exists(path):
        message = f"Missing JSON data file: {path}"
        logging.error(message)
        return [], message

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        rows = obj.get(root_key)
        if not isinstance(rows, list):
            message = f"JSON data file {path} does not contain a list at key '{root_key}'"
            logging.error(message)
            return [], message
        return rows, None
    except Exception as e:
        message = f"Failed reading {path}: {e}"
        logging.error(message)
        return [], message


# -----------------------------
# Data store
# -----------------------------
class DataStore:
    """Loads and joins the simulated airport records used by the kiosk demo.

    The prototype intentionally uses mock JSON data rather than real passenger
    records. A recognised face label is matched to a passport by name, then to
    a boarding pass by passport number, then to a flight by flight number.
    """

    def __init__(self) -> None:
        self.passports: list = []
        self.boarding_passes: list = []
        self.flights: list = []
        self.load_errors: List[str] = []

        self.passports_by_name: Dict[str, Dict[str, Any]] = {}
        self.boarding_by_passport: Dict[str, Dict[str, Any]] = {}
        self.flights_by_number: Dict[str, Dict[str, Any]] = {}

        self.reload()

    def reload(self) -> None:
        self.load_errors = []

        self.passports, pass_error = load_json_list(PASS_FILE, "passports")
        self.boarding_passes, board_error = load_json_list(BOARD_FILE, "boarding_passes")
        self.flights, flight_error = load_json_list(FLIGHT_FILE, "flights")

        self.load_errors = [
            error for error in (pass_error, board_error, flight_error) if error
        ]

        self.passports_by_name = {
            normalize_name(p.get("full_name", "")): p
            for p in self.passports
            if p.get("full_name")
        }
        self.boarding_by_passport = {
            bp.get("passport_number"): bp
            for bp in self.boarding_passes
            if bp.get("passport_number")
        }
        self.flights_by_number = {
            fl.get("flight_number"): fl
            for fl in self.flights
            if fl.get("flight_number")
        }

        logging.info(
            f"Loaded: passports={len(self.passports)} boarding_passes={len(self.boarding_passes)} flights={len(self.flights)}"
        )
        SERVICE_STATUS["data"] = self.health()

    def join_by_recognized_name(
        self, recognized_name: str
    ) -> Tuple[Optional[dict], Optional[dict], Optional[dict], Optional[str]]:
        p = self.passports_by_name.get(normalize_name(recognized_name))
        if not p:
            return None, None, None, "PASSPORT_NOT_FOUND"

        passport_number = p.get("passport_number")
        if not passport_number:
            return p, None, None, "PASSPORT_NUMBER_MISSING"

        bp = self.boarding_by_passport.get(passport_number)
        if not bp:
            return p, None, None, "BOARDING_PASS_NOT_FOUND"

        if normalize_name(bp.get("full_name", "")) != normalize_name(p.get("full_name", "")):
            return p, bp, None, "NAME_MISMATCH"

        flight_no = bp.get("flight_number")
        if not flight_no:
            return p, bp, None, "FLIGHT_NUMBER_MISSING"

        fl = self.flights_by_number.get(flight_no)
        if not fl:
            return p, bp, None, "FLIGHT_NOT_FOUND"

        return p, bp, fl, None

    def health(self) -> Dict[str, Any]:
        return {
            "ready": not self.load_errors,
            "counts": {
                "passports": len(self.passports),
                "boarding_passes": len(self.boarding_passes),
                "flights": len(self.flights),
            },
            "errors": self.load_errors,
        }


store = DataStore()


# -----------------------------
# SSE manager
# -----------------------------
class SSEManager:
    """Small in-process broadcaster for kiosk Server-Sent Events.

    Each browser connection receives its own bounded queue. If a client falls
    behind, the oldest pending event is dropped so the kiosk shows current
    state rather than stale recognition updates.
    """

    def __init__(self) -> None:
        self._clients: Set[asyncio.Queue] = set()

    async def connect(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._clients.add(q)
        return q

    def disconnect(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    async def broadcast(self, event: Dict[str, Any]) -> None:
        for q in list(self._clients):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass


sse = SSEManager()


# -----------------------------
# API Models
# -----------------------------
class RecognizedEventIn(BaseModel):
    name: str = Field(..., min_length=1)
    similarity: Optional[float] = None
    device_id: Optional[str] = "backend-camera"


# -----------------------------
# Kiosk state
# -----------------------------
current_state: Dict[str, Any] = {
    "recognized": None,
    "passport": None,
    "boarding_pass": None,
    "flight": None,
    "error": None,
    "updated_at": now_iso(),
}

_last_update_at = 0.0
_last_name = None


def set_state(
    recognized: Optional[dict],
    passport: Optional[dict],
    boarding_pass: Optional[dict],
    flight: Optional[dict],
    error: Optional[str],
) -> None:
    global current_state
    current_state = {
        "recognized": recognized,
        "passport": passport,
        "boarding_pass": boarding_pass,
        "flight": flight,
        "error": error,
        "updated_at": now_iso(),
    }


def _service_error_code() -> Optional[str]:
    if not SERVICE_STATUS.get("data", {}).get("ready", True):
        return "DATA_UNAVAILABLE"
    if not SERVICE_STATUS.get("camera", {}).get("ready", False):
        return "CAMERA_UNAVAILABLE"
    if not SERVICE_STATUS.get("recognition_ready", False):
        return "RECOGNITION_UNAVAILABLE"
    return None


def state_snapshot() -> Dict[str, Any]:
    """Return the frontend-facing kiosk state with current service readiness."""
    snapshot = dict(current_state)
    if snapshot.get("error") is None:
        snapshot["error"] = _service_error_code()
    snapshot["service"] = SERVICE_STATUS
    return snapshot


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Smart Identity Kiosk Backend", version="1.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": bool(SERVICE_STATUS.get("recognition_ready")) and bool(SERVICE_STATUS.get("camera", {}).get("ready")),
        "data_dir": DATA_DIR,
        "routing_mode": ROUTING_MODE,
        "focusface_mask_thresh": FOCUSFACE_MASK_THRESH,
        "thresholds": {
            "arcface_mbf": ARCFACE_MBF_THRESH,
            "adaface": ADAFACE_THRESH,
            "arcface_r50": ARCFACE_R50_THRESH,
            "focusface": FOCUSFACE_THRESH,
            "maskinv": MASKINV_THRESH,
        },
        "service": SERVICE_STATUS,
    }


@app.post("/admin/reload-data")
async def reload_data():
    store.reload()
    return {
        "ok": True,
        "data": store.health(),
    }


@app.get("/kiosk/state")
async def kiosk_state():
    return state_snapshot()


@app.post("/kiosk/reset")
async def kiosk_reset():
    set_state(None, None, None, None, None)
    await sse.broadcast({"event": "traveller_update", "data": json.dumps(state_snapshot())})
    return {"ok": True}


@app.post("/kiosk/recognized")
async def kiosk_recognized(payload: RecognizedEventIn):
    global _last_update_at, _last_name

    now = time.time()
    if _last_name == payload.name and (now - _last_update_at) < 0.6:
        return {"ok": True, "debounced": True}

    passport, bp, fl, err = store.join_by_recognized_name(payload.name)

    recognized = {
        "name": payload.name,
        "similarity": payload.similarity,
        "device_id": payload.device_id,
    }

    set_state(recognized, passport, bp, fl, err)

    _last_update_at = now
    _last_name = payload.name

    await sse.broadcast({"event": "traveller_update", "data": json.dumps(state_snapshot())})
    return {"ok": True, "error": err}


@app.get("/kiosk/events")
async def kiosk_events(request: Request):
    client_q = await sse.connect()

    async def event_generator():
        try:
            yield {"event": "init", "data": json.dumps(state_snapshot())}

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(client_q.get(), timeout=15)
                    yield event
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "keep-alive"}
        finally:
            sse.disconnect(client_q)

    return EventSourceResponse(event_generator())


@app.get("/traveller/by-name/{name}")
async def traveller_by_name(name: str):
    passport, bp, fl, err = store.join_by_recognized_name(name)
    return {"passport": passport, "boarding_pass": bp, "flight": fl, "error": err}


# -----------------------------
# Camera streaming + recognition worker
# -----------------------------
class CameraService:
    """Continuously captures webcam frames for MJPEG streaming and recognition.

    Camera failures are stored in SERVICE_STATUS rather than raised at startup,
    allowing /health, /docs, SSE, and the frontend to remain available.
    """

    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self.error: Optional[str] = None

    def start(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.index)
        except Exception as exc:
            self.error = f"Could not initialise webcam index={self.index}: {exc}"
            SERVICE_STATUS["camera"] = {"ready": False, "index": self.index, "error": self.error}
            logging.error(self.error)
            return False

        if not self.cap.isOpened():
            self.error = f"Could not open webcam index={self.index}"
            SERVICE_STATUS["camera"] = {"ready": False, "index": self.index, "error": self.error}
            logging.error(self.error)
            return False

        self.error = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        SERVICE_STATUS["camera"] = {"ready": True, "index": self.index, "error": None}
        return True

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame
            time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class RecognitionDebounce:
    """Require repeated recognition before publishing a traveller update.

    Face recognizers can flicker between identities frame-to-frame. This
    debouncer waits for a stable name across several frames and rate-limits
    repeat posts, which keeps the kiosk UI calmer during the live demo.
    """

    def __init__(self, stable_frames: int, min_post_interval: float):
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


def _status_frame(message: str) -> np.ndarray:
    """Create a simple MJPEG placeholder frame when the webcam has no image."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (18, 24, 35)
    cv2.putText(
        frame,
        "Smart Identity Kiosk",
        (36, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (230, 238, 252),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message[:58],
        (36, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (138, 180, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def mjpeg_generator(camera: CameraService):
    while True:
        frame = camera.get_frame()
        if frame is None:
            frame = _status_frame(camera.error or "Waiting for camera frame")
            ok, jpg = cv2.imencode(".jpg", frame)
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                )
            time.sleep(0.05)
            continue

        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        )


camera = CameraService(CAMERA_INDEX)
debouncer = RecognitionDebounce(STABLE_FRAMES, MIN_POST_INTERVAL)

detector: Optional[SCRFD] = None

arcface_mbf: Optional[ArcFace] = None
adaface: Optional[AdaFace] = None
arcface_r50: Optional[ArcFace] = None

focusface: Optional[FocusFaceEncoder] = None
maskinv: Optional[MaskInv] = None

face_db_arc_mbf: Optional[FaceDatabase] = None
face_db_adaface: Optional[FaceDatabase] = None
face_db_arc_r50: Optional[FaceDatabase] = None
face_db_focus: Optional[FaceDatabase] = None
face_db_mask: Optional[FaceDatabase] = None

_last_face_seen_at = 0.0


@app.get("/kiosk/video")
def kiosk_video():
    return StreamingResponse(
        mjpeg_generator(camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _resolve_routing_mode() -> str:
    mode = (ROUTING_MODE or "auto").lower()
    if mode not in ("auto", "arcface_mbf", "adaface", "arcface_r50", "focusface", "maskinv"):
        mode = "auto"
    return mode


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _align_face(frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    aligned_face, _ = face_alignment(frame, kps, image_size=112)
    return aligned_face


def _search_db(db: FaceDatabase, emb: np.ndarray, thresh: float) -> Tuple[str, float]:
    name, sim = db.search(emb, thresh)
    return name, float(sim)


def _get_arcface_embedding(model: ArcFace, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    return model.get_embedding(frame, kps, normalized=True)


def _get_adaface_embedding(aligned_bgr_112: np.ndarray) -> np.ndarray:
    return adaface(aligned_bgr_112)


def _get_focusface_embedding_and_mask_score(aligned_bgr_112: np.ndarray) -> Tuple[np.ndarray, float]:
    aligned_rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(aligned_rgb)

    x = focusface.transform(pil_img.convert("RGB")).unsqueeze(0).to(focusface.device)

    with torch.no_grad():
        _, emb, _, mask_score = focusface.model(x, inference=True)

    emb_np = emb[0].detach().cpu().numpy().astype(np.float32)
    emb_np = _l2_normalize(emb_np)

    if torch.is_tensor(mask_score):
        score = float(mask_score[0].detach().cpu().item())
    else:
        score = float(mask_score[0])

    return emb_np, score


def _get_maskinv_embedding(frame: np.ndarray, kps: np.ndarray, aligned_bgr_112: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Supports both:
    - old MaskInv wrapper: get_embedding(frame, kps)
    - rewritten wrapper: __call__(aligned_bgr_112)
    """
    if hasattr(maskinv, "get_embedding"):
        return maskinv.get_embedding(frame, kps)

    if aligned_bgr_112 is None:
        aligned_bgr_112 = _align_face(frame, kps)
    return maskinv(aligned_bgr_112)


def _recognize_masked(frame: np.ndarray, kps: np.ndarray, aligned_face: np.ndarray, mask_score: float) -> Dict[str, Any]:
    # 1) FocusFace primary
    focus_emb, _ = _get_focusface_embedding_and_mask_score(aligned_face)
    focus_name, focus_sim = _search_db(face_db_focus, focus_emb, FOCUSFACE_THRESH)
    if focus_name != "Unknown":
        return {
            "name": focus_name,
            "similarity": focus_sim,
            "mode": "focusface",
            "masked": True,
            "mask_score": mask_score,
        }

    # 2) MaskInv secondary
    mask_emb = _get_maskinv_embedding(frame, kps, aligned_face)
    mask_name, mask_sim = _search_db(face_db_mask, mask_emb, MASKINV_THRESH)
    if mask_name != "Unknown":
        return {
            "name": mask_name,
            "similarity": mask_sim,
            "mode": "maskinv",
            "masked": True,
            "mask_score": mask_score,
        }

    # If both fail, keep the stronger masked attempt for debugging
    if mask_sim > focus_sim:
        return {
            "name": "Unknown",
            "similarity": mask_sim,
            "mode": "maskinv",
            "masked": True,
            "mask_score": mask_score,
        }

    return {
        "name": "Unknown",
        "similarity": focus_sim,
        "mode": "focusface",
        "masked": True,
        "mask_score": mask_score,
    }


def _recognize_unmasked(frame: np.ndarray, kps: np.ndarray, aligned_face: np.ndarray, mask_score: float) -> Dict[str, Any]:
    # 1) ArcFace MBF primary
    emb_mbf = _get_arcface_embedding(arcface_mbf, frame, kps)
    name_mbf, sim_mbf = _search_db(face_db_arc_mbf, emb_mbf, ARCFACE_MBF_THRESH)
    if name_mbf != "Unknown":
        return {
            "name": name_mbf,
            "similarity": sim_mbf,
            "mode": "arcface_mbf",
            "masked": False,
            "mask_score": mask_score,
        }

    # 2) AdaFace secondary
    emb_ada = _get_adaface_embedding(aligned_face)
    name_ada, sim_ada = _search_db(face_db_adaface, emb_ada, ADAFACE_THRESH)
    if name_ada != "Unknown":
        return {
            "name": name_ada,
            "similarity": sim_ada,
            "mode": "adaface",
            "masked": False,
            "mask_score": mask_score,
        }

    # 3) ArcFace R50 third
    emb_r50 = _get_arcface_embedding(arcface_r50, frame, kps)
    name_r50, sim_r50 = _search_db(face_db_arc_r50, emb_r50, ARCFACE_R50_THRESH)
    if name_r50 != "Unknown":
        return {
            "name": name_r50,
            "similarity": sim_r50,
            "mode": "arcface_r50",
            "masked": False,
            "mask_score": mask_score,
        }

    # Keep strongest failed attempt for debugging
    best = [
        ("arcface_mbf", sim_mbf),
        ("adaface", sim_ada),
        ("arcface_r50", sim_r50),
    ]
    best_mode, best_sim = max(best, key=lambda x: x[1])

    return {
        "name": "Unknown",
        "similarity": best_sim,
        "mode": best_mode,
        "masked": False,
        "mask_score": mask_score,
    }


def _recognize_one_face(frame: np.ndarray, kps: np.ndarray) -> Dict[str, Any]:
    """
    Returns:
      {
        "name": str,
        "similarity": float,
        "mode": str,
        "masked": bool,
        "mask_score": float
      }
    """
    route_mode = _resolve_routing_mode()
    aligned_face = _align_face(frame, kps)

    # manual overrides
    if route_mode == "arcface_mbf":
        emb = _get_arcface_embedding(arcface_mbf, frame, kps)
        name, sim = _search_db(face_db_arc_mbf, emb, ARCFACE_MBF_THRESH)
        return {"name": name, "similarity": sim, "mode": "arcface_mbf", "masked": False, "mask_score": 0.0}

    if route_mode == "adaface":
        emb = _get_adaface_embedding(aligned_face)
        name, sim = _search_db(face_db_adaface, emb, ADAFACE_THRESH)
        return {"name": name, "similarity": sim, "mode": "adaface", "masked": False, "mask_score": 0.0}

    if route_mode == "arcface_r50":
        emb = _get_arcface_embedding(arcface_r50, frame, kps)
        name, sim = _search_db(face_db_arc_r50, emb, ARCFACE_R50_THRESH)
        return {"name": name, "similarity": sim, "mode": "arcface_r50", "masked": False, "mask_score": 0.0}

    if route_mode == "focusface":
        focus_emb, mask_score = _get_focusface_embedding_and_mask_score(aligned_face)
        name, sim = _search_db(face_db_focus, focus_emb, FOCUSFACE_THRESH)
        return {"name": name, "similarity": sim, "mode": "focusface", "masked": True, "mask_score": mask_score}

    if route_mode == "maskinv":
        emb = _get_maskinv_embedding(frame, kps, aligned_face)
        name, sim = _search_db(face_db_mask, emb, MASKINV_THRESH)
        return {"name": name, "similarity": sim, "mode": "maskinv", "masked": True, "mask_score": 1.0}

    # auto mode: decide masked/unmasked using FocusFace mask score
    _, mask_score = _get_focusface_embedding_and_mask_score(aligned_face)

    if mask_score >= FOCUSFACE_MASK_THRESH:
        return _recognize_masked(frame, kps, aligned_face, mask_score)

    return _recognize_unmasked(frame, kps, aligned_face, mask_score)


def recognition_worker(loop: asyncio.AbstractEventLoop):
    """Background loop that detects faces, recognises identities, and emits SSE.

    The worker is deliberately defensive: if any model/database/camera component
    is unavailable, it sleeps and waits instead of interrupting the FastAPI app.
    """
    global _last_face_seen_at

    while True:
        frame = camera.get_frame()
        if (
            frame is None
            or detector is None
            or arcface_mbf is None
            or adaface is None
            or arcface_r50 is None
            or focusface is None
            or maskinv is None
            or face_db_arc_mbf is None
            or face_db_adaface is None
            or face_db_arc_r50 is None
            or face_db_focus is None
            or face_db_mask is None
        ):
            time.sleep(0.1)
            continue

        try:
            bboxes, kpss = detector.detect(frame, MAX_NUM)

            if len(bboxes) == 0:
                if _last_face_seen_at and (time.time() - _last_face_seen_at) > NO_FACE_RESET_SECONDS:
                    _last_face_seen_at = 0.0
                    set_state(None, None, None, None, None)
                    asyncio.run_coroutine_threadsafe(
                        sse.broadcast({"event": "traveller_update", "data": json.dumps(state_snapshot())}),
                        loop,
                    )
                time.sleep(0.1)
                continue

            _last_face_seen_at = time.time()

            per_face_results = []
            for bbox, kps in zip(bboxes, kpss):
                try:
                    result = _recognize_one_face(frame, kps)
                    per_face_results.append(result)
                except Exception as e:
                    logging.debug(f"[recognition] face error: {e}")
                    per_face_results.append({
                        "name": "Unknown",
                        "similarity": 0.0,
                        "mode": "error",
                        "masked": False,
                        "mask_score": 0.0,
                    })

            best = {"name": "Unknown", "similarity": 0.0, "mode": "unknown", "masked": False, "mask_score": 0.0}
            for result in per_face_results:
                if result["name"] != "Unknown" and float(result["similarity"]) > float(best["similarity"]):
                    best = result

            if best["name"] != "Unknown" and debouncer.update(best["name"]):
                passport, bp, fl, err = store.join_by_recognized_name(best["name"])
                recognized = {
                    "name": best["name"],
                    "similarity": best["similarity"],
                    "device_id": "backend-camera",
                    "mode": best["mode"],
                    "masked": best["masked"],
                    "mask_score": best["mask_score"],
                }
                set_state(recognized, passport, bp, fl, err)

                asyncio.run_coroutine_threadsafe(
                    sse.broadcast({"event": "traveller_update", "data": json.dumps(state_snapshot())}),
                    loop,
                )

        except Exception as e:
            logging.debug(f"[backend-recognition] error: {e}")

        time.sleep(0.12)


def _model_status(path: str, ready: bool, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "ready": ready,
        "path": path,
        "exists": os.path.exists(path),
        "error": error,
    }


def _load_model(label: str, factory, weight_path: str):
    if not os.path.exists(weight_path):
        error = f"Missing model weight: {weight_path}"
        logging.error("%s: %s", label, error)
        SERVICE_STATUS["models"][label] = _model_status(weight_path, False, error)
        return None

    try:
        model = factory(weight_path)
        SERVICE_STATUS["models"][label] = _model_status(weight_path, True, None)
        logging.info("%s loaded from %s", label, weight_path)
        return model
    except Exception as exc:
        error = f"Failed to load {label}: {exc}"
        logging.exception(error)
        SERVICE_STATUS["models"][label] = _model_status(weight_path, False, error)
        return None


def _load_face_db(label: str, db_path: str) -> Optional[FaceDatabase]:
    """Load one model-specific FAISS database without creating false readiness.

    Face databases are stored per recognizer because embeddings from different
    models are not interchangeable, even when they share the same identity names.
    """
    db = FaceDatabase(db_path=db_path, max_workers=1)
    loaded = db.load()
    error = None if loaded else f"Face database not loaded from {db_path}"
    SERVICE_STATUS["databases"][label] = {
        "ready": loaded,
        "path": db_path,
        "index_file": db.index_file,
        "metadata_file": db.meta_file,
        "error": error,
    }
    if error:
        logging.warning(error)
    else:
        logging.info("%s database loaded successfully.", label)
    return db if loaded else None


def _refresh_recognition_ready() -> None:
    model_ready = all(item.get("ready") for item in SERVICE_STATUS.get("models", {}).values())
    db_ready = all(item.get("ready") for item in SERVICE_STATUS.get("databases", {}).values())
    data_ready = bool(SERVICE_STATUS.get("data", {}).get("ready", True))
    camera_ready = bool(SERVICE_STATUS.get("camera", {}).get("ready", False))
    expected_models = {"detector", "arcface_mbf", "adaface", "arcface_r50", "focusface", "maskinv"}
    expected_dbs = {"arcface_mbf", "adaface_ir18", "arcface_r50", "focusface", "maskinv"}

    SERVICE_STATUS["recognition_ready"] = (
        camera_ready
        and data_ready
        and expected_models.issubset(SERVICE_STATUS.get("models", {}).keys())
        and expected_dbs.issubset(SERVICE_STATUS.get("databases", {}).keys())
        and model_ready
        and db_ready
    )


@app.on_event("startup")
async def on_startup():
    global detector
    global arcface_mbf, adaface, arcface_r50
    global focusface, maskinv
    global face_db_arc_mbf, face_db_adaface, face_db_arc_r50, face_db_focus, face_db_mask

    camera.start()

    detector = _load_model(
        "detector",
        lambda path: SCRFD(path, input_size=(640, 640), conf_thres=CONFIDENCE_THRESH),
        DET_WEIGHT,
    )

    # Load all embedding models
    arcface_mbf = _load_model("arcface_mbf", ArcFace, ARCFACE_MBF_WEIGHT)
    adaface = _load_model("adaface", AdaFace, ADAFACE_WEIGHT)
    arcface_r50 = _load_model("arcface_r50", ArcFace, ARCFACE_R50_WEIGHT)
    focusface = _load_model("focusface", FocusFaceEncoder, FOCUSFACE_WEIGHT)
    maskinv = _load_model("maskinv", MaskInv, MASKINV_WEIGHT)

    # Load all DBs
    db_arc_mbf_path = os.path.join(DB_ROOT, "arcface_mbf")
    db_adaface_path = os.path.join(DB_ROOT, "adaface_ir18")
    db_arc_r50_path = os.path.join(DB_ROOT, "arcface_r50")
    db_focus_path = os.path.join(DB_ROOT, "focusface")
    db_mask_path = os.path.join(DB_ROOT, "maskinv")

    face_db_arc_mbf = _load_face_db("arcface_mbf", db_arc_mbf_path)
    face_db_adaface = _load_face_db("adaface_ir18", db_adaface_path)
    face_db_arc_r50 = _load_face_db("arcface_r50", db_arc_r50_path)
    face_db_focus = _load_face_db("focusface", db_focus_path)
    face_db_mask = _load_face_db("maskinv", db_mask_path)

    _refresh_recognition_ready()

    loop = asyncio.get_running_loop()
    threading.Thread(target=recognition_worker, args=(loop,), daemon=True).start()

    if SERVICE_STATUS["recognition_ready"]:
        logging.info("Camera stream + backend recognition started.")
    else:
        logging.warning("Backend started with recognition disabled. Check /health for details.")


@app.on_event("shutdown")
async def on_shutdown():
    camera.stop()
