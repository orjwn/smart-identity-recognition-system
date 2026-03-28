# server/main.py

import os
import sys
import json
import time
import asyncio
import threading
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple

import cv2
import numpy as np

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
from models import SCRFD, ArcFace, MaskInv

# -----------------------------
# Config
# -----------------------------
DATA_DIR = os.getenv("DATA_DIR", os.path.abspath(os.path.join(ROOT, "data")))
PASS_FILE = os.path.join(DATA_DIR, "passports.json")
BOARD_FILE = os.path.join(DATA_DIR, "boarding_passes.json")
FLIGHT_FILE = os.path.join(DATA_DIR, "flights.json")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

DET_WEIGHT = os.getenv("DET_WEIGHT", os.path.join(ROOT, "weights", "det_10g.onnx"))
REC_WEIGHT = os.getenv("REC_WEIGHT", os.path.join(ROOT, "weights", "w600k_mbf.onnx"))


EMBEDDER_MODE = os.getenv("EMBEDDER_MODE", "arcface").lower()  # arcface|maskinv|hybrid
MASKINV_WEIGHT = os.getenv("MASKINV_WEIGHT", os.path.join(ROOT, "weights", "maskinv", "maskinv_hg.onnx"))

# DB root folder that will contain per-model subfolders:
#   DB_ROOT/arcface
#   DB_ROOT/maskinv
DB_ROOT = os.getenv("DB_ROOT", os.path.join(ROOT, "database", "face_database"))

SIMILARITY_THRESH = float(os.getenv("SIMILARITY_THRESH", "0.4"))
CONFIDENCE_THRESH = float(os.getenv("CONFIDENCE_THRESH", "0.5"))
MAX_NUM = int(os.getenv("MAX_NUM", "0"))

STABLE_FRAMES = int(os.getenv("STABLE_FRAMES", "5"))
MIN_POST_INTERVAL = float(os.getenv("MIN_POST_INTERVAL", "1.5"))

NO_FACE_RESET_SECONDS = float(os.getenv("NO_FACE_RESET_SECONDS", "3.0"))

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.strip().lower().split())


def load_json_list(path: str, root_key: str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get(root_key, [])
    except Exception as e:
        logging.error(f"Failed reading {path}: {e}")
        return []


# -----------------------------
# Data store (indexes for fast lookup)
# -----------------------------
class DataStore:
    def __init__(self) -> None:
        self.passports: list = []
        self.boarding_passes: list = []
        self.flights: list = []

        self.passports_by_name: Dict[str, Dict[str, Any]] = {}
        self.boarding_by_passport: Dict[str, Dict[str, Any]] = {}
        self.flights_by_number: Dict[str, Dict[str, Any]] = {}

        self.reload()

    def reload(self) -> None:
        self.passports = load_json_list(PASS_FILE, "passports")
        self.boarding_passes = load_json_list(BOARD_FILE, "boarding_passes")
        self.flights = load_json_list(FLIGHT_FILE, "flights")

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


store = DataStore()


# -----------------------------
# SSE manager
# -----------------------------
class SSEManager:
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


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Smart Identity Kiosk Backend", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "data_dir": DATA_DIR, "embedder_mode": EMBEDDER_MODE}


@app.post("/admin/reload-data")
async def reload_data():
    store.reload()
    return {
        "ok": True,
        "counts": {
            "passports": len(store.passports),
            "boarding_passes": len(store.boarding_passes),
            "flights": len(store.flights),
        },
    }


@app.get("/kiosk/state")
async def kiosk_state():
    return current_state


@app.post("/kiosk/reset")
async def kiosk_reset():
    set_state(None, None, None, None, None)
    await sse.broadcast({"event": "traveller_update", "data": json.dumps(current_state)})
    return {"ok": True}


@app.post("/kiosk/recognized")
async def kiosk_recognized(payload: RecognizedEventIn):
    """
    Manual endpoint still available (Swagger/testing).
    Backend recognition loop also updates state directly.
    """
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

    await sse.broadcast({"event": "traveller_update", "data": json.dumps(current_state)})
    return {"ok": True, "error": err}


@app.get("/kiosk/events")
async def kiosk_events(request: Request):
    client_q = await sse.connect()

    async def event_generator():
        try:
            yield {"event": "init", "data": json.dumps(current_state)}

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
    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False

    def start(self):
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index={self.index}")
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

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


def mjpeg_generator(camera: CameraService):
    """MJPEG stream generator for <img src='/kiosk/video'>."""
    while True:
        frame = camera.get_frame()
        if frame is None:
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
arcface: Optional[ArcFace] = None
maskinv: Optional[MaskInv] = None
face_db_arc: Optional[FaceDatabase] = None
face_db_mask: Optional[FaceDatabase] = None

_last_face_seen_at = 0.0


@app.get("/kiosk/video")
def kiosk_video():
    return StreamingResponse(
        mjpeg_generator(camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _resolve_mode() -> str:
    mode = (EMBEDDER_MODE or "arcface").lower()
    if mode not in ("arcface", "maskinv", "hybrid"):
        mode = "arcface"

    # Hybrid for now defaults to maskinv (safe for both masked/unmasked).
    # Later: add a mask detector and switch per-face.
    if mode == "hybrid":
        return "maskinv"
    return mode


def recognition_worker(loop: asyncio.AbstractEventLoop):
    """
    Background thread:
    - reads latest frame
    - detects + recognizes
    - updates state + broadcasts SSE using the main loop
    """
    global _last_face_seen_at

    while True:
        frame = camera.get_frame()
        if (
            frame is None
            or detector is None
            or arcface is None
            or maskinv is None
            or face_db_arc is None
            or face_db_mask is None
        ):
            time.sleep(0.1)
            continue

        try:
            bboxes, kpss = detector.detect(frame, MAX_NUM)

            if len(bboxes) == 0:
                # auto reset if no face for a while
                if _last_face_seen_at and (time.time() - _last_face_seen_at) > NO_FACE_RESET_SECONDS:
                    _last_face_seen_at = 0.0
                    set_state(None, None, None, None, None)
                    asyncio.run_coroutine_threadsafe(
                        sse.broadcast({"event": "traveller_update", "data": json.dumps(current_state)}),
                        loop,
                    )
                time.sleep(0.1)
                continue

            _last_face_seen_at = time.time()

            mode = _resolve_mode()
            embeddings = []
            for bbox, kps in zip(bboxes, kpss):
                if mode == "arcface":
                    emb = arcface.get_embedding(frame, kps)
                else:
                    emb = maskinv.get_embedding(frame, kps)
                embeddings.append(emb)

            db = face_db_arc if mode == "arcface" else face_db_mask
            results = db.batch_search(embeddings, SIMILARITY_THRESH)

            # pick best face in the frame
            best_name, best_sim = "Unknown", 0.0
            for name, sim in results:
                if name != "Unknown" and float(sim) > best_sim:
                    best_name, best_sim = name, float(sim)

            if best_name != "Unknown" and debouncer.update(best_name):
                passport, bp, fl, err = store.join_by_recognized_name(best_name)
                recognized = {
                    "name": best_name,
                    "similarity": best_sim,
                    "device_id": "backend-camera",
                    "mode": mode,
                }
                set_state(recognized, passport, bp, fl, err)

                asyncio.run_coroutine_threadsafe(
                    sse.broadcast({"event": "traveller_update", "data": json.dumps(current_state)}),
                    loop,
                )

        except Exception as e:
            logging.debug(f"[backend-recognition] error: {e}")

        time.sleep(0.12)  # ~8 FPS recognition loop


@app.on_event("startup")
async def on_startup():
    """
    Start camera + load models + load face DBs + start recognition worker.
    """
    global detector, arcface, maskinv, face_db_arc, face_db_mask

    camera.start()

    detector = SCRFD(DET_WEIGHT, input_size=(640, 640), conf_thres=CONFIDENCE_THRESH)

    # Load both embedding models (Option C)
    arcface = ArcFace(REC_WEIGHT)
    maskinv = MaskInv(MASKINV_WEIGHT)

    # Load both DBs (embedding spaces differ!)
    db_arc_path = os.path.join(DB_ROOT, "arcface")
    db_mask_path = os.path.join(DB_ROOT, "maskinv")

    face_db_arc = FaceDatabase(db_path=db_arc_path, max_workers=1)
    face_db_mask = FaceDatabase(db_path=db_mask_path, max_workers=1)

    loaded_arc = face_db_arc.load()
    loaded_mask = face_db_mask.load()

    if not loaded_arc:
        logging.warning("ArcFace DB not loaded from disk. Build it first for arcface.")
    else:
        logging.info("ArcFace DB loaded successfully.")

    if not loaded_mask:
        logging.warning("MaskInv DB not loaded from disk. Build it first for maskinv.")
    else:
        logging.info("MaskInv DB loaded successfully.")

    loop = asyncio.get_running_loop()
    threading.Thread(target=recognition_worker, args=(loop,), daemon=True).start()

    logging.info("Camera stream + backend recognition started.")


@app.on_event("shutdown")
async def on_shutdown():
    camera.stop()
