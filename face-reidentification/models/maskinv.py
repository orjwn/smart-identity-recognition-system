import cv2
import numpy as np
from onnxruntime import InferenceSession
from utils.helpers import face_alignment


class MaskInv:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.session = InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _preprocess(aligned_bgr_112: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def get_embedding(self, frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        kps = np.asarray(kps, dtype=np.float32)
        aligned, _ = face_alignment(frame_bgr, kps, image_size=112)
        inp = self._preprocess(aligned)
        emb = self.session.run(None, {self.input_name: inp})[0][0].astype(np.float32)
        return self._l2norm(emb)
