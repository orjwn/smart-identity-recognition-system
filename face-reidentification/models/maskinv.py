import cv2
import numpy as np
from onnxruntime import InferenceSession


class MaskInv:
    """
    MaskInv student recognizer wrapper.

    Expects an already aligned 112x112 BGR face crop from the existing
    evaluation pipeline, same style as AdaFace / FocusFace evaluation.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.session = InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _preprocess(aligned_bgr_112: np.ndarray) -> np.ndarray:
        if aligned_bgr_112 is None:
            raise ValueError("aligned_bgr_112 is None")

        if aligned_bgr_112.shape != (112, 112, 3):
            raise ValueError(f"Expected aligned face shape (112,112,3), got {aligned_bgr_112.shape}")

        rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def __call__(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        """Return one L2-normalized embedding for a masked aligned face crop."""
        inp = self._preprocess(aligned_bgr_112)
        emb = self.session.run(None, {self.input_name: inp})[0][0].astype(np.float32)
        return self._l2norm(emb)
