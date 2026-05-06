"""
Project wrapper for MaskInv / Masked-Face-Recognition-KD.

Status: project wrapper/integration code.
External dependency/reference: fdbtrs/Masked-Face-Recognition-KD / MaskInv.
Student contribution: connects the external MaskInv ONNX recognizer to this
project's aligned-face, embedding, FAISS and evaluation pipeline.

This file loads a MaskInv ONNX recognizer and adapts it to the aligned-face
evaluation pipeline used by this project.

External reference:
- https://github.com/fdbtrs/Masked-Face-Recognition-KD

See THIRD_PARTY_ATTRIBUTION.md for full source and licence notes.
"""

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
        if aligned_bgr_112 is None:
            raise ValueError("aligned_bgr_112 is None")

        if aligned_bgr_112.shape[:2] != (112, 112):
            aligned_bgr_112 = cv2.resize(aligned_bgr_112, (112, 112))

        if aligned_bgr_112.ndim != 3 or aligned_bgr_112.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape (112,112,3), got {aligned_bgr_112.shape}")

        rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def get_embedding_from_aligned(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        """
        Extract embedding from an already aligned 112x112 BGR face crop.
        Use this for assets/faces_aligned_112 images.
        """
        inp = self._preprocess(aligned_bgr_112)
        emb = self.session.run(None, {self.input_name: inp})[0][0].astype(np.float32)
        return self._l2norm(emb)

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray | None = None) -> np.ndarray:
        """
        Compatibility method used by main.py.

        If landmarks are provided, align the raw image first.
        If landmarks are not provided, assume image is already an aligned face crop.
        """
        if image is None:
            raise ValueError("image is None")

        if landmarks is not None:
            aligned_face, _ = face_alignment(image, landmarks)
            return self.get_embedding_from_aligned(aligned_face)

        return self.get_embedding_from_aligned(image)

    def __call__(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        """
        Backward-compatible call style for evaluation scripts.
        """
        return self.get_embedding_from_aligned(aligned_bgr_112)