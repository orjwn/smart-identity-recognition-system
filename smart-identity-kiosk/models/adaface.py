"""
Project wrapper for AdaFace.

Status: project wrapper/integration code.
External dependency/reference: mk-minchul/AdaFace.
Student contribution: connects the external AdaFace model/reference
implementation to this project's aligned-face, embedding, FAISS and evaluation
pipeline.

External reference:
- https://github.com/mk-minchul/AdaFace

See THIRD_PARTY_ATTRIBUTION.md for full source and licence notes.
"""

import os
import sys
from logging import getLogger

import numpy as np
import torch

__all__ = ["AdaFace"]

logger = getLogger(__name__)


class AdaFace:
    """
    AdaFace R18 recognizer wrapper.

    Expects an already aligned 112x112 BGR face crop from the existing
    SmartIdentity pipeline.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.model = self._load_model()

    # =========================================================================
    # External source attribution
    # Source: mk-minchul/AdaFace
    # URL: https://github.com/mk-minchul/AdaFace
    # Licence: MIT
    # Status: project wrapper/integration code
    # External dependency/reference: mk-minchul/AdaFace
    # Purpose in this project: Load AdaFace IR-18 backbone and checkpoint weights.
    # Student contribution: connects the external model builder/checkpoint format
    # to SmartIdentity's local repository path, wrapper class and runtime device.
    # Start of externally sourced/adapted block
    # =========================================================================
    def _load_model(self):
        """
        Load AdaFace backbone + checkpoint.
        """
        # AdaFace is kept as an external reference repository. Only its network
        # definition is imported here; the kiosk wrapper owns preprocessing and
        # embedding normalization for this project.
        adaface_repo_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "external", "adaface-test", "AdaFace")
        )

        if adaface_repo_dir not in sys.path:
            sys.path.append(adaface_repo_dir)

        import net  # imported after sys.path update

        model = net.build_model("ir_18")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        model_state_dict = {
            key[6:]: val for key, val in state_dict.items() if key.startswith("model.")
        }

        model.load_state_dict(model_state_dict)
        model.eval()
        model.to(self.device)

        logger.info("Loaded AdaFace model from %s", self.model_path)
        return model
    # =========================================================================
    # End of externally sourced/adapted block from mk-minchul/AdaFace
    # =========================================================================

    # =========================================================================
    # External source attribution
    # Source: mk-minchul/AdaFace
    # URL: https://github.com/mk-minchul/AdaFace
    # Licence: MIT
    # Status: copied and lightly modified from external source
    # Purpose in this project: Convert aligned 112x112 face crops to AdaFace input.
    # Student modification: adapted the AdaFace input conversion from PIL/RGB to
    # aligned BGR crops produced by SmartIdentity's alignment pipeline.
    # Start of externally sourced/adapted block
    # =========================================================================
    @staticmethod
    def _to_input(aligned_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert aligned BGR uint8 image of shape (112,112,3) into AdaFace input tensor.
        """
        if aligned_bgr is None:
            raise ValueError("aligned_bgr is None")

        if aligned_bgr.shape != (112, 112, 3):
            raise ValueError(f"Expected aligned face shape (112,112,3), got {aligned_bgr.shape}")

        img = aligned_bgr.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor
    # =========================================================================
    # End of externally sourced/adapted block from mk-minchul/AdaFace
    # =========================================================================

    def __call__(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Generate a normalized 512-d embedding from an aligned 112x112 BGR face crop.
        """
        tensor = self._to_input(aligned_bgr).to(self.device)

        with torch.no_grad():
            feature, _ = self.model(tensor)

        embedding = feature.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
