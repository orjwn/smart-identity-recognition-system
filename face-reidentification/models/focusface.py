import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

FOCUSFACE_DIR = Path(__file__).resolve().parents[1] / "external" / "FocusFace"
if str(FOCUSFACE_DIR) not in sys.path:
    sys.path.insert(0, str(FOCUSFACE_DIR))

from model import FocusFace as FocusFaceNet


class FocusFaceEncoder:
    """Wrapper around the external FocusFace repository used for masked routing.

    The external repository remains in `external/FocusFace`; this class provides
    the stable project-facing API used by the backend and evaluation scripts.
    """

    def __init__(self, model_path: str, device: str = None, identities: int = 85742):
        self.model_path = str(model_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = FocusFaceNet(identities=identities)
        state = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return x
        return x / norm

    def get_embedding_from_pil(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, emb, _, _ = self.model(x, inference=True)
        emb = emb[0].detach().cpu().numpy().astype(np.float32)
        emb = self._l2_normalize(emb)
        return emb

    def get_embedding_from_path(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        return self.get_embedding_from_pil(img)
