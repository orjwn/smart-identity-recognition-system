from pathlib import Path
import random
import cv2
import numpy as np

# Update these paths for your project
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = Path("datasets/test")
OUTPUT_DIR = Path("datasets/test_masked_fixed")
MASK_PATH = SCRIPT_DIR / "assets" / "mask_img.png"

# Standard ArcFace aligned 112x112 landmarks
ARCFACE_LMK = np.array([
    [30.2946, 51.6963],  # left eye
    [65.5318, 51.5014],  # right eye
    [48.0252, 71.7366],  # nose
    [33.5493, 92.3655],  # left mouth
    [62.7299, 92.2041],  # right mouth
], dtype=np.float32)

# =============================================================================
# External source attribution
# Source: fdbtrs/Masked-Face-Recognition-KD
# URL: https://github.com/fdbtrs/Masked-Face-Recognition-KD
# Licence: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# Status: adapted from external source
# Purpose in this project: Reproduce the MaskInv synthetic-mask landmark
# mapping for offline masked-face evaluation images.
# Student modification: substantially modified for SmartIdentity project
# requirements, including fixed ArcFace-aligned 112x112 inputs, local mask
# tinting and recursive dataset tree processing.
# Start of externally sourced/adapted block
# =============================================================================
# Points from the external Masked-Face-Recognition-KD mask template.
SRC_PTS = np.array([
    [678.0, 464.0],
    [548.0, 614.0],
    [991.0, 664.0],
    [1009.0,  64.0],
    [557.0,   64.0],
], dtype=np.float32)

# Exact point order used in the repo's simulatedMask()
# dst_pts = [nose, mouth_left, mouth_right, eye_right, eye_left]
# where repo maps them as [2, 4, 3, 0, 1] on ArcFace landmarks.
DST_PTS_MASKINV = np.array([
    ARCFACE_LMK[2],  # nose
    ARCFACE_LMK[4],  # exact repo mapping
    ARCFACE_LMK[3],  # exact repo mapping
    ARCFACE_LMK[0],  # exact repo mapping
    ARCFACE_LMK[1],  # exact repo mapping
], dtype=np.float32)
# =============================================================================
# End of externally sourced/adapted block from fdbtrs/Masked-Face-Recognition-KD
# =============================================================================

MASK_COLORS = [
    (80, 80, 80),
    (40, 70, 120),
    (20, 120, 80),
    (120, 90, 60),
    (100, 100, 100),
]


def tint_mask_rgba(mask_rgba: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    out = mask_rgba.copy().astype(np.float32)
    alpha = out[:, :, 3:4] / 255.0
    base = out[:, :, :3]
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out[:, :, :3] = 0.35 * base + 0.65 * color
    out[:, :, :3] = np.clip(out[:, :, :3], 0, 255)
    out[:, :, 3:4] = alpha * 255.0
    return out.astype(np.uint8)


def apply_maskinv_style_aligned(img_bgr: np.ndarray, mask_rgba: np.ndarray) -> np.ndarray:
    """
    Apply the MaskInv synthetic mask to an already ArcFace-aligned 112x112 image.
    Do NOT use plain resize on arbitrary face crops.
    """
    h, w = img_bgr.shape[:2]
    if (w, h) != (112, 112):
        raise ValueError(
            f"Expected an ArcFace-aligned 112x112 image, got {(w, h)}. "
            "Align the face first instead of resizing it."
        )

    M, _ = cv2.findHomography(SRC_PTS, DST_PTS_MASKINV, method=0)
    if M is None:
        raise RuntimeError("findHomography failed")

    warped = cv2.warpPerspective(
        mask_rgba,
        M,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    out = img_bgr.astype(np.float32) * (1.0 - alpha) + warped[:, :, :3].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def process_tree():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_rgba = cv2.imread(str(MASK_PATH), cv2.IMREAD_UNCHANGED)
    if mask_rgba is None:
        raise FileNotFoundError(f"Could not read mask template: {MASK_PATH}")
    if mask_rgba.ndim != 3 or mask_rgba.shape[2] != 4:
        raise ValueError("mask_img.png must be RGBA")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for src in INPUT_DIR.rglob("*"):
        rel = src.relative_to(INPUT_DIR)
        dst = OUTPUT_DIR / rel

        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if src.suffix.lower() not in image_exts:
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping unreadable file: {src}")
            continue

        color = random.choice(MASK_COLORS)
        tinted_mask = tint_mask_rgba(mask_rgba, color)

        try:
            masked = apply_maskinv_style_aligned(img, tinted_mask)
        except Exception as e:
            print(f"Skipping {src}: {e}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), masked)
        print(f"Saved: {dst}")


if __name__ == "__main__":
    process_tree()
