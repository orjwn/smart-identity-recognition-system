import cv2
import numpy as np
from skimage.transform import SimilarityTransform
import torch
from typing import Tuple
import os


# =============================================================================
# External source attribution
# Source: yakhyo/face-reidentification
# URL: https://github.com/yakhyo/face-reidentification
# Licence: MIT indicated by repository README badge; licence file not found locally
# Status: copied unchanged from external source
# Purpose in this project: ArcFace reference landmark template for alignment.
# Student modification: none to the landmark values; retained for compatibility
# with the original ArcFace alignment implementation.
# Start of externally sourced/adapted block
# =============================================================================
# Reference alignment for facial landmarks (ArcFace)
reference_alignment: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ],
    dtype=np.float32
)
# =============================================================================
# End of externally sourced/adapted block from yakhyo/face-reidentification
# =============================================================================

def draw_corners(img, bbox, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = bbox[:4].astype(int)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)

def draw_keypoints(img, kps, color=(0,0,255), r=2):
    for (x,y) in kps.astype(int):
        cv2.circle(img, (x,y), r, color, -1)


# =============================================================================
# External source attribution
# Source: yakhyo/face-reidentification
# URL: https://github.com/yakhyo/face-reidentification
# Licence: MIT indicated by repository README badge; licence file not found locally
# Status: copied and lightly modified from external source
# Purpose in this project: Estimate the similarity transform used for face
# alignment before embedding extraction.
# Student modification: minor edits for typing, docstrings, comments and reuse
# across multiple SmartIdentity recognizer wrappers and evaluation scripts.
# Start of externally sourced/adapted block
# =============================================================================
def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the normalization transformation matrix for facial landmarks.

    Args:
        landmark (np.ndarray): Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size (int, optional): The size of the output image. Default is 112.

    Returns:
        np.ndarray: The 2x3 transformation matrix for aligning the landmarks.
        np.ndarray: The 2x3 inverse transformation matrix for aligning the landmarks.

    Raises:
        AssertionError: If the input landmark array does not have the shape (5, 2)
                        or if image_size is not a multiple of 112 or 128.
    """
    assert landmark.shape == (5, 2), "Landmark array must have shape (5, 2)."
    assert image_size % 112 == 0 or image_size % 128 == 0, "Image size must be a multiple of 112 or 128."

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    # Adjust reference alignment based on ratio and diff_x
    alignment = reference_alignment * ratio
    alignment[:, 0] += diff_x

    # Compute the transformation matrix
    transform = SimilarityTransform()
    transform.estimate(landmark, alignment)

    matrix = transform.params[0:2, :]
    inverse_matrix = np.linalg.inv(transform.params)[0:2, :]

    return matrix, inverse_matrix


def face_alignment(image: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align the face in the input image based on the given facial landmarks.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        landmark (np.ndarray): Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size (int, optional): The size of the aligned output image. Default is 112.

    Returns:
        np.ndarray: The aligned face as a NumPy array.
        np.ndarray: The 2x3 transformation matrix used for alignment.
    """
    # Get the transformation matrix
    M, M_inv = estimate_norm(landmark, image_size)

    # Warp the input image to align the face
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
    # temporary debug save
   # os.makedirs("debug_faces", exist_ok=True)
   # cv2.imwrite("debug_faces/aligned_test.jpg", warped)
    #print("Saved aligned face to debug_faces/aligned_test.jpg", warped.shape)
    return warped, M_inv
# =============================================================================
# End of externally sourced/adapted block from yakhyo/face-reidentification
# =============================================================================

def to_adaface_input_from_bgr(aligned_bgr: np.ndarray):
    img = aligned_bgr.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


# =============================================================================
# External source attribution
# Source: yakhyo/face-reidentification
# URL: https://github.com/yakhyo/face-reidentification
# Licence: MIT indicated by repository README badge; licence file not found locally
# Status: copied and lightly modified from external source
# Purpose in this project: Decode SCRFD model distance outputs into boxes and
# facial keypoints.
# Student modification: minor edits for local typing/style and reuse through
# the project SCRFD wrapper.
# Start of externally sourced/adapted block
# =============================================================================
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bounding boxes with shape (n, 4).
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded keypoints with shape (n, 2k).
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)
# =============================================================================
# End of externally sourced/adapted block from yakhyo/face-reidentification
# =============================================================================


# =============================================================================
# External source attribution
# Source: yakhyo/face-reidentification
# URL: https://github.com/yakhyo/face-reidentification
# Licence: MIT indicated by repository README badge; licence file not found locally
# Status: copied and lightly modified from external source
# Purpose in this project: Similarity calculation and recognition overlay
# drawing used in the legacy/live demo path.
# Student modification: minor edits for local visualisation while backend/kiosk
# logic adds separate passenger-state integration.
# Start of externally sourced/adapted block
# =============================================================================
def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.float32:
    """Computing Similarity between two faces.

    Args:
        feat1 (np.ndarray): Face features.
        feat2 (np.ndarray): Face features.

    Returns:
        np.float32: Cosine similarity between face features.
    """
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=3, proportion=0.2):
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)

    return image


def draw_bbox_info(frame, bbox, similarity, name, color):
    x1, y1, x2, y2 = map(int, bbox)

    cv2.putText(
        frame,
        f"{name}: {similarity:.2f}",
        org=(x1, y1-10),
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontScale=1,
        color=color,
        thickness=1
    )

    # Draw bounding box
    draw_bbox(frame, bbox, color)

    # Draw similarity bar
    rect_x_start = x2 + 10
    rect_x_end = rect_x_start + 10
    rect_y_end = y2
    rect_height = int(similarity * (y2 - y1))
    rect_y_start = rect_y_end - rect_height  # Rectangle starts from bottom and goes upward

    # Draw the filled rectangle
    cv2.rectangle(frame, (rect_x_start, rect_y_start), (rect_x_end, rect_y_end), color, cv2.FILLED)
# =============================================================================
# End of externally sourced/adapted block from yakhyo/face-reidentification
# =============================================================================
