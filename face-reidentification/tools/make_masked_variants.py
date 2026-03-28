import os, sys, json, random
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def apply_rect_mask(img_bgr: np.ndarray, frac: float = 0.45) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    y0 = int(h * (1.0 - frac))
    out = img_bgr.copy()
    color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))  # dark BGR
    cv2.rectangle(out, (0, y0), (w, h), color, thickness=-1)
    return out

def _cymk_to_rgb(mask_cymk_0to100: np.ndarray) -> np.ndarray:
    """
    Repo-style conversion: treat 4 channels as C,M,Y,K in [0..100] and convert to RGB [0..255].
    """
    cyan = mask_cymk_0to100[:, :, 0]
    magenta = mask_cymk_0to100[:, :, 1]
    yellow = mask_cymk_0to100[:, :, 2]
    black = mask_cymk_0to100[:, :, 3]
    scale = 100.0

    red = 255.0 * (1.0 - (cyan + black) / scale)
    green = 255.0 * (1.0 - (magenta + black) / scale)
    blue = 255.0 * (1.0 - (yellow + black) / scale)

    rgb = np.stack((red, green, blue), axis=2)
    return np.clip(rgb, 0, 255).astype(np.float32)


def apply_maskinv_homography_mask(aligned_bgr_112: np.ndarray, mask_img_rgba: np.ndarray) -> np.ndarray:
    """
    MaskInv repo-style synthetic mask application using homography warp + alpha blending.
    Works best on 112x112 aligned faces.

    aligned_bgr_112: (112,112,3) uint8
    mask_img_rgba: mask_img.png loaded with cv2.IMREAD_UNCHANGED (H,W,4)
    """
    img = aligned_bgr_112.copy()
    h, w = img.shape[:2]
    if (h, w) != (112, 112):
        img = cv2.resize(img, (112, 112))

    # These are the 5 canonical landmark locations they use in the repo for aligned faces
    # (note: values differ slightly from the common ArcFace template; we match the repo).
    landmarks = np.array(
        [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    # Repo mapping (keep order consistent with their implementation)
    nose = (landmarks[2][0], landmarks[2][1])
    mouth_left = (landmarks[4][0], landmarks[4][1])
    mouth_right = (landmarks[3][0], landmarks[3][1])
    eye_left = (landmarks[1][0], landmarks[1][1])
    eye_right = (landmarks[0][0], landmarks[0][1])

    # Random shifts like the repo (small in template coordinate space)
    rs = np.random.randint(-40, 40)
    rx = np.random.randint(-10, 10)

    # Template keypoints (in mask image coordinate space) — from repo
    src_pts = np.array(
        [
            [678 + rx, 464 + rs],
            [548 + rx, 614 + rs],
            [991 + rx, 664 + rs],
            [1009 + rx, 64 + rs],
            [557 + rx, 64 + rs],
        ],
        dtype=np.float32,
    )

    # Destination points (in aligned face coordinate space)
    dst_pts = np.array(
        [
            [int(nose[0]), int(nose[1])],
            [int(mouth_left[0]), int(mouth_left[1])],
            [int(mouth_right[0]), int(mouth_right[1])],
            [int(eye_right[0]), int(eye_right[1])],
            [int(eye_left[0]), int(eye_left[1])],
        ],
        dtype=np.float32,
    )

    M, _ = cv2.findHomography(src_pts, dst_pts)
    if M is None:
        return img

    # Warp the RGBA mask template into the aligned face frame
    warped = cv2.warpPerspective(
        mask_img_rgba, M, (112, 112), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
    )

    # Alpha blending like repo
    alpha_mask = warped[:, :, 3].astype(np.float32) / 255.0
    alpha_img = np.abs(1.0 - alpha_mask)

    # Repo treats warped mask channels as CMYK-like then converts to RGB + adds random color
    warped_cymk = (warped.astype(np.float32) / 255.0) * 100.0  # [0..100]
    warped_rgb = _cymk_to_rgb(warped_cymk)  # RGB float32 [0..255]

    # Add random color jitter (repo adds random_value)
    random_value = np.random.randint(0, 150, (1, 1, 3)).astype(np.float32)
    warped_rgb = np.clip(warped_rgb + random_value, 0, 255)

    # Convert warped_rgb (RGB) -> BGR for OpenCV blending
    warped_bgr = warped_rgb[:, :, ::-1]

    # Blend per channel
    for c in range(3):
        img[:, :, c] = (alpha_mask * warped_bgr[:, :, c] + alpha_img * img[:, :, c]).astype(
            np.uint8
        )

    return img

def rewrite_split_paths(split_obj: dict, old_sub: str, new_sub: str) -> dict:
    """Replace 'aligned_112' with 'aligned_112_mask_xxx' inside split paths."""
    out = {"seed": split_obj.get("seed"), "enroll": {}, "probe": {}}
    for k, v in split_obj["enroll"].items():
        out["enroll"][k] = [p.replace(old_sub, new_sub) for p in v]
    for k, v in split_obj["probe"].items():
        out["probe"][k] = [p.replace(old_sub, new_sub) for p in v]
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="processed dataset root (contains aligned_112/ and splits/)")
    ap.add_argument("--split", default="split_seed42.json", help="split file inside splits/")
    ap.add_argument("--variant", choices=["rect", "template"], required=True)
    ap.add_argument("--out-aligned", required=True, help="output aligned folder name, e.g. aligned_112_mask_rect")
    ap.add_argument("--template-path", default=None, help="RGBA mask template path (png with alpha) for template variant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rect-frac", type=float, default=0.45)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = Path(args.dataset)
    in_aligned = ds / "aligned_112"
    if not in_aligned.exists():
        raise FileNotFoundError(f"Missing {in_aligned}")

    splits_dir = ds / "splits"
    split_path = splits_dir / args.split
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    out_aligned = ds / args.out_aligned
    out_aligned.mkdir(parents=True, exist_ok=True)

    # load template if needed
    template_rgba = None
    if args.variant == "template":
        if not args.template_path:
            raise ValueError("--template-path is required when --variant template")
        tp = Path(args.template_path)
        if not tp.exists():
            raise FileNotFoundError(f"Template not found: {tp}")
        template_rgba = cv2.imread(str(tp), cv2.IMREAD_UNCHANGED)
        if template_rgba is None or template_rgba.shape[2] != 4:
            raise ValueError("Template must be a PNG with alpha (RGBA).")

    # process all aligned images by mirroring folder structure
    # aligned_112/<id>/*.jpg → out_aligned/<id>/*.jpg
    id_dirs = [p for p in in_aligned.iterdir() if p.is_dir()]
    total = 0
    for id_dir in id_dirs:
        out_id_dir = out_aligned / id_dir.name
        out_id_dir.mkdir(parents=True, exist_ok=True)

        for img_path in id_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXT:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if img.shape[0] != 112 or img.shape[1] != 112:
                img = cv2.resize(img, (112, 112))

            if args.variant == "rect":
                masked = apply_rect_mask(img, frac=args.rect_frac)
            else:
                masked = apply_maskinv_homography_mask(img, template_rgba)

            cv2.imwrite(str(out_id_dir / img_path.name), masked)
            total += 1

    # create a new split file pointing to the new aligned folder
    split_obj = json.loads(split_path.read_text(encoding="utf-8"))
    new_split = rewrite_split_paths(split_obj, "aligned_112", args.out_aligned)

    out_split_name = args.split.replace(".json", f"_{args.variant}.json")
    out_split_path = splits_dir / out_split_name
    out_split_path.write_text(json.dumps(new_split, indent=2), encoding="utf-8")

    print(f"Done. Wrote {total} masked images to: {out_aligned}")
    print(f"New split saved: {out_split_path}")

if __name__ == "__main__":
    main()