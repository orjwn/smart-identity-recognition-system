import os
import shutil
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split aligned_112 dataset into gallery and test folders"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to aligned_112 dataset root"
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        required=True,
        help="Output path for gallery/enrollment split"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Output path for test/query split"
    )
    parser.add_argument(
        "--gallery-per-id",
        type=int,
        default=2,
        help="Number of images per identity to place in gallery"
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=3,
        help="Minimum images required for an identity to be included"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of move"
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_image_files(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(exts)
    )


def copy_or_move(src: str, dst: str, do_copy: bool):
    ensure_dir(os.path.dirname(dst))
    if do_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def main():
    args = parse_args()
    random.seed(args.seed)

    input_dir = os.path.abspath(args.input_dir)
    gallery_dir = os.path.abspath(args.gallery_dir)
    test_dir = os.path.abspath(args.test_dir)

    ensure_dir(gallery_dir)
    ensure_dir(test_dir)

    identities = [
        d for d in sorted(os.listdir(input_dir))
        if os.path.isdir(os.path.join(input_dir, d))
    ]

    included_ids = 0
    skipped_ids = 0
    gallery_count = 0
    test_count = 0

    for identity in identities:
        person_dir = os.path.join(input_dir, identity)
        images = list_image_files(person_dir)

        if len(images) < args.min_images:
            print(f"Skipping {identity}: only {len(images)} image(s)")
            skipped_ids += 1
            continue

        shuffled = images[:]
        random.shuffle(shuffled)

        gallery_imgs = shuffled[:args.gallery_per_id]
        test_imgs = shuffled[args.gallery_per_id:]

        if len(gallery_imgs) < args.gallery_per_id or len(test_imgs) == 0:
            print(f"Skipping {identity}: not enough images after split")
            skipped_ids += 1
            continue

        for img in gallery_imgs:
            src = os.path.join(person_dir, img)
            dst = os.path.join(gallery_dir, identity, img)
            copy_or_move(src, dst, args.copy)
            gallery_count += 1

        for img in test_imgs:
            src = os.path.join(person_dir, img)
            dst = os.path.join(test_dir, identity, img)
            copy_or_move(src, dst, args.copy)
            test_count += 1

        included_ids += 1
        print(
            f"{identity}: gallery={len(gallery_imgs)} | test={len(test_imgs)}"
        )

    print("\nDone.")
    print(f"Included identities: {included_ids}")
    print(f"Skipped identities:  {skipped_ids}")
    print(f"Gallery images:      {gallery_count}")
    print(f"Test images:         {test_count}")
    print(f"Gallery folder:      {gallery_dir}")
    print(f"Test folder:         {test_dir}")


if __name__ == "__main__":
    main()
