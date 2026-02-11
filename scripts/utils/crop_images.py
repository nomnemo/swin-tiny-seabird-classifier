"""
Crop bird images to their bounding boxes with padding, then resize to 224x224.

Reads bounding boxes from Training-256.json, crops each image with configurable
padding (default 25% of bbox size on each side, clamped to image bounds),
and saves the resized result.

Usage:
    python -m scripts.crop_images

    # Custom padding and output size
    python -m scripts.crop_images --pad-frac 0.3 --size 224

    # Only crop ROTEN and SATEN
    python -m scripts.crop_images --species ROTEN SATEN
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ── defaults ──
JSON_PATH  = Path("data/Training-256.json")
IMAGE_DIR  = Path("data/Images-256")
OUTPUT_DIR = Path("data/Images-256-cropped")
PAD_FRAC   = 0.25   # 25% padding on each side of the bbox
OUTPUT_SIZE = 224


def crop_and_resize(
    img: np.ndarray,
    bbox: list,
    pad_frac: float,
    output_size: int,
) -> np.ndarray:
    """Crop image to bbox with fractional padding, then resize.

    Args:
        img:         HxWxC numpy array (BGR).
        bbox:        [x1, y1, x2, y2] bounding box coordinates.
        pad_frac:    Fraction of bbox size to add as padding on each side.
        output_size: Square output dimension in pixels.

    Returns:
        Resized crop as a numpy array (BGR).
    """
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    # Compute padding in pixels
    pad_x = bw * pad_frac
    pad_y = bh * pad_frac

    # Expand bbox with padding, clamped to image bounds
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(img_w, int(x2 + pad_x))
    cy2 = min(img_h, int(y2 + pad_y))

    crop = img[cy1:cy2, cx1:cx2]
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized


def main():
    parser = argparse.ArgumentParser(description="Crop bird images to bbox + padding, resize")
    parser.add_argument("--json", type=str, default=str(JSON_PATH),
                        help="Path to Training-256.json")
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR),
                        help="Input image directory")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for cropped images")
    parser.add_argument("--pad-frac", type=float, default=PAD_FRAC,
                        help="Padding as fraction of bbox size (default: 0.25)")
    parser.add_argument("--size", type=int, default=OUTPUT_SIZE,
                        help="Output image size in pixels (default: 224)")
    parser.add_argument("--species", type=str, nargs="+", default=None,
                        help="Only crop these species (e.g. --species ROTEN SATEN)")
    args = parser.parse_args()

    json_path = Path(args.json)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    images = raw["images"]
    print(f"[info] loaded {len(images)} entries from {json_path}")

    # Filter by species if requested
    if args.species:
        species_set = set(args.species)
        images = [
            img for img in images
            if img.get("detections")
            and img["detections"][0].get("tcws_species") in species_set
        ]
        print(f"[info] filtered to {len(images)} images for species: {args.species}")

    saved = 0
    skipped = 0

    for entry in images:
        fname = entry["file_name"]
        dets = entry.get("detections", [])
        if not dets:
            skipped += 1
            continue

        bbox = dets[0]["bbox"]
        src_path = image_dir / fname

        img = cv2.imread(str(src_path))
        if img is None:
            if skipped < 5:
                print(f"[warn] cannot read: {src_path}")
            skipped += 1
            continue

        cropped = crop_and_resize(img, bbox, args.pad_frac, args.size)
        cv2.imwrite(str(output_dir / fname), cropped)
        saved += 1

    print(f"[done] saved {saved} cropped images to {output_dir}/")
    if skipped:
        print(f"[info] skipped {skipped} (missing file or no detections)")


if __name__ == "__main__":
    main()
