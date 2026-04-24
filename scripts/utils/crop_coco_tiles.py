"""
Crop bird images from COCO-format tiled dataset.

Reads COCO annotations (bbox as [x, y, width, height]), crops each annotation
with configurable padding, resizes to target size, and saves the result.
Also generates a CSV compatible with the SwinTrainer DataLoader.

Usage:
    python -m scripts.utils.crop_coco_tiles \
        --json /path/to/all_annotations.json \
        --tile-dir /path/to/BirdDataset_2025_nonoverlapping_tiles_500 \
        --output-dir /path/to/cropped-dataset-2025

    # Custom padding and output size
    python -m scripts.utils.crop_coco_tiles --pad-frac 0.3 --size 224
"""

import argparse
import json
import csv
from pathlib import Path
from collections import Counter

import cv2
import numpy as np


def crop_and_resize(
    img: np.ndarray,
    bbox_xywh: list,
    pad_frac: float,
    output_size: int,
) -> np.ndarray:
    """Crop image to COCO bbox with fractional padding, then resize.

    Args:
        img:         HxWxC numpy array (BGR).
        bbox_xywh:   [x, y, width, height] COCO-style bounding box.
        pad_frac:    Fraction of bbox size to add as padding on each side.
        output_size: Square output dimension in pixels.

    Returns:
        Resized crop as a numpy array (BGR).
    """
    img_h, img_w = img.shape[:2]
    x, y, w, h = bbox_xywh

    # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # Compute padding in pixels
    pad_x = w * pad_frac
    pad_y = h * pad_frac

    # Expand bbox with padding, clamped to image bounds
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(img_w, int(x2 + pad_x))
    cy2 = min(img_h, int(y2 + pad_y))

    # Ensure we have a valid crop region
    if cx2 <= cx1 or cy2 <= cy1:
        # Fallback to original bbox without padding
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(img_w, int(x2)), min(img_h, int(y2))

    crop = img[cy1:cy2, cx1:cx2]

    # Handle edge case of empty crop
    if crop.size == 0:
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized


def main():
    parser = argparse.ArgumentParser(description="Crop birds from COCO-format tiled dataset")
    parser.add_argument("--json", type=str, required=True,
                        help="Path to COCO annotations JSON (e.g., all_annotations.json)")
    parser.add_argument("--tile-dir", type=str, required=True,
                        help="Root directory containing tile images (organized by OM_xxx/tiles/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for cropped images and CSV")
    parser.add_argument("--pad-frac", type=float, default=0.25,
                        help="Padding as fraction of bbox size (default: 0.25)")
    parser.add_argument("--size", type=int, default=224,
                        help="Output image size in pixels (default: 224)")
    parser.add_argument("--species", type=str, nargs="+", default=None,
                        help="Only crop these species (e.g., --species ROTEA SATEA)")
    args = parser.parse_args()

    json_path = Path(args.json)
    tile_dir = Path(args.tile_dir)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    print(f"[info] Loading annotations from {json_path}")
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookup tables
    # image_id -> image info
    id_to_image = {img["id"]: img for img in coco["images"]}
    # category_id -> species name
    id_to_category = {cat["id"]: cat["name"] for cat in coco["categories"]}

    print(f"[info] Found {len(coco['images'])} images, {len(coco['annotations'])} annotations, {len(coco['categories'])} categories")

    # Filter by species if requested
    annotations = coco["annotations"]
    if args.species:
        species_set = set(args.species)
        # Get category IDs for the requested species
        valid_cat_ids = {cat_id for cat_id, name in id_to_category.items() if name in species_set}
        annotations = [ann for ann in annotations if ann["category_id"] in valid_cat_ids]
        print(f"[info] Filtered to {len(annotations)} annotations for species: {args.species}")

    # Process annotations
    csv_rows = []
    saved = 0
    skipped = 0
    species_counts = Counter()

    # Cache for loaded images (to avoid reloading same tile multiple times)
    image_cache = {}
    MAX_CACHE_SIZE = 100

    for ann in annotations:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        bbox = ann["bbox"]  # [x, y, width, height]
        ann_id = ann["id"]

        # Get image info
        img_info = id_to_image.get(image_id)
        if img_info is None:
            skipped += 1
            continue

        # Get species name
        species = id_to_category.get(category_id, "UNKNOWN")

        # Build path to tile image
        # file_name is like "OM_001/tiles/OM_001_00000_00000.png"
        tile_path = tile_dir / img_info["file_name"]

        # Load image (with caching)
        if image_id in image_cache:
            img = image_cache[image_id]
        else:
            img = cv2.imread(str(tile_path))
            if img is None:
                if skipped < 5:
                    print(f"[warn] Cannot read: {tile_path}")
                skipped += 1
                continue
            # Add to cache
            if len(image_cache) < MAX_CACHE_SIZE:
                image_cache[image_id] = img
            else:
                # Clear cache if too large
                image_cache.clear()
                image_cache[image_id] = img

        # Crop and resize
        cropped = crop_and_resize(img, bbox, args.pad_frac, args.size)

        # Extract orthomosaic_id from file_name (e.g., "OM_001/tiles/OM_001_00000_00000.png")
        orthomosaic_id = img_info.get("orthomosaic_id") or img_info["file_name"].split("/")[0]

        # Create species subfolder
        species_dir = crops_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename: {species}_{ann_id}.png
        out_filename = f"{species}_{ann_id:06d}.png"
        out_path = species_dir / out_filename

        cv2.imwrite(str(out_path), cropped)
        saved += 1
        species_counts[species] += 1

        # Record for CSV (crop_path relative to crops_dir, includes species subfolder)
        csv_rows.append({
            "crop_path": f"{species}/{out_filename}",
            "species_name": species,
            "source_tile": img_info["file_name"],
            "orthomosaic_id": orthomosaic_id,
            "bbox_coco": bbox,  # [x, y, w, h]
            "annotation_id": ann_id,
        })

        if saved % 1000 == 0:
            print(f"[progress] Saved {saved} crops...")

    # Write CSV
    csv_path = output_dir / "annotations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_path", "species_name", "source_tile", "orthomosaic_id", "bbox_coco", "annotation_id"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Write species distribution summary
    summary_path = output_dir / "species_distribution.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_crops": saved,
            "total_skipped": skipped,
            "species_counts": dict(species_counts.most_common()),
            "num_species": len(species_counts),
        }, f, indent=2)

    print(f"\n[done] Saved {saved} cropped images to {crops_dir}/")
    print(f"[done] Annotations CSV written to {csv_path}")
    print(f"[done] Species distribution written to {summary_path}")
    if skipped:
        print(f"[info] Skipped {skipped} (missing file or invalid)")

    print("\n[info] Species distribution:")
    for species, count in species_counts.most_common():
        print(f"  {species}: {count}")


if __name__ == "__main__":
    main()
