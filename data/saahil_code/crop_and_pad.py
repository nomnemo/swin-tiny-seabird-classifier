import os
import json
import cv2
import re

# CONFIGURATION
json_path = r"C:/Users/Audub/Training Image Sets/20251020 512 Species/Training.json"
images_root = r"C:/Users/Audub/Training Image Sets/20251020 512 Species/Images"
output_root = r"C:/Users/Audub/saahil_classification/data/crops"
metadata_out = r"C:/Users/Audub/saahil_classification/data/metadata_full.json"

# HELPER FUNCTIONS 

def load_categories_from_malformed_json(json_path):
    """Extract all (id, name) pairs from the malformed Training.json."""
    with open(json_path, "r") as f:
        text = f.read()

    pattern = r'"categories"\s*:\s*\[\s*\{\s*"id"\s*:\s*(\d+)\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}\s*\]'
    matches = re.findall(pattern, text)
    if not matches:
        raise ValueError("No categories found in JSON. Check file structure.")

    id_to_species = {int(cat_id): cat_name for cat_id, cat_name in matches}
    return id_to_species


def resize_and_pad_to_224(image):
    """Resize keeping aspect ratio, pad to 224x224 using ImageNet mean color."""
    h, w = image.shape[:2]
    scale = 224 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = 224 - new_w
    pad_h = 224 - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    # ImageNet mean in BGR
    mean_bgr = [103.53, 116.28, 123.675]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=mean_bgr)
    return padded


# MAIN SCRIPT

def main():
    # Step 1: Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    id_to_species = load_categories_from_malformed_json(json_path)

    id_to_filename = {img["id"]: img["file_name"] for img in images}

    # Step 2: Select first 2 images
    sample_images = images
    metadata_records = []

    for img in sample_images:
        img_id = img["id"]
        img_path = os.path.join(images_root, os.path.basename(img["file_name"]))
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping {img_path} (not found or unreadable)")
            continue

        H, W = image.shape[:2]
        anns_for_img = [ann for ann in annotations if ann["image_id"] == img_id]
        print(f"Processing {img['file_name']} with {len(anns_for_img)} detections")

        for ann in anns_for_img:
            ann_id = ann["id"]
            cat_id = ann["category_id"]
            bbox = ann["bbox"]  # [x, y, w, h]
            species = id_to_species.get(cat_id, "UNKNOWN")

            # Add 10% margin around bbox 
            x, y, w, h = map(int, bbox)
            margin_x = int(0.1 * w)
            margin_y = int(0.1 * h)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(W, x + w + margin_x)
            y2 = min(H, y + h + margin_y)

            expanded_bbox = [x1, y1, x2 - x1, y2 - y1]

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"Empty crop for annotation {ann_id} in {img['file_name']}")
                continue

            # Resize & pad
            crop_padded = resize_and_pad_to_224(crop)

            # Save to species folder
            species_dir = os.path.join(output_root, species)
            os.makedirs(species_dir, exist_ok=True)
            save_name = f"{img_id}_{ann_id}.jpg"
            save_path = os.path.join(species_dir, save_name)
            cv2.imwrite(save_path, crop_padded)

            # Record metadata
            record = {
                "crop_path": os.path.relpath(save_path, start=os.path.dirname(output_root)),
                "species_name": species,
                "species_id": cat_id,
                "source_image": img["file_name"],
                "bbox_original": bbox,
                "bbox_expanded": expanded_bbox
            }
            metadata_records.append(record)

    # Step 4: Save metadata JSON
    os.makedirs(os.path.dirname(metadata_out), exist_ok=True)
    with open(metadata_out, "w") as f:
        json.dump(metadata_records, f, indent=4)

    print(f"\nDone! Processed {len(metadata_records)} crops.")
    print(f"Metadata saved to: {metadata_out}")


if __name__ == "__main__":
    main()
