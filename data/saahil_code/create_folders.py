import os
import json
import re

json_path = r"C:/Users/Audub/Training Image Sets/20251020 512 Species/Training.json"
output_root = r"C:/Users/Audub/saahil_classification/data/crops"

def create_species_folders_from_malformed(json_path, output_root):
    # Read file as raw text
    with open(json_path, "r") as f:
        text = f.read()

    # Find every {"id": X, "name": "YYY"} inside any "categories": [...] block
    pattern = r'"categories"\s*:\s*\[\s*\{\s*"id"\s*:\s*(\d+)\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}\s*\]'
    matches = re.findall(pattern, text)

    if not matches:
        raise ValueError("No categories found. Check JSON structure.")

    id_to_name = {int(cat_id): cat_name for cat_id, cat_name in matches}

    # Make root and subfolders
    os.makedirs(output_root, exist_ok=True)
    for name in sorted(id_to_name.values()):
        os.makedirs(os.path.join(output_root, name), exist_ok=True)

    print(f"Created {len(id_to_name)} folders under:\n   {output_root}")
    print("\nSpecies folders:")
    for name in sorted(id_to_name.values()):
        print("  -", name)

    return id_to_name


if __name__ == "__main__":
    create_species_folders_from_malformed(json_path, output_root)
