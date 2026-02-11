"""
Prepare ROTEN vs SATEN binary classification data from Training-256.json.

Reads Hank's masked 256x256 bird images, filters to Royal Terns (ROTEN)
and Sandwich Terns (SATEN), downsamples ROTEN to match SATEN (635 each),
and creates train/val/test CSVs with stratified splitting (80/10/10).

Note: Only 4 unique parent aerial photos exist in this dataset, so
GroupShuffleSplit is not feasible. We use StratifiedShuffleSplit instead,
which preserves class balance across splits.

Usage:
    python -m scripts.prepare_roten_saten
"""

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from scripts.DataSplitter import _write_csv_from_list_of_dicts

# ── config ──
JSON_PATH = Path("data/Training-256.json")
OUT_DIR = Path("data/roten_saten")
SEED = 42
TARGET_SPECIES = {"ROTEN", "SATEN"}


def _get_parent_group(file_name: str) -> str:
    """Extract parent group from filename.

    Example: '20250510 10k-03-11 000001.jpg' -> '20250510 10k-03-11'
    """
    m = re.match(r"(.+)\s+\d+\.jpg$", file_name, re.IGNORECASE)
    return m.group(1) if m else Path(file_name).stem


def main() -> None:
    # ── 1. Load and filter ──
    raw = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    images = raw["images"]
    print(f"[info] loaded {len(images)} images from {JSON_PATH}")

    records = []
    for img in images:
        dets = img.get("detections", [])
        if not dets:
            continue
        species = dets[0].get("tcws_species", "")
        if species in TARGET_SPECIES:
            records.append({
                "crop_path": img["file_name"],
                "species_name": species,
                "source_image": img["file_name"],
            })

    df = pd.DataFrame(records)
    print(f"[info] filtered to {len(df)} ROTEN/SATEN images")
    print(f"       {dict(Counter(df['species_name']))}")

    # ── 2. Downsample ROTEN to match SATEN ──
    n_saten = len(df[df["species_name"] == "SATEN"])
    roten_mask = df["species_name"] == "ROTEN"
    roten_down = df[roten_mask].sample(n=n_saten, random_state=SEED)
    df = pd.concat([roten_down, df[~roten_mask]], ignore_index=True)
    print(f"[info] after downsampling ROTEN -> {n_saten}: {len(df)} total")
    print(f"       {dict(Counter(df['species_name']))}")

    # ── 3. Parent group info (for reference) ──
    parent_groups = np.array([_get_parent_group(fn) for fn in df["crop_path"]])
    n_groups = len(set(parent_groups))
    print(f"[info] unique parent groups: {n_groups}")
    print(f"[info] using StratifiedShuffleSplit (too few groups for GroupShuffleSplit)")

    # ── 4. StratifiedShuffleSplit 80/10/10 ──
    labels = df["species_name"].values

    # First split: 80% train, 20% holdout
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, hold_idx = next(sss1.split(df, labels))

    # Second split: 50/50 on holdout → 10% val, 10% test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED + 1)
    val_rel, test_rel = next(sss2.split(hold_idx, labels[hold_idx]))
    val_idx = hold_idx[val_rel]
    test_idx = hold_idx[test_rel]

    train_rows = df.iloc[train_idx].to_dict(orient="records")
    val_rows = df.iloc[val_idx].to_dict(orient="records")
    test_rows = df.iloc[test_idx].to_dict(orient="records")

    # ── 5. Write CSVs ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv_from_list_of_dicts(OUT_DIR / "split_train.csv", train_rows)
    _write_csv_from_list_of_dicts(OUT_DIR / "split_val.csv", val_rows)
    _write_csv_from_list_of_dicts(OUT_DIR / "split_test.csv", test_rows)

    # ── 6. Summary ──
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        cnt = Counter(r["species_name"] for r in rows)
        parents = set(_get_parent_group(r["crop_path"]) for r in rows)
        print(f"[{name:5s}] n={len(rows):4d} | parents={len(parents)} | {dict(cnt)}")

    print(f"\n[done] CSVs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
