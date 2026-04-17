"""
Create train/val/test splits for the 2025 cropped bird dataset.

Splits are done at the ORTHOMOSAIC level to avoid data leakage
(birds at tile boundaries may appear in multiple tiles from the same orthomosaic).

Supports multiple experiment configurations with label remapping:
- exp1_11class: Classes with >=50 samples kept, rare classes -> OTHERS
- exp2_10class_terns: Same as exp1 but terns merged into TERNS
- exp3_hank_coarse: Hank's 8-class coarse grouping

Usage:
    python scripts/utils/create_splits_2025.py \
        --annotations /path/to/annotations.csv \
        --output-dir /path/to/splits \
        --experiment exp1_11class
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Minimum samples to keep a class (for exp1 and exp2)
MIN_SAMPLES_THRESHOLD = 50

# Experiment 1: 11-class baseline
# Keep all classes with >= 50 samples, merge rest into OTHERS
EXP1_CONFIG = {
    "name": "exp1_11class",
    "description": "11-class baseline: classes with >=50 samples kept, rare classes merged to OTHERS",
    "merge_groups": None,  # Will be computed dynamically based on counts
    "min_samples": MIN_SAMPLES_THRESHOLD,
}

# Experiment 2: 10-class tern-merged
# Same as exp1 but merge all tern-related classes
EXP2_CONFIG = {
    "name": "exp2_10class_terns",
    "description": "10-class: terns merged into TERNS, rare classes merged to OTHERS",
    "tern_merge": ["ROTEA", "ROTEF", "SATEA", "SATEF", "MTRNS"],
    "tern_target": "TERNS",
    "min_samples": MIN_SAMPLES_THRESHOLD,
}

# Experiment 3: Hank coarse grouping (8 classes)
EXP3_CONFIG = {
    "name": "exp3_hank_coarse",
    "description": "Hank's 8-class coarse grouping",
    "explicit_mapping": {
        # Target class -> list of source classes
        "PELICAN_ADULT": ["BRPEA", "BRPEF"],
        "PELICAN_CHICK": ["BRPEC"],
        "TERNS": ["ROTEA", "ROTEF", "SATEA", "SATEF", "MTRNS", "CATEA"],
        "LAUGHING_GULL": ["LAGUA", "LAGUF"],
        "LARGE_WHITE_BIRDS": [
            "GREGA", "GREGC", "GREGF", "SNEGA", "WHIBA", "WHIBF",
            "REEGA", "REEGF", "REEGWMA", "LWBBA", "LWBBC"  # LWBBC if present
        ],
        "GBHE_ADULT": ["GBHEA"],
        "GBHE_CHICK": ["GBHEC"],
        # OTHERS: everything else
    },
    "others_class": "OTHERS",
}

EXPERIMENT_CONFIGS = {
    "exp1_11class": EXP1_CONFIG,
    "exp2_10class_terns": EXP2_CONFIG,
    "exp3_hank_coarse": EXP3_CONFIG,
}


def create_orthomosaic_splits(
    orthomosaics: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Split orthomosaics into train/val/test sets.

    Args:
        orthomosaics: List of orthomosaic IDs (e.g., ["OM_001", "OM_002", ...])
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test = 1 - train - val)
        seed: Random seed for reproducibility

    Returns:
        Dict with keys "train", "val", "test" mapping to lists of orthomosaic IDs
    """
    random.seed(seed)
    oms = sorted(list(set(orthomosaics)))
    random.shuffle(oms)

    n = len(oms)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": oms[:n_train],
        "val": oms[n_train:n_train + n_val],
        "test": oms[n_train + n_val:],
    }

    return splits


def compute_label_mapping_exp1(species_counts: Counter, min_samples: int) -> Dict[str, str]:
    """
    Compute label mapping for Experiment 1 (11-class baseline).
    Classes with >= min_samples kept, others -> OTHERS.
    """
    mapping = {}
    for species, count in species_counts.items():
        if count >= min_samples:
            mapping[species] = species
        else:
            mapping[species] = "OTHERS"
    return mapping


def compute_label_mapping_exp2(
    species_counts: Counter,
    min_samples: int,
    tern_classes: List[str],
    tern_target: str
) -> Dict[str, str]:
    """
    Compute label mapping for Experiment 2 (10-class tern-merged).
    Terns merged first, then rare classes -> OTHERS.
    """
    # First, merge terns
    merged_counts = Counter()
    for species, count in species_counts.items():
        if species in tern_classes:
            merged_counts[tern_target] += count
        else:
            merged_counts[species] += count

    # Now apply min_samples threshold
    mapping = {}
    for species in species_counts.keys():
        if species in tern_classes:
            # Terns always go to TERNS (which has many samples)
            mapping[species] = tern_target
        elif merged_counts.get(species, 0) >= min_samples:
            mapping[species] = species
        else:
            mapping[species] = "OTHERS"

    return mapping


def compute_label_mapping_exp3(
    all_species: Set[str],
    explicit_mapping: Dict[str, List[str]],
    others_class: str
) -> Dict[str, str]:
    """
    Compute label mapping for Experiment 3 (Hank coarse grouping).
    Uses explicit mapping, everything else -> OTHERS.
    """
    # Invert the mapping: source -> target
    mapping = {}
    mapped_sources = set()

    for target, sources in explicit_mapping.items():
        for src in sources:
            mapping[src] = target
            mapped_sources.add(src)

    # Everything not explicitly mapped -> OTHERS
    for species in all_species:
        if species not in mapped_sources:
            mapping[species] = others_class

    return mapping


def apply_label_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Apply label mapping to create remapped_label column."""
    df = df.copy()
    df["remapped_label"] = df["species_name"].map(mapping)
    # Handle any unmapped species (shouldn't happen, but safety)
    df["remapped_label"] = df["remapped_label"].fillna("OTHERS")
    return df


def generate_report(
    df: pd.DataFrame,
    splits: Dict[str, List[str]],
    mapping: Dict[str, str],
    experiment_name: str
) -> str:
    """Generate a text report of the split and class distribution."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"SPLIT REPORT: {experiment_name}")
    lines.append("=" * 70)

    # Orthomosaic split info
    lines.append("\n## Orthomosaic Split")
    for split_name, oms in splits.items():
        lines.append(f"  {split_name}: {len(oms)} orthomosaics - {sorted(oms)}")

    # Label mapping
    lines.append("\n## Label Mapping (original -> remapped)")
    unique_remapped = sorted(set(mapping.values()))
    for target in unique_remapped:
        sources = sorted([k for k, v in mapping.items() if v == target])
        lines.append(f"  {target} <- {sources}")

    # Overall class distribution (after remapping)
    lines.append("\n## Overall Class Distribution (after remapping)")
    overall_counts = df["remapped_label"].value_counts().sort_values(ascending=False)
    for cls, count in overall_counts.items():
        lines.append(f"  {cls}: {count}")
    lines.append(f"  TOTAL: {len(df)}")
    lines.append(f"  NUM CLASSES: {len(overall_counts)}")

    # Per-split distribution
    lines.append("\n## Per-Split Class Distribution")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        lines.append(f"\n### {split_name.upper()} ({len(split_df)} samples)")
        split_counts = split_df["remapped_label"].value_counts().sort_values(ascending=False)
        for cls, count in split_counts.items():
            pct = 100.0 * count / len(split_df) if len(split_df) > 0 else 0
            lines.append(f"    {cls}: {count} ({pct:.1f}%)")

    # Check for missing classes in val/test
    lines.append("\n## Missing Classes Check")
    all_classes = set(overall_counts.index)
    for split_name in ["val", "test"]:
        split_df = df[df["split"] == split_name]
        split_classes = set(split_df["remapped_label"].unique())
        missing = all_classes - split_classes
        if missing:
            lines.append(f"  WARNING: {split_name} is missing classes: {sorted(missing)}")
        else:
            lines.append(f"  {split_name}: all {len(all_classes)} classes present")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for 2025 dataset")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to annotations.csv from cropping")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for split files")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "exp1_11class", "exp2_10class_terns", "exp3_hank_coarse"],
                        help="Which experiment configuration to generate")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load annotations
    print(f"[info] Loading annotations from {args.annotations}")
    df = pd.read_csv(args.annotations)
    print(f"[info] Loaded {len(df)} annotations")

    # Get all orthomosaics and create splits
    all_orthomosaics = df["orthomosaic_id"].unique().tolist()
    print(f"[info] Found {len(all_orthomosaics)} orthomosaics: {sorted(all_orthomosaics)}")

    splits = create_orthomosaic_splits(
        all_orthomosaics,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Assign each crop to a split based on its orthomosaic
    om_to_split = {}
    for split_name, oms in splits.items():
        for om in oms:
            om_to_split[om] = split_name

    df["split"] = df["orthomosaic_id"].map(om_to_split)

    # Check for unmapped orthomosaics
    unmapped = df[df["split"].isna()]
    if len(unmapped) > 0:
        print(f"[warn] {len(unmapped)} crops have unmapped orthomosaics!")
        print(unmapped["orthomosaic_id"].value_counts())

    # Get species counts for threshold-based mapping
    species_counts = Counter(df["species_name"])
    all_species = set(species_counts.keys())

    # Determine which experiments to run
    if args.experiment == "all":
        experiments = list(EXPERIMENT_CONFIGS.keys())
    else:
        experiments = [args.experiment]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save orthomosaic splits (shared across experiments)
    splits_path = output_dir / "orthomosaic_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[info] Saved orthomosaic splits to {splits_path}")

    # Process each experiment
    for exp_name in experiments:
        print(f"\n[info] Processing experiment: {exp_name}")
        config = EXPERIMENT_CONFIGS[exp_name]
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Compute label mapping based on experiment type
        if exp_name == "exp1_11class":
            mapping = compute_label_mapping_exp1(species_counts, config["min_samples"])
        elif exp_name == "exp2_10class_terns":
            mapping = compute_label_mapping_exp2(
                species_counts,
                config["min_samples"],
                config["tern_merge"],
                config["tern_target"]
            )
        elif exp_name == "exp3_hank_coarse":
            mapping = compute_label_mapping_exp3(
                all_species,
                config["explicit_mapping"],
                config["others_class"]
            )
        else:
            raise ValueError(f"Unknown experiment: {exp_name}")

        # Apply mapping
        exp_df = apply_label_mapping(df, mapping)

        # Save per-split CSVs
        for split_name in ["train", "val", "test"]:
            split_df = exp_df[exp_df["split"] == split_name]
            split_path = exp_dir / f"{split_name}.csv"
            split_df.to_csv(split_path, index=False)
            print(f"  [saved] {split_path} ({len(split_df)} samples)")

        # Save label mapping
        mapping_path = exp_dir / "label_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2, sort_keys=True)

        # Save class counts per split
        counts_per_split = {}
        for split_name in ["train", "val", "test"]:
            split_df = exp_df[exp_df["split"] == split_name]
            # Convert to native Python int for JSON serialization
            counts_per_split[split_name] = {
                k: int(v) for k, v in split_df["remapped_label"].value_counts().items()
            }
        counts_path = exp_dir / "class_counts_per_split.json"
        with open(counts_path, "w") as f:
            json.dump(counts_per_split, f, indent=2)

        # Generate and save report
        report = generate_report(exp_df, splits, mapping, exp_name)
        report_path = exp_dir / "split_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  [saved] {report_path}")

        # Print summary
        print(f"\n{report}")

    print("\n[done] All experiments processed successfully!")


if __name__ == "__main__":
    main()
