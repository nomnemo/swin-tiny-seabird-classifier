"""
Create train/val/test splits for the 2025 cropped bird dataset.

Splits are done at the CROP level with stratified sampling per remapped class.
Since tiles were extracted without overlap, there is no spatial leakage risk,
and stratified splitting guarantees every class appears in all three splits.

Supports multiple experiment configurations with label remapping:
- exp1_11class: Classes with >=50 samples kept, rare classes -> OTHERS
- exp2_10class_terns: Same as exp1 but terns merged into TERNS
- exp3_hank_coarse: Hank's 8-class coarse grouping
- exp4_fine_grained: 11-class fine-grained grouping (split terns, split egret life stages)
- exp5_split_terns: Variant of exp3: ROTE/SATE split out; TRHEA its own class; LARGE_WHITE_BIRDS unchanged

Usage:
    python scripts/utils/create_splits_2025.py \
        --annotations /path/to/annotations.csv \
        --output-dir /path/to/splits \
        --experiment exp1_11class
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set

import pandas as pd


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

MIN_SAMPLES_THRESHOLD = 50

EXP1_CONFIG = {
    "name": "exp1_11class",
    "description": "11-class baseline: classes with >=50 samples kept, rare classes merged to OTHERS",
    "min_samples": MIN_SAMPLES_THRESHOLD,
}

EXP2_CONFIG = {
    "name": "exp2_10class_terns",
    "description": "10-class: terns merged into TERNS, rare classes merged to OTHERS",
    "tern_merge": ["ROTEA", "ROTEF", "SATEA", "SATEF", "MTRNS"],
    "tern_target": "TERNS",
    "min_samples": MIN_SAMPLES_THRESHOLD,
}

EXP3_CONFIG = {
    "name": "exp3_hank_coarse",
    "description": "Hank's 8-class coarse grouping",
    "explicit_mapping": {
        "PELICAN_ADULT": ["BRPEA", "BRPEF"],
        "PELICAN_CHICK": ["BRPEC"],
        "TERNS": ["ROTEA", "ROTEF", "SATEA", "SATEF", "MTRNS", "CATEA"],
        "LAUGHING_GULL": ["LAGUA", "LAGUF"],
        "LARGE_WHITE_BIRDS": [
            "GREGA", "GREGC", "GREGF", "SNEGA", "WHIBA", "WHIBF",
            "REEGA", "REEGF", "REEGWMA", "LWBBA", "LWBBC"
        ],
        "GBHE_ADULT": ["GBHEA"],
        "GBHE_CHICK": ["GBHEC"],
    },
    "others_class": "OTHERS",
}

EXP4_CONFIG = {
    "name": "exp4_fine_grained",
    "description": "Fine-grained 11-class grouping: split tern species, split egret life stages",
    "explicit_mapping": {
        "ROTEA_ROTEF": ["ROTEA", "ROTEF"],
        "SATEA_SATEF": ["SATEA", "SATEF"],
        "BRPEA_BRPEF": ["BRPEA", "BRPEF"],
        "BRPEC":       ["BRPEC"],
        "LAGUA_LAGUF": ["LAGUA", "LAGUF"],
        "TRHEA":       ["TRHEA"],
        "GREGA_GREGF": ["GREGA", "GREGF"],
        "GREGC":       ["GREGC"],
        "GBHEC":       ["GBHEC"],
        "WHIBA_WHIBF": ["WHIBA", "WHIBF"],
    },
    "others_class": "OTHERS",
}

EXP5_CONFIG = {
    "name": "exp5_split_terns",
    "description": "Variant of exp3: ROTE/SATE split out; TRHEA separated; LARGE_WHITE_BIRDS retains WHIB",
    "explicit_mapping": {
        "PELICAN_ADULT":     ["BRPEA", "BRPEF"],
        "PELICAN_CHICK":     ["BRPEC"],
        "ROTEA_ROTEF":       ["ROTEA", "ROTEF"],
        "SATEA_SATEF":       ["SATEA", "SATEF"],
        "LAUGHING_GULL":     ["LAGUA", "LAGUF"],
        "LARGE_WHITE_BIRDS": [
            "GREGA", "GREGC", "GREGF", "SNEGA", "WHIBA", "WHIBF",
            "REEGA", "REEGF", "REEGWMA", "LWBBA", "LWBBC",
        ],
        "GBHE_ADULT":        ["GBHEA"],
        "GBHE_CHICK":        ["GBHEC"],
        "TRHEA":             ["TRHEA"],
    },
    "others_class": "OTHERS",
}

SUBCLASS_TERNS_CONFIG = {
    "name": "subclass_terns",
    "description": "Sub-classifier under TERNS super-class: 3-way split among tern species",
    "explicit_mapping": {
        "ROTEA_ROTEF": ["ROTEA", "ROTEF"],
        "SATEA_SATEF": ["SATEA", "SATEF"],
        "CATEA_MTRNS": ["CATEA", "MTRNS"],
    },
    "others_class": None,  # drop rows whose species isn't in the mapping
}

SUBCLASS_LARGE_WHITE_BIRDS_CONFIG = {
    "name": "subclass_large_white_birds",
    "description": "Sub-classifier under LARGE_WHITE_BIRDS super-class: GREG adults, GREG chicks, LWBBA",
    "explicit_mapping": {
        "GREGA_GREGF": ["GREGA", "GREGF"],
        "GREGC":       ["GREGC"],
        "LWBBA":       ["LWBBA"],
    },
    "others_class": None,
}

EXPERIMENT_CONFIGS = {
    "exp1_11class": EXP1_CONFIG,
    "exp2_10class_terns": EXP2_CONFIG,
    "exp3_hank_coarse": EXP3_CONFIG,
    "exp4_fine_grained": EXP4_CONFIG,
    "exp5_split_terns": EXP5_CONFIG,
    "subclass_terns": SUBCLASS_TERNS_CONFIG,
    "subclass_large_white_birds": SUBCLASS_LARGE_WHITE_BIRDS_CONFIG,
}


# ============================================================================
# LABEL MAPPING
# ============================================================================

def compute_label_mapping_exp1(species_counts: Counter, min_samples: int) -> Dict[str, str]:
    mapping = {}
    for species, count in species_counts.items():
        mapping[species] = species if count >= min_samples else "OTHERS"
    return mapping


def compute_label_mapping_exp2(
    species_counts: Counter,
    min_samples: int,
    tern_classes: List[str],
    tern_target: str,
) -> Dict[str, str]:
    merged_counts = Counter()
    for species, count in species_counts.items():
        key = tern_target if species in tern_classes else species
        merged_counts[key] += count

    mapping = {}
    for species in species_counts.keys():
        if species in tern_classes:
            mapping[species] = tern_target
        elif merged_counts.get(species, 0) >= min_samples:
            mapping[species] = species
        else:
            mapping[species] = "OTHERS"
    return mapping


def compute_label_mapping_exp3(
    all_species: Set[str],
    explicit_mapping: Dict[str, List[str]],
    others_class: str,
) -> Dict[str, str]:
    mapping = {}
    mapped_sources = set()
    for target, sources in explicit_mapping.items():
        for src in sources:
            mapping[src] = target
            mapped_sources.add(src)
    for species in all_species:
        if species not in mapped_sources:
            mapping[species] = others_class
    return mapping


def compute_label_mapping_subclass(
    explicit_mapping: Dict[str, List[str]],
) -> Dict[str, str]:
    """Mapping for sub-classifier splits: only the listed species get a target;
    unmapped species are dropped from the dataset downstream."""
    mapping = {}
    for target, sources in explicit_mapping.items():
        for src in sources:
            mapping[src] = target
    return mapping


def apply_label_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    others_class: "str | None" = "OTHERS",
) -> pd.DataFrame:
    df = df.copy()
    df["remapped_label"] = df["species_name"].map(mapping)
    if others_class is not None:
        df["remapped_label"] = df["remapped_label"].fillna(others_class)
    else:
        df = df.dropna(subset=["remapped_label"]).reset_index(drop=True)
    return df


# ============================================================================
# STRATIFIED SPLIT
# ============================================================================

def create_stratified_splits(
    df: pd.DataFrame,
    label_key: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split crops into train/val/test with stratified sampling per class.

    Each class is shuffled then partitioned proportionally so every class
    with >= 3 samples is guaranteed representation in all three splits.
    Classes with fewer than 3 samples go entirely to train.
    """
    parts = []
    for label, group in df.groupby(label_key):
        group = group.sample(frac=1.0, random_state=seed)
        n = len(group)

        if n < 3:
            group = group.copy()
            group["split"] = "train"
            parts.append(group)
            continue

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(round(n * val_ratio)))
        # guarantee at least 1 test sample
        if n - n_train - n_val < 1:
            n_val = max(0, n_val - 1)

        g_train = group.iloc[:n_train].copy()
        g_val   = group.iloc[n_train:n_train + n_val].copy()
        g_test  = group.iloc[n_train + n_val:].copy()

        g_train["split"] = "train"
        g_val["split"]   = "val"
        g_test["split"]  = "test"

        parts.append(pd.concat([g_train, g_val, g_test]))

    return pd.concat(parts, ignore_index=True)


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(df: pd.DataFrame, mapping: Dict[str, str], experiment_name: str) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"SPLIT REPORT: {experiment_name}")
    lines.append("=" * 70)

    lines.append("\n## Split Method: stratified per remapped class (crop-level)")

    lines.append("\n## Label Mapping (original -> remapped)")
    for target in sorted(set(mapping.values())):
        sources = sorted(k for k, v in mapping.items() if v == target)
        lines.append(f"  {target} <- {sources}")

    lines.append("\n## Overall Class Distribution (after remapping)")
    overall_counts = df["remapped_label"].value_counts().sort_values(ascending=False)
    for cls, count in overall_counts.items():
        lines.append(f"  {cls}: {count}")
    lines.append(f"  TOTAL: {len(df)}")
    lines.append(f"  NUM CLASSES: {len(overall_counts)}")

    lines.append("\n## Per-Split Class Distribution")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        lines.append(f"\n### {split_name.upper()} ({len(split_df)} samples)")
        split_counts = split_df["remapped_label"].value_counts().sort_values(ascending=False)
        for cls, count in split_counts.items():
            pct = 100.0 * count / len(split_df) if len(split_df) > 0 else 0
            lines.append(f"    {cls}: {count} ({pct:.1f}%)")

    lines.append("\n## Missing Classes Check")
    all_classes = set(overall_counts.index)
    for split_name in ["val", "test"]:
        split_df = df[df["split"] == split_name]
        missing = all_classes - set(split_df["remapped_label"].unique())
        if missing:
            lines.append(f"  WARNING: {split_name} is missing classes: {sorted(missing)}")
        else:
            lines.append(f"  {split_name}: all {len(all_classes)} classes present")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for 2025 dataset")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to annotations.csv from cropping")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for split files")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all"] + list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[info] Loading annotations from {args.annotations}")
    df = pd.read_csv(args.annotations)
    print(f"[info] Loaded {len(df)} annotations")

    species_counts = Counter(df["species_name"])
    all_species = set(species_counts.keys())
    print(f"[info] Found {len(all_species)} species")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = list(EXPERIMENT_CONFIGS.keys()) if args.experiment == "all" else [args.experiment]

    for exp_name in experiments:
        print(f"\n[info] Processing experiment: {exp_name}")
        config = EXPERIMENT_CONFIGS[exp_name]
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Compute label mapping
        others_class = "OTHERS"
        if exp_name == "exp1_11class":
            mapping = compute_label_mapping_exp1(species_counts, config["min_samples"])
        elif exp_name == "exp2_10class_terns":
            mapping = compute_label_mapping_exp2(
                species_counts, config["min_samples"],
                config["tern_merge"], config["tern_target"],
            )
        elif exp_name in ("exp3_hank_coarse", "exp4_fine_grained", "exp5_split_terns"):
            mapping = compute_label_mapping_exp3(
                all_species, config["explicit_mapping"], config["others_class"],
            )
            others_class = config["others_class"]
        elif exp_name in ("subclass_terns", "subclass_large_white_birds"):
            mapping = compute_label_mapping_subclass(config["explicit_mapping"])
            others_class = config["others_class"]  # None -> drop unmapped species

        # Apply mapping then stratified split
        exp_df = apply_label_mapping(df, mapping, others_class=others_class)
        exp_df = create_stratified_splits(
            exp_df,
            label_key="remapped_label",
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        # Save per-split CSVs
        for split_name in ["train", "val", "test"]:
            split_df = exp_df[exp_df["split"] == split_name]
            split_path = exp_dir / f"{split_name}.csv"
            split_df.to_csv(split_path, index=False)
            print(f"  [saved] {split_path} ({len(split_df)} samples)")

        # Save label mapping and counts
        with open(exp_dir / "label_mapping.json", "w") as f:
            json.dump(mapping, f, indent=2, sort_keys=True)

        counts_per_split = {}
        for split_name in ["train", "val", "test"]:
            split_df = exp_df[exp_df["split"] == split_name]
            counts_per_split[split_name] = {
                k: int(v) for k, v in split_df["remapped_label"].value_counts().items()
            }
        with open(exp_dir / "class_counts_per_split.json", "w") as f:
            json.dump(counts_per_split, f, indent=2)

        # Report
        report = generate_report(exp_df, mapping, exp_name)
        with open(exp_dir / "split_report.txt", "w") as f:
            f.write(report)
        print(f"\n{report}")

    print("\n[done] All experiments processed successfully!")


if __name__ == "__main__":
    main()
