#!/usr/bin/env python3
"""
COCO-format dataset annotation analyzer with plots.

Usage:
    python scripts/utils/analyze_dataset.py [path_to_json] [--output-dir DIR]

Defaults to data/classification_original_training.json if no argument given.
Plots are saved to --output-dir (default: plots/).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_coco_json(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise KeyError(f"Missing required COCO key: '{key}'")
    return data


def _cat_map(data: dict) -> dict:
    return {c["id"]: c["name"] for c in data["categories"]}


# ---------------------------------------------------------------------------
# Console report helpers
# ---------------------------------------------------------------------------

def basic_stats(data: dict) -> None:
    n_images = len(data["images"])
    n_annots = len(data["annotations"])
    n_cats = len(data["categories"])
    unique_cat_ids = {a["category_id"] for a in data["annotations"]}

    print("=" * 60)
    print("1. BASIC DATASET STATISTICS")
    print("=" * 60)
    print(f"  Images:              {n_images:,}")
    print(f"  Annotations (birds): {n_annots:,}")
    print(f"  Categories defined:  {n_cats}")
    print(f"  Unique category IDs in annotations: {len(unique_cat_ids)}")
    print()


def species_distribution(data: dict) -> Counter:
    cm = _cat_map(data)
    counts = Counter(a["category_id"] for a in data["annotations"])
    total = sum(counts.values())

    print("=" * 60)
    print("2. SPECIES DISTRIBUTION")
    print("=" * 60)
    print(f"  {'ID':>4}  {'Name':<20} {'Count':>8}  {'Pct':>6}")
    print(f"  {'-'*4}  {'-'*20} {'-'*8}  {'-'*6}")
    for cat_id, count in counts.most_common():
        name = cm.get(cat_id, f"<UNMAPPED:{cat_id}>")
        pct = 100.0 * count / total
        print(f"  {cat_id:>4}  {name:<20} {count:>8,}  {pct:>5.1f}%")
    print()
    return counts


def bird_density(data: dict) -> Counter:
    birds_per_image = Counter(a["image_id"] for a in data["annotations"])
    vals = list(birds_per_image.values())

    n = len(vals)
    avg = sum(vals) / n if n else 0
    vals_sorted = sorted(vals)
    median = vals_sorted[n // 2] if n else 0
    max_birds = max(vals) if vals else 0
    min_birds = min(vals) if vals else 0

    buckets = [1, 2, 3, 5, 10, 20, 50, 100]
    hist = Counter()
    for c in vals:
        placed = False
        for b in buckets:
            if c <= b:
                hist[b] += 1
                placed = True
                break
        if not placed:
            hist["> 100"] += 1

    print("=" * 60)
    print("3. BIRD DENSITY PER IMAGE")
    print("=" * 60)
    print(f"  Images with annotations: {n:,}")
    images_without = len(data["images"]) - n
    print(f"  Images with 0 birds:     {images_without:,}")
    print(f"  Avg birds/image:         {avg:.2f}")
    print(f"  Median birds/image:      {median}")
    print(f"  Min birds/image:         {min_birds}")
    print(f"  Max birds/image:         {max_birds}")
    print()
    print("  Distribution:")
    prev = 0
    for b in buckets:
        label = f"  {prev+1}-{b}" if b > 1 else "  1"
        print(f"    {label:>8} birds: {hist.get(b, 0):>6,} images")
        prev = b
    if hist.get("> 100", 0):
        print(f"    {'> 100':>8} birds: {hist['> 100']:>6,} images")
    print()
    return birds_per_image


def top_images(data: dict, birds_per_image: Counter, top_n: int = 10) -> None:
    image_map = {img["id"]: img["file_name"] for img in data["images"]}
    top = birds_per_image.most_common(top_n)

    print("=" * 60)
    print(f"4. TOP {top_n} IMAGES WITH MOST BIRDS")
    print("=" * 60)
    for rank, (img_id, count) in enumerate(top, 1):
        fname = image_map.get(img_id, f"<unknown id={img_id}>")
        print(f"  {rank:>3}. {fname:<45} {count:>4} birds")
    print()


def dataset_imbalance(species_counts: Counter, data: dict) -> None:
    cm = _cat_map(data)

    if not species_counts:
        print("  No annotations to analyze.\n")
        return

    most_common_id, most_count = species_counts.most_common(1)[0]
    least_common_id, least_count = species_counts.most_common()[-1]

    print("=" * 60)
    print("5. DATASET IMBALANCE")
    print("=" * 60)
    most_name = cm.get(most_common_id, f"<UNMAPPED:{most_common_id}>")
    least_name = cm.get(least_common_id, f"<UNMAPPED:{least_common_id}>")
    ratio = most_count / least_count if least_count else float("inf")
    print(f"  Largest class:  {most_name} (id={most_common_id}) — {most_count:,}")
    print(f"  Smallest class: {least_name} (id={least_common_id}) — {least_count:,}")
    print(f"  Imbalance ratio (max/min): {ratio:.1f}x")
    print()

    threshold = 50
    rare = [(cid, cnt) for cid, cnt in species_counts.most_common() if cnt < threshold]
    if rare:
        print(f"  Species with fewer than {threshold} examples:")
        for cid, cnt in rare:
            name = cm.get(cid, f"<UNMAPPED:{cid}>")
            print(f"    {name:<20} (id={cid:>3}): {cnt:>4}")
    else:
        print(f"  All species have >= {threshold} examples.")
    print()


def sanity_checks(data: dict) -> None:
    image_ids = {img["id"] for img in data["images"]}
    cm = _cat_map(data)

    bad_image_refs = []
    for a in data["annotations"]:
        if a["image_id"] not in image_ids:
            bad_image_refs.append(a)

    unmapped_cat_ids = sorted(
        {a["category_id"] for a in data["annotations"]} - set(cm.keys())
    )

    print("=" * 60)
    print("6. SANITY CHECKS")
    print("=" * 60)

    if bad_image_refs:
        print(f"  [FAIL] {len(bad_image_refs):,} annotations reference invalid image IDs")
        for a in bad_image_refs[:5]:
            print(f"         annotation id={a['id']}, image_id={a['image_id']}")
        if len(bad_image_refs) > 5:
            print(f"         ... and {len(bad_image_refs)-5} more")
    else:
        print("  [PASS] All annotations reference valid image IDs")

    if unmapped_cat_ids:
        bad_count = sum(1 for a in data["annotations"] if a["category_id"] in unmapped_cat_ids)
        print(f"  [FAIL] {len(unmapped_cat_ids)} category IDs in annotations have no category entry")
        print(f"         Unmapped IDs: {unmapped_cat_ids}")
        print(f"         Affected annotations: {bad_count:,}")
        print(f"         Defined categories: {list(cm.items())}")
    else:
        print("  [PASS] All category IDs map to a defined category name")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _species_label(cat_id: int, cm: dict) -> str:
    name = cm.get(cat_id, f"id={cat_id}")
    return f"{name} ({cat_id})"


def plot_species_distribution(species_counts: Counter, data: dict, out_dir: Path) -> None:
    cm = _cat_map(data)
    ordered = species_counts.most_common()
    labels = [_species_label(cid, cm) for cid, _ in ordered]
    values = [cnt for _, cnt in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(labels) * 0.35)))

    # --- Bar chart (all species) ---
    ax = axes[0]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Annotation count")
    ax.set_title("Species distribution (all)")

    # --- Bar chart (log scale, highlights tail) ---
    ax2 = axes[1]
    ax2.barh(y_pos, values, color="coral")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlabel("Annotation count (log scale)")
    ax2.set_title("Species distribution (log scale)")

    plt.tight_layout()
    path = out_dir / "species_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_species_pie(species_counts: Counter, data: dict, out_dir: Path) -> None:
    cm = _cat_map(data)
    ordered = species_counts.most_common()
    total = sum(species_counts.values())

    # Group tiny slices (<1%) into "Other"
    main, other_sum = [], 0
    for cid, cnt in ordered:
        if cnt / total >= 0.01:
            main.append((cid, cnt))
        else:
            other_sum += cnt
    if other_sum:
        main.append((-1, other_sum))

    labels = [
        _species_label(cid, cm) if cid != -1 else f"Other ({other_sum:,})"
        for cid, _ in main
    ]
    sizes = [cnt for _, cnt in main]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, textprops={"fontsize": 8})
    ax.set_title("Species proportion")
    plt.tight_layout()
    path = out_dir / "species_pie.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bird_density_histogram(birds_per_image: Counter, out_dir: Path) -> None:
    vals = list(birds_per_image.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Raw histogram ---
    ax = axes[0]
    ax.hist(vals, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Birds per image")
    ax.set_ylabel("Number of images")
    ax.set_title("Bird density per image")
    ax.axvline(np.mean(vals), color="red", linestyle="--", label=f"mean={np.mean(vals):.1f}")
    ax.axvline(np.median(vals), color="orange", linestyle="--", label=f"median={np.median(vals):.0f}")
    ax.legend()

    # --- CDF ---
    ax2 = axes[1]
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2.plot(sorted_vals, cdf, color="steelblue")
    ax2.set_xlabel("Birds per image")
    ax2.set_ylabel("Cumulative fraction of images")
    ax2.set_title("Cumulative distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "bird_density.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_top_images(data: dict, birds_per_image: Counter, out_dir: Path, top_n: int = 20) -> None:
    image_map = {img["id"]: img["file_name"] for img in data["images"]}
    top = birds_per_image.most_common(top_n)

    labels = [Path(image_map.get(img_id, f"id={img_id}")).stem for img_id, _ in top]
    values = [cnt for _, cnt in top]

    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.35)))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="teal")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Number of birds")
    ax.set_title(f"Top {top_n} images by bird count")
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
    plt.tight_layout()
    path = out_dir / "top_images.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_class_imbalance(species_counts: Counter, data: dict, out_dir: Path) -> None:
    cm = _cat_map(data)
    ordered = species_counts.most_common()
    labels = [_species_label(cid, cm) for cid, _ in ordered]
    values = [cnt for _, cnt in ordered]
    max_val = values[0]

    fig, ax = plt.subplots(figsize=(14, max(6, len(labels) * 0.35)))
    y_pos = np.arange(len(labels))
    colors = ["#d9534f" if v < 50 else "#5cb85c" if v > max_val * 0.1 else "#f0ad4e" for v in values]
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Annotation count (log)")
    ax.set_title("Class imbalance (red < 50, yellow < 10% of max, green >= 10%)")
    ax.axvline(50, color="red", linestyle=":", alpha=0.5, label="threshold=50")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "class_imbalance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze(path: str, output_dir: str = "plots") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset: {path}")
    print(f"Plots:   {out_dir.resolve()}\n")

    data = load_coco_json(path)

    # Console report
    basic_stats(data)
    species_counts = species_distribution(data)
    bpi = bird_density(data)
    top_images(data, bpi)
    dataset_imbalance(species_counts, data)
    sanity_checks(data)

    # Generate plots
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_species_distribution(species_counts, data, out_dir)
    plot_species_pie(species_counts, data, out_dir)
    plot_bird_density_histogram(bpi, out_dir)
    plot_top_images(data, bpi, out_dir)
    plot_class_imbalance(species_counts, data, out_dir)
    print()
    print(f"All plots saved to {out_dir.resolve()}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a COCO-format dataset JSON.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="data/classification_original_training.json",
        help="Path to the COCO JSON file (default: data/classification_original_training.json)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="plots",
        help="Directory to save plots (default: plots/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.json_path).exists():
        print(f"Error: file not found: {args.json_path}")
        sys.exit(1)
    analyze(args.json_path, args.output_dir)
