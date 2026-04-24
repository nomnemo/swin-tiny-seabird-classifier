"""
Estimate duplication rate in the crop dataset using perceptual hashing.

For each species folder, computes a perceptual hash (pHash) of every crop image
using multiprocessing, then clusters crops with hamming distance <= threshold
as likely duplicates of the same physical bird.

Outputs:
  - Per-species duplication stats (total crops, unique birds, dup rate)
  - Overall summary
  - CSV report saved to data_exploration/

Usage:
    python scripts/estimate_duplicates.py [--threshold 8] [--data-dir data/crops] [--workers 16]
"""

import argparse
import csv
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm


def _hash_one_file(file_path: str) -> tuple[str, str] | None:
    """Compute pHash for a single image. Returns (filename, hash_hex) or None."""
    try:
        img = Image.open(file_path)
        h = imagehash.phash(img)
        return (Path(file_path).name, str(h))
    except Exception:
        return None


def compute_hashes_parallel(
    species_dir: Path, n_workers: int
) -> list[tuple[str, imagehash.ImageHash]]:
    """Compute pHash for every image in a species folder using multiprocessing."""
    files = sorted(str(f) for f in species_dir.glob("*.jpg"))
    if not files:
        return []

    results = []
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(_hash_one_file, files, chunksize=64):
            if result is not None:
                name, hash_hex = result
                results.append((name, imagehash.hex_to_hash(hash_hex)))

    return results


def cluster_duplicates(
    hashes: list[tuple[str, imagehash.ImageHash]], threshold: int
) -> list[list[str]]:
    """
    Greedy clustering: iterate through hashes, assign each to the first
    existing cluster within hamming distance <= threshold, or start a new cluster.
    Each cluster represents one likely unique bird.
    """
    clusters: list[tuple[imagehash.ImageHash, list[str]]] = []

    for name, h in hashes:
        matched = False
        for rep_hash, members in clusters:
            if abs(h - rep_hash) <= threshold:
                members.append(name)
                matched = True
                break
        if not matched:
            clusters.append((h, [name]))

    return [members for _, members in clusters]


def main():
    parser = argparse.ArgumentParser(description="Estimate crop duplication rate")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/crops",
        help="Path to crops directory",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Hamming distance threshold for considering two crops as duplicates (default: 8)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, cpu_count()),
        help="Number of parallel workers for hashing (default: min(16, cpu_count))",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_exploration/duplication_report.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    crops_dir = Path(args.data_dir)
    if not crops_dir.exists():
        print(f"ERROR: {crops_dir} not found")
        sys.exit(1)

    species_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()])

    print(f"Crops directory: {crops_dir}")
    print(f"Hamming distance threshold: {args.threshold}")
    print(f"Workers: {args.workers}")
    print(f"Species folders found: {len(species_dirs)}")
    print()

    rows = []
    total_crops = 0
    total_unique = 0

    for sp_dir in species_dirs:
        species = sp_dir.name
        print(f"Processing {species}...", end=" ", flush=True)

        hashes = compute_hashes_parallel(sp_dir, args.workers)
        n_crops = len(hashes)

        if n_crops == 0:
            print("no images, skipping.")
            continue

        clusters = cluster_duplicates(hashes, args.threshold)
        n_unique = len(clusters)
        n_duplicated = n_crops - n_unique
        dup_rate = n_duplicated / n_crops * 100 if n_crops > 0 else 0

        max_cluster_size = max(len(c) for c in clusters)
        clusters_with_dups = sum(1 for c in clusters if len(c) > 1)

        print(
            f"crops={n_crops:,}  unique≈{n_unique:,}  "
            f"dup_rate={dup_rate:.1f}%  "
            f"max_cluster={max_cluster_size}  "
            f"birds_with_dups={clusters_with_dups}"
        )

        rows.append(
            {
                "species": species,
                "total_crops": n_crops,
                "unique_birds_approx": n_unique,
                "duplicate_crops": n_duplicated,
                "duplication_rate_pct": round(dup_rate, 1),
                "max_cluster_size": max_cluster_size,
                "birds_with_duplicates": clusters_with_dups,
            }
        )

        total_crops += n_crops
        total_unique += n_unique

    # Summary
    print()
    print("=" * 60)
    overall_dup_rate = (
        (total_crops - total_unique) / total_crops * 100 if total_crops > 0 else 0
    )
    print(f"OVERALL: {total_crops:,} crops → ~{total_unique:,} unique birds")
    print(f"         ~{total_crops - total_unique:,} duplicate crops ({overall_dup_rate:.1f}%)")
    print("=" * 60)

    # Save CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "species": "TOTAL",
                "total_crops": total_crops,
                "unique_birds_approx": total_unique,
                "duplicate_crops": total_crops - total_unique,
                "duplication_rate_pct": round(overall_dup_rate, 1),
                "max_cluster_size": "",
                "birds_with_duplicates": "",
            }
        )

    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
