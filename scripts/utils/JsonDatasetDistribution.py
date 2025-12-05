from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import csv
import json
from collections import Counter

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

"""
GENERIC JSON DATASET DISTRIBUTION

This module mirrors CsvDatasetDistribution.py but for JSON files that contain
a list of dict-like records. It computes the frequency distribution of values
for a given key, writes a summary CSV, and produces both a bar chart and a
pie chart of the distribution.

Usage pattern in this project:
  - Edit the DEFAULT_* constants below to point to the desired JSON and key.
  - Run this script to generate a distribution summary CSV + plots.
"""

# === DEFAULT CONFIG ===
DEFAULT_JSON = Path("data/metadata_balanced_t100.json")
DEFAULT_KEY = "species_name"
DEFAULT_OUT_DIR = Path("data_exploration/comibined_t100_dataset")
DEFAULT_OUTPUT_NAME: Optional[str] = "metadata_species_distribution.csv"
# Title templates for bar and pie charts.
# Use "{column}" as a placeholder for the key/column name.
DEFAULT_BAR_TITLE_TEMPLATE = "Distribution of the dataset after combining rare species into OTHERS"
DEFAULT_PIE_TITLE_TEMPLATE = "Distribution of the dataset after combining rare species into OTHERS"

def compute_key_distribution(
    json_path: Path,
    key: str,
    out_dir: Path,
    output_name: Optional[str] = None,
) -> Path:
    """
    Compute a frequency distribution for `key` in a JSON list-of-dicts file.

    The function:
      1) Loads the JSON file as a list of records (dicts).
      2) Extracts `record[key]` for each record (missing keys count as None).
      3) Counts occurrences of each distinct value.
      4) Computes the percentage of the total for each value.
      5) Writes a summary CSV with columns:
           [key, "count", "percent"]

    Args:
        json_path:  Path to the input JSON file.
        key:        Dict key whose distribution you want to compute.
        out_dir:    Directory where the output CSV will be written.
        output_name: Optional explicit filename for the output CSV. If None,
                     a name of the form "{stem}_{key}_distribution.csv"
                     is used, where `stem` is `json_path.stem`.

    Returns:
        Path to the written summary CSV.
    """
    # Load JSON records
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {json_path}, got {type(data)}")

    # Extract values; use None for missing keys explicitly
    values = []
    for rec in data:
        if isinstance(rec, dict):
            values.append(rec.get(key))
        else:
            # If a record is not a dict, treat as missing
            values.append(None)

    counts = Counter(values)
    total = sum(counts.values())

    out_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"{json_path.stem}_{key}_distribution.csv"
    out_path = out_dir / output_name

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([key, "count", "percent"])
        for value, count in counts.most_common():
            pct = (count / total * 100.0) if total > 0 else 0.0
            writer.writerow([value, int(count), f"{pct:.3f}"])

    return out_path


def plot_distribution(
    summary_csv: Path,
    value_column: Optional[str] = None,
    out_png: Optional[Path] = None,
) -> Path:
    """
    Plot a simple bar chart from a summary CSV produced by compute_key_distribution.

    Args:
        summary_csv: Path to the summary CSV.
        value_column: Name of the value column (first column) in the summary CSV.
                      If None, the first column in the file is used.
        out_png: Optional explicit path for the output PNG. If None, the plot
                 is saved next to the summary CSV with a .png extension.

    Returns:
        Path to the saved PNG file.
    """
    df = pd.read_csv(summary_csv)
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError(f"{summary_csv} must have at least two columns.")

    if value_column is None:
        value_column = cols[0]
    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in {summary_csv}")
    if "count" not in df.columns:
        raise ValueError(f"'count' column not found in {summary_csv}")

    if out_png is None:
        out_png = summary_csv.with_suffix(".png")

    x = df[value_column].astype(str)
    y = df["count"].astype(int)

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    bars = ax.bar(x, y)
    ax.set_xlabel(value_column)
    ax.set_ylabel("count")
    # Title can be customized via DEFAULT_BAR_TITLE_TEMPLATE.
    ax.set_title(DEFAULT_BAR_TITLE_TEMPLATE.format(column=value_column))

    # Add count labels on top of each bar for readability.
    for bar, count in zip(bars, y):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return out_png


def compute_and_plot_distribution(
    json_path: Path,
    key: str,
    out_dir: Path,
    output_name: Optional[str] = None,
    plot_name: Optional[str] = None,
    pie_name: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Convenience wrapper that computes the key distribution and then plots it
    as both a bar chart and a pie chart.

    Args:
        json_path:   Input JSON file (list of dicts).
        key:         Dict key to compute distribution over.
        out_dir:     Output directory for summary CSV and plots.
        output_name: Optional explicit summary CSV filename.
        plot_name:   Optional explicit bar‑plot filename. If None, the PNG name
                     is derived from the summary CSV name.
        pie_name:    Optional explicit pie‑chart filename. If None, the PNG name
                     is derived from the summary CSV name.

    Returns:
        (summary_csv_path, bar_png_path, pie_png_path)
    """
    summary_csv = compute_key_distribution(
        json_path=json_path,
        key=key,
        out_dir=out_dir,
        output_name=output_name,
    )

    if plot_name is None:
        bar_path = None
    else:
        bar_path = out_dir / plot_name

    bar_png = plot_distribution(
        summary_csv=summary_csv,
        value_column=key,
        out_png=bar_path,
    )

    # Pie chart filename
    if pie_name is None:
        pie_path = summary_csv.with_suffix(".pie.png")
    else:
        pie_path = out_dir / pie_name

    # Load distribution once and create a pie chart with a separate legend
    df = pd.read_csv(summary_csv)
    x = df[key].astype(str)
    y = df["count"].astype(int)
    total = int(y.sum()) if len(y) > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 6))

    wedges = ax.pie(
        y,
        labels=None,
        startangle=90,
    )[0]

    if total > 0:
        legend_labels = [f"{name} ({cnt / total * 100:.1f}%)" for name, cnt in zip(x, y)]
    else:
        legend_labels = [name for name in x]

    # Title can be customized via DEFAULT_PIE_TITLE_TEMPLATE.
    ax.set_title(DEFAULT_PIE_TITLE_TEMPLATE.format(column=key))
    ax.legend(
        wedges,
        legend_labels,
        title=key,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
    )

    plt.tight_layout()
    plt.savefig(pie_path, dpi=200, bbox_inches="tight")
    plt.close()

    return summary_csv, bar_png, pie_path


if __name__ == "__main__":
    # Simple script mode: uses DEFAULT_* constants defined above.
    compute_and_plot_distribution(
        json_path=DEFAULT_JSON,
        key=DEFAULT_KEY,
        out_dir=DEFAULT_OUT_DIR,
        output_name=DEFAULT_OUTPUT_NAME,
    )
