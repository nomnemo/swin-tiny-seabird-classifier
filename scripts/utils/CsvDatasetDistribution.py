from pathlib import Path
from typing import Optional, Tuple
import csv

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

"""
1. GENERIC DATASET DISTRIBUTION

This module provides a small utility to compute value distributions for any
column in a CSV file. It counts how many times each distinct value appears,
computes percentages, and writes the result to a new CSV in the specified
output directory.

Usage pattern in this project:
  - Edit the DEFAULT_* constants below to point to the desired CSV and column.
  - Run this script to generate a distribution summary CSV.
"""

# === DEFAULT CONFIG ===
DEFAULT_CSV = Path("data/split_test.csv")
DEFAULT_COLUMN = "species_name"
FOLDER_NAME = "test_split_distribution"
DEFAULT_OUT_DIR = Path(f"data_exploration/{FOLDER_NAME}")
DEFAULT_OUTPUT_NAME: Optional[str] = "test_split_species_distribution.csv"
# Title templates for bar and pie charts.
# Use "{column}" as a placeholder for the column/key name.
DEFAULT_BAR_TITLE_TEMPLATE = FOLDER_NAME
DEFAULT_PIE_TITLE_TEMPLATE = FOLDER_NAME

def compute_column_distribution(
    csv_path: Path,
    column: str,
    out_dir: Path,
    output_name: Optional[str] = None,
) -> Path:
    """
    Compute a frequency distribution for `column` in `csv_path`.

    The function:
      1) Reads the CSV into a pandas DataFrame.
      2) Counts occurrences of each distinct value in the target column.
      3) Computes the percentage of the total for each value.
      4) Writes a summary CSV with columns:
           [column, "count", "percent"]

    Args:
        csv_path: Path to the input CSV file.
        column:   Name of the column whose distribution you want.
        out_dir:  Directory where the output CSV will be written.
        output_name: Optional explicit filename for the output CSV. If None,
                     a name of the form "{stem}_{column}_distribution.csv"
                     is used, where `stem` is `csv_path.stem`.

    Returns:
        Path to the written summary CSV.
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")

    # Ensure output directory exists and determine output path
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"{csv_path.stem}_{column}_distribution.csv"
    out_path = out_dir / output_name

    # get value counts for this column
    counts = df[column].value_counts(dropna=False) 
    
    # total count of all values (including NaN if present)
    total = int(counts.sum())

    # write out the summary CSV
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([column, "count", "percent"])
        for value, count in counts.items():
            pct = (count / total * 100.0) if total > 0 else 0.0
            writer.writerow([value, int(count), f"{pct:.3f}"])

    return out_path

def plot_distribution(
    summary_csv: Path,
    value_column: Optional[str] = None,
    out_png: Optional[Path] = None,
) -> Path:
    """
    Plot a simple bar chart from the output of compute_column_distribution.

    Args:
        summary_csv: Path to a CSV produced by compute_column_distribution.
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

    # Basic bar plot: distinct values vs counts.
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
    csv_path: Path,
    column: str,
    out_dir: Path,
    output_name: Optional[str] = None,
    plot_name: Optional[str] = None,
    pie_name: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Convenience wrapper that computes the column distribution and then plots it
    as both a bar chart and a pie chart.

    Args:
        csv_path:     Input CSV file.
        column:       Column name to compute distribution over.
        out_dir:      Output directory for both the summary CSV and plot PNG.
        output_name:  Optional explicit summary CSV filename.
        plot_name:    Optional explicit bar‑plot filename. If None, the PNG name
                      is derived from the summary CSV name.
        pie_name:     Optional explicit pie‑chart filename. If None, the PNG name
                      is derived from the summary CSV name.

    Returns:
        (summary_csv_path, bar_png_path, pie_png_path)
    """
    summary_csv = compute_column_distribution(
        csv_path=csv_path,
        column=column,
        out_dir=out_dir,
        output_name=output_name,
    )

    if plot_name is None:
        bar_path = None
    else:
        bar_path = out_dir / plot_name

    bar_png = plot_distribution(
        summary_csv=summary_csv,
        value_column=column,
        out_png=bar_path,
    )
    # Pie chart filename
    if pie_name is None:
        pie_path = summary_csv.with_suffix(".pie.png")
    else:
        pie_path = out_dir / pie_name

    # Load distribution once and create a pie chart with a separate legend
    # so labels do not overlap on the figure.
    df = pd.read_csv(summary_csv)
    x = df[column].astype(str)
    y = df["count"].astype(int)
    total = int(y.sum()) if len(y) > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 6))

    # Simple pie with no labels/percentages drawn directly on slices.
    wedges = ax.pie(
        y,
        labels=None,
        startangle=90,
    )[0]

    # Build legend labels with percentages.
    if total > 0:
        legend_labels = [f"{name} ({cnt / total * 100:.1f}%)" for name, cnt in zip(x, y)]
    else:
        legend_labels = [name for name in x]

    # Title can be customized via DEFAULT_PIE_TITLE_TEMPLATE.
    ax.set_title(DEFAULT_PIE_TITLE_TEMPLATE.format(column=column))
    ax.legend(
        wedges,
        legend_labels,
        title=column,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
    )

    plt.tight_layout()
    plt.savefig(pie_path, dpi=200, bbox_inches="tight")
    plt.close()

    return summary_csv, bar_png, pie_path


if __name__ == "__main__":
    # Simple script mode: uses the DEFAULT_* constants defined above
    # to compute and plot the distribution in one step.
    compute_and_plot_distribution(
        csv_path=DEFAULT_CSV,
        column=DEFAULT_COLUMN,
        out_dir=DEFAULT_OUT_DIR,
        output_name=DEFAULT_OUTPUT_NAME,
    )
