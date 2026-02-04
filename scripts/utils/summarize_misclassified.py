"""
Summarize a *_misclassified.json file produced by the training scripts.

Usage:
    # Overall summary (all classes)
    python -m scripts.utils.summarize_misclassified path/to/test_misclassified.json

    # Filter by true label
    python -m scripts.utils.summarize_misclassified path/to/test_misclassified.json --true-label MTRN

    # Generate visual contact sheet (grid of misclassified images)
    python -m scripts.utils.summarize_misclassified path/to/test_misclassified.json --true-label MTRN --visual
"""

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_ROOT = Path("data/crops")


def summarize(data, true_label=None, out_dir=None):
    if true_label is not None:
        data = [d for d in data if d["true_label"] == true_label]

    if not data:
        label_msg = f" with true_label={true_label}" if true_label else ""
        print(f"No misclassified images found{label_msg}.")
        return

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"Total misclassified: {len(data)}")

    if true_label is not None:
        log(f"True label: {true_label}")
        pred_counts = Counter(d["predicted_label"] for d in data)
        log(f"Confused into {len(pred_counts)} different classes:")
        for pred, count in pred_counts.most_common():
            examples = [d for d in data if d["predicted_label"] == pred]
            avg_pred_conf = sum(d["predicted_class_conf"] for d in examples) / len(examples)
            avg_true_conf = sum(d["true_class_conf"] for d in examples) / len(examples)
            log(f"  {pred:>12s}: {count:3d}  (avg pred_conf={avg_pred_conf:.3f}, avg true_conf={avg_true_conf:.3f})")
            for d in examples:
                log(f"    {d['crop_path']}  pred_conf={d['predicted_class_conf']:.4f}  true_conf={d['true_class_conf']:.4f}")
    else:
        # group by true label
        by_true = Counter(d["true_label"] for d in data)
        log(f"\nBreakdown by true label ({len(by_true)} classes with errors):")
        for label, count in by_true.most_common():
            preds = Counter(d["predicted_label"] for d in data if d["true_label"] == label)
            pred_str = ", ".join(f"{p}({n})" for p, n in preds.most_common())
            log(f"  {label:>12s}: {count:3d} misclassified -> {pred_str}")

    # save to file
    if out_dir is not None:
        suffix = f"_{true_label}" if true_label else "_all"
        out_path = out_dir / f"misclassified_summary{suffix}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved to {out_path}")


def _resolve_crop_path(crop_path_str):
    """Resolve a crop_path from the JSON to an actual file path."""
    p = Path(crop_path_str.replace("\\", "/"))
    # strip leading "crops/" if present since IMAGE_ROOT already points there
    parts = p.parts
    if len(parts) > 0 and parts[0].lower() == "crops":
        p = Path(*parts[1:])
    return (IMAGE_ROOT / p).resolve()


def make_visual(data, true_label, out_dir):
    """Generate a contact sheet PDF of misclassified images grouped by predicted label."""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if true_label is not None:
        data = [d for d in data if d["true_label"] == true_label]

    if not data:
        label_msg = f" with true_label={true_label}" if true_label else ""
        print(f"No misclassified images found{label_msg}.")
        return

    # group by (true_label, predicted_label)
    groups = {}
    for d in data:
        key = (d["true_label"], d["predicted_label"])
        groups.setdefault(key, []).append(d)

    # sort groups: by true_label, then by count descending
    sorted_keys = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k[0], k[1]))

    COLS = 6
    THUMB = 180  # thumbnail size in pixels

    suffix = f"_{true_label}" if true_label else "_all"
    pdf_path = out_dir / f"misclassified_visual{suffix}.pdf"

    with PdfPages(pdf_path) as pdf:
        for (true_lbl, pred_lbl) in sorted_keys:
            items = groups[(true_lbl, pred_lbl)]
            n = len(items)
            rows = (n + COLS - 1) // COLS

            fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3, rows * 3.5))
            fig.suptitle(
                f"True: {true_lbl}  ->  Predicted: {pred_lbl}  ({n} images)",
                fontsize=14, fontweight="bold", y=0.99,
            )

            # flatten axes for easy indexing
            if rows == 1 and COLS == 1:
                axes = np.array([axes])
            axes = np.array(axes).reshape(-1)

            for i, d in enumerate(items):
                ax = axes[i]
                img_path = _resolve_crop_path(d["crop_path"])
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (THUMB, THUMB))
                else:
                    img = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)

                ax.imshow(img)
                fname = Path(d["crop_path"]).name
                ax.set_title(
                    f"{fname}\npred={d['predicted_class_conf']:.2f}  true={d['true_class_conf']:.2f}",
                    fontsize=7,
                )
                ax.axis("off")

            # hide unused axes
            for i in range(n, len(axes)):
                axes[i].axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"Visual contact sheet saved to {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize misclassified images JSON")
    parser.add_argument("json_path", type=str, help="Path to *_misclassified.json")
    parser.add_argument("--true-label", type=str, default=None,
                        help="Filter by true label (e.g. MTRN)")
    parser.add_argument("--visual", action="store_true",
                        help="Generate a visual contact sheet PDF of misclassified images")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(args.json_path).parent
    summarize(data, args.true_label, out_dir)

    if args.visual:
        make_visual(data, args.true_label, out_dir)


if __name__ == "__main__":
    main()
