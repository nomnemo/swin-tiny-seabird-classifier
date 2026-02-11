"""
Summarize misclassified images from the ROTEN vs SATEN binary classifier.

Generates visual contact sheets for both val and test splits in a single run.

Usage:
    python -m scripts.utils.summarize_misclassified_rosa
"""

import json
from collections import Counter
from pathlib import Path

IMAGE_ROOT = Path("data/Images-256")
RUN_DIR = Path("runs_roten_saten/swin_mpcNone_ep30_lr0100_wd0100_as1_3")


def _resolve_crop_path(crop_path_str):
    """Resolve a crop_path from the JSON to an actual file path."""
    p = Path(crop_path_str.replace("\\", "/"))
    parts = p.parts
    if len(parts) > 0 and parts[0].lower() == "crops":
        p = Path(*parts[1:])
    return (IMAGE_ROOT / p).resolve()


def summarize(data, split_name, out_dir):
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"=== {split_name.upper()} — {len(data)} misclassified ===")

    by_true = Counter(d["true_label"] for d in data)
    for label, count in by_true.most_common():
        preds = Counter(d["predicted_label"] for d in data if d["true_label"] == label)
        pred_str = ", ".join(f"{p}({n})" for p, n in preds.most_common())
        log(f"  {label:>12s}: {count:3d} misclassified -> {pred_str}")

    for d in data:
        log(f"    {d['crop_path']}  true={d['true_label']}  pred={d['predicted_label']}  "
            f"pred_conf={d['predicted_class_conf']:.4f}  true_conf={d['true_class_conf']:.4f}")

    out_path = out_dir / f"misclassified_summary_{split_name}.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nSaved to {out_path}")


def make_visual(data, split_name, out_dir):
    """Generate a contact sheet PDF of misclassified images grouped by error direction."""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if not data:
        print(f"No misclassified images for {split_name}.")
        return

    groups = {}
    for d in data:
        key = (d["true_label"], d["predicted_label"])
        groups.setdefault(key, []).append(d)

    sorted_keys = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k[0], k[1]))

    COLS = 6
    THUMB = 180

    pdf_path = out_dir / f"misclassified_visual_{split_name}.pdf"

    with PdfPages(pdf_path) as pdf:
        for (true_lbl, pred_lbl) in sorted_keys:
            items = groups[(true_lbl, pred_lbl)]
            n = len(items)
            rows = (n + COLS - 1) // COLS

            fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3, rows * 3.5))
            fig.suptitle(
                f"{split_name.upper()} — True: {true_lbl}  ->  Predicted: {pred_lbl}  ({n} images)",
                fontsize=14, fontweight="bold", y=0.99,
            )

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

            for i in range(n, len(axes)):
                axes[i].axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"Visual contact sheet saved to {pdf_path}")


def main():
    for split in ("val", "test"):
        json_path = RUN_DIR / f"{split}_misclassified.json"
        data = json.load(open(json_path, encoding="utf-8"))
        summarize(data, split, RUN_DIR)
        make_visual(data, split, RUN_DIR)
        print()


if __name__ == "__main__":
    main()
