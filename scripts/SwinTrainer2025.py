"""
Swin-Tiny trainer for the 2025 Chester Island seabird dataset.

Supports 4 main + 2 sub-classifier experiment presets:
  - exp1_11class:               11-class baseline (>=50 samples kept, rare -> OTHERS)
  - exp2_10class_terns:         10-class with terns merged into TERNS
  - exp3_hank_coarse:           8-class Hank coarse grouping
  - exp4_fine_grained:          11-class fine-grained (split terns, split egret life stages)
  - exp5_split_terns:           10-class: exp3 with ROTE/SATE split out and TRHEA separated
  - exp6_drop_mtrns:            same as exp5 but MTRNS rows dropped from dataset entirely
  - subclass_terns:             3-class sub-classifier under TERNS super-class
  - subclass_large_white_birds: 3-class sub-classifier under LARGE_WHITE_BIRDS super-class

Usage:
    python scripts/SwinTrainer2025.py --experiment exp1_11class
    python scripts/SwinTrainer2025.py --experiment exp2_10class_terns --epochs 50
    python scripts/SwinTrainer2025.py --experiment exp4_fine_grained
"""

import time
from datetime import timedelta
from pathlib import Path
import json
import argparse
from typing import Optional
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
import timm
from torch import nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    classification_report,
    f1_score,
    average_precision_score,
)
from scripts.DataLoader import set_up_data_loaders


# ============================================================================
# EXPERIMENT PRESETS
# ============================================================================

DATASET_ROOT = Path("cropped-dataset-2025")
SPLITS_ROOT = DATASET_ROOT / "splits"
IMAGE_ROOT = DATASET_ROOT / "crops"

EXPERIMENT_PRESETS = {
    "exp1_11class": {
        "name": "exp1_11class",
        "description": "11-class baseline: classes with >=50 samples, rare -> OTHERS",
        "train_csv": SPLITS_ROOT / "exp1_11class" / "train.csv",
        "val_csv": SPLITS_ROOT / "exp1_11class" / "val.csv",
        "test_csv": SPLITS_ROOT / "exp1_11class" / "test.csv",
        "image_root": IMAGE_ROOT,
        "label_key": "remapped_label",
        "num_classes": 11,
    },
    "exp2_10class_terns": {
        "name": "exp2_10class_terns",
        "description": "10-class: terns merged into TERNS, rare -> OTHERS",
        "train_csv": SPLITS_ROOT / "exp2_10class_terns" / "train.csv",
        "val_csv": SPLITS_ROOT / "exp2_10class_terns" / "val.csv",
        "test_csv": SPLITS_ROOT / "exp2_10class_terns" / "test.csv",
        "image_root": IMAGE_ROOT,
        "label_key": "remapped_label",
        "num_classes": 10,
    },
    "exp3_hank_coarse": {
        "name": "exp3_hank_coarse",
        "description": "8-class Hank coarse grouping",
        "train_csv": SPLITS_ROOT / "exp3_hank_coarse" / "train.csv",
        "val_csv": SPLITS_ROOT / "exp3_hank_coarse" / "val.csv",
        "test_csv": SPLITS_ROOT / "exp3_hank_coarse" / "test.csv",
        "image_root": IMAGE_ROOT,
        "label_key": "remapped_label",
        "num_classes": 8,
    },
    "exp4_fine_grained": {
        "name": "exp4_fine_grained",
        "description": "11-class fine-grained: split terns, split egret life stages",
        "train_csv": SPLITS_ROOT / "exp4_fine_grained" / "train.csv",
        "val_csv": SPLITS_ROOT / "exp4_fine_grained" / "val.csv",
        "test_csv": SPLITS_ROOT / "exp4_fine_grained" / "test.csv",
        "image_root": IMAGE_ROOT,
        "label_key": "remapped_label",
        "num_classes": 11,
    },
    "exp5_split_terns": {
        "name": "exp5_split_terns",
        "description": "10-class exp3 variant: ROTE/SATE split; TRHEA separated; LARGE_WHITE_BIRDS keeps WHIB",
        "train_csv": SPLITS_ROOT / "exp5_split_terns" / "train.csv",
        "val_csv": SPLITS_ROOT / "exp5_split_terns" / "val.csv",
        "test_csv": SPLITS_ROOT / "exp5_split_terns" / "test.csv",
        "image_root": IMAGE_ROOT,
        "label_key": "remapped_label",
        "num_classes": 10,
    },
    "exp6_drop_mtrns": {
        "name": "exp6_drop_mtrns",
        "description": "Same as exp5_split_terns but MTRNS rows dropped from dataset",
        "train_csv": SPLITS_ROOT / "exp6_drop_mtrns" / "train.csv",
        "val_csv": SPLITS_ROOT / "exp6_drop_mtrns" / "val.csv",
        "test_csv": SPLITS_ROOT / "exp6_drop_mtrns" / "test.csv",
        "image_root": IMAGE_ROOT,
        "label_key": "remapped_label",
        "num_classes": 10,
    },
    "subclass_terns": {
        "name": "subclass_terns",
        "description": "3-class sub-classifier for TERNS: ROTEA_ROTEF, SATEA_SATEF, CATEA_MTRNS",
        "train_csv": SPLITS_ROOT / "subclass_terns" / "train.csv",
        "val_csv": SPLITS_ROOT / "subclass_terns" / "val.csv",
        "test_csv": SPLITS_ROOT / "subclass_terns" / "test.csv",
        "image_root": IMAGE_ROOT,
        "label_key": "remapped_label",
        "num_classes": 3,
    },
    "subclass_large_white_birds": {
        "name": "subclass_large_white_birds",
        "description": "3-class sub-classifier for LARGE_WHITE_BIRDS: GREGA_GREGF, GREGC, LWBBA",
        "train_csv": SPLITS_ROOT / "subclass_large_white_birds" / "train.csv",
        "val_csv": SPLITS_ROOT / "subclass_large_white_birds" / "val.csv",
        "test_csv": SPLITS_ROOT / "subclass_large_white_birds" / "test.csv",
        "image_root": IMAGE_ROOT,
        "label_key": "remapped_label",
        "num_classes": 3,
    },
}


# ============================================================================
# DEFAULT HYPERPARAMETERS
# ============================================================================

MODEL_NAME = "swin_tiny_patch4_window7_224"
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 0.01
ACCUM_STEPS = 1
WARMUP_EPOCHS = 10
BATCH_TRAIN = 32
BATCH_EVAL = 128
NUM_WORKERS = 16
INPUT_SIZE = 224
USE_SAMPLER = True
SAMPLER_POWER = 0.5  # 1/sqrt(count) reweighting: softer correction for long-tailed classes
AMP = True

# Hardware (edit when switching GPU instances)
HARDWARE = "1x NVIDIA A10 (Lambda)"

# Will be set in main()
DEVICE = None
OUT_DIR = None
LOG_PATH = None


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:
    """Log message to stdout and optionally to file."""
    print(message)
    if LOG_PATH is not None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


# ============================================================================
# PLOTTING
# ============================================================================

def plot_curves(history, path):
    """Plot training vs validation loss and accuracy."""
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)

    ax_loss.plot(ep, history["train_loss"], label="train_loss", color="blue")
    ax_loss.plot(ep, history["val_loss"], label="val_loss", color="red")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(ep, history["train_acc"], label="train_acc", color="blue")
    ax_acc.plot(ep, history["val_acc"], label="val_acc", color="red")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_two_cms(y1, p1, y2, p2, classes, path, titles=("Validation", "Test")):
    """Plot two row-normalized confusion matrices side-by-side."""
    from sklearn.metrics import confusion_matrix
    labels = list(range(len(classes)))

    cm1 = confusion_matrix(y1, p1, labels=labels)
    cm2 = confusion_matrix(y2, p2, labels=labels)

    def _row_normalize(cm):
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.divide(cm, row_sums, where=row_sums > 0) * 100.0
        pct[np.isnan(pct)] = 0.0
        return pct

    cm1_pct = _row_normalize(cm1)
    cm2_pct = _row_normalize(cm2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    vmin, vmax = 0.0, 100.0
    im = None
    for ax, cm_pct, title in zip(axes, (cm1_pct, cm2_pct), titles):
        im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        thresh = (cm_pct.max() + cm_pct.min()) / 2.0 if cm_pct.size else 0.0
        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                val = cm_pct[i, j]
                text = f"{val:.1f}" if val > 0 else "0.0"
                color = "white" if val > thresh else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=6)

    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.046, pad=0.02)
    cbar.set_label("Percentage (%)")
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ============================================================================
# EVALUATION
# ============================================================================

def eval_collect(model, dl, num_classes, device, amp_enabled):
    """Return (y_true_list, y_pred_list, y_proba_array)"""
    model.eval()
    y_true, y_pred = [], []
    probs = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(xb)
                p = torch.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    probs = np.concatenate(probs, axis=0) if probs else np.zeros((0, num_classes), dtype=np.float32)
    return y_true, y_pred, probs


def compute_map_ovr(y_true, probs, num_classes):
    """One-vs-rest AP per class, macro mAP."""
    if len(y_true) == 0:
        return 0.0, np.zeros(num_classes)
    y_true = np.array(y_true)
    Y = np.zeros((len(y_true), num_classes), dtype=np.int32)
    Y[np.arange(len(y_true)), y_true] = 1
    ap_per_class = []
    for k in range(num_classes):
        try:
            ap = average_precision_score(Y[:, k], probs[:, k])
        except ValueError:
            ap = 0.0
        ap_per_class.append(ap if np.isfinite(ap) else 0.0)
    return float(np.mean(ap_per_class)), np.array(ap_per_class)


def evaluate_full(model, dl, classes, header, save_prefix, out_dir, device, amp_enabled):
    """Full evaluation report + mAP, returns metrics and predictions."""
    y_true, y_pred, probs = eval_collect(model, dl, len(classes), device, amp_enabled)

    labels = list(range(len(classes)))
    log(f"\n{header}:")
    log(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=classes,
        digits=3,
        zero_division=0
    ))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    log(f"{header} macro-F1: {macro_f1:.3f}")

    mAP_macro, ap_cls = compute_map_ovr(y_true, probs, len(classes))
    log(f"{header} mAP (macro, one-vs-rest): {mAP_macro:.3f}")

    # Save per-class AP
    per_class_ap = {cls: float(ap) for cls, ap in zip(classes, ap_cls.tolist())}
    with (out_dir / f"{save_prefix}_ap_per_class.json").open("w", encoding="utf-8") as f:
        json.dump(per_class_ap, f, indent=2)
    log(f"{header} per-class AP written to {save_prefix}_ap_per_class.json")

    # Save misclassified images
    ds = dl.dataset
    misclassified = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != yp:
            row = ds.rows[i]
            misclassified.append({
                "crop_path": row["crop_path"],
                "true_label": classes[yt],
                "predicted_label": classes[yp],
                "true_class_conf": round(float(probs[i][yt]), 4),
                "predicted_class_conf": round(float(probs[i][yp]), 4),
            })
    with open(out_dir / f"{save_prefix}_misclassified.json", "w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=2)
    log(f"{header}: {len(misclassified)} misclassified saved to {save_prefix}_misclassified.json")

    metrics = {
        "macro_f1": float(macro_f1),
        "map_macro": float(mAP_macro),
        "n_samples": int(len(y_true)),
    }
    return metrics, y_true, y_pred


def split_composition(ds, classes, label_key):
    """Get class distribution in a dataset split."""
    cnt = Counter([r[label_key] for r in ds.rows])
    return {c: int(cnt.get(c, 0)) for c in classes}


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(
    experiment: str,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    accum_steps: int = ACCUM_STEPS,
    warmup_epochs: int = WARMUP_EPOCHS,
    batch_train: int = BATCH_TRAIN,
    batch_eval: int = BATCH_EVAL,
    num_workers: int = NUM_WORKERS,
    model_name: str = MODEL_NAME,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    amp_enabled: bool = AMP,
    out_dir: Optional[Path] = None,
):
    """Main training function."""
    global LOG_PATH, OUT_DIR

    run_start = time.perf_counter()

    # Get experiment preset
    if experiment not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown experiment: {experiment}. Choose from: {list(EXPERIMENT_PRESETS.keys())}")
    preset = EXPERIMENT_PRESETS[experiment]

    # Set up output directory
    if out_dir is None:
        out_dir = Path(f"runs_2025_{experiment}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create run subdirectory with hyperparameters
    lr_str = f"{int(round(lr * 1e6)):04d}"
    wd_str = f"{int(round(weight_decay * 10000)):04d}"
    run_name = f"swin_cropsplit_full_ep{epochs}_lr{lr_str}_wd{wd_str}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    OUT_DIR = run_dir
    LOG_PATH = run_dir / "train.log"
    ckpt_path = run_dir / "best.pt"

    log(f"[info] Experiment: {experiment}")
    log(f"[info] Description: {preset['description']}")
    log(f"[info] Run directory: {run_dir}")

    # Set up data loaders
    dl_train, dl_val, dl_test, meta = set_up_data_loaders(
        train_csv=preset["train_csv"],
        val_csv=preset["val_csv"],
        test_csv=preset["test_csv"],
        image_root=preset["image_root"],
        label_key=preset["label_key"],
        input_size=INPUT_SIZE,
        use_sampler=USE_SAMPLER,
        batch_train=batch_train,
        batch_eval=batch_eval,
        num_workers=num_workers,
        max_per_class=None,  # Use all training data; no per-class cap
        merge_groups=None,  # Merging already done in split CSVs
        sampler_power=SAMPLER_POWER,
    )

    classes = meta["classes"]
    num_classes = len(classes)
    dl_cfg = meta["dataloader_config"]

    # Log configuration
    log("=" * 60)
    log("TRAINING CONFIGURATION")
    log("=" * 60)
    log(f"  experiment:       {experiment}")
    log(f"  model:            {model_name}")
    log(f"  loss:             CrossEntropy")
    log(f"  epochs:           {epochs}")
    log(f"  warmup_epochs:    {warmup_epochs}")
    log(f"  lr:               {lr}")
    log(f"  weight_decay:     {weight_decay}")
    log(f"  accum_steps:      {accum_steps}")
    log(f"  AMP:              {amp_enabled}")
    log(f"  device:           {device}")
    log(f"  hardware:         {HARDWARE}")
    log(f"  input_size:       {dl_cfg['input_size']}")
    log(f"  batch_train:      {dl_cfg['batch_train']}")
    log(f"  batch_eval:       {dl_cfg['batch_eval']}")
    log(f"  use_sampler:      {dl_cfg['use_sampler']}")
    log(f"  sampler_power:    {dl_cfg['sampler_power']}  (1.0 = 1/count, 0.5 = 1/sqrt(count), 0.0 = uniform)")
    log(f"  num_workers:      {dl_cfg['num_workers']}")
    log(f"  image_root:       {dl_cfg['image_root']}")
    log(f"  label_key:        {dl_cfg['label_key']}")
    log(f"  num_classes:      {num_classes}")
    log(f"  classes:          {classes}")
    log(f"  train_size:       {meta['sizes']['train']}")
    log(f"  val_size:         {meta['sizes']['val']}")
    log(f"  test_size:        {meta['sizes']['test']}")
    log("=" * 60)

    # Save split composition
    label_key = preset["label_key"]
    comp_train = split_composition(dl_train.dataset, classes, label_key)
    comp_val = split_composition(dl_val.dataset, classes, label_key)
    comp_test = split_composition(dl_test.dataset, classes, label_key)
    with open(run_dir / "split_composition.json", "w", encoding="utf-8") as f:
        json.dump({"train": comp_train, "val": comp_val, "test": comp_test}, f, indent=2)
    log("[info] saved split_composition.json")

    # Model / optimizer
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(epochs - warmup_epochs, 1))
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_metric = float("-inf")

    # Training loop
    for ep in range(1, epochs + 1):
        ep_start = time.perf_counter()

        # Learning rate warmup / schedule
        if ep <= warmup_epochs:
            warmup_factor = ep / max(warmup_epochs, 1)
            lr_now = lr * warmup_factor
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        else:
            sched.step()
            lr_now = opt.param_groups[0]["lr"]

        # Train pass
        model.train()
        running_loss = running_correct = running_count = 0
        opt.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(dl_train, start=1):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb)

            scaler.scale(loss / accum_steps).backward()
            if step % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                pred = logits.argmax(1)
                running_correct += (pred == yb).sum().item()
                running_count += yb.size(0)
                running_loss += loss.item() * yb.size(0)

        train_acc = running_correct / max(1, running_count)
        train_loss = running_loss / max(1, running_count)

        # Validation pass
        t_val = time.perf_counter()
        model.eval()
        v_loss = v_correct = v_count = 0
        y_val_true, y_val_pred = [], []

        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(xb)
                    loss = nn.functional.cross_entropy(logits, yb)
                v_loss += loss.item() * yb.size(0)
                preds = logits.argmax(1)
                v_correct += (preds == yb).sum().item()
                v_count += yb.size(0)
                y_val_true.extend(yb.cpu().tolist())
                y_val_pred.extend(preds.cpu().tolist())

        val_acc = v_correct / max(1, v_count)
        val_loss = v_loss / max(1, v_count)

        labels = list(range(num_classes))
        val_macro_f1 = f1_score(y_val_true, y_val_pred, labels=labels, average="macro", zero_division=0)
        t_ep_val = time.perf_counter() - t_val
        ep_dt = time.perf_counter() - ep_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log(
            f"ep {ep:02d} | "
            f"train acc {train_acc:.3f} loss {train_loss:.3f} | "
            f"val acc {val_acc:.3f} loss {val_loss:.3f} | macroF1 {val_macro_f1:.3f} | "
            f"lr {lr_now:.2e} | "
            f"val_time {t_ep_val:.1f}s | ep_time {ep_dt:.1f}s"
        )

        # Save best model
        if val_macro_f1 > best_metric:
            best_metric = val_macro_f1
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "name": model_name,
                "experiment": experiment,
            }, ckpt_path)
            log(f"[info] new best val macro-F1 {best_metric:.3f} at epoch {ep:02d}")

    # Save curves
    plot_curves(history, run_dir / "curves.pdf")
    log("[info] saved curves.pdf")

    # Final evaluation
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    val_summary, val_y, val_p = evaluate_full(
        model, dl_val, classes, "Validation report", "val", run_dir, device, amp_enabled
    )
    test_summary, test_y, test_p = evaluate_full(
        model, dl_test, classes, "Test report", "test", run_dir, device, amp_enabled
    )

    # Combined confusion matrices
    plot_two_cms(val_y, val_p, test_y, test_p, classes, run_dir / "val_test_cms.pdf")
    log("[info] saved val_test_cms.pdf")

    # Save final summary
    summary = {
        "experiment": experiment,
        "model": model_name,
        "epochs": epochs,
        "best_val_macro_f1": float(best_metric),
        "val_summary": val_summary,
        "test_summary": test_summary,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_time = time.perf_counter() - run_start
    log(f"[info] total run time: {str(timedelta(seconds=int(total_time)))}")

    return summary


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Swin-Tiny on 2025 seabird dataset with experiment presets"
    )
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=list(EXPERIMENT_PRESETS.keys()),
        help="Experiment preset to run"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--batch-train", type=int, default=BATCH_TRAIN)
    parser.add_argument("--batch-eval", type=int, default=BATCH_EVAL)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")

    args = parser.parse_args()

    train(
        experiment=args.experiment,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accum_steps=args.accum_steps,
        warmup_epochs=args.warmup_epochs,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        num_workers=args.num_workers,
        model_name=args.model_name,
        device=args.device,
        amp_enabled=not args.no_amp,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
