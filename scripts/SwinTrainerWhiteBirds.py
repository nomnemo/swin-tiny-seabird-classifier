"""
Swin Tiny trainer for Large White Bird classification.

Classifies: GREG, MEGRT, SNEG, WHIB

Standalone training script that reuses utility functions from SwinTrainer.py
but filters the main split CSVs to only include the target white bird species.
Uses the same grouped splits (leakage-safe by orthomosaic parent ID).

Usage:
    python -m scripts.SwinTrainerWhiteBirds
    python -m scripts.SwinTrainerWhiteBirds --epochs 20 --lr 3e-4
"""

import time
from datetime import timedelta
from pathlib import Path
import json
import argparse
from typing import Optional, List
from collections import Counter

import numpy as np
import pandas as pd
import torch, timm
from torch import nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    classification_report,
    f1_score,
)
from scripts.SwinTrainer import (
    make_run_dir_name,
    plot_curves,
    plot_two_cms,
    eval_collect,
    compute_map_ovr,
    split_composition,
)
from scripts.ImageTransformer import get_transforms
from scripts.BirdDataset import BirdDataset

# ===== config =====
MODEL_NAME   = "swin_tiny_patch4_window7_224"
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 0.01
ACCUM_STEPS  = 1
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
AMP          = True
MAX_PER_CLASS = 3000

# Target species for this classifier
TARGET_SPECIES = ["GREG", "MEGRT", "SNEG", "WHIB"]

# Data paths (reuse the main split CSVs, filter to target species)
DATA_DIR   = Path("data")
IMAGE_ROOT = Path("data/crops")
TRAIN_CSV  = DATA_DIR / "split_train.csv"
VAL_CSV    = DATA_DIR / "split_val.csv"
TEST_CSV   = DATA_DIR / "split_test.csv"

# Output
OUT_DIR    = Path("runs_swin_white_birds")
OUT_DIR.mkdir(exist_ok=True)
LOG_PATH: Optional[Path] = None
CKPT_PATH  = "best_white_birds.pt"

# Learning rate warmup
WARMUP_EPOCHS = 10

# Hardware
HARDWARE = "1x NVIDIA A10 (Lambda)"

# DataLoader settings
INPUT_SIZE  = 224
USE_SAMPLER = True
BATCH_TRAIN = 32
BATCH_EVAL  = 128
NUM_WORKERS = 16
# ==================


def log(message: str) -> None:
    """Log to stdout and optionally to the run log file."""
    print(message)
    if LOG_PATH is not None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def evaluate_full(model, dl, classes, header, save_prefix, out_dir):
    """Full classification report + mAP, saves artefacts to out_dir."""
    y_true, y_pred, probs = eval_collect(model, dl, len(classes))

    labels = list(range(len(classes)))
    log(f"\n{header}:")
    log(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=classes,
        digits=3,
        zero_division=0,
    ))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    log(f"{header} macro-F1: {macro_f1:.3f}")

    mAP_macro, ap_cls = compute_map_ovr(y_true, probs, len(classes))
    log(f"{header} mAP (macro, one-vs-rest): {mAP_macro:.3f}")

    per_class_ap = {cls: float(ap) for cls, ap in zip(classes, ap_cls.tolist())}
    with (out_dir / f"{save_prefix}_ap_per_class.json").open("w", encoding="utf-8") as f:
        json.dump(per_class_ap, f, indent=2)
    log(f"{header} per-class AP written to {save_prefix}_ap_per_class.json")

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
    log(f"{header}: {len(misclassified)} misclassified images saved to {save_prefix}_misclassified.json")

    metrics = {
        "macro_f1": float(macro_f1),
        "map_macro": float(mAP_macro),
        "n_samples": int(len(y_true)),
    }
    return metrics, y_true, y_pred


def _filter_and_cap(df: pd.DataFrame, species: List[str], max_per_class: Optional[int]) -> pd.DataFrame:
    """Filter DataFrame to target species and optionally cap per class."""
    df = df[df["species_name"].isin(species)].copy()
    if max_per_class is not None:
        df = df.sample(frac=1.0, random_state=42)
        df = (
            df.groupby("species_name", group_keys=False)
            .head(max_per_class)
            .reset_index(drop=True)
        )
    return df


def _build_label_map(rows, label_key="species_name"):
    """Build consistent label map from label -> class index."""
    classes = sorted({row[label_key] for row in rows})
    class2id = {c: i for i, c in enumerate(classes)}
    return class2id, classes


def main():
    global OUT_DIR, LOG_PATH, CKPT_PATH
    run_start = time.perf_counter()

    # ── run directory ──
    run_name = make_run_dir_name(MODEL_NAME, MAX_PER_CLASS, EPOCHS, LR, WEIGHT_DECAY, ACCUM_STEPS)
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR = run_dir
    CKPT_PATH = OUT_DIR / "best.pt"
    LOG_PATH = OUT_DIR / "train.log"
    log(f"[info] run directory: {OUT_DIR}")

    # ── load and filter data ──
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    # Filter to target white bird species only
    cap_val_test = max(1, MAX_PER_CLASS // 5) if MAX_PER_CLASS is not None else None
    train_df = _filter_and_cap(train_df, TARGET_SPECIES, MAX_PER_CLASS)
    val_df   = _filter_and_cap(val_df,   TARGET_SPECIES, cap_val_test)
    test_df  = _filter_and_cap(test_df,  TARGET_SPECIES, cap_val_test)

    train_rows = train_df.to_dict(orient="records")
    val_rows   = val_df.to_dict(orient="records")
    test_rows  = test_df.to_dict(orient="records")

    # Build label map from training split
    class2id, classes = _build_label_map(train_rows)
    num_classes = len(classes)

    # Transforms
    train_tf = get_transforms(INPUT_SIZE, train=True)
    eval_tf  = get_transforms(INPUT_SIZE, train=False)

    # Datasets
    ds_train = BirdDataset(train_rows, class2id, IMAGE_ROOT, train_tf, missing_size=INPUT_SIZE)
    ds_val   = BirdDataset(val_rows,   class2id, IMAGE_ROOT, eval_tf,  missing_size=INPUT_SIZE)
    ds_test  = BirdDataset(test_rows,  class2id, IMAGE_ROOT, eval_tf,  missing_size=INPUT_SIZE)

    # Sampler
    class_counts = Counter([r["species_name"] for r in train_rows])
    sampler = None
    if USE_SAMPLER:
        weights = np.array([1.0 / class_counts[r["species_name"]] for r in train_rows], dtype=np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=BATCH_TRAIN, shuffle=(sampler is None),
                          sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=(NUM_WORKERS > 0))
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_EVAL, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=(NUM_WORKERS > 0))
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_EVAL, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=(NUM_WORKERS > 0))

    # Class weights
    cls_weights = np.array([1.0 / max(class_counts[c], 1) for c in classes], dtype=np.float32)
    normalized_cls_weights = cls_weights / cls_weights.mean()

    # ── training config log ──
    log("=" * 60)
    log("TRAINING CONFIGURATION  (White Birds: GREG / MEGRT / SNEG / WHIB)")
    log("=" * 60)
    log(f"  model:            {MODEL_NAME}")
    log(f"  loss:             CrossEntropy")
    log(f"  epochs:           {EPOCHS}")
    log(f"  warmup_epochs:    {WARMUP_EPOCHS}")
    log(f"  lr:               {LR}")
    log(f"  weight_decay:     {WEIGHT_DECAY}")
    log(f"  accum_steps:      {ACCUM_STEPS}")
    log(f"  AMP:              {AMP}")
    log(f"  device:           {DEVICE}")
    log(f"  hardware:         {HARDWARE}")
    log(f"  image_root:       {IMAGE_ROOT}")
    log(f"  input_size:       {INPUT_SIZE}")
    log(f"  batch_train:      {BATCH_TRAIN}")
    log(f"  batch_eval:       {BATCH_EVAL}")
    log(f"  use_sampler:      {USE_SAMPLER}")
    log(f"  num_workers:      {NUM_WORKERS}")
    log(f"  max_per_class:    {MAX_PER_CLASS}")
    log(f"  target_species:   {TARGET_SPECIES}")
    log(f"  num_classes:      {num_classes}")
    log(f"  classes:          {classes}")
    log(f"  train_size:       {len(train_rows)}")
    log(f"  val_size:         {len(val_rows)}")
    log(f"  test_size:        {len(test_rows)}")
    log("=" * 60)

    # Split composition
    comp_train = split_composition(ds_train, classes)
    comp_val   = split_composition(ds_val,   classes)
    comp_test  = split_composition(ds_test,  classes)
    with open(OUT_DIR / "split_composition.json", "w", encoding="utf-8") as f:
        json.dump({"train": comp_train, "val": comp_val, "test": comp_test}, f, indent=2)
    log("[info] saved split_composition.json")

    # ── model / optimizer ──
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes).to(DEVICE)
    opt   = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=max(EPOCHS - WARMUP_EPOCHS, 1))
    scaler = GradScaler(device="cuda", enabled=AMP)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_metric = float("-inf")

    # ── training loop ──
    for ep in range(1, EPOCHS + 1):
        ep_start = time.perf_counter()

        # warmup / schedule
        if ep <= WARMUP_EPOCHS:
            warmup_factor = ep / max(WARMUP_EPOCHS, 1)
            lr_now = LR * warmup_factor
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        else:
            sched.step()
            lr_now = opt.param_groups[0]["lr"]

        # train
        model.train()
        running_loss = running_correct = running_count = 0
        opt.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(dl_train, start=1):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            with autocast(device_type="cuda", enabled=AMP):
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb)

            scaler.scale(loss / ACCUM_STEPS).backward()
            if step % ACCUM_STEPS == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                pred = logits.argmax(1)
                running_correct += (pred == yb).sum().item()
                running_count += yb.size(0)
                running_loss += loss.item() * yb.size(0)

        train_acc  = running_correct / max(1, running_count)
        train_loss = running_loss / max(1, running_count)

        # validation
        t_val = time.perf_counter()
        model.eval()
        v_loss = v_correct = v_count = 0
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with autocast(device_type="cuda", enabled=AMP):
                    logits = model(xb)
                    loss = nn.functional.cross_entropy(logits, yb)
                v_loss += loss.item() * yb.size(0)
                preds = logits.argmax(1)
                v_correct += (preds == yb).sum().item()
                v_count += yb.size(0)
                y_val_true.extend(yb.cpu().tolist())
                y_val_pred.extend(preds.cpu().tolist())
        val_acc  = v_correct / max(1, v_count)
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

        if val_macro_f1 > best_metric:
            best_metric = val_macro_f1
            torch.save({"model": model.state_dict(), "classes": classes, "name": MODEL_NAME}, CKPT_PATH)
            log(f"[info] new best val macro-F1 {best_metric:.3f} at epoch {ep:02d}")

    # ── curves ──
    plot_curves(history, OUT_DIR / "curves.pdf")
    log("[info] saved curves.pdf")

    # ── final evaluation ──
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    val_summary, val_y, val_p = evaluate_full(
        model, dl_val, classes, header="Validation report", save_prefix="val", out_dir=OUT_DIR
    )
    test_summary, test_y, test_p = evaluate_full(
        model, dl_test, classes, header="Test report", save_prefix="test", out_dir=OUT_DIR
    )

    plot_two_cms(val_y, val_p, test_y, test_p, classes, OUT_DIR / "val_test_cms.pdf",
                 titles=("Validation", "Test"))
    log("[info] saved val_test_cms.pdf")

    total_time = time.perf_counter() - run_start
    log(f"[info] total run time: {str(timedelta(seconds=int(total_time)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin Tiny: White Birds (GREG/MEGRT/SNEG/WHIB)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    MAX_PER_CLASS = args.max_per_class
    ACCUM_STEPS = args.accum_steps
    MODEL_NAME = args.model_name
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = args.device

    main()
