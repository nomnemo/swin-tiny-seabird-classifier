"""
Swin Tiny trainer for Classifier 1 (15-class pipeline stage).

This is the first-stage classifier in the detection+classification pipeline:
  Co-DETR detection -> Classifier 1 (this) -> Classifier 2 (tern sub-classifier)

Uses the original 21-class data splits and merges into 15 classes:
  - 13 individual: BCNH, BLSK, BRPE, BRPEC, CAEG, GBHE, LAGU, REEG, REEGWM,
                    ROSP, TRHE, WHIBC
  - 3 merged:      GREG_SNEG_WHIB, MTRN_ROTE_SATE, MEGRT_OTHR_OTHERS

MTRN_ROTE_SATE is later split by Classifier 2 into ROTE, SATE, OTHER_TERNS.

Usage:
    python scripts/SwinTrainerClassifier1.py
    python scripts/SwinTrainerClassifier1.py --epochs 30 --lr 1e-4
"""

import time
from datetime import timedelta
from pathlib import Path
import json
import argparse
from typing import Optional

import numpy as np
import torch, timm
from torch import nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    classification_report,
    f1_score,
)
from scripts.DataLoader import set_up_data_loaders
from scripts.SwinTrainer import (
    make_run_dir_name,
    plot_curves,
    plot_two_cms,
    eval_collect,
    compute_map_ovr,
    split_composition,
)

# ===== config =====
MODEL_NAME   = "swin_tiny_patch4_window7_224"
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 0.01
ACCUM_STEPS  = 1
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
AMP          = True
MAX_PER_CLASS = 1000

# Merge groups: 21 classes -> 15 classes
# - Visually similar egrets/ibis merged (same as 14-class run, minus MEGRT)
# - Terns + mixed terns merged (for Classifier 2 to split later)
# - Catch-all for mixed/other classes
MERGE_GROUPS = [
    ['GREG', 'SNEG', 'WHIB'],          # -> GREG_SNEG_WHIB
    ['MTRN', 'ROTE', 'SATE'],          # -> MTRN_ROTE_SATE
    ['MEGRT', 'OTHR', 'OTHERS'],        # -> MEGRT_OTHR_OTHERS
]

# Output
OUT_DIR = Path("runs_classifier1")
OUT_DIR.mkdir(exist_ok=True)
LOG_PATH: Optional[Path] = None
CKPT_PATH = "best.pt"

# Learning rate warmup
WARMUP_EPOCHS = 10

# Hardware
HARDWARE = "1x NVIDIA A10 (Lambda)"
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

    # ── data loaders (original 21-class splits, merged to 15) ──
    dl_train, dl_val, dl_test, meta = set_up_data_loaders(
        max_per_class=MAX_PER_CLASS,
        merge_groups=MERGE_GROUPS,
    )

    classes = meta["classes"]
    num_classes = len(classes)
    dl_cfg = meta["dataloader_config"]

    # ── training config log ──
    log("=" * 60)
    log("TRAINING CONFIGURATION  (Classifier 1 – 15 classes)")
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
    log(f"  input_size:       {dl_cfg['input_size']}")
    log(f"  batch_train:      {dl_cfg['batch_train']}")
    log(f"  batch_eval:       {dl_cfg['batch_eval']}")
    log(f"  use_sampler:      {dl_cfg['use_sampler']}")
    log(f"  num_workers:      {dl_cfg['num_workers']}")
    log(f"  max_per_class:    {dl_cfg['max_per_class']}")
    log(f"  merge_groups:     {dl_cfg['merge_groups']}")
    if dl_cfg['merge_groups']:
        for group in dl_cfg['merge_groups']:
            merged_name = "_".join(sorted(group))
            log(f"    {', '.join(group)} -> {merged_name}")
    log(f"  num_classes:      {num_classes}")
    log(f"  classes:          {classes}")
    log(f"  train_size:       {meta['sizes']['train']}")
    log(f"  val_size:         {meta['sizes']['val']}")
    log(f"  test_size:        {meta['sizes']['test']}")
    log("=" * 60)

    # split composition
    comp_train = split_composition(dl_train.dataset, classes)
    comp_val   = split_composition(dl_val.dataset,   classes)
    comp_test  = split_composition(dl_test.dataset,  classes)
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
    parser = argparse.ArgumentParser(description="Train Swin Tiny: Classifier 1 (15 classes)")
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
