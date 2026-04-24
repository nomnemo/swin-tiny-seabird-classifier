#!/usr/bin/env bash
# Train the two sub-classifiers back-to-back.
# Run from the project root:   bash scripts/helper/run_subclassifiers.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[$(date '+%H:%M:%S')] ===== 1/2: subclass_terns ====="
python -m scripts.SwinTrainer2025 --experiment subclass_terns "$@"

echo "[$(date '+%H:%M:%S')] ===== 2/2: subclass_large_white_birds ====="
python -m scripts.SwinTrainer2025 --experiment subclass_large_white_birds "$@"

echo "[$(date '+%H:%M:%S')] ===== both runs done ====="
