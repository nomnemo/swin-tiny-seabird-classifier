#!/usr/bin/env bash
# Setup script for this repo on a fresh machine (e.g., Lambda GPU instance).
#
# What it does:
# 1) Creates a Python virtual environment at ./.venv if it doesn't already exist.
# 2) Activates the virtual environment for the current shell session.
# 3) Upgrades pip inside the venv.
# 4) Installs Python dependencies from requirements.txt into the venv.
# 5) Installs Claude Code CLI (system/user-level) if it is not already installed.
#
# Notes:
# - Python dependencies go into the venv; the Claude CLI is installed outside the venv.
# - Re-running the script is safe: it reuses the existing .venv and only installs Claude if missing.
# - After the script finishes, your venv remains active only for that shell session.

# 0) Configure git identity (global)
GIT_NAME="nomnemo"
GIT_EMAIL="ng53@rice.edu"

CURRENT_NAME="$(git config --global user.name || true)"
CURRENT_EMAIL="$(git config --global user.email || true)"

if [ "$CURRENT_NAME" != "$GIT_NAME" ]; then
  git config --global user.name "$GIT_NAME"
fi

if [ "$CURRENT_EMAIL" != "$GIT_EMAIL" ]; then
  git config --global user.email "$GIT_EMAIL"
fi


set -euo pipefail

VENV_DIR=".venv"

# 1) Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# 2) Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3) Upgrade pip + install python deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) Install Claude Code (CLI) if missing (system/user install, not in venv)
if ! command -v claude >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh | bash
fi

echo "✅ Setup complete. Venv active: $VENV_DIR"
echo "Tip: next time just run: source $VENV_DIR/bin/activate"
