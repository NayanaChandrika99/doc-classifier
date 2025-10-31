#!/usr/bin/env bash
# Bootstraps the Tennr classifier development environment using uv.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install via 'brew install uv' and rerun." >&2
  exit 1
fi

UV_VENV_DIR=".venv"

if [[ ! -d "${UV_VENV_DIR}" ]]; then
  echo "Creating virtual environment with uv..."
  uv venv "${UV_VENV_DIR}"
else
  echo "Virtual environment already exists at ${UV_VENV_DIR}."
fi

echo "Syncing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Environment ready. Activate with 'source ${UV_VENV_DIR}/bin/activate' or use 'uv run'."
echo "Run 'uv run pytest' to execute tests or 'uv run python -m src.__main__ --show-settings' to inspect configuration."
