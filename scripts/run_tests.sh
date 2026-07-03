#!/usr/bin/env bash
set -euo pipefail

if ! python -c "import pytest" >/dev/null 2>&1; then
  echo "pytest is not installed."
  echo "Set up a local dev environment first:"
  echo "  python -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  python -m pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu"
  exit 2
fi

python -m pytest -m "not gpu"
