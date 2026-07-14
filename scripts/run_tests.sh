#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed."
  echo "Install uv first, then sync the project environment:"
  echo "  python -m pip install uv"
  echo "  UV_HTTP_TIMEOUT=600 uv sync --locked --group dev"
  exit 2
fi

uv run pytest -m "not gpu"
