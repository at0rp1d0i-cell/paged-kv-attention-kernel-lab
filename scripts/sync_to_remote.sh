#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-autodl-pkv}"
REMOTE_DIR="${REMOTE_DIR:-/root/paged-kv-attention-kernel-lab}"

rsync -az --delete \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude ".ruff_cache/" \
  --exclude ".DS_Store" \
  --exclude "artifacts/" \
  --exclude "results/" \
  --exclude "profiles/" \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo "Synced local workspace to ${REMOTE}:${REMOTE_DIR}"

