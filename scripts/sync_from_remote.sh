#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-autodl-pkv}"
REMOTE_DIR="${REMOTE_DIR:-/root/paged-kv-attention-kernel-lab}"

mkdir -p artifacts/remote results profiles

rsync -az \
  "${REMOTE}:${REMOTE_DIR}/docs/env-notes.md" docs/env-notes.md
rsync -az \
  "${REMOTE}:${REMOTE_DIR}/docs/lab-notes/" docs/lab-notes/

rsync -az --ignore-missing-args \
  "${REMOTE}:${REMOTE_DIR}/results/" results/
rsync -az --ignore-missing-args \
  "${REMOTE}:${REMOTE_DIR}/profiles/" profiles/
rsync -az --ignore-missing-args \
  "${REMOTE}:${REMOTE_DIR}/artifacts/" artifacts/remote/

echo "Pulled remote notes/results/profiles from ${REMOTE}:${REMOTE_DIR}"

