#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MSG'
sync_to_remote.sh is intentionally disabled.

GitHub is the source of truth for code. The AutoDL machine should update code with:

  cd /root/paged-kv-attention-kernel-lab
  git pull --ff-only

Use scripts/sync_from_remote.sh only for pulling experiment outputs back to the local machine.
MSG

exit 2
