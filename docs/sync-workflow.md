# Sync Workflow

目标：本机和 AutoDL 远端保持一个清晰的数据通道，不把代码同步、实验产物同步和临时缓存混在一起。

## Recommended Split

- GitHub：代码、README、docs、tests、scripts、轻量 benchmark CSV。
- SSH / rsync：GPU profiling traces、临时 artifacts、大文件结果、缓存。
- 不同步：`.venv`、`__pycache__`、`.pytest_cache`、`.DS_Store`、大型 profiler 原始文件。

## GitHub Code Sync

Local and remote should share the same `origin`:

```bash
git remote -v
git status
```

Typical flow:

```bash
# local
git add .
git commit -m "..."
git push

# remote
cd /root/paged-kv-attention-kernel-lab
git pull --ff-only
```

When editing directly on the remote machine, commit and push from remote, then pull locally.

## Rsync Data Channel

Push current local workspace snapshot to AutoDL:

```bash
bash scripts/sync_to_remote.sh
```

Pull remote notes and experiment outputs back:

```bash
bash scripts/sync_from_remote.sh
```

The default remote is:

```text
autodl-pkv:/root/paged-kv-attention-kernel-lab
```

Override if needed:

```bash
REMOTE=autodl-pkv REMOTE_DIR=/root/paged-kv-attention-kernel-lab bash scripts/sync_to_remote.sh
```

## Rule of Thumb

- If it is source of truth, commit it.
- If it is generated but small and needed for the report, commit it deliberately.
- If it is large, temporary, machine-specific, or reproducible, keep it under `artifacts/`, `results/`, or `profiles/` and move it with rsync.

