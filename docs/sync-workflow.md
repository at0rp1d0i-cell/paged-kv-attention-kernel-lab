# Sync Workflow

目标：GitHub 作为代码 source of truth，AutoDL 远端作为主要开发和执行环境。本机只保留临时 checkout / 管理入口，不作为长期代码源。

## Recommended Split

- GitHub：代码、README、docs、tests、scripts、轻量 benchmark CSV。
- AutoDL：主要开发、GPU 运行、profiling、benchmark。
- SSH / rsync：只拉回 GPU profiling traces、临时 artifacts、大文件结果、缓存。
- 本机：不作为可运行环境；只做仓库管理、临时查看和结果归档。
- 不进 GitHub：`.venv`、`__pycache__`、`.pytest_cache`、`.DS_Store`、大型 profiler 原始文件。

## GitHub Code Sync

GitHub repo:

```text
https://github.com/at0rp1d0i-cell/paged-kv-attention-kernel-lab
```

Remote AutoDL checkout:

```text
/root/paged-kv-attention-kernel-lab
```

Remote origin:

```text
git@github.com-pkv:at0rp1d0i-cell/paged-kv-attention-kernel-lab.git
```

The remote machine has a deploy key named `autodl-pkv` with read/write access.

Check status:

```bash
git remote -v
git status
```

Normal remote-first flow:

```bash
# remote
cd /root/paged-kv-attention-kernel-lab
git pull --ff-only

# edit / run / verify on GPU

git add .
git commit -m "..."
git push
```

Local checkout is optional and should not be treated as executable truth. If local state is needed:

```bash
git clone https://github.com/at0rp1d0i-cell/paged-kv-attention-kernel-lab.git
```

## Rsync Data Channel

Do not push source code to AutoDL with rsync. `scripts/sync_to_remote.sh` is intentionally disabled so it cannot overwrite the remote Git checkout.

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
REMOTE=autodl-pkv REMOTE_DIR=/root/paged-kv-attention-kernel-lab bash scripts/sync_from_remote.sh
```

## Local Cleanup Policy

After GitHub and AutoDL are verified, the local seed checkout may be removed or archived. Do not delete it until the latest remote commit is pushed and verified on GitHub.

## Rule of Thumb

- If it is source of truth, commit it.
- If it is generated but small and needed for the report, commit it deliberately.
- If it is large, temporary, machine-specific, or reproducible, keep it under `artifacts/`, `results/`, or `profiles/` and move it with rsync.
