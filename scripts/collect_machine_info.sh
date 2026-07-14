#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-docs/machine-info.md}"
PYBIN="$(command -v python || command -v python3 || true)"
NOW_UTC="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

mkdir -p "$(dirname "${OUT}")"

{
  echo "# Machine Info（机器信息）"
  echo
  echo "Last updated（更新时间）: ${NOW_UTC}"
  echo
  echo "> This file is a machine snapshot（机器快照）. Re-run \`bash scripts/collect_machine_info.sh\` whenever the project moves to a different machine, GPU instance（GPU 实例）, or CUDA environment（CUDA 环境）."
  echo
  echo "## Update Checklist（更新清单）"
  echo
  echo "- Re-run this snapshot after changing machine or GPU instance（GPU 实例）."
  echo "- Copy key GPU/CUDA facts into \`docs/env-notes.md\` when they affect benchmark（基准测试） or profiling（性能剖析） decisions."
  echo "- Keep benchmark CSV files tied to the machine snapshot used for that run."
  echo

  echo "## System（系统）"
  echo
  echo '```text'
  echo "hostname: $(hostname 2>/dev/null || true)"
  echo "user: $(id -un 2>/dev/null || true)"
  echo "uid/gid: $(id 2>/dev/null || true)"
  echo "working_dir: $(pwd)"
  echo "kernel: $(uname -a 2>/dev/null || true)"
  echo
  if [ -f /etc/os-release ]; then
    cat /etc/os-release
  else
    echo "/etc/os-release: not found"
  fi
  echo '```'
  echo

  echo "## CPU（处理器）"
  echo
  echo '```text'
  if command -v lscpu >/dev/null 2>&1; then
    lscpu
  else
    echo "lscpu: not found"
    sysctl -a 2>/dev/null | grep -E 'machdep.cpu|hw.ncpu|hw.memsize' || true
  fi
  echo '```'
  echo

  echo "## Memory（内存）"
  echo
  echo '```text'
  if command -v free >/dev/null 2>&1; then
    free -h
  else
    echo "free: not found"
    vm_stat 2>/dev/null || true
  fi
  echo
  if [ -f /proc/meminfo ]; then
    grep -E 'MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree' /proc/meminfo || true
  fi
  echo '```'
  echo

  echo "## Disk And Filesystem（磁盘和文件系统）"
  echo
  echo '```text'
  df -h . /tmp 2>/dev/null || df -h . 2>/dev/null || true
  echo
  if command -v lsblk >/dev/null 2>&1; then
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE,MODEL
  else
    echo "lsblk: not found"
  fi
  echo '```'
  echo

  echo "## GPU And CUDA（GPU 和 CUDA）"
  echo
  echo '```text'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
    echo
    echo "nvidia-smi query:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,pci.bus_id,power.limit,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,compute_cap --format=csv,noheader 2>/dev/null \
      || nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,pci.bus_id,power.limit,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory --format=csv,noheader 2>/dev/null \
      || true
  else
    echo "nvidia-smi: not found"
  fi
  echo
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
  else
    echo "nvcc: not found"
  fi
  echo
  if command -v ncu >/dev/null 2>&1; then
    ncu --version
  else
    echo "ncu: not found"
  fi
  echo
  if command -v nsys >/dev/null 2>&1; then
    nsys --version
  else
    echo "nsys: not found"
  fi
  echo
  if command -v lspci >/dev/null 2>&1; then
    lspci | grep -Ei 'nvidia|vga|3d|display' || true
  else
    echo "lspci: not found"
  fi
  echo '```'
  echo

  echo "## Python And Packages（Python 和包）"
  echo
  echo '```text'
  if command -v uv >/dev/null 2>&1; then
    uv --version
  else
    echo "uv: not found"
  fi
  echo
  if command -v python >/dev/null 2>&1; then
    python --version
    which python
  else
    echo "python: not found"
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 --version
    which python3
  else
    echo "python3: not found"
  fi
  echo
  if [ -n "${PYBIN}" ]; then
    "${PYBIN}" - <<'PY'
import importlib.util
from importlib import metadata
import os
import platform
import sys

print("sys.executable", sys.executable)
print("sys.version", sys.version.replace("\n", " "))
print("platform", platform.platform())
print("python_prefix", sys.prefix)
print("virtual_env", os.environ.get("VIRTUAL_ENV", ""))

for name in ["torch", "triton", "pytest", "ruff", "ninja", "flashinfer"]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"{name}: not installed")
        continue
    try:
        mod = __import__(name)
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = getattr(mod, "__version__", "unknown")
        print(f"{name}: {version}")
        if name == "torch":
            print("torch.cuda", getattr(mod.version, "cuda", None))
            print("torch.cuda.is_available", mod.cuda.is_available())
            print("torch.cuda.device_count", mod.cuda.device_count())
            if mod.cuda.is_available():
                for i in range(mod.cuda.device_count()):
                    props = mod.cuda.get_device_properties(i)
                    print(f"torch.cuda.device[{i}].name", props.name)
                    print(f"torch.cuda.device[{i}].capability", f"{props.major}.{props.minor}")
                    print(f"torch.cuda.device[{i}].total_memory", props.total_memory)
    except Exception as exc:
        print(f"{name}: import failed: {exc!r}")
PY
  else
    echo "No Python interpreter found for package probe."
  fi
  echo '```'
  echo

  echo "## Project Environment（项目环境）"
  echo
  echo '```text'
  if [ -f pyproject.toml ]; then
    sed -n '1,160p' pyproject.toml
  else
    echo "pyproject.toml: not found"
  fi
  echo '```'
  echo

  echo "## Git And Repo State（Git 和仓库状态）"
  echo
  echo '```text'
  git rev-parse --show-toplevel 2>/dev/null || true
  git rev-parse --short HEAD 2>/dev/null || true
  git branch --show-current 2>/dev/null || true
  git status --short 2>/dev/null || true
  echo
  echo "Tracked project files:"
  git ls-files 2>/dev/null | sed -n '1,200p' || true
  echo '```'
  echo

  echo "## Validation Commands（验证命令）"
  echo
  echo '```bash'
  echo "UV_HTTP_TIMEOUT=600 uv sync --locked --group dev"
  echo "bash scripts/check_env.sh"
  echo "bash scripts/run_tests.sh"
  echo "uv run python scripts/gpu_smoke.py"
  echo '```'
} > "${OUT}"

echo "Wrote ${OUT}"
