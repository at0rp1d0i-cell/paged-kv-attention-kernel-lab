#!/usr/bin/env bash
set -euo pipefail

echo "== Toolchain =="
if ! command -v uv >/dev/null 2>&1; then
  echo "uv: missing"
  echo "Install uv first:"
  echo "  python -m pip install uv"
  exit 2
fi
uv --version

echo
echo "== Python / package environment =="
uv run python - <<'PY'
import importlib.metadata as md
import os
import sys

import numpy
import torch
import triton

print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("triton", triton.__version__)
for name in ["pytest", "ruff", "ninja", "paged-kv-attention-kernel-lab"]:
    print(name, md.version(name))

print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    capability = f"{props.major}.{props.minor}"
    print("gpu", props.name)
    print("compute capability", capability)
    print("TORCH_CUDA_ARCH_LIST", os.environ.get("TORCH_CUDA_ARCH_LIST") or capability)
PY

echo
echo "== NVIDIA / CUDA tools =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null \
    || nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    || nvidia-smi
else
  echo "nvidia-smi: missing"
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | sed -n '1,4p'
else
  echo "nvcc: missing"
fi

echo
echo "== NCU permission probe =="
if command -v ncu >/dev/null 2>&1; then
  ncu --version | sed -n '1,4p'
  if [ -r /proc/driver/nvidia/params ]; then
    if command -v rg >/dev/null 2>&1; then
      rg -n "RmProfilingAdminOnly|RestrictProfiling" /proc/driver/nvidia/params || true
    else
      grep -nE "RmProfilingAdminOnly|RestrictProfiling" /proc/driver/nvidia/params || true
    fi
  fi

  tmp_log="$(mktemp)"
  set +e
  ncu --set full --target-processes all uv run python - <<'PY' >"${tmp_log}" 2>&1
import torch

x = torch.randn(1024, device="cuda")
y = x * 2
torch.cuda.synchronize()
print(float(y[0].item()))
PY
  ncu_status=$?
  set -e

  if grep -q "ERR_NVGPUCTRPERM" "${tmp_log}"; then
    echo "ncu counters: blocked (ERR_NVGPUCTRPERM)"
    echo "fallback: use CUDA events for latency, torch.profiler for timeline, and analytical bandwidth."
  elif [ "${ncu_status}" -eq 0 ]; then
    echo "ncu counters: available"
  else
    echo "ncu probe: failed with status ${ncu_status}"
    sed -n '1,120p' "${tmp_log}"
  fi
  rm -f "${tmp_log}"
else
  echo "ncu: missing"
  echo "fallback: use CUDA events for latency, torch.profiler for timeline, and analytical bandwidth."
fi

echo
echo "== Validation commands =="
echo "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple UV_HTTP_TIMEOUT=600 uv sync --locked --group dev"
echo "bash scripts/check_env.sh"
echo "bash scripts/run_tests.sh"
echo "uv run python scripts/gpu_smoke.py"
