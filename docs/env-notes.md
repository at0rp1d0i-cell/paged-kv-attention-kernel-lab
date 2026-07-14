# Environment Notes

记录原则：这里写事实，不写猜测。命令输出如果太长，只保留关键行；失败时保留错误关键词和下一步 fallback。

## 1. Local Machine

| Item | Value |
| --- | --- |
| Date | 2026-07-03 |
| Local OS | macOS 26.5.2 (25F84) |
| Python | Python 3.14.5 (`/opt/homebrew/bin/python3`) |
| Repo path | `/Users/torpedo/Workspace/paged-kv-attention-kernel-lab` |
| Git commit | Not initialized yet |

Commands:

```bash
python --version
which python
git rev-parse --short HEAD
```

Local validation:

| Check | Status | Notes |
| --- | --- | --- |
| Python syntax compile | Passed | `scripts/gpu_smoke.py`, package init, and import test compile locally. |
| CPU pytest | Not run | Local Python 3.14 environment does not currently have `pytest` installed. |
| Test entrypoint | Passed as diagnostic | `scripts/run_tests.sh` exits with setup instructions when `pytest` is missing. |

Local dev setup:

```bash
UV_HTTP_TIMEOUT=600 uv sync --locked --group dev
bash scripts/run_tests.sh
```

## 2. GPU Machine

| Item | Value |
| --- | --- |
| Provider | AutoDL |
| Instance type | RTX 5090 container (`autodl-container-88e54ca00f-12d54101`) |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU memory | 32607 MiB reported by `nvidia-smi`; 33668988928 bytes reported by PyTorch |
| Compute capability | 12.0 |
| Driver | 580.105.08 |
| CUDA runtime | 13.0 reported by `nvidia-smi`; 12.8 used by PyTorch |
| CUDA compiler / nvcc | 12.8, V12.8.93 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA | 12.8 |
| Triton | 3.4.0 |
| Python | 3.12.3 from uv-managed `.venv` |
| Package manager | uv 0.11.26 |
| Dev tools | pytest 9.1.1, ruff 0.15.20, ninja 1.13.0 |
| FlashInfer | Not required in Week 0 |

Commands:

```bash
nvidia-smi
nvcc --version || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
try:
    import triton
    print("triton", triton.__version__)
except Exception as exc:
    print("triton import failed", repr(exc))
PY
```

## 3. Week 0 GPU Smoke Result

| Check | Status | Notes |
| --- | --- | --- |
| PyTorch CUDA tensor op | Passed | `uv run python scripts/gpu_smoke.py` completed tensor op on NVIDIA GeForce RTX 5090. |
| Triton vector add kernel | Passed | Triton 3.4.0 JIT compiled and launched vector-add kernel. |
| CUDA extension compile | Passed | `torch.utils.cpp_extension.load` compiled and ran CUDA extension after installing `ninja`. |
| CPU pytest | Passed | `bash scripts/run_tests.sh`: 1 test passed. |
| Ruff lint | Passed | `uv run ruff check .`: all checks passed. |
| NCU installed | Present | `/usr/local/cuda-12.8/bin/ncu`, Nsight Compute 2025.1.1.0. |
| NCU counter permission | Blocked | `ERR_NVGPUCTRPERM` when wrapping `uv run python scripts/gpu_smoke.py`; `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`. |
| Profiling fallback | Enabled | Container-only permission means NCU counters are non-fatal; use CUDA events, `torch.profiler`, and analytical bandwidth. |

Smoke command:

```bash
uv run python scripts/gpu_smoke.py
```

NCU command:

```bash
ncu --set full --target-processes all uv run python scripts/gpu_smoke.py
```

## 4. NCU / Profiling Decision

Decision:

```text
Nsight Compute is installed but GPU performance counter access is blocked on this container.
Use CUDA events for latency measurements now. For Week 4 profiling, either enable NVIDIA GPU
performance counter permissions or fallback to torch.profiler + analytical bandwidth model.
`nsys` is not installed in the current image.
```

Evidence:

```text
ncu --set full --target-processes all uv run python scripts/gpu_smoke.py
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
The smoke workload itself still completed: PyTorch CUDA ok, Triton vector add ok, CUDA extension compile/run ok.
```

Fallback if NCU is unavailable:

- Use CUDA events for latency.
- Use `torch.profiler` for operator-level timeline.
- Use `nsys` if available for timeline-level profiling.
- Compute effective bandwidth analytically:

```text
bytes_read = seq_len * 2 * num_kv_heads * head_dim * dtype_size
effective_bandwidth = bytes_read / latency
bandwidth_utilization = effective_bandwidth / hardware_peak_bandwidth
```

Container-only permission note:

```text
The current user is root inside the container, but the host driver still restricts
GPU performance counters. Treat ERR_NVGPUCTRPERM as expected on this machine.
Do not block Week 1-3 correctness or benchmark harness work on NCU access.
```

## 5. Version Pin Candidate

Candidate environment:

```text
uv==0.11.26
python==3.12.3
numpy>=2.0
torch==2.8.0+cu128
triton==3.4.0
cuda runtime reported by torch==12.8
cuda compiler==12.8, V12.8.93
ninja==1.13.0
```

Reason:

```text
This is the simplest current uv-managed environment that passed import tests, CPU pytest, Ruff,
PyTorch CUDA tensor op, Triton JIT launch, and CUDA extension compile/run.
```

## 6. Open Issues

- `ncu` counter collection is blocked by `ERR_NVGPUCTRPERM`; needs host/container permission change for full Nsight Compute profiling.
- `nsys` is not installed in the current image.
- Optional baseline `flashinfer-python==0.6.14` installs from the `baseline` dependency group, but paged decode cannot plan on RTX 5090 / SM 12.0 with PyTorch CUDA 12.8. FlashInfer reports that SM 12.x requires CUDA >= 12.9, then its JIT architecture check raises `RuntimeError: FlashInfer requires GPUs with sm75 or higher`.
- Project commands now use `uv sync --locked --group dev` and `uv run`; the earlier root/base conda `pip install -e '.[dev]'` path is superseded.
- The project-level default index is `https://pypi.tuna.tsinghua.edu.cn/simple` for ordinary PyPI packages; PyTorch remains pinned to the explicit cu128 PyTorch index.
