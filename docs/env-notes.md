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
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
bash scripts/run_tests.sh
```

## 2. GPU Machine

| Item | Value |
| --- | --- |
| Provider | AutoDL |
| Instance type | TODO |
| GPU | TODO |
| GPU memory | TODO |
| Driver | TODO |
| CUDA runtime | TODO |
| CUDA compiler / nvcc | TODO |
| PyTorch | TODO |
| PyTorch CUDA | TODO |
| Triton | TODO |
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
| PyTorch CUDA tensor op | TODO | TODO |
| Triton vector add kernel | TODO | TODO |
| CUDA extension compile | TODO | TODO |
| NCU installed | TODO | TODO |
| NCU counter permission | TODO | TODO |

Smoke command:

```bash
python scripts/gpu_smoke.py
```

NCU command:

```bash
ncu --set full --target-processes all python scripts/gpu_smoke.py
```

## 4. NCU / Profiling Decision

Decision:

```text
TODO: Use NCU in Week 4 / fallback to nsys + torch.profiler + analytical bandwidth model.
```

Evidence:

```text
TODO: Paste short success summary or key error text, e.g. ERR_NVGPUCTRPERM.
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

## 5. Version Pin Candidate

Candidate environment:

```text
python==TODO
torch==TODO
triton==TODO
cuda==TODO
```

Reason:

```text
TODO: Keep the simplest combination that passed Triton smoke + CUDA extension compile.
```

## 6. Open Issues

- TODO
