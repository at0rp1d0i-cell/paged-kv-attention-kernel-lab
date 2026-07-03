"""Week 0 GPU smoke checks.

This script is intentionally small. It verifies:
1. PyTorch can run a CUDA tensor operation.
2. Triton can JIT and launch a simple vector-add kernel.
3. A minimal CUDA extension can compile and run.

Run on the rented GPU machine:
    python scripts/gpu_smoke.py

Optionally profile it:
    ncu --set full --target-processes all python scripts/gpu_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def check_torch_cuda() -> None:
    import torch

    print("== PyTorch CUDA ==")
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    device = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(device))
    x = torch.arange(1024, device=device, dtype=torch.float32)
    y = x * 2.0 + 1.0
    torch.cuda.synchronize()
    expected = torch.arange(1024, device=device, dtype=torch.float32) * 2.0 + 1.0
    torch.testing.assert_close(y, expected)
    print("torch cuda tensor op: ok")


def check_triton_vector_add() -> None:
    import torch
    import triton
    import triton.language as tl

    print("\n== Triton Vector Add ==")
    print("triton:", triton.__version__)

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    n = 1 << 20
    block_size = 256
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, block_size),)
    add_kernel[grid](x, y, out, n, block_size)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y)
    print("triton vector add: ok")


def check_cuda_extension() -> None:
    import torch
    from torch.utils.cpp_extension import load

    print("\n== CUDA Extension Compile ==")
    source = r'''
#include <torch/extension.h>

__global__ void add_one_kernel(const float* x, float* y, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] + 1.0f;
  }
}

torch::Tensor add_one(torch::Tensor x) {
  auto y = torch::empty_like(x);
  const int threads = 256;
  const int64_t n = x.numel();
  const int blocks = (n + threads - 1) / threads;
  add_one_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one, "Add one to a CUDA float tensor");
}
'''
    with tempfile.TemporaryDirectory(prefix="week0_cuda_ext_") as tmp:
        tmp_path = Path(tmp)
        cu_path = tmp_path / "add_one_extension.cu"
        cu_path.write_text(source)
        module = load(
            name="week0_add_one_extension",
            sources=[str(cu_path)],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
        x = torch.randn(4096, device="cuda", dtype=torch.float32)
        y = module.add_one(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, x + 1.0)
    print("cuda extension compile/run: ok")


def main() -> None:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "")
    check_torch_cuda()
    check_triton_vector_add()
    check_cuda_extension()
    print("\nweek0 gpu smoke: ok")


if __name__ == "__main__":
    main()
