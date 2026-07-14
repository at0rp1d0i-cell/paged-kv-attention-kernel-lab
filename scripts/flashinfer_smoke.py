#!/usr/bin/env python3
"""Probe whether FlashInfer paged decode can plan and run on this GPU stack."""

from __future__ import annotations

import sys

import torch


def main() -> None:
    try:
        import flashinfer
    except ImportError as exc:
        raise SystemExit(
            "FlashInfer is not installed; run `uv sync --locked --group baseline`."
        ) from exc

    batch_size = 1
    num_heads = 8
    head_dim = 128
    context_len = 128
    page_size = 16
    num_pages = context_len // page_size

    print("flashinfer", getattr(flashinfer, "__version__", "unknown"))
    print("torch", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("gpu", torch.cuda.get_device_name(0))
    print("compute_capability", ".".join(map(str, torch.cuda.get_device_capability(0))))

    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="auto")
    indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda")
    q = torch.randn(
        batch_size,
        num_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    kv = torch.randn(
        num_pages,
        2,
        page_size,
        num_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    try:
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            data_type=torch.float16,
        )
        out = wrapper.run(q, kv)
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"FlashInfer paged decode unavailable: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    print("output", tuple(out.shape), out.dtype, bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    sys.exit(main())
