#!/usr/bin/env python3
"""Check FlashInfer paged decode against the FP32 dense reference."""

from __future__ import annotations

import os

import torch

from paged_kv_attention.flashinfer_baseline import plan_flashinfer_paged_decode
from paged_kv_attention.reference import default_attention_scale, dense_decode_attention


def main() -> None:
    batch_size = 2
    num_heads = 8
    head_dim = 128
    max_context_len = 47
    page_size = 16
    context_lens = torch.tensor([31, 47], dtype=torch.int32, device="cuda")
    block_tables = torch.tensor([[2, 0, -1], [4, 1, 3]], dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(20260715)
    q = torch.randn(
        batch_size,
        num_heads,
        head_dim,
        generator=generator,
        dtype=torch.float16,
        device="cuda",
    )
    dense_k = torch.randn(
        batch_size,
        max_context_len,
        num_heads,
        head_dim,
        generator=generator,
        dtype=torch.float16,
        device="cuda",
    )
    dense_v = torch.randn(
        dense_k.shape,
        generator=generator,
        dtype=dense_k.dtype,
        device=dense_k.device,
    )
    cache_shape = (5, page_size, num_heads, head_dim)
    k_cache = torch.randn(cache_shape, generator=generator, dtype=torch.float16, device="cuda")
    v_cache = torch.randn(cache_shape, generator=generator, dtype=torch.float16, device="cuda")

    for batch_idx, context_len in enumerate(context_lens.tolist()):
        page_count = (context_len + page_size - 1) // page_size
        for logical_page in range(page_count):
            start = logical_page * page_size
            end = min(start + page_size, context_len)
            physical_page = int(block_tables[batch_idx, logical_page])
            k_cache[physical_page, : end - start] = dense_k[batch_idx, start:end]
            v_cache[physical_page, : end - start] = dense_v[batch_idx, start:end]

    try:
        operation = plan_flashinfer_paged_decode(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            block_size=page_size,
            scale=default_attention_scale(head_dim),
        )
        actual = operation.run()
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"FlashInfer paged decode unavailable: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    expected = dense_decode_attention(q, dense_k, dense_v, context_lens)
    torch.testing.assert_close(actual.to(torch.float32), expected, atol=3e-3, rtol=3e-3)

    import flashinfer

    print("flashinfer", flashinfer.__version__)
    print("torch", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("cuda_home", os.environ["CUDA_HOME"])
    print("gpu", torch.cuda.get_device_name(0))
    print("compute_capability", ".".join(map(str, torch.cuda.get_device_capability(0))))
    print("output", tuple(actual.shape), actual.dtype, bool(torch.isfinite(actual).all()))
    print("max_abs_error", float((actual.to(torch.float32) - expected).abs().max()))
    print("correctness", "passed")


if __name__ == "__main__":
    main()
