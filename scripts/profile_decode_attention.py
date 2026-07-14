#!/usr/bin/env python3
"""Profile representative decode-attention providers with torch.profiler."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from paged_kv_attention.reference import default_attention_scale
from paged_kv_attention.triton_decode import (
    _launch_dense_decode_attention_triton,
    _launch_paged_decode_attention_triton,
)
from run_benchmarks import make_paged_cache


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=16384)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--block-t", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("profiles"))
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.head_dim != 128:
        raise SystemExit("the current Triton kernel requires head_dim=128")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    generator = torch.Generator(device="cuda").manual_seed(20260715)
    q = torch.randn(
        args.batch_size,
        args.num_heads,
        args.head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    dense_k = torch.randn(
        args.batch_size,
        args.context_len,
        args.num_heads,
        args.head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    dense_v = torch.randn_like(dense_k)
    context_lens = torch.full(
        (args.batch_size,),
        args.context_len,
        device="cuda",
        dtype=torch.int32,
    )
    k_cache, v_cache, block_tables = make_paged_cache(
        dense_k,
        dense_v,
        context_lens,
        block_size=args.block_size,
        seed=20260715,
    )
    scale = default_attention_scale(args.head_dim)
    dense_out = torch.empty_like(q, dtype=torch.float32)
    paged_out = torch.empty_like(q, dtype=torch.float32)

    operations = {
        "dense_triton": lambda: _launch_dense_decode_attention_triton(
            q,
            dense_k,
            dense_v,
            context_lens,
            dense_out,
            max_context_len=args.context_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            scale=scale,
            block_t=args.block_t,
        ),
        "paged_triton": lambda: _launch_paged_decode_attention_triton(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            paged_out,
            max_blocks_per_seq=block_tables.shape[1],
            block_size=args.block_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            scale=scale,
            block_t=args.block_t,
        ),
        "pytorch_dense_sdpa": lambda: F.scaled_dot_product_attention(
            q.unsqueeze(2),
            dense_k.transpose(1, 2),
            dense_v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"decode_B{args.batch_size}_S{args.context_len}"
    text_path = args.output_dir / f"{stem}.txt"
    trace_path = args.output_dir / f"{stem}.json"

    for _ in range(args.warmup):
        for operation in operations.values():
            operation()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profiler:
        for _ in range(args.iterations):
            for name, operation in operations.items():
                with torch.profiler.record_function(name):
                    operation()
        torch.cuda.synchronize()

    table = profiler.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
    text_path.write_text(table + "\n", encoding="utf-8")
    profiler.export_chrome_trace(str(trace_path))
    print(table)
    print(f"wrote {text_path}")
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
