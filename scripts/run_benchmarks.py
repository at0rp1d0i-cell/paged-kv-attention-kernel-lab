#!/usr/bin/env python3
"""Run reproducible decode-attention latency benchmarks and write CSV results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import math
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
import torch.nn.functional as F

from paged_kv_attention.benchmark_utils import (
    LatencyStats,
    analytical_kv_bytes,
    bandwidth_utilization_percent,
    collect_environment_metadata,
    effective_bandwidth_gbps,
    measure_cuda_latency,
    measure_synchronized_wall_latency,
)
from paged_kv_attention.block_table import make_random_block_tables
from paged_kv_attention.reference import default_attention_scale, dense_decode_attention
from paged_kv_attention.triton_decode import (
    _launch_dense_decode_attention_triton,
    _launch_paged_decode_attention_triton,
    dense_decode_attention_triton,
    paged_decode_attention_triton,
)


CSV_FIELDS = [
    "run_id",
    "timestamp_utc",
    "provider",
    "measurement_scope",
    "timing_method",
    "batch_size",
    "num_query_heads",
    "num_kv_heads",
    "context_len",
    "block_size",
    "head_dim",
    "dtype",
    "block_t",
    "warmup",
    "repeat",
    "p50_ms",
    "p95_ms",
    "mean_ms",
    "min_ms",
    "max_ms",
    "analytical_kv_bytes",
    "effective_bandwidth_p50_gbps",
    "effective_bandwidth_p95_gbps",
    "peak_memory_bandwidth_gbps",
    "bandwidth_utilization_p50_pct",
    "bandwidth_utilization_p95_pct",
    "tokens_per_second_p50",
    "correctness_guard",
    "gpu_name",
    "gpu_compute_capability",
    "gpu_memory_bytes",
    "driver_version",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "clock_state",
    "python_version",
    "pytorch_version",
    "pytorch_cuda_version",
    "triton_version",
]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_providers(value: str) -> list[str]:
    providers = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"dense_triton", "paged_triton", "pytorch_dense_sdpa", "pytorch_paged_reference"}
    unknown = sorted(set(providers) - valid)
    if not providers or unknown:
        raise argparse.ArgumentTypeError(f"unknown providers: {', '.join(unknown) or 'none'}")
    return providers


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=parse_int_list, default=[1, 4, 16])
    parser.add_argument("--contexts", type=parse_int_list, default=[128, 512, 2048, 8192, 16384])
    parser.add_argument("--block-sizes", type=parse_int_list, default=[16, 32])
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-t", type=int, default=128)
    parser.add_argument(
        "--peak-bandwidth-gbps",
        type=float,
        default=None,
        help="Optional nominal hardware peak used to calculate utilization percentages.",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--providers",
        type=parse_providers,
        default=["dense_triton", "paged_triton", "pytorch_dense_sdpa"],
    )
    parser.add_argument(
        "--reference-max-context",
        type=int,
        default=128,
        help="Skip the intentionally slow paged reference above this context length.",
    )
    parser.add_argument(
        "--reference-repeat",
        type=int,
        default=3,
        help="Use fewer wall-clock samples for the intentionally slow Python reference.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-correctness-guard", action="store_true")
    return parser


def make_paged_cache(
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack equal-length dense K/V into a random-order paged cache outside timing."""

    batch_size, context_len, num_heads, head_dim = dense_k.shape
    block_tables, num_blocks = make_random_block_tables(
        context_lens,
        block_size=block_size,
        seed=seed,
        device=dense_k.device,
    )
    block_tables = block_tables.to(torch.int32).contiguous()
    blocks_per_seq = math.ceil(context_len / block_size)
    padded_context = blocks_per_seq * block_size

    if padded_context != context_len:
        pad_shape = (batch_size, padded_context - context_len, num_heads, head_dim)
        dense_k = torch.cat(
            [dense_k, torch.randn(pad_shape, device=dense_k.device, dtype=dense_k.dtype)]
        )
        dense_v = torch.cat(
            [dense_v, torch.randn(pad_shape, device=dense_v.device, dtype=dense_v.dtype)]
        )

    logical_k = dense_k.view(batch_size * blocks_per_seq, block_size, num_heads, head_dim)
    logical_v = dense_v.view(batch_size * blocks_per_seq, block_size, num_heads, head_dim)
    physical_ids = block_tables[:, :blocks_per_seq].reshape(-1).to(torch.long)

    cache_shape = (num_blocks, block_size, num_heads, head_dim)
    k_cache = torch.randn(cache_shape, device=dense_k.device, dtype=dense_k.dtype)
    v_cache = torch.randn(cache_shape, device=dense_v.device, dtype=dense_v.dtype)
    k_cache[physical_ids] = logical_k
    v_cache[physical_ids] = logical_v
    return k_cache.contiguous(), v_cache.contiguous(), block_tables


def check_correctness(
    q: torch.Tensor,
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    block_t: int,
) -> None:
    expected = dense_decode_attention(q, dense_k, dense_v, context_lens)
    dense_actual = dense_decode_attention_triton(
        q,
        dense_k,
        dense_v,
        context_lens,
        block_t=block_t,
    )
    paged_actual = paged_decode_attention_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
        block_t=block_t,
    )
    sdpa_actual = F.scaled_dot_product_attention(
        q.unsqueeze(2),
        dense_k.transpose(1, 2),
        dense_v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=default_attention_scale(q.shape[-1]),
    ).squeeze(2)

    torch.testing.assert_close(dense_actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(paged_actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(sdpa_actual.to(torch.float32), expected, atol=2e-3, rtol=2e-3)


def benchmark_operations(
    providers: Iterable[str],
    *,
    q: torch.Tensor,
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    block_t: int,
    reference_max_context: int,
) -> Iterable[tuple[str, str, str, Callable[[], object]]]:
    batch_size, num_heads, head_dim = q.shape
    context_len = dense_k.shape[1]
    scale = default_attention_scale(head_dim)

    dense_out = torch.empty_like(q, dtype=torch.float32)
    paged_out = torch.empty_like(q, dtype=torch.float32)
    dense_k_by_head = dense_k.transpose(1, 2)
    dense_v_by_head = dense_v.transpose(1, 2)
    q_with_seq = q.unsqueeze(2)

    operations: dict[str, tuple[str, str, Callable[[], object]]] = {
        "dense_triton": (
            "raw_kernel_preallocated_output",
            "cuda_events",
            lambda: _launch_dense_decode_attention_triton(
                q,
                dense_k,
                dense_v,
                context_lens,
                dense_out,
                max_context_len=context_len,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                block_t=block_t,
            ),
        ),
        "paged_triton": (
            "raw_kernel_preallocated_output",
            "cuda_events",
            lambda: _launch_paged_decode_attention_triton(
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                paged_out,
                max_blocks_per_seq=block_tables.shape[1],
                block_size=block_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                block_t=block_t,
            ),
        ),
        "pytorch_dense_sdpa": (
            "framework_operator_output_allocation",
            "cuda_events",
            lambda: F.scaled_dot_product_attention(
                q_with_seq,
                dense_k_by_head,
                dense_v_by_head,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            ),
        ),
        "pytorch_paged_reference": (
            "python_reference_end_to_end",
            "synchronized_wall_clock",
            lambda: __import__(
                "paged_kv_attention.reference", fromlist=["paged_decode_attention"]
            ).paged_decode_attention(
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                block_size=block_size,
                scale=scale,
            ),
        ),
    }

    for provider in providers:
        if provider == "pytorch_paged_reference" and context_len > reference_max_context:
            continue
        scope, timing_method, operation = operations[provider]
        yield provider, scope, timing_method, operation


def result_row(
    *,
    run_id: str,
    timestamp_utc: str,
    provider: str,
    measurement_scope: str,
    timing_method: str,
    stats: LatencyStats,
    batch_size: int,
    num_heads: int,
    context_len: int,
    block_size: int,
    head_dim: int,
    block_t: int,
    warmup: int,
    repeat: int,
    peak_bandwidth_gbps: float | None,
    correctness_guard: str,
    environment: dict[str, str],
) -> dict[str, object]:
    kv_bytes = analytical_kv_bytes(
        [context_len] * batch_size,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype_size=torch.tensor([], dtype=torch.float16).element_size(),
    )
    bandwidth_p50 = effective_bandwidth_gbps(kv_bytes, stats.p50_ms)
    bandwidth_p95 = effective_bandwidth_gbps(kv_bytes, stats.p95_ms)
    utilization_p50 = (
        bandwidth_utilization_percent(bandwidth_p50, peak_bandwidth_gbps)
        if peak_bandwidth_gbps is not None
        else None
    )
    utilization_p95 = (
        bandwidth_utilization_percent(bandwidth_p95, peak_bandwidth_gbps)
        if peak_bandwidth_gbps is not None
        else None
    )
    return {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "provider": provider,
        "measurement_scope": measurement_scope,
        "timing_method": timing_method,
        "batch_size": batch_size,
        "num_query_heads": num_heads,
        "num_kv_heads": num_heads,
        "context_len": context_len,
        "block_size": block_size,
        "head_dim": head_dim,
        "dtype": "float16",
        "block_t": block_t,
        "warmup": warmup,
        "repeat": repeat,
        "p50_ms": f"{stats.p50_ms:.6f}",
        "p95_ms": f"{stats.p95_ms:.6f}",
        "mean_ms": f"{stats.mean_ms:.6f}",
        "min_ms": f"{stats.min_ms:.6f}",
        "max_ms": f"{stats.max_ms:.6f}",
        "analytical_kv_bytes": kv_bytes,
        "effective_bandwidth_p50_gbps": f"{bandwidth_p50:.3f}",
        "effective_bandwidth_p95_gbps": f"{bandwidth_p95:.3f}",
        "peak_memory_bandwidth_gbps": (
            f"{peak_bandwidth_gbps:.3f}" if peak_bandwidth_gbps is not None else ""
        ),
        "bandwidth_utilization_p50_pct": (
            f"{utilization_p50:.3f}" if utilization_p50 is not None else ""
        ),
        "bandwidth_utilization_p95_pct": (
            f"{utilization_p95:.3f}" if utilization_p95 is not None else ""
        ),
        "tokens_per_second_p50": f"{batch_size / (stats.p50_ms * 1e-3):.3f}",
        "correctness_guard": correctness_guard,
        **environment,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = make_parser().parse_args()
    if args.num_heads <= 0 or args.head_dim != 128:
        raise SystemExit("the current benchmark requires num_heads > 0 and head_dim=128")
    if args.warmup < 0 or args.repeat <= 0 or args.reference_repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat counts must be positive")
    if args.peak_bandwidth_gbps is not None and args.peak_bandwidth_gbps <= 0:
        raise SystemExit("peak bandwidth must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    timestamp = dt.datetime.now(dt.timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("benchmarks/results") / f"decode_attention_{run_id}.csv"
    environment = collect_environment_metadata()
    rows: list[dict[str, object]] = []
    guarded = args.skip_correctness_guard

    for batch_size in args.batches:
        for context_len in args.contexts:
            for block_size in args.block_sizes:
                generator = torch.Generator(device="cuda").manual_seed(args.seed)
                q = torch.randn(
                    batch_size,
                    args.num_heads,
                    args.head_dim,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float16,
                )
                dense_k = torch.randn(
                    batch_size,
                    context_len,
                    args.num_heads,
                    args.head_dim,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float16,
                )
                dense_v = torch.randn_like(dense_k)
                context_lens = torch.full(
                    (batch_size,), context_len, device="cuda", dtype=torch.int32
                )
                k_cache, v_cache, block_tables = make_paged_cache(
                    dense_k,
                    dense_v,
                    context_lens,
                    block_size=block_size,
                    seed=args.seed + block_size,
                )

                guard_status = "skipped_by_flag" if args.skip_correctness_guard else "not_run"
                if not guarded:
                    check_correctness(
                        q,
                        dense_k,
                        dense_v,
                        k_cache,
                        v_cache,
                        block_tables,
                        context_lens,
                        block_size=block_size,
                        block_t=args.block_t,
                    )
                    guarded = True
                    guard_status = "passed_representative_case"
                    print(
                        f"correctness guard passed: B={batch_size}, S={context_len}, block={block_size}"
                    )
                elif not args.skip_correctness_guard:
                    guard_status = "passed_earlier_in_run"

                for provider, scope, timing_method, operation in benchmark_operations(
                    args.providers,
                    q=q,
                    dense_k=dense_k,
                    dense_v=dense_v,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    block_size=block_size,
                    block_t=args.block_t,
                    reference_max_context=args.reference_max_context,
                ):
                    if block_size != args.block_sizes[0] and provider in {
                        "dense_triton",
                        "pytorch_dense_sdpa",
                    }:
                        continue
                    timer = (
                        measure_cuda_latency
                        if timing_method == "cuda_events"
                        else measure_synchronized_wall_latency
                    )
                    operation_warmup = args.warmup if timing_method == "cuda_events" else 0
                    operation_repeat = (
                        args.repeat if timing_method == "cuda_events" else args.reference_repeat
                    )
                    stats = timer(
                        operation,
                        warmup=operation_warmup,
                        repeat=operation_repeat,
                    )
                    rows.append(
                        result_row(
                            run_id=run_id,
                            timestamp_utc=timestamp.isoformat(),
                            provider=provider,
                            measurement_scope=scope,
                            timing_method=timing_method,
                            stats=stats,
                            batch_size=batch_size,
                            num_heads=args.num_heads,
                            context_len=context_len,
                            block_size=block_size,
                            head_dim=args.head_dim,
                            block_t=args.block_t,
                            warmup=operation_warmup,
                            repeat=operation_repeat,
                            peak_bandwidth_gbps=args.peak_bandwidth_gbps,
                            correctness_guard=guard_status,
                            environment=environment,
                        )
                    )
                    print(
                        f"{provider:26s} B={batch_size:2d} S={context_len:5d} "
                        f"block={block_size:2d} p50={stats.p50_ms:.4f} ms "
                        f"p95={stats.p95_ms:.4f} ms"
                    )

                del q, dense_k, dense_v, k_cache, v_cache, block_tables, context_lens
                gc.collect()
                torch.cuda.empty_cache()

    write_rows(output, rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
