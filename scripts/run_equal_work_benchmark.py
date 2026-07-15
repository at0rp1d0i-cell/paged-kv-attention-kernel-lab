#!/usr/bin/env python3
"""Benchmark program-matched split-KV cases with equal analytical KV work."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from paged_kv_attention.benchmark_utils import (
    EqualWorkCase,
    LatencyStats,
    analytical_kv_bytes,
    bandwidth_utilization_percent,
    collect_environment_metadata,
    effective_bandwidth_gbps,
    make_equal_work_cases,
    summarize_latencies,
)
from paged_kv_attention.block_table import make_random_block_tables
from paged_kv_attention.reference import default_attention_scale, dense_decode_attention
from paged_kv_attention.triton_decode import (
    _launch_paged_decode_attention_split_partial_triton,
    _launch_paged_decode_attention_split_reduce_triton,
    _launch_paged_decode_attention_triton,
)


DEFAULT_BATCH_SPLIT_PAIRS: tuple[tuple[int, int | None], ...] = (
    (1, 16),
    (2, 8),
    (4, 4),
    (16, None),
)

ENVIRONMENT_FIELDS = [
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
    "flashinfer_version",
    "cuda_compiler_version",
]

CSV_FIELDS = [
    "run_id",
    "timestamp_utc",
    "analysis_type",
    "provider",
    "case_label",
    "batch_size",
    "context_len",
    "total_context_tokens",
    "num_query_heads",
    "num_kv_heads",
    "head_dim",
    "dtype",
    "block_size",
    "block_t",
    "num_splits",
    "main_program_count",
    "reduce_program_count",
    "tokens_per_main_program",
    "partial_state_bytes",
    "kernel_launches",
    "analytical_kv_bytes",
    "cache_mode",
    "measurement_scope",
    "timing_method",
    "measurement_order",
    "warmup",
    "repeat",
    "p50_ms",
    "p95_ms",
    "mean_ms",
    "min_ms",
    "max_ms",
    "effective_bandwidth_p50_gbps",
    "effective_bandwidth_p95_gbps",
    "peak_memory_bandwidth_gbps",
    "bandwidth_utilization_p50_pct",
    "bandwidth_utilization_p95_pct",
    "correctness_guard",
    *ENVIRONMENT_FIELDS,
]


def parse_case_specs(value: str) -> list[tuple[int, int | None]]:
    """Parse ``batch:split`` pairs, using ``single`` for the single-pass path."""

    pairs = []
    try:
        for item in value.split(","):
            batch_text, implementation_text = item.strip().split(":", maxsplit=1)
            implementation_text = implementation_text.strip()
            num_splits = None if implementation_text == "single" else int(implementation_text)
            pairs.append((int(batch_text.strip()), num_splits))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated batch:split entries such as 1:16,2:8,4:4,16:single"
        ) from error
    if not pairs:
        raise argparse.ArgumentTypeError("at least one case is required")
    return pairs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-context-tokens", type=int, default=16384)
    parser.add_argument(
        "--cases",
        type=parse_case_specs,
        default=list(DEFAULT_BATCH_SPLIT_PAIRS),
        help="Program-matched batch:split cases; use 'single' for single-pass.",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--block-t", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument("--peak-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _make_split_operation(
    case: EqualWorkCase,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    block_t: int,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> tuple[Callable[[], None], torch.Tensor]:
    assert case.num_splits is not None
    partial_shape = (case.batch_size, num_heads, case.num_splits)
    partial_m = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
    partial_l = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
    partial_acc = torch.empty((*partial_shape, head_dim), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q, dtype=torch.float32)

    def operation() -> None:
        _launch_paged_decode_attention_split_partial_triton(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            partial_m,
            partial_l,
            partial_acc,
            max_blocks_per_seq=block_tables.shape[1],
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_splits=case.num_splits,
            scale=scale,
            block_t=block_t,
        )
        _launch_paged_decode_attention_split_reduce_triton(
            partial_m,
            partial_l,
            partial_acc,
            out,
            num_heads=num_heads,
            head_dim=head_dim,
            num_splits=case.num_splits,
        )

    return operation, out


def _make_single_operation(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    block_t: int,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> tuple[Callable[[], None], torch.Tensor]:
    out = torch.empty_like(q, dtype=torch.float32)

    def operation() -> None:
        _launch_paged_decode_attention_triton(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            out,
            max_blocks_per_seq=block_tables.shape[1],
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            block_t=block_t,
        )

    return operation, out


def _check_correctness(
    operation: Callable[[], None],
    out: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    dense_k = k_cache[block_tables.to(torch.long)].flatten(1, 2)
    dense_v = v_cache[block_tables.to(torch.long)].flatten(1, 2)
    expected = dense_decode_attention(q, dense_k, dense_v, context_lens)
    operation()
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)


def _measure_interleaved_cuda_latencies(
    operations: dict[str, Callable[[], None]],
    *,
    warmup: int,
    repeat: int,
    seed: int,
) -> dict[str, LatencyStats]:
    """Measure each operation once per shuffled cycle to balance clock and order effects."""

    names = list(operations)
    rng = random.Random(seed)
    for _ in range(warmup):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            operations[name]()
    torch.cuda.synchronize()

    event_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in names
    }
    for _ in range(repeat):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operations[name]()
            end.record()
            event_pairs[name].append((start, end))

    torch.cuda.synchronize()
    return {
        name: summarize_latencies([start.elapsed_time(end) for start, end in pairs])
        for name, pairs in event_pairs.items()
    }


def _result_row(
    *,
    case: EqualWorkCase,
    stats: LatencyStats,
    run_id: str,
    timestamp_utc: str,
    total_context_tokens: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    block_t: int,
    warmup: int,
    repeat: int,
    peak_bandwidth_gbps: float | None,
    environment: dict[str, str],
) -> dict[str, object]:
    kv_bytes = analytical_kv_bytes(
        [case.context_len] * case.batch_size,
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
        "analysis_type": "equal_work_program_matched",
        "provider": case.provider,
        "case_label": case.label,
        "batch_size": case.batch_size,
        "context_len": case.context_len,
        "total_context_tokens": total_context_tokens,
        "num_query_heads": num_heads,
        "num_kv_heads": num_heads,
        "head_dim": head_dim,
        "dtype": "float16",
        "block_size": block_size,
        "block_t": block_t,
        "num_splits": case.num_splits if case.num_splits is not None else "",
        "main_program_count": case.main_program_count(num_heads),
        "reduce_program_count": case.reduce_program_count(num_heads),
        "tokens_per_main_program": case.tokens_per_main_program(),
        "partial_state_bytes": case.partial_state_bytes(
            num_heads=num_heads,
            head_dim=head_dim,
        ),
        "kernel_launches": 1 if case.num_splits is None else 2,
        "analytical_kv_bytes": kv_bytes,
        "cache_mode": "shared_kv_repeated_steady_state",
        "measurement_scope": "raw_kernels_preallocated_buffers",
        "timing_method": "cuda_events",
        "measurement_order": "deterministic_shuffled_cycles",
        "warmup": warmup,
        "repeat": repeat,
        "p50_ms": f"{stats.p50_ms:.6f}",
        "p95_ms": f"{stats.p95_ms:.6f}",
        "mean_ms": f"{stats.mean_ms:.6f}",
        "min_ms": f"{stats.min_ms:.6f}",
        "max_ms": f"{stats.max_ms:.6f}",
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
        "correctness_guard": "passed_fp32_dense_reference",
        **environment,
    }


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = make_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.num_heads <= 0 or args.head_dim != 128:
        raise SystemExit("the current benchmark requires num_heads > 0 and head_dim=128")
    if args.block_size <= 0 or args.block_t <= 0:
        raise SystemExit("block_size and block_t must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat must be positive")
    if args.peak_bandwidth_gbps is not None and args.peak_bandwidth_gbps <= 0:
        raise SystemExit("peak bandwidth must be positive")

    cases = make_equal_work_cases(args.total_context_tokens, args.cases)
    if any(case.context_len % args.block_size != 0 for case in cases):
        raise SystemExit("every equal-work context length must be divisible by block_size")

    timestamp = dt.datetime.now(dt.timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("benchmarks/results") / f"split_kv_equal_work_{run_id}.csv"
    environment = collect_environment_metadata()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    scale = default_attention_scale(args.head_dim)

    base_context_lens = torch.tensor(
        [args.total_context_tokens],
        device="cuda",
        dtype=torch.int32,
    )
    base_table, num_blocks = make_random_block_tables(
        base_context_lens,
        block_size=args.block_size,
        seed=args.seed,
        device="cuda",
    )
    base_table = base_table.to(torch.int32).contiguous()
    cache_shape = (num_blocks, args.block_size, args.num_heads, args.head_dim)
    k_cache = torch.randn(
        cache_shape,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    v_cache = torch.randn(
        cache_shape,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )

    operations: dict[str, Callable[[], None]] = {}
    for case in cases:
        blocks_per_sequence = case.context_len // args.block_size
        block_tables = base_table.view(case.batch_size, blocks_per_sequence).contiguous()
        context_lens = torch.full(
            (case.batch_size,),
            case.context_len,
            device="cuda",
            dtype=torch.int32,
        )
        q = torch.randn(
            case.batch_size,
            args.num_heads,
            args.head_dim,
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )

        if case.num_splits is None:
            operation, out = _make_single_operation(
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                block_size=args.block_size,
                block_t=args.block_t,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                scale=scale,
            )
        else:
            operation, out = _make_split_operation(
                case,
                q,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                block_size=args.block_size,
                block_t=args.block_t,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                scale=scale,
            )

        _check_correctness(
            operation,
            out,
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
        )
        operations[case.label] = operation

    stats_by_case = _measure_interleaved_cuda_latencies(
        operations,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
    )
    rows = []
    for case in cases:
        stats = stats_by_case[case.label]
        rows.append(
            _result_row(
                case=case,
                stats=stats,
                run_id=run_id,
                timestamp_utc=timestamp.isoformat(),
                total_context_tokens=args.total_context_tokens,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                block_t=args.block_t,
                warmup=args.warmup,
                repeat=args.repeat,
                peak_bandwidth_gbps=args.peak_bandwidth_gbps,
                environment=environment,
            )
        )
        print(
            f"{case.label:28s} programs={case.main_program_count(args.num_heads):4d} "
            f"tokens/program={case.tokens_per_main_program():4d} "
            f"p50={stats.p50_ms:.4f} ms p95={stats.p95_ms:.4f} ms"
        )
    _write_rows(output, rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
