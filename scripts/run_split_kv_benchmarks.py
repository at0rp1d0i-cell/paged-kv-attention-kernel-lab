#!/usr/bin/env python3
"""Run same-shape single-pass versus split-KV Triton benchmarks."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from paged_kv_attention.benchmark_utils import (
    LatencyStats,
    analytical_kv_bytes,
    bandwidth_utilization_percent,
    collect_environment_metadata,
    effective_bandwidth_gbps,
    measure_interleaved_cuda_latencies,
)
from paged_kv_attention.block_table import make_random_block_tables
from paged_kv_attention.reference import default_attention_scale
from paged_kv_attention.triton_decode import (
    _launch_paged_decode_attention_split_partial_triton,
    _launch_paged_decode_attention_split_reduce_triton,
    _launch_paged_decode_attention_triton,
    select_paged_decode_num_splits,
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
    "implementation",
    "case_label",
    "batch_size",
    "context_len",
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
    "speedup_vs_single_p50",
    "is_best_p50",
    "adaptive_implementation",
    "adaptive_speedup_vs_single_p50",
    "adaptive_loss_vs_best_p50",
    "is_adaptive_choice",
    "effective_bandwidth_p50_gbps",
    "effective_bandwidth_p95_gbps",
    "peak_memory_bandwidth_gbps",
    "bandwidth_utilization_p50_pct",
    "bandwidth_utilization_p95_pct",
    "correctness_guard",
    *ENVIRONMENT_FIELDS,
]


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=parse_int_list, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument(
        "--contexts",
        type=parse_int_list,
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--splits", type=parse_int_list, default=[1, 4, 8, 16])
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


def _make_single_operation(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int,
    block_t: int,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> Callable[[], None]:
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

    return operation


def _make_split_operation(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int,
    block_t: int,
    num_heads: int,
    head_dim: int,
    num_splits: int,
    scale: float,
) -> Callable[[], None]:
    partial_shape = (q.shape[0], num_heads, num_splits)
    partial_m = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
    partial_l = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
    partial_acc = torch.empty((*partial_shape, head_dim), device=q.device, dtype=torch.float32)

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
            num_splits=num_splits,
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
            num_splits=num_splits,
        )

    return operation


def _result_row(
    *,
    run_id: str,
    timestamp_utc: str,
    batch_size: int,
    context_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    block_t: int,
    num_splits: int | None,
    stats: LatencyStats,
    single_p50_ms: float,
    best_p50_ms: float,
    adaptive_implementation: str,
    adaptive_p50_ms: float,
    warmup: int,
    repeat: int,
    peak_bandwidth_gbps: float | None,
    environment: dict[str, str],
) -> dict[str, object]:
    partitions = num_splits if num_splits is not None else 1
    provider = "paged_triton_single" if num_splits is None else "paged_triton_split"
    implementation = "single" if num_splits is None else f"split={num_splits}"
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
    partial_state_bytes = (
        batch_size * num_heads * num_splits * (head_dim + 2) * 4 if num_splits is not None else 0
    )
    return {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "analysis_type": "same_shape_split_sweep",
        "provider": provider,
        "implementation": implementation,
        "case_label": f"B={batch_size},S={context_len},{implementation}",
        "batch_size": batch_size,
        "context_len": context_len,
        "num_query_heads": num_heads,
        "num_kv_heads": num_heads,
        "head_dim": head_dim,
        "dtype": "float16",
        "block_size": block_size,
        "block_t": block_t,
        "num_splits": num_splits if num_splits is not None else "",
        "main_program_count": batch_size * num_heads * partitions,
        "reduce_program_count": batch_size * num_heads if num_splits is not None else 0,
        "tokens_per_main_program": (context_len + partitions - 1) // partitions,
        "partial_state_bytes": partial_state_bytes,
        "kernel_launches": 1 if num_splits is None else 2,
        "analytical_kv_bytes": kv_bytes,
        "cache_mode": "per_shape_shared_kv_repeated_steady_state",
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
        "speedup_vs_single_p50": f"{single_p50_ms / stats.p50_ms:.6f}",
        "is_best_p50": str(stats.p50_ms == best_p50_ms).lower(),
        "adaptive_implementation": adaptive_implementation,
        "adaptive_speedup_vs_single_p50": f"{single_p50_ms / adaptive_p50_ms:.6f}",
        "adaptive_loss_vs_best_p50": f"{adaptive_p50_ms / best_p50_ms:.6f}",
        "is_adaptive_choice": str(implementation == adaptive_implementation).lower(),
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
        "correctness_guard": "passed_single_pass_alignment",
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
    if args.block_size <= 0:
        raise SystemExit("block_size must be positive")
    if args.block_t <= 0 or args.block_t & (args.block_t - 1):
        raise SystemExit("block_t must be a positive power of two")
    if any(num_splits not in (1, 4, 8, 16) for num_splits in args.splits):
        raise SystemExit("splits must contain only 1, 4, 8, or 16")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat must be positive")
    if args.peak_bandwidth_gbps is not None and args.peak_bandwidth_gbps <= 0:
        raise SystemExit("peak bandwidth must be positive")

    timestamp = dt.datetime.now(dt.timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("benchmarks/results") / f"split_kv_same_shape_{run_id}.csv"
    environment = collect_environment_metadata()
    scale = default_attention_scale(args.head_dim)
    rows = []

    for batch_size in args.batches:
        for context_len in args.contexts:
            case_seed = args.seed + batch_size * 100_000 + context_len
            generator = torch.Generator(device="cuda").manual_seed(case_seed)
            q = torch.randn(
                batch_size,
                args.num_heads,
                args.head_dim,
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            context_lens = torch.full(
                (batch_size,),
                context_len,
                device="cuda",
                dtype=torch.int32,
            )
            block_tables, num_blocks = make_random_block_tables(
                context_lens,
                block_size=args.block_size,
                seed=case_seed,
                device="cuda",
            )
            block_tables = block_tables.to(torch.int32).contiguous()
            cache_shape = (
                num_blocks,
                args.block_size,
                args.num_heads,
                args.head_dim,
            )
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

            outputs = {"single": torch.empty_like(q, dtype=torch.float32)}
            operations: dict[str, Callable[[], object]] = {
                "single": _make_single_operation(
                    q,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    outputs["single"],
                    block_size=args.block_size,
                    block_t=args.block_t,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                    scale=scale,
                )
            }
            for num_splits in args.splits:
                name = f"split={num_splits}"
                outputs[name] = torch.empty_like(q, dtype=torch.float32)
                operations[name] = _make_split_operation(
                    q,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    outputs[name],
                    block_size=args.block_size,
                    block_t=args.block_t,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                    num_splits=num_splits,
                    scale=scale,
                )

            for operation in operations.values():
                operation()
            torch.cuda.synchronize()
            for name, actual in outputs.items():
                if name == "single":
                    continue
                torch.testing.assert_close(actual, outputs["single"], atol=2e-3, rtol=2e-3)

            stats_by_name = measure_interleaved_cuda_latencies(
                operations,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=case_seed,
            )
            single_p50_ms = stats_by_name["single"].p50_ms
            best_p50_ms = min(stats.p50_ms for stats in stats_by_name.values())
            adaptive_num_splits = select_paged_decode_num_splits(
                batch_size=batch_size,
                num_heads=args.num_heads,
                max_context_len=context_len,
                block_size=args.block_size,
            )
            adaptive_implementation = (
                "single" if adaptive_num_splits is None else f"split={adaptive_num_splits}"
            )
            adaptive_p50_ms = stats_by_name[adaptive_implementation].p50_ms
            configurations: list[tuple[str, int | None]] = [("single", None)] + [
                (f"split={num_splits}", num_splits) for num_splits in args.splits
            ]
            for name, num_splits in configurations:
                stats = stats_by_name[name]
                rows.append(
                    _result_row(
                        run_id=run_id,
                        timestamp_utc=timestamp.isoformat(),
                        batch_size=batch_size,
                        context_len=context_len,
                        num_heads=args.num_heads,
                        head_dim=args.head_dim,
                        block_size=args.block_size,
                        block_t=args.block_t,
                        num_splits=num_splits,
                        stats=stats,
                        single_p50_ms=single_p50_ms,
                        best_p50_ms=best_p50_ms,
                        adaptive_implementation=adaptive_implementation,
                        adaptive_p50_ms=adaptive_p50_ms,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        peak_bandwidth_gbps=args.peak_bandwidth_gbps,
                        environment=environment,
                    )
                )
            best_name, best_stats = min(
                stats_by_name.items(),
                key=lambda item: item[1].p50_ms,
            )
            print(
                f"B={batch_size:2d} S={context_len:5d} best={best_name:8s} "
                f"adaptive={adaptive_implementation:8s} "
                f"speedup={single_p50_ms / adaptive_p50_ms:.3f}x "
                f"best_loss={adaptive_p50_ms / best_stats.p50_ms:.3f}x"
            )

            del q, context_lens, block_tables, k_cache, v_cache
            del outputs, operations, stats_by_name
            gc.collect()
            torch.cuda.empty_cache()

    _write_rows(output, rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
