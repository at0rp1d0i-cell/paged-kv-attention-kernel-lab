"""Reusable measurement and reporting helpers for decode-attention benchmarks."""

from __future__ import annotations

import math
import os
import platform
import re
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import triton


@dataclass(frozen=True)
class LatencyStats:
    """Latency distribution summarized in milliseconds."""

    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    samples: int


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for non-empty samples."""

    if not samples:
        raise ValueError("samples must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(float(sample) for sample in samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latencies(samples_ms: Sequence[float]) -> LatencyStats:
    """Summarize per-iteration CUDA-event samples."""

    if not samples_ms:
        raise ValueError("samples_ms must not be empty")
    samples = [float(sample) for sample in samples_ms]
    return LatencyStats(
        p50_ms=percentile(samples, 0.50),
        p95_ms=percentile(samples, 0.95),
        mean_ms=statistics.fmean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        samples=len(samples),
    )


def measure_cuda_latency(
    operation: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
) -> LatencyStats:
    """Measure one CUDA operation using one event pair per sample."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark timing")

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    event_pairs = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        event_pairs.append((start, end))

    torch.cuda.synchronize()
    samples_ms = [start.elapsed_time(end) for start, end in event_pairs]
    return summarize_latencies(samples_ms)


def measure_synchronized_wall_latency(
    operation: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
) -> LatencyStats:
    """Measure CPU-driven CUDA code including Python and synchronization overhead."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark timing")

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)
    return summarize_latencies(samples_ms)


def analytical_kv_bytes(
    context_lens: Sequence[int],
    *,
    num_kv_heads: int,
    head_dim: int,
    dtype_size: int,
) -> int:
    """Estimate bytes read from K and V for one decode step."""

    if any(context_len < 0 for context_len in context_lens):
        raise ValueError("context lengths must be non-negative")
    if min(num_kv_heads, head_dim, dtype_size) <= 0:
        raise ValueError("num_kv_heads, head_dim, and dtype_size must be positive")
    return sum(context_lens) * 2 * num_kv_heads * head_dim * dtype_size


def effective_bandwidth_gbps(bytes_read: int, latency_ms: float) -> float:
    """Convert analytical bytes and latency to decimal GB/s."""

    if bytes_read < 0:
        raise ValueError("bytes_read must be non-negative")
    if latency_ms <= 0:
        raise ValueError("latency_ms must be positive")
    return bytes_read / (latency_ms * 1e-3) / 1e9


def bandwidth_utilization_percent(
    effective_bandwidth: float,
    peak_bandwidth: float,
) -> float:
    """Return effective bandwidth as a percentage of nominal hardware peak."""

    if effective_bandwidth < 0:
        raise ValueError("effective_bandwidth must be non-negative")
    if peak_bandwidth <= 0:
        raise ValueError("peak_bandwidth must be positive")
    return effective_bandwidth / peak_bandwidth * 100.0


def _nvidia_smi_value(query: str) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.splitlines()[0].strip() or "unknown"


def _cuda_compiler_version() -> str:
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = Path(cuda_home) / "bin" / "nvcc" if cuda_home else Path("nvcc")
    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    match = re.search(r"release (\d+\.\d+)", result.stdout)
    return match.group(1) if match else "unknown"


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not_installed"


def collect_environment_metadata(device_index: int = 0) -> dict[str, str]:
    """Collect stable software, GPU, driver, and clock facts for CSV rows."""

    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": properties.name,
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_memory_bytes": str(properties.total_memory),
        "driver_version": _nvidia_smi_value("driver_version"),
        "graphics_clock_mhz": _nvidia_smi_value("clocks.current.graphics"),
        "memory_clock_mhz": _nvidia_smi_value("clocks.current.memory"),
        "clock_state": "recorded_not_locked",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda or "unknown",
        "triton_version": triton.__version__,
        "flashinfer_version": _package_version("flashinfer-python"),
        "cuda_compiler_version": _cuda_compiler_version(),
    }
