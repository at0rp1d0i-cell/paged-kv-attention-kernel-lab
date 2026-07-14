#!/usr/bin/env python3
"""Generate static benchmark charts from the canonical CSV output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROVIDER_LABELS = {
    "dense_triton": "Dense Triton",
    "paged_triton": "Paged Triton",
    "pytorch_dense_sdpa": "PyTorch dense SDPA",
    "pytorch_paged_reference": "PyTorch paged reference",
}

SERIES_STYLES = {
    ("dense_triton", 16): ("Dense Triton", "#1f77b4", "-"),
    ("paged_triton", 16): ("Paged Triton, block=16", "#d62728", "-"),
    ("paged_triton", 32): ("Paged Triton, block=32", "#2ca02c", "--"),
    ("pytorch_dense_sdpa", 16): ("PyTorch dense SDPA", "#9467bd", "-"),
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_metric(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    grouped: dict[tuple[str, int, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (row["provider"], int(row["batch_size"]), int(row["block_size"]))
        grouped[key].append((int(row["context_len"]), float(row[metric])))

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for (provider, batch_size, block_size), points in sorted(grouped.items()):
        points.sort()
        contexts, values = zip(*points, strict=True)
        label = f"{PROVIDER_LABELS.get(provider, provider)}, B={batch_size}, block={block_size}"
        axis.plot(contexts, values, marker="o", linewidth=1.6, label=label)

    axis.set_xscale("log", base=2)
    axis.set_xlabel("Context length")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_metric_by_batch(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    batches = sorted({int(row["batch_size"]) for row in rows})
    figure, axes = plt.subplots(
        1,
        len(batches),
        figsize=(15, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(batches) == 1:
        axes = [axes]

    for axis, batch_size in zip(axes, batches, strict=True):
        for (provider, block_size), (label, color, linestyle) in SERIES_STYLES.items():
            points = sorted(
                (int(row["context_len"]), float(row[metric]))
                for row in rows
                if row["provider"] == provider
                and int(row["batch_size"]) == batch_size
                and int(row["block_size"]) == block_size
            )
            if not points:
                continue
            contexts, values = zip(*points, strict=True)
            axis.plot(
                contexts,
                values,
                marker="o",
                linewidth=1.8,
                linestyle=linestyle,
                color=color,
                label=label,
            )
        axis.set_xscale("log", base=2)
        axis.set_title(f"Batch = {batch_size}")
        axis.set_xlabel("Context length")
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=9)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_paged_dense_ratio(rows: list[dict[str, str]], output: Path) -> None:
    lookup = {
        (
            row["provider"],
            int(row["batch_size"]),
            int(row["context_len"]),
            int(row["block_size"]),
        ): float(row["p50_ms"])
        for row in rows
    }
    batches = sorted({int(row["batch_size"]) for row in rows})
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

    colors = {1: "#1f77b4", 4: "#d62728", 16: "#2ca02c"}
    for batch_size in batches:
        for block_size, linestyle in ((16, "-"), (32, "--")):
            points = []
            for context_len in sorted({int(row["context_len"]) for row in rows}):
                dense_key = ("dense_triton", batch_size, context_len, 16)
                paged_key = ("paged_triton", batch_size, context_len, block_size)
                if dense_key in lookup and paged_key in lookup:
                    points.append((context_len, lookup[paged_key] / lookup[dense_key]))
            contexts, ratios = zip(*points, strict=True)
            axis.plot(
                contexts,
                ratios,
                marker="o",
                color=colors.get(batch_size),
                linestyle=linestyle,
                linewidth=1.8,
                label=f"B={batch_size}, block={block_size}",
            )

    axis.axhline(1.0, color="#444444", linewidth=1.0, linestyle=":")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Context length")
    axis.set_ylabel("Paged / dense Triton p50 latency")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_batch_scaling(rows: list[dict[str, str]], output: Path) -> None:
    max_context = max(int(row["context_len"]) for row in rows)
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

    for (provider, block_size), (label, color, linestyle) in SERIES_STYLES.items():
        points = sorted(
            (int(row["batch_size"]), float(row["p50_ms"]))
            for row in rows
            if row["provider"] == provider
            and int(row["block_size"]) == block_size
            and int(row["context_len"]) == max_context
        )
        if not points:
            continue
        batches, values = zip(*points, strict=True)
        axis.plot(
            batches,
            values,
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )

    axis.set_xscale("log", base=2)
    axis.set_xlabel("Batch size")
    axis.set_ylabel("p50 latency (ms)")
    axis.set_title(f"Context length = {max_context}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=9)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = make_parser().parse_args()
    rows = read_rows(args.csv_path)
    if not rows:
        raise SystemExit("CSV contains no benchmark rows")
    output_dir = args.output_dir or args.csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.csv_path.stem

    plot_metric(
        rows,
        metric="p50_ms",
        ylabel="p50 latency (ms)",
        output=output_dir / f"{stem}_latency_p50.png",
    )
    plot_metric(
        rows,
        metric="p95_ms",
        ylabel="p95 latency (ms)",
        output=output_dir / f"{stem}_latency_p95.png",
    )
    plot_metric(
        rows,
        metric="effective_bandwidth_p50_gbps",
        ylabel="Effective KV bandwidth (GB/s)",
        output=output_dir / f"{stem}_bandwidth.png",
    )
    plot_metric_by_batch(
        rows,
        metric="p50_ms",
        ylabel="p50 latency (ms)",
        output=output_dir / f"{stem}_latency_p50_by_batch.png",
    )
    plot_metric_by_batch(
        rows,
        metric="effective_bandwidth_p50_gbps",
        ylabel="Effective KV bandwidth (GB/s)",
        output=output_dir / f"{stem}_bandwidth_by_batch.png",
    )
    if all(row.get("bandwidth_utilization_p50_pct") for row in rows):
        plot_metric_by_batch(
            rows,
            metric="bandwidth_utilization_p50_pct",
            ylabel="Nominal peak bandwidth utilization (%)",
            output=output_dir / f"{stem}_bandwidth_utilization_by_batch.png",
        )
    plot_paged_dense_ratio(rows, output_dir / f"{stem}_paged_dense_ratio.png")
    plot_batch_scaling(rows, output_dir / f"{stem}_batch_scaling.png")
    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
