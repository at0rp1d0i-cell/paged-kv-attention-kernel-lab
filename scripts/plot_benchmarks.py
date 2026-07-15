#!/usr/bin/env python3
"""Generate static benchmark charts from the canonical CSV output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


PROVIDER_LABELS = {
    "dense_triton": "Dense Triton",
    "flashinfer_paged": "FlashInfer paged",
    "paged_triton": "Paged Triton",
    "pytorch_dense_sdpa": "PyTorch dense SDPA",
    "pytorch_paged_reference": "PyTorch paged reference",
    "paged_triton_single": "Paged Triton single",
    "paged_triton_split": "Paged Triton split-KV",
}

SERIES_STYLES = {
    ("dense_triton", 16): ("Dense Triton", "#1f77b4", "-"),
    ("paged_triton", 16): ("Paged Triton, block=16", "#d62728", "-"),
    ("paged_triton", 32): ("Paged Triton, block=32", "#2ca02c", "--"),
    ("pytorch_dense_sdpa", 16): ("PyTorch dense SDPA", "#9467bd", "-"),
    ("flashinfer_paged", 16): ("FlashInfer, block=16", "#ff7f0e", "-"),
    ("flashinfer_paged", 32): ("FlashInfer, block=32", "#8c564b", "--"),
}

SPLIT_SWEEP_STYLES = {
    "single": ("Single-pass", "#bc4749", "-"),
    "split=1": ("Split=1", "#6c757d", ":"),
    "split=4": ("Split=4", "#2a6f97", "--"),
    "split=8": ("Split=8", "#588157", "-."),
    "split=16": ("Split=16", "#8f5d9f", "-"),
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


def plot_equal_work_metric(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["batch_size"]))
    positions = list(range(len(ordered)))
    labels = [row["case_label"].replace(",", "\n") for row in ordered]
    values = [float(row[metric]) for row in ordered]
    colors = [
        "#2a6f97" if row["provider"] == "paged_triton_split" else "#bc4749" for row in ordered
    ]

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.bar(positions, values, color=colors, width=0.68)
    axis.set_xticks(positions, labels)
    axis.set_ylabel(ylabel)
    axis.set_title("Equal KV work and matched main-program topology")
    axis.grid(True, axis="y", alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_split_sweep_metric_by_batch(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    output: Path,
    baseline: float | None = None,
) -> None:
    batches = sorted({int(row["batch_size"]) for row in rows})
    num_columns = min(3, len(batches))
    num_rows = (len(batches) + num_columns - 1) // num_columns
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(5.2 * num_columns, 4.2 * num_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    for axis, batch_size in zip(axes.flat, batches, strict=False):
        for implementation, (label, color, linestyle) in SPLIT_SWEEP_STYLES.items():
            points = sorted(
                (int(row["context_len"]), float(row[metric]))
                for row in rows
                if int(row["batch_size"]) == batch_size and row["implementation"] == implementation
            )
            if not points:
                continue
            contexts, values = zip(*points, strict=True)
            axis.plot(
                contexts,
                values,
                marker="o",
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
        if baseline is not None:
            axis.axhline(baseline, color="#333333", linewidth=1.0, linestyle=":")
        axis.set_xscale("log", base=2)
        axis.set_title(f"Batch = {batch_size}")
        axis.set_xlabel("Context length")
        axis.grid(True, which="both", alpha=0.25)

    for axis in list(axes.flat)[len(batches) :]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=5, fontsize=9)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_split_dispatch_map(
    rows: list[dict[str, str]],
    output: Path,
    *,
    choice_field: str,
    title: str,
) -> None:
    batches = sorted({int(row["batch_size"]) for row in rows})
    contexts = sorted({int(row["context_len"]) for row in rows})
    implementations = list(SPLIT_SWEEP_STYLES)
    implementation_ids = {name: index for index, name in enumerate(implementations)}
    best = {
        (int(row["batch_size"]), int(row["context_len"])): row["implementation"]
        for row in rows
        if row[choice_field] == "true"
    }
    matrix = [
        [implementation_ids[best[(batch_size, context_len)]] for context_len in contexts]
        for batch_size in batches
    ]
    colors = [SPLIT_SWEEP_STYLES[name][1] for name in implementations]
    color_map = ListedColormap(colors)

    figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    axis.imshow(matrix, cmap=color_map, vmin=-0.5, vmax=len(implementations) - 0.5, aspect="auto")
    axis.set_xticks(range(len(contexts)), [str(context_len) for context_len in contexts])
    axis.set_yticks(range(len(batches)), [str(batch_size) for batch_size in batches])
    axis.set_xlabel("Context length")
    axis.set_ylabel("Batch size")
    axis.set_title(title)
    for batch_index, batch_size in enumerate(batches):
        for context_index, context_len in enumerate(contexts):
            implementation = best[(batch_size, context_len)]
            axis.text(
                context_index,
                batch_index,
                implementation.replace("split=", "s"),
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
    legend = [
        Patch(color=SPLIT_SWEEP_STYLES[name][1], label=SPLIT_SWEEP_STYLES[name][0])
        for name in implementations
    ]
    figure.legend(handles=legend, loc="outside upper center", ncol=5, fontsize=9)
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

    if all(row.get("analysis_type") == "same_shape_split_sweep" for row in rows):
        plot_split_sweep_metric_by_batch(
            rows,
            metric="p50_ms",
            ylabel="p50 latency (ms)",
            output=output_dir / f"{stem}_latency_p50_by_batch.png",
        )
        plot_split_sweep_metric_by_batch(
            rows,
            metric="speedup_vs_single_p50",
            ylabel="Speedup vs single-pass",
            output=output_dir / f"{stem}_speedup_by_batch.png",
            baseline=1.0,
        )
        plot_split_dispatch_map(
            rows,
            output_dir / f"{stem}_best_dispatch_map.png",
            choice_field="is_best_p50",
            title="Best p50 path from the same-shape sweep",
        )
        plot_split_dispatch_map(
            rows,
            output_dir / f"{stem}_adaptive_dispatch_map.png",
            choice_field="is_adaptive_choice",
            title="Evidence-based adaptive dispatch policy",
        )
        print(f"wrote same-shape split-KV plots to {output_dir}")
        return

    if all(row.get("analysis_type") == "equal_work_program_matched" for row in rows):
        plot_equal_work_metric(
            rows,
            metric="p50_ms",
            ylabel="p50 latency (ms)",
            output=output_dir / f"{stem}_latency_p50.png",
        )
        plot_equal_work_metric(
            rows,
            metric="effective_bandwidth_p50_gbps",
            ylabel="Effective KV bandwidth (GB/s)",
            output=output_dir / f"{stem}_bandwidth.png",
        )
        print(f"wrote equal-work plots to {output_dir}")
        return

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
