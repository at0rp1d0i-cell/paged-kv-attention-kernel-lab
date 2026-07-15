import csv
from pathlib import Path

import pytest

from paged_kv_attention.triton_decode import select_paged_decode_num_splits


CANONICAL_SWEEP = (
    Path(__file__).parents[1] / "benchmarks" / "results" / "split_kv_same_shape.csv"
)


@pytest.mark.parametrize(
    ("batch_size", "num_heads", "context_len", "expected_num_splits"),
    [
        (1, 8, 512, None),
        (1, 8, 2048, 4),
        (1, 8, 4096, 16),
        (2, 8, 32768, 16),
        (4, 8, 4096, 8),
        (4, 8, 16384, 4),
        (8, 8, 2048, 4),
        (16, 8, 2048, 4),
        (16, 8, 4096, None),
        (32, 8, 2048, None),
    ],
)
def test_adaptive_split_selector_matches_measured_policy(
    batch_size: int,
    num_heads: int,
    context_len: int,
    expected_num_splits: int | None,
) -> None:
    assert (
        select_paged_decode_num_splits(
            batch_size=batch_size,
            num_heads=num_heads,
            max_context_len=context_len,
            block_size=32,
        )
        == expected_num_splits
    )


def test_adaptive_split_selector_uses_single_for_unmeasured_block_size() -> None:
    assert (
        select_paged_decode_num_splits(
            batch_size=1,
            num_heads=8,
            max_context_len=16384,
            block_size=16,
        )
        is None
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("batch_size", 0),
        ("num_heads", 0),
        ("max_context_len", 0),
        ("block_size", 0),
    ],
)
def test_adaptive_split_selector_rejects_non_positive_values(name: str, value: int) -> None:
    values = {
        "batch_size": 1,
        "num_heads": 8,
        "max_context_len": 16384,
        "block_size": 32,
    }
    values[name] = value

    with pytest.raises(ValueError, match=f"{name} must be positive"):
        select_paged_decode_num_splits(**values)


@pytest.mark.parametrize("value", [True, 4.0])
def test_adaptive_split_selector_rejects_non_integer_values(value: object) -> None:
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        select_paged_decode_num_splits(
            batch_size=value,  # type: ignore[arg-type]
            num_heads=8,
            max_context_len=16384,
            block_size=32,
        )


def test_canonical_same_shape_sweep_supports_adaptive_policy() -> None:
    with CANONICAL_SWEEP.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    expected_batches = {1, 2, 4, 8, 16, 32}
    expected_contexts = {512, 1024, 2048, 4096, 8192, 16384, 32768}
    expected_implementations = {"single", "split=1", "split=4", "split=8", "split=16"}
    rows_by_shape: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        shape = (int(row["batch_size"]), int(row["context_len"]))
        rows_by_shape.setdefault(shape, []).append(row)

    expected_shapes = {
        (batch_size, context_len)
        for batch_size in expected_batches
        for context_len in expected_contexts
    }
    assert len(rows) == 210
    assert set(rows_by_shape) == expected_shapes
    assert {row["analysis_type"] for row in rows} == {"same_shape_split_sweep"}
    assert {row["num_query_heads"] for row in rows} == {"8"}
    assert {row["num_kv_heads"] for row in rows} == {"8"}
    assert {row["head_dim"] for row in rows} == {"128"}
    assert {row["dtype"] for row in rows} == {"float16"}
    assert {row["block_size"] for row in rows} == {"32"}
    assert {row["warmup"] for row in rows} == {"50"}
    assert {row["repeat"] for row in rows} == {"300"}
    assert {row["correctness_guard"] for row in rows} == {"passed_single_pass_alignment"}

    for (batch_size, context_len), shape_rows in rows_by_shape.items():
        assert {row["implementation"] for row in shape_rows} == expected_implementations
        expected_num_splits = select_paged_decode_num_splits(
            batch_size=batch_size,
            num_heads=8,
            max_context_len=context_len,
            block_size=32,
        )
        expected_implementation = (
            "single" if expected_num_splits is None else f"split={expected_num_splits}"
        )
        adaptive_rows = [row for row in shape_rows if row["is_adaptive_choice"] == "true"]

        assert len(adaptive_rows) == 1
        assert adaptive_rows[0]["implementation"] == expected_implementation
        assert adaptive_rows[0]["adaptive_implementation"] == expected_implementation
        assert float(adaptive_rows[0]["adaptive_speedup_vs_single_p50"]) >= 1.0
        if batch_size >= 16 and context_len >= 4096:
            assert expected_implementation == "single"
