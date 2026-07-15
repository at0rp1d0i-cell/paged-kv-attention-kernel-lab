import pytest

from paged_kv_attention.benchmark_utils import (
    EqualWorkCase,
    analytical_kv_bytes,
    bandwidth_utilization_percent,
    effective_bandwidth_gbps,
    make_equal_work_cases,
    percentile,
    summarize_latencies,
)


def test_percentile_uses_linear_interpolation() -> None:
    samples = [1.0, 2.0, 3.0, 4.0]

    assert percentile(samples, 0.50) == pytest.approx(2.5)
    assert percentile(samples, 0.95) == pytest.approx(3.85)


def test_latency_summary_reports_distribution() -> None:
    stats = summarize_latencies([1.0, 2.0, 3.0])

    assert stats.p50_ms == pytest.approx(2.0)
    assert stats.p95_ms == pytest.approx(2.9)
    assert stats.mean_ms == pytest.approx(2.0)
    assert stats.min_ms == pytest.approx(1.0)
    assert stats.max_ms == pytest.approx(3.0)
    assert stats.samples == 3


def test_analytical_bandwidth_counts_batch_heads_k_and_v() -> None:
    bytes_read = analytical_kv_bytes(
        [128, 64],
        num_kv_heads=4,
        head_dim=128,
        dtype_size=2,
    )

    assert bytes_read == (128 + 64) * 2 * 4 * 128 * 2
    assert effective_bandwidth_gbps(1_000_000_000, 1_000.0) == pytest.approx(1.0)


def test_effective_bandwidth_rejects_non_positive_latency() -> None:
    with pytest.raises(ValueError, match="latency_ms must be positive"):
        effective_bandwidth_gbps(1, 0.0)


def test_bandwidth_utilization_compares_with_nominal_peak() -> None:
    assert bandwidth_utilization_percent(1_680.0, 1_792.0) == pytest.approx(93.75)


def test_bandwidth_utilization_rejects_invalid_peak() -> None:
    with pytest.raises(ValueError, match="peak_bandwidth must be positive"):
        bandwidth_utilization_percent(1.0, 0.0)


def test_equal_work_cases_match_kv_work_and_program_topology() -> None:
    cases = make_equal_work_cases(
        16384,
        [(1, 16), (2, 8), (4, 4), (16, None)],
    )

    assert cases == [
        EqualWorkCase(batch_size=1, context_len=16384, num_splits=16),
        EqualWorkCase(batch_size=2, context_len=8192, num_splits=8),
        EqualWorkCase(batch_size=4, context_len=4096, num_splits=4),
        EqualWorkCase(batch_size=16, context_len=1024, num_splits=None),
    ]
    assert {case.batch_size * case.context_len for case in cases} == {16384}
    assert {case.main_program_count(num_heads=8) for case in cases} == {128}
    assert {case.tokens_per_main_program() for case in cases} == {1024}
    assert cases[0].partial_state_bytes(num_heads=8, head_dim=128) == 16 * 8 * 130 * 4
    assert cases[-1].partial_state_bytes(num_heads=8, head_dim=128) == 0
    assert cases[-1].reduce_program_count(num_heads=8) == 0


def test_equal_work_cases_reject_mismatched_program_counts() -> None:
    with pytest.raises(ValueError, match=r"batch_size \* context_partitions"):
        make_equal_work_cases(16384, [(1, 8), (16, None)])


def test_equal_work_cases_reject_unsupported_split_count() -> None:
    with pytest.raises(ValueError, match="split counts must be one of"):
        make_equal_work_cases(16384, [(8, 2)])


@pytest.mark.parametrize("quantile", [-0.1, 1.1])
def test_percentile_rejects_invalid_quantile(quantile: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1.0], quantile)
