import pytest
import torch

from paged_kv_attention.block_table import make_random_block_tables
from paged_kv_attention.reference import dense_decode_attention
from paged_kv_attention.triton_decode import (
    dense_decode_attention_triton,
    paged_decode_attention_adaptive_triton,
    paged_decode_attention_split_partials_triton,
    paged_decode_attention_split_triton,
    paged_decode_attention_triton,
)


pytestmark = pytest.mark.gpu


def _make_paged_split_case(
    context_lens: list[int],
    *,
    num_heads: int,
    block_size: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = len(context_lens)
    max_context_len = max(context_lens)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    q = torch.randn(
        batch_size,
        num_heads,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    dense_k = torch.randn(
        batch_size,
        max_context_len,
        num_heads,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    dense_v = torch.randn_like(dense_k)
    context_lens_tensor = torch.tensor(context_lens, device="cuda", dtype=torch.int32)
    block_tables, num_blocks = make_random_block_tables(
        context_lens_tensor,
        block_size=block_size,
        seed=seed + 1,
        device="cuda",
    )
    block_tables = block_tables.to(torch.int32).contiguous()

    k_cache = torch.full(
        (num_blocks, block_size, num_heads, 128),
        10_000.0,
        device="cuda",
        dtype=torch.float16,
    )
    v_cache = torch.full_like(k_cache, -10_000.0)
    for batch_idx, context_len in enumerate(context_lens):
        for token_idx in range(context_len):
            logical_block = token_idx // block_size
            slot = token_idx % block_size
            physical_block = int(block_tables[batch_idx, logical_block].item())
            k_cache[physical_block, slot] = dense_k[batch_idx, token_idx]
            v_cache[physical_block, slot] = dense_v[batch_idx, token_idx]

    return q, dense_k, dense_v, k_cache, v_cache, block_tables, context_lens_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batch_size", "num_heads", "max_context_len", "context_lens", "block_t"),
    [
        (1, 1, 17, [17], 32),
        (1, 4, 129, [129], 128),
        (2, 3, 257, [193, 257], 128),
    ],
)
def test_dense_decode_attention_triton_matches_fp32_reference(
    batch_size: int,
    num_heads: int,
    max_context_len: int,
    context_lens: list[int],
    block_t: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(31)
    q = torch.randn(
        batch_size,
        num_heads,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    k = torch.randn(
        batch_size,
        max_context_len,
        num_heads,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    context_lens_tensor = torch.tensor(context_lens, device="cuda", dtype=torch.int32)

    # Garbage past each valid length catches missing token masks in partial/full tail tiles.
    for batch_idx, context_len in enumerate(context_lens):
        k[batch_idx, context_len:] = 10_000.0
        v[batch_idx, context_len:] = -10_000.0

    expected = dense_decode_attention(q, k, v, context_lens_tensor)
    actual = dense_decode_attention_triton(
        q,
        k,
        v,
        context_lens_tensor,
        block_t=block_t,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batch_size", "num_heads", "max_context_len", "context_lens", "block_size", "block_t"),
    [
        (1, 1, 7, [7], 4, 8),
        (1, 4, 17, [17], 8, 16),
        (2, 3, 33, [19, 33], 8, 16),
    ],
)
def test_paged_decode_attention_triton_matches_fp32_reference(
    batch_size: int,
    num_heads: int,
    max_context_len: int,
    context_lens: list[int],
    block_size: int,
    block_t: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(47)
    q = torch.randn(
        batch_size,
        num_heads,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    dense_k = torch.randn(
        batch_size,
        max_context_len,
        num_heads,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    dense_v = torch.randn_like(dense_k)
    context_lens_tensor = torch.tensor(context_lens, device="cuda", dtype=torch.int32)
    block_tables, num_blocks = make_random_block_tables(
        context_lens_tensor,
        block_size=block_size,
        seed=53,
        device="cuda",
    )
    block_tables = block_tables.to(torch.int32).contiguous()

    # Unused blocks and slots remain garbage so wrong indirect reads corrupt the result.
    k_cache = torch.full(
        (num_blocks, block_size, num_heads, 128),
        10_000.0,
        device="cuda",
        dtype=torch.float16,
    )
    v_cache = torch.full_like(k_cache, -10_000.0)
    for batch_idx, context_len in enumerate(context_lens):
        for token_idx in range(context_len):
            logical_block = token_idx // block_size
            slot = token_idx % block_size
            physical_block = int(block_tables[batch_idx, logical_block].item())
            k_cache[physical_block, slot] = dense_k[batch_idx, token_idx]
            v_cache[physical_block, slot] = dense_v[batch_idx, token_idx]

    expected = dense_decode_attention(q, dense_k, dense_v, context_lens_tensor)
    actual = paged_decode_attention_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens_tensor,
        block_size=block_size,
        block_t=block_t,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_split_partial_single_split_matches_single_pass() -> None:
    q, _dense_k, _dense_v, k_cache, v_cache, block_tables, context_lens = _make_paged_split_case(
        [33, 19],
        num_heads=2,
        block_size=8,
        seed=71,
    )

    expected = paged_decode_attention_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=8,
        block_t=16,
    )
    partial_m, partial_l, partial_acc = paged_decode_attention_split_partials_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=8,
        num_splits=1,
        block_t=16,
    )

    assert partial_m.shape == (2, 2, 1)
    assert partial_l.shape == (2, 2, 1)
    assert partial_acc.shape == (2, 2, 1, 128)
    assert partial_m.dtype == torch.float32
    assert partial_l.dtype == torch.float32
    assert partial_acc.dtype == torch.float32

    actual = partial_acc[:, :, 0] / partial_l[:, :, 0, None]
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_split_partials_merge_and_empty_split_semantics() -> None:
    q, _dense_k, _dense_v, k_cache, v_cache, block_tables, context_lens = _make_paged_split_case(
        [3, 17],
        num_heads=2,
        block_size=8,
        seed=83,
    )

    expected = paged_decode_attention_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=8,
        block_t=8,
    )
    partial_m, partial_l, partial_acc = paged_decode_attention_split_partials_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=8,
        num_splits=4,
        block_t=8,
    )

    assert torch.isneginf(partial_m[0, :, 3]).all()
    torch.testing.assert_close(partial_l[0, :, 3], torch.zeros_like(partial_l[0, :, 3]))
    torch.testing.assert_close(
        partial_acc[0, :, 3],
        torch.zeros_like(partial_acc[0, :, 3]),
    )

    merged_m = partial_m.max(dim=2).values
    partial_scale = torch.exp(partial_m - merged_m.unsqueeze(2))
    merged_l = (partial_l * partial_scale).sum(dim=2)
    merged_acc = (partial_acc * partial_scale.unsqueeze(-1)).sum(dim=2)
    actual = merged_acc / merged_l.unsqueeze(-1)

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_splits", [0, 2, 3, 32])
def test_split_partials_reject_unsupported_split_counts(num_splits: int) -> None:
    q, _dense_k, _dense_v, k_cache, v_cache, block_tables, context_lens = _make_paged_split_case(
        [17],
        num_heads=1,
        block_size=8,
        seed=97,
    )

    with pytest.raises(ValueError, match="num_splits must be one of"):
        paged_decode_attention_split_partials_triton(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            block_size=8,
            num_splits=num_splits,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_splits", [True, 4.0])
def test_split_partials_reject_non_integer_split_counts(num_splits: object) -> None:
    q, _dense_k, _dense_v, k_cache, v_cache, block_tables, context_lens = _make_paged_split_case(
        [17],
        num_heads=1,
        block_size=8,
        seed=101,
    )

    with pytest.raises(TypeError, match="num_splits must be an integer"):
        paged_decode_attention_split_partials_triton(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            block_size=8,
            num_splits=num_splits,  # type: ignore[arg-type]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_split_partials_reuse_paged_input_validation() -> None:
    q, _dense_k, _dense_v, k_cache, v_cache, block_tables, context_lens = _make_paged_split_case(
        [7],
        num_heads=1,
        block_size=8,
        seed=107,
    )

    with pytest.raises(TypeError, match="must have dtype torch.float16"):
        paged_decode_attention_split_partials_triton(
            q.float(),
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            block_size=8,
            num_splits=4,
        )

    with pytest.raises(ValueError, match="block_t must be a positive power of two"):
        paged_decode_attention_split_partials_triton(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            block_size=8,
            num_splits=4,
            block_t=3,
        )

    invalid_context_lens = torch.tensor([9], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="cannot exceed block_tables capacity"):
        paged_decode_attention_split_partials_triton(
            q,
            k_cache,
            v_cache,
            block_tables,
            invalid_context_lens,
            block_size=8,
            num_splits=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_splits", [1, 4, 8, 16])
def test_split_decode_matches_fp32_reference(num_splits: int) -> None:
    q, dense_k, dense_v, k_cache, v_cache, block_tables, context_lens = _make_paged_split_case(
        [3, 19, 33],
        num_heads=2,
        block_size=8,
        seed=113,
    )

    expected = dense_decode_attention(
        q,
        dense_k,
        dense_v,
        context_lens,
    )
    actual = paged_decode_attention_split_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=8,
        num_splits=num_splits,
        block_t=8,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("context_lens", "block_size", "seed"),
    [
        ([33, 19], 32, 127),
        ([4096, 3073], 32, 131),
        ([257, 193], 16, 137),
    ],
)
def test_adaptive_decode_matches_fp32_reference(
    context_lens: list[int],
    block_size: int,
    seed: int,
) -> None:
    q, dense_k, dense_v, k_cache, v_cache, block_tables, context_lens_tensor = (
        _make_paged_split_case(
            context_lens,
            num_heads=2,
            block_size=block_size,
            seed=seed,
        )
    )
    expected = dense_decode_attention(q, dense_k, dense_v, context_lens_tensor)
    actual = paged_decode_attention_adaptive_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens_tensor,
        block_size=block_size,
        block_t=128,
    )

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
