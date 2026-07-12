import pytest
import torch

from paged_kv_attention.block_table import make_random_block_tables
from paged_kv_attention.reference import dense_decode_attention
from paged_kv_attention.triton_decode import (
    dense_decode_attention_triton,
    paged_decode_attention_triton,
)


pytestmark = pytest.mark.gpu


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
