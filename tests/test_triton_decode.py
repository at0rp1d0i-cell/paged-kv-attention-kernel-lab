import pytest
import torch

from paged_kv_attention.reference import dense_decode_attention
from paged_kv_attention.triton_decode import dense_decode_attention_triton


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
