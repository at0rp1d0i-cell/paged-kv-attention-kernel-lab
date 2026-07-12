import torch

from paged_kv_attention.block_table import make_random_block_tables
from paged_kv_attention.reference import (
    dense_decode_attention,
    dense_decode_attention_online,
    paged_decode_attention,
)


def test_dense_decode_attention_matches_hand_checked_case() -> None:
    q = torch.tensor([[[2.0, -1.0]]])  # [B=1, H=1, D=2]

    k = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
        ]]
    )  # [B=1, S=3, H=1, D=2]

    v = torch.tensor(
        [[
            [[10.0, -2.0]],
            [[-3.0, 4.0]],
            [[5.0, 7.0]],
        ]]
    )  # [B=1, S=3, H=1, D=2]

    context_lens = torch.tensor([3])

    out = dense_decode_attention(q, k, v, context_lens, scale=1.0)

    probs = torch.softmax(torch.tensor([2.0, -1.0, 1.0]), dim=0)
    expected = torch.tensor(
        [
            [
                [
                    probs[0] * 10.0 + probs[1] * -3.0 + probs[2] * 5.0,
                    probs[0] * -2.0 + probs[1] * 4.0 + probs[2] * 7.0,
                ]
            ]
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(out, expected)


def test_dense_decode_attention_ignores_tokens_past_context_len() -> None:
    q = torch.tensor([[[2.0, -1.0]]])  # [B=1, H=1, D=2]

    k = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[1000.0, 1000.0]],
            [[-1000.0, 500.0]],
        ]]
    )  # [B=1, S=3, H=1, D=2]

    v = torch.tensor(
        [[
            [[10.0, -2.0]],
            [[9999.0, 9999.0]],
            [[-9999.0, 9999.0]],
        ]]
    )  # [B=1, S=3, H=1, D=2]

    context_lens = torch.tensor([1])

    out = dense_decode_attention(q, k, v, context_lens, scale=1.0)

    expected = torch.tensor([[[10.0, -2.0]]], dtype=torch.float32)

    torch.testing.assert_close(out, expected)


def test_dense_decode_attention_handles_multi_batch_and_multi_head() -> None:
    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, -1.0], [-1.0, 2.0]],
        ]
    )  # [B=2, H=2, D=2]

    k = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[50.0, 50.0], [-50.0, 50.0]],
            ],
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
        ]
    )  # [B=2, S=3, H=2, D=2]

    v = torch.tensor(
        [
            [
                [[10.0, 0.0], [0.0, 10.0]],
                [[0.0, 20.0], [20.0, 0.0]],
                [[9999.0, 9999.0], [-9999.0, 9999.0]],
            ],
            [
                [[1.0, -1.0], [2.0, -2.0]],
                [[3.0, -3.0], [4.0, -4.0]],
                [[5.0, -5.0], [6.0, -6.0]],
            ],
        ]
    )  # [B=2, S=3, H=2, D=2]

    context_lens = torch.tensor([2, 3])

    out = dense_decode_attention(q, k, v, context_lens, scale=1.0)

    expected = torch.empty_like(out)
    for b in range(q.shape[0]):
        valid_len = int(context_lens[b].item())
        for h in range(q.shape[1]):
            scores = (k[b, :valid_len, h] * q[b, h]).sum(dim=-1)
            probs = torch.softmax(scores, dim=0)
            expected[b, h] = (probs.unsqueeze(-1) * v[b, :valid_len, h]).sum(dim=0)

    torch.testing.assert_close(out, expected)


def test_paged_decode_attention_matches_dense_with_manual_block_table() -> None:
    q = torch.tensor([[[2.0, -1.0]]])  # [B=1, H=1, D=2]

    dense_k = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
        ]]
    )  # [B=1, S=3, H=1, D=2]

    dense_v = torch.tensor(
        [[
            [[10.0, -2.0]],
            [[-3.0, 4.0]],
            [[5.0, 7.0]],
        ]]
    )  # [B=1, S=3, H=1, D=2]

    block_size = 2
    block_tables = torch.tensor([[2, 0]])
    context_lens = torch.tensor([3])

    k_cache = torch.full((3, block_size, 1, 2), 9999.0)
    v_cache = torch.full((3, block_size, 1, 2), -9999.0)

    # logical block 0 -> physical block 2: token 0, token 1
    k_cache[2, 0] = dense_k[0, 0]
    k_cache[2, 1] = dense_k[0, 1]
    v_cache[2, 0] = dense_v[0, 0]
    v_cache[2, 1] = dense_v[0, 1]

    # logical block 1 -> physical block 0: token 2, then one unused garbage slot
    k_cache[0, 0] = dense_k[0, 2]
    v_cache[0, 0] = dense_v[0, 2]

    dense_out = dense_decode_attention(q, dense_k, dense_v, context_lens, scale=1.0)
    paged_out = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
        scale=1.0,
    )

    torch.testing.assert_close(paged_out, dense_out)


def test_paged_decode_attention_matches_dense_for_variable_length_batch() -> None:
    q = torch.tensor(
        [
            [[1.0, -1.0], [0.5, 2.0]],
            [[2.0, 0.5], [-1.0, 1.5]],
        ]
    )  # [B=2, H=2, D=2]

    dense_k = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 1.0], [1.0, -1.0]],
                [[99.0, 99.0], [-99.0, 99.0]],
                [[-99.0, 99.0], [99.0, -99.0]],
            ],
            [
                [[2.0, -1.0], [1.0, 1.0]],
                [[-1.0, 2.0], [2.0, -1.0]],
                [[0.5, 1.5], [-0.5, 1.5]],
                [[1.0, 0.5], [0.5, 1.0]],
                [[-1.5, 0.5], [1.5, -0.5]],
            ],
        ]
    )  # [B=2, S=5, H=2, D=2]

    dense_v = torch.tensor(
        [
            [
                [[10.0, 0.0], [0.0, 10.0]],
                [[0.0, 20.0], [20.0, 0.0]],
                [[5.0, -5.0], [-5.0, 5.0]],
                [[9999.0, 9999.0], [-9999.0, 9999.0]],
                [[-9999.0, 9999.0], [9999.0, -9999.0]],
            ],
            [
                [[1.0, -1.0], [2.0, -2.0]],
                [[3.0, -3.0], [4.0, -4.0]],
                [[5.0, -5.0], [6.0, -6.0]],
                [[7.0, -7.0], [8.0, -8.0]],
                [[9.0, -9.0], [10.0, -10.0]],
            ],
        ]
    )  # [B=2, S=5, H=2, D=2]

    block_size = 2
    block_tables = torch.tensor(
        [
            [4, 1, -1],
            [3, 0, 5],
        ]
    )
    context_lens = torch.tensor([3, 5])

    k_cache = torch.full((6, block_size, 2, 2), 7777.0)
    v_cache = torch.full((6, block_size, 2, 2), -7777.0)

    for b in range(dense_k.shape[0]):
        for t in range(int(context_lens[b].item())):
            logical_block = t // block_size
            slot = t % block_size
            physical_block = int(block_tables[b, logical_block].item())
            k_cache[physical_block, slot] = dense_k[b, t]
            v_cache[physical_block, slot] = dense_v[b, t]

    dense_out = dense_decode_attention(q, dense_k, dense_v, context_lens, scale=1.0)
    paged_out = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
        scale=1.0,
    )

    torch.testing.assert_close(paged_out, dense_out)


def test_paged_decode_attention_matches_dense_with_generated_block_tables() -> None:
    generator = torch.Generator().manual_seed(7)
    batch_size = 3
    max_context_len = 7
    num_heads = 2
    head_dim = 4
    block_size = 3

    q = torch.randn(batch_size, num_heads, head_dim, generator=generator)
    dense_k = torch.randn(batch_size, max_context_len, num_heads, head_dim, generator=generator)
    dense_v = torch.randn(batch_size, max_context_len, num_heads, head_dim, generator=generator)
    context_lens = torch.tensor([1, 4, 7])

    block_tables, num_physical_blocks = make_random_block_tables(
        context_lens,
        block_size=block_size,
        seed=11,
    )

    k_cache = torch.full(
        (num_physical_blocks, block_size, num_heads, head_dim),
        12345.0,
    )
    v_cache = torch.full(
        (num_physical_blocks, block_size, num_heads, head_dim),
        -12345.0,
    )

    for b in range(batch_size):
        for t in range(int(context_lens[b].item())):
            logical_block = t // block_size
            slot = t % block_size
            physical_block = int(block_tables[b, logical_block].item())
            k_cache[physical_block, slot] = dense_k[b, t]
            v_cache[physical_block, slot] = dense_v[b, t]

    dense_out = dense_decode_attention(q, dense_k, dense_v, context_lens)
    paged_out = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
    )

    torch.testing.assert_close(paged_out, dense_out)


def test_dense_decode_attention_online_matches_dense_across_tiles() -> None:
    generator = torch.Generator().manual_seed(17)
    batch_size = 3
    max_context_len = 9
    num_heads = 2
    head_dim = 8

    q = torch.randn(batch_size, num_heads, head_dim, generator=generator, dtype=torch.float16)
    k = torch.randn(
        batch_size,
        max_context_len,
        num_heads,
        head_dim,
        generator=generator,
        dtype=torch.float16,
    )
    v = torch.randn(
        batch_size,
        max_context_len,
        num_heads,
        head_dim,
        generator=generator,
        dtype=torch.float16,
    )
    context_lens = torch.tensor([1, 5, 9])

    dense_out = dense_decode_attention(q, k, v, context_lens)
    online_out = dense_decode_attention_online(
        q,
        k,
        v,
        context_lens,
        block_size=4,
    )

    torch.testing.assert_close(online_out, dense_out, rtol=1e-5, atol=1e-5)
