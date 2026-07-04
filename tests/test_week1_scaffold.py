import torch

from paged_kv_attention.block_table import blocks_per_sequence, make_random_block_tables
from paged_kv_attention.layouts import DecodeLayout
from paged_kv_attention.reference import dense_decode_attention, paged_decode_attention


def test_decode_layout_week1_contract() -> None:
    layout = DecodeLayout(
        batch_size=2,
        max_context_len=17,
        num_query_heads=4,
        num_kv_heads=4,
        head_dim=8,
        block_size=16,
    )

    layout.validate_week1()

    assert layout.max_blocks_per_seq == 2


def test_blocks_per_sequence_handles_week1_boundaries() -> None:
    context_lens = torch.tensor([0, 1, 15, 16, 17, 32])

    assert blocks_per_sequence(context_lens, block_size=16).tolist() == [0, 1, 1, 1, 2, 2]


def test_dense_decode_attention_contract_or_todo() -> None:
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 5, 3, 8)
    v = torch.randn(2, 5, 3, 8)
    context_lens = torch.tensor([3, 5])

    try:
        out = dense_decode_attention(q, k, v, context_lens)
    except NotImplementedError as exc:
        assert "Week 1 TODO" in str(exc)
        return

    assert out.shape == q.shape
    assert out.dtype == torch.float32


def test_paged_decode_attention_contract_or_todo() -> None:
    q = torch.randn(2, 3, 8)
    k_cache = torch.randn(4, 4, 3, 8)
    v_cache = torch.randn(4, 4, 3, 8)
    block_tables = torch.tensor([[1, 3], [2, -1]])
    context_lens = torch.tensor([5, 3])

    try:
        out = paged_decode_attention(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            block_size=4,
        )
    except NotImplementedError as exc:
        assert "Week 1 TODO" in str(exc)
        return

    assert out.shape == q.shape
    assert out.dtype == torch.float32


def test_block_table_generator_contract_or_todo() -> None:
    context_lens = torch.tensor([1, 16, 17])

    try:
        block_tables, num_physical_blocks = make_random_block_tables(
            context_lens,
            block_size=16,
            seed=0,
        )
    except NotImplementedError as exc:
        assert "Week 1 TODO" in str(exc)
        return

    assert block_tables.shape == (3, 2)
    assert num_physical_blocks >= 4
