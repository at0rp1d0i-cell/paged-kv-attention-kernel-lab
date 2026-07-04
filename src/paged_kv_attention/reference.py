"""Week 1 PyTorch reference implementations.

The learning order is:

1. Implement ``dense_decode_attention`` as the FP32 ground truth.
2. Implement ``paged_decode_attention`` by matching dense semantics exactly.
3. Use tests to prove random block-table order and garbage slots do not affect output.

The core loops are intentionally left as TODOs for the first learning pass.
"""

from __future__ import annotations

import torch


def default_attention_scale(head_dim: int) -> float:
    """Return the standard ``1 / sqrt(head_dim)`` attention scale."""

    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    return head_dim**-0.5


def dense_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute dense decode attention for ``q_len=1``.

    Args:
        q: Query tensor with shape ``[batch, num_heads, head_dim]``.
        k: Dense key tensor with shape ``[batch, max_context_len, num_heads, head_dim]``.
        v: Dense value tensor with shape ``[batch, max_context_len, num_heads, head_dim]``.
        context_lens: Valid context length per batch item, shape ``[batch]``.
        scale: Optional attention scale. Defaults to ``1 / sqrt(head_dim)``.

    Returns:
        FP32 tensor with shape ``[batch, num_heads, head_dim]``.

    Correctness target:
        This is the ground truth for all Week 1 paged-reference tests. Each batch item
        must mask tokens at positions ``>= context_lens[b]`` before softmax.
    """

    _ = (q, k, v, context_lens, scale)
    raise NotImplementedError(
        "Week 1 TODO: implement dense_decode_attention before paged_decode_attention"
    )


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute paged decode attention for ``q_len=1``.

    Args:
        q: Query tensor with shape ``[batch, num_heads, head_dim]``.
        k_cache: Paged key cache with shape ``[num_blocks, block_size, num_heads, head_dim]``.
        v_cache: Paged value cache with shape ``[num_blocks, block_size, num_heads, head_dim]``.
        block_tables: Physical block ids per sequence, shape ``[batch, max_blocks_per_seq]``.
        context_lens: Valid context length per batch item, shape ``[batch]``.
        block_size: Number of token slots per physical block.
        scale: Optional attention scale. Defaults to ``1 / sqrt(head_dim)``.

    Returns:
        FP32 tensor with shape ``[batch, num_heads, head_dim]``.

    Correctness target:
        For token ``t`` in batch ``b``, read physical block
        ``block_tables[b, t // block_size]`` and slot ``t % block_size``. Never read
        slots at positions ``>= context_lens[b]``.
    """

    _ = (q, k_cache, v_cache, block_tables, context_lens, block_size, scale)
    raise NotImplementedError(
        "Week 1 TODO: implement paged_decode_attention after dense_decode_attention"
    )
