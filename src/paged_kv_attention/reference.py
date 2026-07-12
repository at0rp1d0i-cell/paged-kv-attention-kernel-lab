"""PyTorch reference implementations.

The learning order is:

1. Implement ``dense_decode_attention`` as the FP32 ground truth.
2. Implement ``paged_decode_attention`` by matching dense semantics exactly.
3. Use tests to prove random block-table order and garbage slots do not affect output.
4. Implement ``dense_decode_attention_online`` as the bridge to Triton.

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
    # Correctness checks: validate the dense layout and valid-token bounds first.
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, num_heads, head_dim]")
    if k.ndim != 4:
        raise ValueError("k must have shape [batch, max_context_len, num_heads, head_dim]")
    if v.ndim != 4:
        raise ValueError("v must have shape [batch, max_context_len, num_heads, head_dim]")

    B, H, D = q.shape
    k_B, S, k_H, k_D = k.shape
    v_B, v_S, v_H, v_D = v.shape

    if (k_B, k_H, k_D) != (B, H, D):
        raise ValueError("k must have shape [batch, max_context_len, num_heads, head_dim]")
    if (v_B, v_S, v_H, v_D) != (B, S, H, D):
        raise ValueError("v must have the same shape as k")

    if context_lens.ndim != 1 or context_lens.shape[0] != B:
        raise ValueError("context_lens must have shape [batch]")
    if torch.any(context_lens < 0):
        raise ValueError("context_lens must be non-negative")
    if torch.any(context_lens > S):
        raise ValueError("context_lens cannot exceed max_context_len")

    # Prepare FP32 tensors and output storage for the reference calculation.
    if scale is None:
        scale = default_attention_scale(D)

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)

    # Compute one decode attention output per batch item.
    for b in range(B):
        valid_len = int(context_lens[b].item())

        q_b = q_f[b]  # [H, D]
        k_b = k_f[b, :valid_len]  # [T, H, D], T = valid_len
        v_b = v_f[b, :valid_len]  # [T, H, D]

        # q_b.unsqueeze(0): [1, H, D]
        # k_b:              [T, H, D]
        # 相乘后:            [T, H, D]
        # sum(dim=-1):      [T, H]
        # transpose(0, 1):  [H, T]
        scores = (k_b * q_b.unsqueeze(0)).sum(dim=-1).transpose(0, 1) * scale

        # Softmax over the token dimension: [H, T] -> [H, T].
        probs = torch.softmax(scores, dim=-1)  # [H, T]
        v_by_head = v_b.transpose(0, 1)  # [T, H, D] -> [H, T, D]

        # probs.unsqueeze(-1): [H, T, 1]
        # v_by_head:           [H, T, D]
        # 相乘后:               [H, T, D]
        # sum(dim=1):          [H, D]
        out[b] = (probs.unsqueeze(-1) * v_by_head).sum(dim=1)

    return out


def dense_decode_attention_online(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute dense decode attention with block-wise online softmax.

    This is the Week 2 learning bridge between the dense PyTorch reference and
    the first Triton decode kernel. It intentionally keeps the same dense K/V
    layout as ``dense_decode_attention`` and changes only the softmax algorithm:
    instead of materializing all scores for one sequence, it should stream over
    token blocks and maintain running ``m``, ``l``, and ``acc``.

    Args:
        q: Query tensor with shape ``[batch, num_heads, head_dim]``.
        k: Dense key tensor with shape ``[batch, max_context_len, num_heads, head_dim]``.
        v: Dense value tensor with shape ``[batch, max_context_len, num_heads, head_dim]``.
        context_lens: Valid context length per batch item, shape ``[batch]``.
        block_size: Number of tokens to process per online-softmax step.
        scale: Optional attention scale. Defaults to ``1 / sqrt(head_dim)``.

    Returns:
        FP32 tensor with shape ``[batch, num_heads, head_dim]``.

    Correctness target:
        Match ``dense_decode_attention`` in FP32 for the same inputs. The core
        update should maintain:

        - ``m``: running max, shape ``[num_heads]``.
        - ``l``: running exp sum, shape ``[num_heads]``.
        - ``acc``: running weighted value sum, shape ``[num_heads, head_dim]``.
    """

    if q.ndim != 3:
        raise ValueError("q must have shape [batch, num_heads, head_dim]")
    if k.ndim != 4:
        raise ValueError("k must have shape [batch, max_context_len, num_heads, head_dim]")
    if v.ndim != 4:
        raise ValueError("v must have shape [batch, max_context_len, num_heads, head_dim]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    B, H, D = q.shape
    k_B, S, k_H, k_D = k.shape
    v_B, v_S, v_H, v_D = v.shape

    if (k_B, k_H, k_D) != (B, H, D):
        raise ValueError("k must have shape [batch, max_context_len, num_heads, head_dim]")
    if (v_B, v_S, v_H, v_D) != (B, S, H, D):
        raise ValueError("v must have the same shape as k")
    if context_lens.ndim != 1 or context_lens.shape[0] != B:
        raise ValueError("context_lens must have shape [batch]")
    if torch.any(context_lens < 0):
        raise ValueError("context_lens must be non-negative")
    if torch.any(context_lens > S):
        raise ValueError("context_lens cannot exceed max_context_len")

    if scale is None:
        scale = default_attention_scale(D)

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)

    for b in range(B):
        valid_len = int(context_lens[b].item())

        if valid_len == 0:
            out[b].zero_()
            continue

        q_b = q_f[b]  # [H, D]

        # Running online-softmax state, one independent stream per attention head.
        running_max = torch.full((H,), -torch.inf, dtype=torch.float32, device=q.device)
        running_sum = torch.zeros((H,), dtype=torch.float32, device=q.device)
        accumulator = torch.zeros((H, D), dtype=torch.float32, device=q.device)

        for start in range(0, valid_len, block_size):
            end = min(start + block_size, valid_len)
            k_tile = k_f[b, start:end]  # [TILE, H, D]
            v_tile = v_f[b, start:end]  # [TILE, H, D]

            # k_tile:             [TILE, H, D]
            # q_b.unsqueeze(0):   [1, H, D]
            # scores_tile:        [H, TILE]
            scores_tile = (k_tile * q_b.unsqueeze(0)).sum(dim=-1).transpose(0, 1) * scale

            # Local tile summary in its own max scale.
            m_tile = scores_tile.max(dim=-1).values  # [H]
            p_tile = torch.exp(scores_tile - m_tile.unsqueeze(-1))  # [H, TILE]
            l_tile = p_tile.sum(dim=-1)  # [H]

            # p_tile.unsqueeze(-1): [H, TILE, 1]
            # v_tile_by_head:       [H, TILE, D]
            # acc_tile:             [H, D]
            v_tile_by_head = v_tile.transpose(0, 1)
            acc_tile = (p_tile.unsqueeze(-1) * v_tile_by_head).sum(dim=1)

            # Update running max, exp sum, and weighted value sum.
            m_new = torch.maximum(running_max, m_tile)  # [H]
            old_scale = torch.exp(running_max - m_new)  # [H]
            tile_scale = torch.exp(m_tile - m_new)  # [H]

            running_sum = running_sum * old_scale + l_tile * tile_scale  # [H]
            accumulator = accumulator * old_scale.unsqueeze(-1) + acc_tile * tile_scale.unsqueeze(
                -1
            )  # [H, D]
            running_max = m_new

        out[b] = accumulator / running_sum.unsqueeze(-1)

    return out


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

    # Correctness checks: validate the paged layout and valid-token bounds first.
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, num_heads, head_dim]")
    if k_cache.ndim != 4:
        raise ValueError("k_cache must have shape [num_blocks, block_size, num_heads, head_dim]")
    if v_cache.ndim != 4:
        raise ValueError("v_cache must have shape [num_blocks, block_size, num_heads, head_dim]")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [batch, max_blocks_per_seq]")
    if context_lens.ndim != 1:
        raise ValueError("context_lens must have shape [batch]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    B, H, D = q.shape
    num_blocks, cache_block_size, k_H, k_D = k_cache.shape
    table_B, max_blocks_per_seq = block_tables.shape

    if cache_block_size != block_size:
        raise ValueError("k_cache block dimension must match block_size")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must have the same shape as k_cache")
    if (k_H, k_D) != (H, D):
        raise ValueError("k_cache must match q num_heads and head_dim")
    if table_B != B:
        raise ValueError("block_tables must have shape [batch, max_blocks_per_seq]")
    if context_lens.shape[0] != B:
        raise ValueError("context_lens must have shape [batch]")
    if torch.any(context_lens < 0):
        raise ValueError("context_lens must be non-negative")
    if torch.any(context_lens > max_blocks_per_seq * block_size):
        raise ValueError("context_lens cannot exceed block_tables capacity")

    used_block_ids = block_tables[block_tables >= 0]
    if used_block_ids.numel() > 0 and torch.any(used_block_ids >= num_blocks):
        raise ValueError("block_tables contains physical block ids outside k_cache")

    if scale is None:
        scale = default_attention_scale(D)

    q_f = q.to(torch.float32)
    k_cache_f = k_cache.to(torch.float32)
    v_cache_f = v_cache.to(torch.float32)

    out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)

    for b in range(B):
        valid_len = int(context_lens[b].item())

        if valid_len == 0:
            out[b].zero_()
            continue

        k_tokens = []
        v_tokens = []

        for t in range(valid_len):
            logical_block = t // block_size
            slot = t % block_size
            physical_block = int(block_tables[b, logical_block].item())

            if physical_block < 0:
                raise ValueError("block_tables is missing a physical block for a valid token")

            k_tokens.append(k_cache_f[physical_block, slot])
            v_tokens.append(v_cache_f[physical_block, slot])

        k_b = torch.stack(k_tokens, dim=0)  # [T, H, D]
        v_b = torch.stack(v_tokens, dim=0)  # [T, H, D]
        q_b = q_f[b]  # [H, D]

        # q_b.unsqueeze(0): [1, H, D]
        # k_b:              [T, H, D]
        # 相乘后:            [T, H, D]
        # sum(dim=-1):      [T, H]
        # transpose(0, 1):  [H, T]
        scores = (k_b * q_b.unsqueeze(0)).sum(dim=-1).transpose(0, 1) * scale

        # Softmax over the token dimension: [H, T] -> [H, T].
        probs = torch.softmax(scores, dim=-1)  # [H, T]
        v_by_head = v_b.transpose(0, 1)  # [T, H, D] -> [H, T, D]

        # probs.unsqueeze(-1): [H, T, 1]
        # v_by_head:           [H, T, D]
        # 相乘后:               [H, T, D]
        # sum(dim=1):          [H, D]
        out[b] = (probs.unsqueeze(-1) * v_by_head).sum(dim=1)

    return out
