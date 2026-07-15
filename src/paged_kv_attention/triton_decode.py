"""Triton implementation of dense decode attention."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from paged_kv_attention.reference import default_attention_scale


@triton.jit
def _dense_decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    context_lens_ptr,
    out_ptr,
    max_context_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    block_t: tl.constexpr,
    block_d: tl.constexpr,
):
    """Compute one ``out[batch, head, :]`` vector per Triton program."""

    batch_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)

    offs_d = tl.arange(0, block_d)  # [D]
    valid_d = offs_d < head_dim  # [D]

    q_ptrs = q_ptr + (batch_idx * num_heads + head_idx) * head_dim + offs_d
    q_vec = tl.load(q_ptrs, mask=valid_d, other=0.0).to(tl.float32)  # [D]
    context_len = tl.load(context_lens_ptr + batch_idx)  # scalar

    # One program owns one head, so m/l are scalars while acc remains a [D] vector.
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_d,), dtype=tl.float32)

    # Learning task: stream over token tiles, load K/V, compute scores, and apply
    # the online-softmax update before storing acc / l to out_ptr.
    for start in range(0, max_context_len, block_t):
        offs_t = start + tl.arange(0, block_t)  # [T]
        valid_t = offs_t < context_len  # [T]
        kv_mask = valid_t[:, None] & valid_d[None, :]  # [T, D]

        # fmt: off
        offs_kv = (
            ((batch_idx * max_context_len + offs_t[:, None])
            * num_heads + head_idx)
            * head_dim + offs_d[None, :]
        )  # K/V 使用相同 offset  # [T, D]
        # fmt: on

        k_ptrs = k_ptr + offs_kv
        v_ptrs = v_ptr + offs_kv

        k_tile = tl.load(
            k_ptrs,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)  # [T, D]
        v_tile = tl.load(
            v_ptrs,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)  # [T, D]

        # fmt: off
        scores = tl.sum(
            k_tile * q_vec[None, :],
            axis=1,
        ) * scale  # [T]
        # fmt: on
        scores = tl.where(valid_t, scores, -float("inf"))  # [T]

        m_tile = tl.max(scores, axis=0)  # [T] -> scalar
        p_tile = tl.where(
            valid_t,
            tl.exp(scores - m_tile),
            0.0,
        )  # [T]
        l_tile = tl.sum(p_tile, axis=0)  # [T] -> scalar
        # fmt: off
        acc_tile = tl.sum(
            p_tile[:, None] * v_tile,
            axis=0,
        )  # [T, D] -> [D]
        # fmt: on

        m_new = tl.maximum(running_max, m_tile)  # scalar
        old_scale = tl.exp(running_max - m_new)  # scalar
        tile_scale = tl.exp(m_tile - m_new)  # scalar
        # fmt: off
        l_new = (
            running_sum * old_scale
            + l_tile * tile_scale
        )  # scalar
        acc_new = (
            accumulator * old_scale
            + acc_tile * tile_scale
        )  # [D]
        # fmt: on

        running_max = m_new
        running_sum = l_new
        accumulator = acc_new

    output = accumulator / running_sum  # [D]
    # fmt: off
    out_ptrs = (
        out_ptr
        + (batch_idx * num_heads + head_idx) * head_dim
        + offs_d
    )
    # fmt: on
    tl.store(out_ptrs, output, mask=valid_d)


def dense_decode_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_t: int = 128,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute dense decode attention using a Triton kernel under study.

    Args:
        q: Contiguous FP16 CUDA tensor with shape ``[batch, num_heads, 128]``.
        k: Contiguous FP16 CUDA tensor with shape
            ``[batch, max_context_len, num_heads, 128]``.
        v: Contiguous FP16 CUDA tensor with the same shape as ``k``.
        context_lens: CUDA int32/int64 tensor with shape ``[batch]``. The current
            learning kernel requires every context length to be positive.
        block_t: Power-of-two number of context tokens processed per tile.
        scale: Optional attention scale. Defaults to ``1 / sqrt(head_dim)``.

    Returns:
        FP32 CUDA tensor with shape ``[batch, num_heads, 128]``.
    """

    if q.ndim != 3:
        raise ValueError("q must have shape [batch, num_heads, head_dim]")
    if k.ndim != 4:
        raise ValueError("k must have shape [batch, max_context_len, num_heads, head_dim]")
    if v.ndim != 4:
        raise ValueError("v must have shape [batch, max_context_len, num_heads, head_dim]")

    batch_size, num_heads, head_dim = q.shape
    k_batch, max_context_len, k_heads, k_head_dim = k.shape

    if (k_batch, k_heads, k_head_dim) != (batch_size, num_heads, head_dim):
        raise ValueError("k must have shape [batch, max_context_len, num_heads, head_dim]")
    if v.shape != k.shape:
        raise ValueError("v must have the same shape as k")
    if context_lens.ndim != 1 or context_lens.shape[0] != batch_size:
        raise ValueError("context_lens must have shape [batch]")
    if head_dim != 128:
        raise ValueError("the current Triton kernel requires head_dim=128")
    if max_context_len <= 0:
        raise ValueError("max_context_len must be positive")

    tensors = (q, k, v, context_lens)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("q, k, v, and context_lens must be on the same device")
    if not q.is_cuda:
        raise ValueError("q, k, v, and context_lens must be CUDA tensors")
    if any(tensor.dtype != torch.float16 for tensor in (q, k, v)):
        raise TypeError("q, k, and v must have dtype torch.float16")
    if context_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("context_lens must have dtype torch.int32 or torch.int64")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("q, k, v, and context_lens must be contiguous")

    if torch.any(context_lens <= 0):
        raise ValueError("the current Triton kernel requires positive context lengths")
    if torch.any(context_lens > max_context_len):
        raise ValueError("context_lens cannot exceed max_context_len")
    if block_t <= 0 or block_t & (block_t - 1):
        raise ValueError("block_t must be a positive power of two")

    if scale is None:
        scale = default_attention_scale(head_dim)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")

    out = torch.empty_like(q, dtype=torch.float32)
    _launch_dense_decode_attention_triton(
        q,
        k,
        v,
        context_lens,
        out,
        max_context_len=max_context_len,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
        block_t=block_t,
    )
    return out


def _launch_dense_decode_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    max_context_len: int,
    num_heads: int,
    head_dim: int,
    scale: float,
    block_t: int,
) -> None:
    """Launch the validated dense kernel into preallocated output storage."""

    _dense_decode_attention_kernel[(q.shape[0], num_heads)](
        q,
        k,
        v,
        context_lens,
        out,
        max_context_len=max_context_len,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
        block_t=block_t,
        block_d=128,
    )


@triton.jit
def _paged_decode_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    out_ptr,
    max_blocks_per_seq: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    block_t: tl.constexpr,
    block_d: tl.constexpr,
):
    """Compute one paged ``out[batch, head, :]`` vector per Triton program."""

    batch_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)

    offs_d = tl.arange(0, block_d)  # [D]
    valid_d = offs_d < head_dim  # [D]

    q_ptrs = q_ptr + (batch_idx * num_heads + head_idx) * head_dim + offs_d
    q_vec = tl.load(q_ptrs, mask=valid_d, other=0.0).to(tl.float32)  # [D]
    context_len = tl.load(context_lens_ptr + batch_idx)  # scalar

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_d,), dtype=tl.float32)

    max_context_len: tl.constexpr = max_blocks_per_seq * block_size
    for start in range(0, max_context_len, block_t):
        offs_t = start + tl.arange(0, block_t)  # [T]
        valid_t = offs_t < context_len  # [T]

        logical_blocks = offs_t // block_size  # [T]
        slots = offs_t % block_size  # [T]

        block_table_ptrs = block_tables_ptr + batch_idx * max_blocks_per_seq + logical_blocks  # [T]
        physical_blocks = tl.load(
            block_table_ptrs,
            mask=valid_t,
            other=0,
        )  # [T]

        # fmt: off
        offs_kv = (
            ((physical_blocks[:, None] * block_size + slots[:, None])
            * num_heads + head_idx)
            * head_dim + offs_d[None, :]
        )  # [T, D]
        # fmt: on

        k_ptrs = k_cache_ptr + offs_kv
        v_ptrs = v_cache_ptr + offs_kv

        kv_mask = valid_t[:, None] & valid_d[None, :]  # [T, D]

        k_tile = tl.load(
            k_ptrs,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)  # [T, D]
        v_tile = tl.load(
            v_ptrs,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)  # [T, D]

        # fmt: off
        scores = tl.sum(
            k_tile * q_vec[None, :],
            axis=1,
        ) * scale  # [T]
        # fmt: on
        scores = tl.where(valid_t, scores, -float("inf"))  # [T]

        m_tile = tl.max(scores, axis=0)  # [T] -> scalar
        p_tile = tl.where(
            valid_t,
            tl.exp(scores - m_tile),
            0.0,
        )  # [T]
        l_tile = tl.sum(p_tile, axis=0)  # [T] -> scalar
        # fmt: off
        acc_tile = tl.sum(
            p_tile[:, None] * v_tile,
            axis=0,
        )  # [T, D] -> [D]
        # fmt: on

        m_new = tl.maximum(running_max, m_tile)  # scalar
        old_scale = tl.exp(running_max - m_new)  # scalar
        tile_scale = tl.exp(m_tile - m_new)  # scalar
        # fmt: off
        l_new = (
            running_sum * old_scale
            + l_tile * tile_scale
        )  # scalar
        acc_new = (
            accumulator * old_scale
            + acc_tile * tile_scale
        )  # [D]
        # fmt: on

        running_max = m_new
        running_sum = l_new
        accumulator = acc_new

    output = accumulator / running_sum  # [D]
    # fmt: off
    out_ptrs = (
        out_ptr
        + (batch_idx * num_heads + head_idx) * head_dim
        + offs_d
    )
    # fmt: on
    tl.store(out_ptrs, output, mask=valid_d)


def _validate_paged_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    block_t: int,
    scale: float | None,
) -> tuple[int, int, int, float]:
    """Validate the shared contract for paged Triton decode kernels."""

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

    batch_size, num_heads, head_dim = q.shape
    num_blocks, cache_block_size, cache_heads, cache_head_dim = k_cache.shape
    table_batch, max_blocks_per_seq = block_tables.shape

    if cache_block_size != block_size:
        raise ValueError("k_cache block dimension must match block_size")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must have the same shape as k_cache")
    if (cache_heads, cache_head_dim) != (num_heads, head_dim):
        raise ValueError("k_cache must match q num_heads and head_dim")
    if table_batch != batch_size:
        raise ValueError("block_tables must have shape [batch, max_blocks_per_seq]")
    if context_lens.shape[0] != batch_size:
        raise ValueError("context_lens must have shape [batch]")
    if head_dim != 128:
        raise ValueError("the current Triton kernel requires head_dim=128")
    if num_blocks <= 0:
        raise ValueError("k_cache must contain at least one physical block")
    if max_blocks_per_seq <= 0:
        raise ValueError("block_tables must contain at least one logical block column")

    tensors = (q, k_cache, v_cache, block_tables, context_lens)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all inputs must be on the same device")
    if not q.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if any(tensor.dtype != torch.float16 for tensor in (q, k_cache, v_cache)):
        raise TypeError("q, k_cache, and v_cache must have dtype torch.float16")
    if block_tables.dtype not in (torch.int32, torch.int64):
        raise TypeError("block_tables must have dtype torch.int32 or torch.int64")
    if context_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("context_lens must have dtype torch.int32 or torch.int64")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous")

    if torch.any(context_lens <= 0):
        raise ValueError("the current Triton kernel requires positive context lengths")
    if torch.any(context_lens > max_blocks_per_seq * block_size):
        raise ValueError("context_lens cannot exceed block_tables capacity")
    if block_t <= 0 or block_t & (block_t - 1):
        raise ValueError("block_t must be a positive power of two")

    required_blocks = torch.div(
        context_lens + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    logical_block_ids = torch.arange(max_blocks_per_seq, device=q.device)
    required_entries = logical_block_ids.unsqueeze(0) < required_blocks.unsqueeze(1)
    required_block_ids = block_tables[required_entries]
    if torch.any(required_block_ids < 0):
        raise ValueError("block_tables is missing a physical block for a valid token")
    if torch.any(required_block_ids >= num_blocks):
        raise ValueError("block_tables contains physical block ids outside k_cache")

    if scale is None:
        scale = default_attention_scale(head_dim)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")

    return num_heads, head_dim, max_blocks_per_seq, scale


def paged_decode_attention_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    block_t: int = 128,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute paged decode attention using the Triton kernel under study.

    Args:
        q: Contiguous FP16 CUDA tensor with shape ``[batch, num_heads, 128]``.
        k_cache: Contiguous FP16 CUDA tensor with shape
            ``[num_blocks, block_size, num_heads, 128]``.
        v_cache: Contiguous FP16 CUDA tensor with the same shape as ``k_cache``.
        block_tables: Contiguous CUDA int32/int64 tensor with shape
            ``[batch, max_blocks_per_seq]``.
        context_lens: Contiguous CUDA int32/int64 tensor with shape ``[batch]``.
            The current learning kernel requires every context length to be positive.
        block_size: Number of token slots in each physical block.
        block_t: Power-of-two number of logical tokens processed per tile.
        scale: Optional attention scale. Defaults to ``1 / sqrt(head_dim)``.

    Returns:
        FP32 CUDA tensor with shape ``[batch, num_heads, 128]``.
    """

    num_heads, head_dim, max_blocks_per_seq, scale = _validate_paged_decode_inputs(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
        block_t=block_t,
        scale=scale,
    )

    out = torch.empty_like(q, dtype=torch.float32)
    _launch_paged_decode_attention_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
        block_t=block_t,
    )

    return out


@triton.jit
def _paged_decode_attention_split_partial_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    max_blocks_per_seq: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
    scale: tl.constexpr,
    block_t: tl.constexpr,
    block_d: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)
    split_idx = tl.program_id(axis=2)

    offs_d = tl.arange(0, block_d)  # [D]
    valid_d = offs_d < head_dim  # [D]

    q_ptrs = q_ptr + (batch_idx * num_heads + head_idx) * head_dim + offs_d
    q_vec = tl.load(q_ptrs, mask=valid_d, other=0.0).to(tl.float32)  # [D]
    context_len = tl.load(context_lens_ptr + batch_idx)  # scalar

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_d,), dtype=tl.float32)

    max_context_len: tl.constexpr = max_blocks_per_seq * block_size
    max_tokens_per_split: tl.constexpr = (
        max_context_len + num_splits - 1
    ) // num_splits  # 向上取整
    tokens_per_split = (context_len + num_splits - 1) // num_splits  # 向上取整

    split_start = split_idx * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, context_len)

    for local_start in range(0, max_tokens_per_split, block_t):
        local_offs_t = local_start + tl.arange(0, block_t)  # [T]
        offs_t = split_start + local_offs_t  # [T]

        # fmt: off
        valid_t = (
            (local_offs_t < max_tokens_per_split)
            & (offs_t < split_end)
        )  # [T]
        # fmt: on

        split_len = split_end - split_start
        tile_has_tokens = local_start < split_len

        logical_blocks = offs_t // block_size  # [T]
        slots = offs_t % block_size  # [T]

        # fmt: off
        block_table_ptrs = (
            block_tables_ptr
            + batch_idx * max_blocks_per_seq
            + logical_blocks
        )  # [T]

        physical_blocks = tl.load(
            block_table_ptrs,
            mask=valid_t,
            other=0,
        )  # [T]

        offs_kv = (
            ((physical_blocks[:, None] * block_size + slots[:, None])
            * num_heads + head_idx)
            * head_dim + offs_d[None, :]
        )  # [T, D]
        # fmt: on

        k_ptrs = k_cache_ptr + offs_kv
        v_ptrs = v_cache_ptr + offs_kv

        kv_mask = valid_t[:, None] & valid_d[None, :]  # [T, D]

        k_tile = tl.load(
            k_ptrs,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)  # [T, D]
        v_tile = tl.load(
            v_ptrs,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)  # [T, D]

        # fmt: off
        scores = tl.sum(
            k_tile * q_vec[None, :],
            axis=1,
        ) * scale  # [T]
        # fmt: on
        scores = tl.where(valid_t, scores, -float("inf"))  # [T]

        m_tile = tl.max(scores, axis=0)  # [T] -> scalar
        p_tile = tl.where(
            valid_t,
            tl.exp(scores - m_tile),
            0.0,
        )  # [T]
        l_tile = tl.sum(p_tile, axis=0)  # [T] -> scalar
        # fmt: off
        acc_tile = tl.sum(
            p_tile[:, None] * v_tile,
            axis=0,
        )  # [T, D] -> [D]
        # fmt: on

        m_new = tl.maximum(running_max, m_tile)  # scalar
        old_scale = tl.exp(running_max - m_new)  # scalar
        tile_scale = tl.exp(m_tile - m_new)  # scalar
        # fmt: off
        l_new = (
            running_sum * old_scale
            + l_tile * tile_scale
        )  # scalar
        acc_new = (
            accumulator * old_scale
            + acc_tile * tile_scale
        )  # [D]
        # fmt: on

        running_max = tl.where(
            tile_has_tokens,
            m_new,
            running_max,
        )
        running_sum = tl.where(
            tile_has_tokens,
            l_new,
            running_sum,
        )
        accumulator = tl.where(
            tile_has_tokens,
            acc_new,
            accumulator,
        )

    partial_idx = (
        batch_idx * num_heads + head_idx
    ) * num_splits + split_idx  # partial index for the current program

    split_has_tokens = split_start < split_end

    stored_m = tl.where(
        split_has_tokens,
        running_max,
        -float("inf"),
    )
    stored_l = tl.where(
        split_has_tokens,
        running_sum,
        0.0,
    )
    stored_acc = tl.where(
        split_has_tokens,
        accumulator,
        0.0,
    )

    partial_m_ptrs = partial_m_ptr + partial_idx
    partial_l_ptrs = partial_l_ptr + partial_idx
    partial_acc_ptrs = partial_acc_ptr + partial_idx * head_dim + offs_d

    tl.store(partial_m_ptrs, stored_m)
    tl.store(partial_l_ptrs, stored_l)
    tl.store(partial_acc_ptrs, stored_acc, mask=valid_d)


@triton.jit
def _paged_decode_attention_split_reduce_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    out_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
):
    """Reduce one set of split-KV partial states per ``(batch, head)`` program."""

    batch_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)

    offs_s = tl.arange(0, block_s)  # [S]
    valid_s = offs_s < num_splits  # [S]
    offs_d = tl.arange(0, block_d)  # [D]
    valid_d = offs_d < head_dim  # [D]

    partial_base = (batch_idx * num_heads + head_idx) * num_splits
    partial_m_ptrs = partial_m_ptr + partial_base + offs_s  # [S]
    partial_l_ptrs = partial_l_ptr + partial_base + offs_s  # [S]
    # fmt: off
    partial_acc_ptrs = (
        partial_acc_ptr
        + (partial_base + offs_s[:, None]) * head_dim
        + offs_d[None, :]
    )  # [S, D]
    # fmt: on

    _partial_m = tl.load(
        partial_m_ptrs,
        mask=valid_s,
        other=-float("inf"),
    ).to(tl.float32)  # [S]
    _partial_l = tl.load(
        partial_l_ptrs,
        mask=valid_s,
        other=0.0,
    ).to(tl.float32)  # [S]
    _partial_acc = tl.load(
        partial_acc_ptrs,
        mask=valid_s[:, None] & valid_d[None, :],
        other=0.0,
    ).to(tl.float32)  # [S, D]

    merged_m = tl.max(_partial_m, axis=0)  # [S] -> scalar

    # _partial_m [S] - merged_m scalar -> [S]; exp preserves [S].
    split_scales = tl.exp(_partial_m - merged_m)  # [S]

    # _partial_l [S] * split_scales [S] -> [S]; reduce S -> scalar.
    merged_l = tl.sum(_partial_l * split_scales, axis=0)  # scalar

    # split_scales[:, None]: [S, 1]; _partial_acc: [S, D].
    # Multiplication broadcasts to [S, D], then reducing S produces [D].
    merged_acc = tl.sum(
        _partial_acc * split_scales[:, None],
        axis=0,
    )  # [D]

    output = merged_acc / merged_l  # [D] / scalar -> [D]
    _out_ptrs = out_ptr + (batch_idx * num_heads + head_idx) * head_dim + offs_d

    tl.store(_out_ptrs, output, mask=valid_d)


def _launch_paged_decode_attention_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    max_blocks_per_seq: int,
    block_size: int,
    num_heads: int,
    head_dim: int,
    scale: float,
    block_t: int,
) -> None:
    """Launch the validated paged kernel into preallocated output storage."""

    _paged_decode_attention_kernel[(q.shape[0], num_heads)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
        block_t=block_t,
        block_d=128,
    )


def _launch_paged_decode_attention_split_partial_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    *,
    max_blocks_per_seq: int,
    block_size: int,
    num_heads: int,
    head_dim: int,
    num_splits: int,
    scale: float,
    block_t: int,
) -> None:
    """Launch the split-KV partial kernel into preallocated state buffers."""

    _paged_decode_attention_split_partial_kernel[(q.shape[0], num_heads, num_splits)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        partial_m,
        partial_l,
        partial_acc,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_splits=num_splits,
        scale=scale,
        block_t=block_t,
        block_d=128,
    )


def _launch_paged_decode_attention_split_reduce_triton(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    out: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    num_splits: int,
) -> None:
    """Launch the split-KV reduce kernel into preallocated output storage."""

    _paged_decode_attention_split_reduce_kernel[(partial_m.shape[0], num_heads)](
        partial_m,
        partial_l,
        partial_acc,
        out,
        num_heads=num_heads,
        head_dim=head_dim,
        num_splits=num_splits,
        block_s=triton.next_power_of_2(num_splits),
        block_d=128,
    )


def paged_decode_attention_split_partials_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    num_splits: int,
    block_t: int = 128,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute validated split-KV partial online-softmax states.

    Returns:
        FP32 CUDA tensors ``partial_m`` and ``partial_l`` with shape
        ``[batch, num_heads, num_splits]``, plus ``partial_acc`` with shape
        ``[batch, num_heads, num_splits, 128]``.
    """

    if not isinstance(num_splits, int) or isinstance(num_splits, bool):
        raise TypeError("num_splits must be an integer")
    if num_splits not in (1, 4, 8, 16):
        raise ValueError("num_splits must be one of 1, 4, 8, or 16")

    num_heads, head_dim, max_blocks_per_seq, scale = _validate_paged_decode_inputs(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
        block_t=block_t,
        scale=scale,
    )

    partial_shape = (q.shape[0], num_heads, num_splits)
    partial_m = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
    partial_l = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
    partial_acc = torch.empty(
        (*partial_shape, head_dim),
        device=q.device,
        dtype=torch.float32,
    )

    _launch_paged_decode_attention_split_partial_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        partial_m,
        partial_l,
        partial_acc,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_splits=num_splits,
        scale=scale,
        block_t=block_t,
    )

    return partial_m, partial_l, partial_acc


def paged_decode_attention_split_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    num_splits: int,
    block_t: int = 128,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute paged decode attention with split-KV partial and reduce kernels."""

    partial_m, partial_l, partial_acc = paged_decode_attention_split_partials_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
        num_splits=num_splits,
        block_t=block_t,
        scale=scale,
    )

    _, num_heads, num_splits = partial_m.shape
    head_dim = partial_acc.shape[3]
    out = torch.empty_like(q, dtype=torch.float32)
    _launch_paged_decode_attention_split_reduce_triton(
        partial_m,
        partial_l,
        partial_acc,
        out,
        num_heads=num_heads,
        head_dim=head_dim,
        num_splits=num_splits,
    )

    return out


def select_paged_decode_num_splits(
    *,
    batch_size: int,
    num_heads: int,
    max_context_len: int,
    block_size: int,
) -> int | None:
    """Select the measured RTX 5090 split-KV policy, or ``None`` for single-pass.

    The policy is calibrated from the canonical FP16, ``head_dim=128``,
    ``block_size=32`` same-shape sweep. Other block sizes conservatively use the
    single-pass path until they have their own benchmark evidence.
    """

    values = {
        "batch_size": batch_size,
        "num_heads": num_heads,
        "max_context_len": max_context_len,
        "block_size": block_size,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    base_programs = batch_size * num_heads
    if block_size != 32 or max_context_len <= 1024 or base_programs >= 256:
        return None
    if base_programs >= 128:
        return 4 if max_context_len <= 2048 else None
    if base_programs >= 64:
        return 4
    if base_programs >= 32:
        if max_context_len <= 2048:
            return 4
        if max_context_len <= 8192:
            return 8
        return 4
    return 4 if max_context_len <= 2048 else 16


def paged_decode_attention_adaptive_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    block_t: int = 128,
    scale: float | None = None,
) -> torch.Tensor:
    """Dispatch paged decode attention to the measured single-pass or split-KV path.

    The maximum valid context length drives the current variable-length batch policy.
    This preserves correctness for uneven batches but is only performance-calibrated
    for the equal-length shapes in the canonical RTX 5090 benchmark.
    """

    num_heads, head_dim, max_blocks_per_seq, scale = _validate_paged_decode_inputs(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
        block_t=block_t,
        scale=scale,
    )
    max_context_len = int(torch.max(context_lens).item())
    num_splits = select_paged_decode_num_splits(
        batch_size=q.shape[0],
        num_heads=num_heads,
        max_context_len=max_context_len,
        block_size=block_size,
    )
    out = torch.empty_like(q, dtype=torch.float32)
    if num_splits is None:
        _launch_paged_decode_attention_triton(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            out,
            max_blocks_per_seq=max_blocks_per_seq,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            block_t=block_t,
        )
        return out

    partial_shape = (q.shape[0], num_heads, num_splits)
    partial_m = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
    partial_l = torch.empty(partial_shape, device=q.device, dtype=torch.float32)
    partial_acc = torch.empty(
        (*partial_shape, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    _launch_paged_decode_attention_split_partial_triton(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        partial_m,
        partial_l,
        partial_acc,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_splits=num_splits,
        scale=scale,
        block_t=block_t,
    )
    _launch_paged_decode_attention_split_reduce_triton(
        partial_m,
        partial_l,
        partial_acc,
        out,
        num_heads=num_heads,
        head_dim=head_dim,
        num_splits=num_splits,
    )
    return out
