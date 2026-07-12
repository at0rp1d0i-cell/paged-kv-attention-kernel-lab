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
    grid = (batch_size, num_heads)

    _dense_decode_attention_kernel[grid](
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
    return out
