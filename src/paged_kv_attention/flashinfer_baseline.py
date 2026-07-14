"""Optional FlashInfer paged-decode baseline integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

import torch


def _ensure_symlink(link: Path, target: Path, *, relative_target: str) -> None:
    if link.exists() or link.is_symlink():
        return
    if not target.exists():
        raise RuntimeError(f"FlashInfer CUDA component is missing: {target}")
    link.symlink_to(relative_target, target_is_directory=target.is_dir())


def prepare_cuda_home_layout(cuda_home: Path) -> Path:
    """Add traditional CUDA toolkit paths missing from NVIDIA's pip wheels."""

    cuda_home = cuda_home.resolve()
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA compiler not found at {nvcc}")

    _ensure_symlink(
        cuda_home / "nvvm" / "bin",
        cuda_home / "bin",
        relative_target="../bin",
    )
    _ensure_symlink(
        cuda_home / "lib64",
        cuda_home / "lib",
        relative_target="lib",
    )
    _ensure_symlink(
        cuda_home / "lib" / "libcudart.so",
        cuda_home / "lib" / "libcudart.so.13",
        relative_target="libcudart.so.13",
    )
    return cuda_home


def configure_flashinfer_cuda_home() -> Path:
    """Select the pinned CUDA 13 pip toolkit before importing FlashInfer."""

    override = os.environ.get("PAGED_KV_FLASHINFER_CUDA_HOME")
    if override:
        cuda_home = Path(override)
    else:
        try:
            nvcc_distribution = distribution("nvidia-cuda-nvcc")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "FlashInfer requires the baseline dependency group; run "
                "`uv sync --locked --group baseline`."
            ) from exc
        cuda_home = Path(nvcc_distribution.locate_file("nvidia/cu13"))

    cuda_home = prepare_cuda_home_layout(cuda_home)
    os.environ["CUDA_HOME"] = str(cuda_home)
    return cuda_home


def make_flashinfer_page_metadata(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert padded block tables to FlashInfer's CSR page metadata."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_tables.ndim != 2 or context_lens.ndim != 1:
        raise ValueError("block_tables must be 2D and context_lens must be 1D")
    if block_tables.shape[0] != context_lens.shape[0]:
        raise ValueError("block_tables and context_lens batch dimensions must match")
    if block_tables.device != context_lens.device:
        raise ValueError("block_tables and context_lens must be on the same device")
    if bool(torch.any(context_lens <= 0)):
        raise ValueError("FlashInfer decode requires positive context lengths")

    page_counts = torch.div(context_lens + block_size - 1, block_size, rounding_mode="floor")
    if bool(torch.any(page_counts > block_tables.shape[1])):
        raise ValueError("block table is too short for at least one context length")

    indptr = torch.zeros(
        context_lens.shape[0] + 1,
        dtype=torch.int32,
        device=context_lens.device,
    )
    indptr[1:] = torch.cumsum(page_counts.to(torch.int32), dim=0)
    logical_pages = torch.arange(block_tables.shape[1], device=block_tables.device)
    valid_pages = logical_pages[None, :] < page_counts[:, None]
    indices = block_tables[valid_pages].to(torch.int32).contiguous()
    if bool(torch.any(indices < 0)):
        raise ValueError("valid block-table entries must be non-negative")
    last_page_len = (
        context_lens - (page_counts - 1) * block_size
    ).to(torch.int32).contiguous()
    return indptr, indices, last_page_len


@dataclass
class FlashInferPagedDecodeOperation:
    """Planned FlashInfer decode operation with preallocated output."""

    wrapper: Any
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    out: torch.Tensor
    workspace: torch.Tensor

    def run(self) -> torch.Tensor:
        return self.wrapper.run(
            self.q,
            (self.k_cache, self.v_cache),
            out=self.out,
        )


def plan_flashinfer_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    scale: float,
    workspace_bytes: int = 32 * 1024 * 1024,
) -> FlashInferPagedDecodeOperation:
    """Plan a FlashInfer paged decode outside the measured execution path."""

    configure_flashinfer_cuda_home()
    try:
        import flashinfer
    except ImportError as exc:
        raise RuntimeError(
            "FlashInfer is not installed; run `uv sync --locked --group baseline`."
        ) from exc

    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("expected q [B,H,D] and matching K/V cache [N,T,H,D]")
    if q.dtype != torch.float16 or k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise ValueError("the FlashInfer baseline currently requires FP16 Q/K/V")
    tensors = (q, k_cache, v_cache, block_tables, context_lens)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("the FlashInfer baseline requires CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("all FlashInfer baseline tensors must be on the same CUDA device")
    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be positive")

    _, num_heads, head_dim = q.shape
    if q.shape[0] != block_tables.shape[0] or q.shape[0] != context_lens.shape[0]:
        raise ValueError("q, block_tables, and context_lens batch dimensions must match")
    if k_cache.shape[1:] != (block_size, num_heads, head_dim):
        raise ValueError("paged K/V layout must be [num_blocks, block_size, H, D]")

    indptr, indices, last_page_len = make_flashinfer_page_metadata(
        block_tables,
        context_lens,
        block_size=block_size,
    )
    if bool(torch.any(indices >= k_cache.shape[0])):
        raise ValueError("block table references a physical page outside the K/V cache")
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend="auto",
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_heads,
        num_heads,
        head_dim,
        block_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
        o_data_type=q.dtype,
        sm_scale=scale,
    )
    return FlashInferPagedDecodeOperation(
        wrapper=wrapper,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        out=torch.empty_like(q),
        workspace=workspace,
    )
