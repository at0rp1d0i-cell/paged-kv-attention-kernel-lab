from pathlib import Path

import pytest
import torch

from paged_kv_attention.flashinfer_baseline import (
    make_flashinfer_page_metadata,
    prepare_cuda_home_layout,
)


def test_prepare_cuda_home_layout_adds_pip_wheel_compatibility_links(tmp_path: Path) -> None:
    cuda_home = tmp_path / "cu13"
    (cuda_home / "bin").mkdir(parents=True)
    (cuda_home / "nvvm" / "libdevice").mkdir(parents=True)
    (cuda_home / "lib").mkdir(parents=True)
    (cuda_home / "bin" / "nvcc").touch()
    (cuda_home / "bin" / "cicc").touch()
    (cuda_home / "lib" / "libcudart.so.13").touch()

    prepare_cuda_home_layout(cuda_home)
    prepare_cuda_home_layout(cuda_home)

    assert (cuda_home / "nvvm" / "bin" / "cicc").is_file()
    assert (cuda_home / "lib64").resolve() == (cuda_home / "lib").resolve()
    assert (cuda_home / "lib" / "libcudart.so").resolve() == (
        cuda_home / "lib" / "libcudart.so.13"
    ).resolve()


def test_make_flashinfer_page_metadata_compacts_variable_length_tables() -> None:
    block_tables = torch.tensor([[4, 1, -1], [3, 0, 2]], dtype=torch.int32)
    context_lens = torch.tensor([16, 33], dtype=torch.int32)

    indptr, indices, last_page_len = make_flashinfer_page_metadata(
        block_tables,
        context_lens,
        block_size=16,
    )

    torch.testing.assert_close(indptr, torch.tensor([0, 1, 4], dtype=torch.int32))
    torch.testing.assert_close(indices, torch.tensor([4, 3, 0, 2], dtype=torch.int32))
    torch.testing.assert_close(last_page_len, torch.tensor([16, 1], dtype=torch.int32))


def test_make_flashinfer_page_metadata_rejects_invalid_used_slot() -> None:
    block_tables = torch.tensor([[2, -1]], dtype=torch.int32)
    context_lens = torch.tensor([17], dtype=torch.int32)

    with pytest.raises(ValueError, match="must be non-negative"):
        make_flashinfer_page_metadata(block_tables, context_lens, block_size=16)
