"""Block-table helpers for Week 1 paged KV tests.

The generator should eventually create random-order and fragmented physical
block mappings, then fill unused slots in the cache with garbage values. That
is how the tests catch accidental out-of-bounds reads.
"""

from __future__ import annotations

import torch


def blocks_per_sequence(context_lens: torch.Tensor, block_size: int) -> torch.Tensor:
    """Return the number of logical blocks required for each sequence."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if context_lens.ndim != 1:
        raise ValueError("context_lens must be a 1D tensor")
    if torch.any(context_lens < 0):
        raise ValueError("context_lens must be non-negative")
    return torch.div(context_lens + block_size - 1, block_size, rounding_mode="floor")


def make_random_block_tables(
    context_lens: torch.Tensor,
    *,
    block_size: int,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, int]:
    """Create random physical block tables for a variable-length batch.

    Args:
        context_lens: Valid context length per batch item, shape ``[batch]``.
        block_size: Number of token slots per physical block.
        seed: Optional random seed for reproducible tests.
        device: Optional output device. Defaults to ``context_lens.device``.

    Returns:
        A tuple ``(block_tables, num_physical_blocks)`` where ``block_tables`` has
        shape ``[batch, max_blocks_per_seq]`` and contains physical block ids.

    Week 1 requirement:
        The final implementation should produce non-contiguous, random-order block
        ids whenever possible. Unused table entries may use ``-1`` because
        ``context_lens`` defines the valid logical block range.
    """

    output_device = context_lens.device if device is None else torch.device(device)
    blocks_per_seq = blocks_per_sequence(context_lens, block_size).to(torch.long)
    batch_size = int(context_lens.shape[0])
    max_blocks_per_seq = int(blocks_per_seq.max().item()) if batch_size > 0 else 0
    total_required_blocks = int(blocks_per_seq.sum().item())

    # Allocate spare physical blocks so valid ids are not just a dense 0..N-1 prefix.
    # Tests can fill the spare blocks with garbage to catch accidental wrong reads.
    spare_blocks = 1 if total_required_blocks > 0 else 0
    num_physical_blocks = total_required_blocks + spare_blocks

    block_tables = torch.full(
        (batch_size, max_blocks_per_seq),
        -1,
        dtype=torch.long,
        device=output_device,
    )

    if total_required_blocks == 0:
        return block_tables, num_physical_blocks

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    physical_ids = torch.randperm(num_physical_blocks, generator=generator)[:total_required_blocks]
    cursor = 0
    for b, required_blocks in enumerate(blocks_per_seq.tolist()):
        if required_blocks == 0:
            continue
        block_tables[b, :required_blocks] = physical_ids[cursor : cursor + required_blocks].to(
            output_device
        )
        cursor += required_blocks

    return block_tables, num_physical_blocks
