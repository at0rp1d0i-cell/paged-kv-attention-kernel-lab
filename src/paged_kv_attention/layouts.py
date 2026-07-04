"""Layout contracts shared by Week 1 reference implementations.

Week 1 intentionally keeps the tensor layout small and explicit:

- q: ``[batch, num_heads, head_dim]``
- dense k/v: ``[batch, max_context_len, num_heads, head_dim]``
- paged k/v cache: ``[num_blocks, block_size, num_heads, head_dim]``
- block_tables: ``[batch, max_blocks_per_seq]``
- context_lens: ``[batch]``

These helpers are scaffolding. The attention math itself belongs in
``reference.py`` after you work through the dense reference first.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeLayout:
    """Static dimensions for a decode attention test case.

    The project starts with MHA, so ``num_query_heads`` and ``num_kv_heads``
    should match until the later GQA/MQA extension.
    """

    batch_size: int
    max_context_len: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int

    @property
    def max_blocks_per_seq(self) -> int:
        """Maximum logical blocks needed by one sequence."""

        return (self.max_context_len + self.block_size - 1) // self.block_size

    def validate_week1(self) -> None:
        """Validate the narrow Week 1 layout assumptions."""

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_context_len <= 0:
            raise ValueError("max_context_len must be positive")
        if self.num_query_heads <= 0:
            raise ValueError("num_query_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.num_query_heads != self.num_kv_heads:
            raise ValueError("Week 1 starts with MHA: num_query_heads must equal num_kv_heads")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
