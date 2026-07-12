"""Paged-KV attention kernel lab package."""

__all__ = [
    "__version__",
    "dense_decode_attention",
    "dense_decode_attention_online",
    "dense_decode_attention_triton",
    "make_random_block_tables",
    "paged_decode_attention",
]

__version__ = "0.0.0"

from paged_kv_attention.block_table import make_random_block_tables
from paged_kv_attention.reference import (
    dense_decode_attention,
    dense_decode_attention_online,
    paged_decode_attention,
)
from paged_kv_attention.triton_decode import dense_decode_attention_triton
