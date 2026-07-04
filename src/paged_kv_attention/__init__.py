"""Paged-KV attention kernel lab package."""

__all__ = [
    "__version__",
    "dense_decode_attention",
    "make_random_block_tables",
    "paged_decode_attention",
]

__version__ = "0.0.0"

from paged_kv_attention.block_table import make_random_block_tables
from paged_kv_attention.reference import dense_decode_attention, paged_decode_attention
