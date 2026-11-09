"""Query transforms: producing the variants to search with, and merging their results (§7.8)."""
from .base import QueryTransform, QueryVariant, TransformContext
from .fanout import SecondaryQueryFanOut
from .fusion import fair_interleave, fuse_variants

__all__ = [
    "QueryTransform",
    "QueryVariant",
    "TransformContext",
    "SecondaryQueryFanOut",
    "fair_interleave",
    "fuse_variants",
]
