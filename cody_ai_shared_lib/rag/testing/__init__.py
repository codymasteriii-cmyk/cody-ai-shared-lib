"""Conformance test kit, canary helpers, retrieval diff, and broken fixtures for RAGStore (§4.6, §14.2, AC-RAG-23)."""
from .canary import assert_canary_not_leaked, assert_no_text_past_boundary
from .contract import BrokenStore, RAGStoreContractTests
from .diff import diff_retrievals

__all__ = [
    "BrokenStore",
    "RAGStoreContractTests",
    "assert_canary_not_leaked",
    "assert_no_text_past_boundary",
    "diff_retrievals",
]
