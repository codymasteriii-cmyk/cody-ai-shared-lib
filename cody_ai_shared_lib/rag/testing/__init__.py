"""Conformance test kit, canary helpers, retrieval diff, and broken fixtures for RAGStore (§4.6, §14.2, AC-RAG-23).

Tools an app uses to prove its own integration is right, shipped with the library so the app tests against the same rules:

    RAGStoreContractTests     a pytest mixin: subclass it, implement `make_store()`, and every store rule is checked
    BrokenStore               a store with known defects, used to prove the kit really catches them
    assert_no_text_past_boundary / assert_canary_not_leaked    spoiler-guard checks: a secret word placed past the reader's
                              position must appear nowhere in results, prompt, answer, citations or stream
    diff_retrievals           compare two retrieval runs (before and after a profile or store change)
"""
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
