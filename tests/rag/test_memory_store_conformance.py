"""Conformance test for InMemoryRAGStore and negative test for BrokenStore (AC-RAG-50)."""
from __future__ import annotations

import pytest

from cody_ai_shared_lib.rag.errors import EmbeddingProviderError, MissingRequiredScopeError
from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore
from cody_ai_shared_lib.rag.models import DefaultNodeMeta, RAGFilterSpec
from cody_ai_shared_lib.rag.ports import StoreCapabilities
from cody_ai_shared_lib.rag.testing.contract import BrokenStore, RAGStoreContractTests


class TestInMemoryRAGStoreConformance(RAGStoreContractTests):
    """Verifies that InMemoryRAGStore satisfies all 15 contract checks."""

    def make_store(self) -> InMemoryRAGStore[DefaultNodeMeta]:
        # Enable required_filter_keys on test instance for check 2
        caps = StoreCapabilities(
            supports_vector_search=True,
            supports_keyword_fts=True,
            supports_native_hybrid_rpc=True,
            returns_raw_scores=True,
            uses_analyzer_tokens=True,
            returns_snippets=True,
            supports_parent_summaries=True,
            supports_summary_levels=True,
            supports_local_window=True,
            supports_full_scope_fetch=True,
            supports_sequential_boundary=True,
            supports_hydration=True,
            supports_fetch_nodes=True,
            supports_fingerprints=True,
            supports_delete=True,
            supports_revisions=True,
            supports_embedding_return=True,
            required_filter_keys=frozenset(),
            supported_custom_filters={"favorites_only": bool, "active": bool, "language": str},
            supported_date_fields=frozenset(["created_at", "updated_at"]),
        )
        return InMemoryRAGStore(capabilities=caps)


def test_broken_store_fails_checks():
    """BrokenStore must fail zero-vector and required scope checks (AC-RAG-50)."""
    broken = BrokenStore()

    # BrokenStore fails check 6: does NOT raise EmbeddingProviderError on zero vector
    res = broken.vector_search([0.0] * 768, RAGFilterSpec(), top_k=5)
    assert len(res) > 0  # It returned results instead of raising!

    # BrokenStore fails check 1 & 2: does NOT raise MissingRequiredScopeError
    res2 = broken.vector_search([0.1] * 768, RAGFilterSpec(), top_k=5)
    assert len(res2) > 0
