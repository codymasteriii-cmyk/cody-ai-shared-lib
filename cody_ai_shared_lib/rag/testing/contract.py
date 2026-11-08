"""RAGStore conformance kit and BrokenStore fixture (§4.6, AC-RAG-50).

Provides:
  - RAGStoreContractTests: pytest mixin with 15 contract validation checks.
  - BrokenStore: deliberately non-conforming store that must fail checks.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
from typing import Any

import pytest

from ..errors import (
    EmbeddingProviderError,
    MissingRequiredScopeError,
    UnsupportedFilterError,
)
from ..models import (
    DefaultNodeMeta,
    RAGFilterSpec,
    RAGNode,
    SourceSpan,
)
from ..ports import (
    EmbeddingSpace,
    Fingerprint,
    RAGStore,
    ScopeStats,
    StoreCapabilities,
)
from ..text.analyzer import TokenGroup


class BrokenStore(RAGStore[DefaultNodeMeta]):
    """Deliberately non-conforming store used to verify that the test kit catches defects."""

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            supports_vector_search=True,
            supports_keyword_fts=True,
            required_filter_keys=frozenset(["tenant_id"]),
            supported_custom_filters={"active": bool},
        )

    def embedding_space(self) -> EmbeddingSpace:
        return EmbeddingSpace(model="gemini-embedding-2", dims=768)

    def vector_search(self, query_vector: list[float], filters: RAGFilterSpec, top_k: int) -> list[RAGNode[DefaultNodeMeta]]:
        # Defect 1: does NOT reject zero vectors
        # Defect 2: does NOT check required_filter_keys
        # Defect 3: ignores doc_ids filter
        return [
            RAGNode(
                node_id="broken_node_1",
                text="Some text",
                span=SourceSpan(doc_id="unrelated_doc", char_start=0, char_end=10),
                score=0.5,
                score_kind="cosine",
                metadata=DefaultNodeMeta(),
            )
        ]


class RAGStoreContractTests:
    """Pytest mixin checking all 15 RAGStore contract requirements (§4.6).

    Subclass this and implement `make_store()` in test suites.
    """

    def make_store(self) -> RAGStore[Any]:
        """Factory creating a clean store instance populated with test data if necessary."""
        raise NotImplementedError

    # ── Check 1: Filters honoured ──────────────────────────────────────────────
    def test_check_1_filters_honoured(self):
        store = self.make_store()
        span1 = SourceSpan(doc_id="doc1", char_start=0, char_end=100)
        span2 = SourceSpan(doc_id="doc2", char_start=0, char_end=100)
        n1 = RAGNode(node_id="n1", text="Node 1 text", span=span1, metadata=DefaultNodeMeta(extra={"tenant_id": "t1"}))
        n2 = RAGNode(node_id="n2", text="Node 2 text", span=span2, metadata=DefaultNodeMeta(extra={"tenant_id": "t2"}))

        vec = [0.1] * 768
        store.upsert_section_nodes("doc1", 0, [n1], [vec])
        store.upsert_section_nodes("doc2", 0, [n2], [vec])

        # doc_ids filter
        res = store.vector_search(vec, RAGFilterSpec(doc_ids=["doc1"]), top_k=5)
        assert len(res) == 1
        assert res[0].node_id == "n1"

        # sequential boundary
        res_bound = store.vector_search(vec, RAGFilterSpec(doc_ids=["doc1"], max_char_offset=0), top_k=5)
        assert len(res_bound) == 0

    # ── Check 2: Required scope enforced ───────────────────────────────────────
    def test_check_2_required_scope_enforced(self):
        store = self.make_store()
        caps = store.capabilities()
        if not caps.required_filter_keys:
            # If store has no required keys, skip or configure one for check
            return
        # Query missing required key must raise MissingRequiredScopeError
        with pytest.raises(MissingRequiredScopeError):
            store.vector_search([0.1] * 768, RAGFilterSpec(), top_k=5)

    # ── Check 3: Ordering descending and ranks 1-based ─────────────────────────
    def test_check_3_ordering_and_ranks(self):
        store = self.make_store()
        n1 = RAGNode(node_id="n1", text="Text 1", span=SourceSpan(doc_id="d1"), metadata=DefaultNodeMeta())
        n2 = RAGNode(node_id="n2", text="Text 2", span=SourceSpan(doc_id="d1"), metadata=DefaultNodeMeta())

        # Give n1 vector closer to query than n2
        store.upsert_section_nodes("d1", 0, [n1, n2], [[0.9] * 768, [0.1] * 768])
        res = store.vector_search([0.9] * 768, RAGFilterSpec(), top_k=2)
        assert len(res) == 2
        assert res[0].score >= res[1].score
        assert res[0].vector_rank == 1
        assert res[1].vector_rank == 2

    # ── Check 4: Idempotence, deletes, fingerprints ────────────────────────────
    def test_check_4_upsert_idempotence_and_deletes(self):
        store = self.make_store()
        n1 = RAGNode(node_id="n_idemp", text="Original text", span=SourceSpan(doc_id="d_idemp"), metadata=DefaultNodeMeta())
        store.upsert_section_nodes("d_idemp", 0, [n1], [[0.1] * 768])
        # Repeat upsert with same node_id
        store.upsert_section_nodes("d_idemp", 0, [n1], [[0.1] * 768])

        fp = store.get_fingerprints("d_idemp")
        assert "n_idemp" in fp

        deleted = store.delete_document("d_idemp")
        assert deleted == 1
        assert store.get_fingerprints("d_idemp") == {}

    # ── Check 5: Declared capabilities match working methods ───────────────────
    def test_check_5_capabilities_match_methods(self):
        store = self.make_store()
        caps = store.capabilities()
        if caps.supports_full_scope_fetch:
            res = store.fetch_full_scope(RAGFilterSpec())
            assert isinstance(res, list)

    # ── Check 6: Zero-vector query rejected ────────────────────────────────────
    def test_check_6_zero_vector_query_rejected(self):
        store = self.make_store()
        zero_vec = [0.0] * 768
        with pytest.raises(EmbeddingProviderError):
            store.vector_search(zero_vec, RAGFilterSpec(), top_k=5)

    # ── Check 7: Filtered recall ───────────────────────────────────────────────
    def test_check_7_filtered_recall(self):
        store = self.make_store()
        # Insert target document and multiple distractor documents
        target = RAGNode(node_id="target_n", text="Target text", span=SourceSpan(doc_id="target_doc"), metadata=DefaultNodeMeta())
        distractors = [
            RAGNode(node_id=f"dist_{i}", text=f"Distractor {i}", span=SourceSpan(doc_id=f"dist_doc_{i}"), metadata=DefaultNodeMeta())
            for i in range(10)
        ]
        store.upsert_section_nodes("target_doc", 0, [target], [[0.5] * 768])
        for i, d in enumerate(distractors):
            store.upsert_section_nodes(f"dist_doc_{i}", 0, [d], [[0.5] * 768])

        # Filter strictly for target_doc: must return target node
        res = store.vector_search([0.5] * 768, RAGFilterSpec(doc_ids=["target_doc"]), top_k=5)
        assert len(res) == 1
        assert res[0].node_id == "target_n"

    # ── Check 8: Native hybrid raw scores ──────────────────────────────────────
    def test_check_8_native_hybrid_raw_scores(self):
        store = self.make_store()
        if not store.capabilities().supports_native_hybrid_rpc:
            return
        n = RAGNode(node_id="h_node", text="hybrid content", span=SourceSpan(doc_id="h_doc"), metadata=DefaultNodeMeta())
        store.upsert_section_nodes("h_doc", 0, [n], [[0.5] * 768])
        res = store.hybrid_search("hybrid", [0.5] * 768, RAGFilterSpec(), top_k=1)
        assert len(res) == 1
        if store.capabilities().returns_raw_scores:
            assert res[0].vector_score is not None
            assert res[0].keyword_score is not None
        assert res[0].score_kind == "rrf"

    # ── Check 9: Hydrate and fetch_nodes honour filters ────────────────────────
    def test_check_9_hydrate_and_fetch_nodes_honour_filters(self):
        store = self.make_store()
        n = RAGNode(node_id="fn_1", text="fetch content", span=SourceSpan(doc_id="fn_doc"), metadata=DefaultNodeMeta(extra={"active": True}))
        store.upsert_section_nodes("fn_doc", 0, [n], [[0.1] * 768])

        # Passes when matching
        res = store.fetch_nodes(["fn_1"], RAGFilterSpec(doc_ids=["fn_doc"]))
        assert len(res) == 1

        # Omitted when filter does not match
        res_empty = store.fetch_nodes(["fn_1"], RAGFilterSpec(doc_ids=["other_doc"]))
        assert len(res_empty) == 0

        # Hydrate
        meta_map = store.hydrate(["fn_1"], projection="context")
        assert "fn_1" in meta_map

    # ── Check 10: CJK keyword search finds exact name ──────────────────────────
    def test_check_10_cjk_keyword_search(self):
        store = self.make_store()
        if not store.capabilities().supports_keyword_fts:
            return
        node = RAGNode(node_id="cjk_n", text="张三来到了北京大学", span=SourceSpan(doc_id="cjk_doc"), metadata=DefaultNodeMeta())
        store.upsert_section_nodes("cjk_doc", 0, [node], [[0.1] * 768])

        res = store.keyword_search("张三", RAGFilterSpec(), top_k=5)
        assert len(res) >= 1
        assert res[0].node_id == "cjk_n"

    # ── Check 11: Keyword-only hit survives cosine floor ───────────────────────
    def test_check_11_keyword_hit_survives_cosine_floor(self):
        store = self.make_store()
        if not store.capabilities().supports_native_hybrid_rpc:
            return
        # Node with keyword match but zero vector
        n_kw = RAGNode(node_id="kw_only", text="SpecialUniqueTerm", span=SourceSpan(doc_id="k_doc"), metadata=DefaultNodeMeta())
        store.upsert_section_nodes("k_doc", 0, [n_kw], [[0.0] * 768])

        # Filter with min_similarity=0.8
        res = store.hybrid_search("SpecialUniqueTerm", [0.5] * 768, RAGFilterSpec(min_similarity=0.8), top_k=5)
        # Keyword-only hit should survive because min_similarity applies to vector branch only
        assert any(hit.node_id == "kw_only" for hit in res)

    # ── Check 12: Atomic section writes ────────────────────────────────────────
    def test_check_12_atomic_section_writes(self):
        store = self.make_store()
        nodes = [
            RAGNode(node_id=f"atom_{i}", text=f"Text {i}", span=SourceSpan(doc_id="atom_doc"), metadata=DefaultNodeMeta())
            for i in range(5)
        ]
        embeddings = [[0.1] * 768 for _ in range(5)]

        # Read while concurrent write is happening
        def _writer():
            store.upsert_section_nodes("atom_doc", 0, nodes, embeddings)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut = executor.submit(_writer)
            read_res = store.vector_search([0.1] * 768, RAGFilterSpec(doc_ids=["atom_doc"]), top_k=10)
            fut.result()

        # Reader must see either 0 or all 5 nodes, never partial
        assert len(read_res) in (0, 5)

    # ── Check 13: CJK phrase precision ─────────────────────────────────────────
    def test_check_13_cjk_phrase_precision(self):
        store = self.make_store()
        if not store.capabilities().uses_analyzer_tokens:
            return
        # Full name vs partial name
        n_full = RAGNode(node_id="full_n", text="这里有李小明同学", span=SourceSpan(doc_id="d1"), metadata=DefaultNodeMeta())
        n_part = RAGNode(node_id="part_n", text="这里只有小明同学", span=SourceSpan(doc_id="d1"), metadata=DefaultNodeMeta())
        store.upsert_section_nodes("d1", 0, [n_full, n_part], [[0.1] * 768, [0.1] * 768])

        group = TokenGroup(tokens=["李小", "小明"], mode="phrase", origin="exact_term")
        res = store.keyword_search("李小明", RAGFilterSpec(), top_k=5, analyzer_tokens=[group])
        assert len(res) >= 1
        assert res[0].node_id == "full_n"

    # ── Check 14: Snippets arrived unchanged ───────────────────────────────────
    def test_check_14_snippets_arrive_unchanged(self):
        store = self.make_store()
        if not store.capabilities().returns_snippets:
            return
        node = RAGNode(node_id="snip_n", text="Snippet test text", span=SourceSpan(doc_id="s_doc"), metadata=DefaultNodeMeta())
        store.upsert_section_nodes("s_doc", 0, [node], [[0.1] * 768])
        res = store.keyword_search("Snippet", RAGFilterSpec(), top_k=1)
        assert len(res) == 1
        assert res[0].snippet is not None
        assert res[0].snippet_kind == "match_excerpt"

    # ── Check 15: Fingerprint parity ───────────────────────────────────────────
    def test_check_15_fingerprint_parity(self):
        store = self.make_store()
        node = RAGNode(node_id="fp_node", text="Fingerprint text", span=SourceSpan(doc_id="fp_doc"), metadata=DefaultNodeMeta())
        store.upsert_section_nodes("fp_doc", 0, [node], [[0.1] * 768])
        fps = store.get_fingerprints("fp_doc")
        assert "fp_node" in fps
        fp = fps["fp_node"]
        assert fp.content_hash is not None
        assert store.embedding_space().model in fp.embedding_signature
