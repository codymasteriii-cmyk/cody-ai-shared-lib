"""Tests for RAG core contracts, error hierarchy, granularity traits, and canonical data models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from cody_ai_shared_lib.rag import (
    BUILTIN_GRANULARITIES,
    TRAIT_ATOMIC,
    TRAIT_OFFSET_ADDRESSABLE,
    TRAIT_ORDERED,
    TRAIT_RERANKABLE,
    TRAIT_SUMMARY,
    BudgetExceeded,
    CitationCoverage,
    ConversationState,
    DeadlineExceeded,
    DefaultNodeMeta,
    DefaultRenderer,
    Degradation,
    EmbeddingProviderError,
    EmbeddingSignatureMismatch,
    EmbeddingSpace,
    Fingerprint,
    GenerationError,
    GranularityRegistry,
    IntentParseError,
    MissingRequiredScopeError,
    NodeRef,
    NodeRenderer,
    NoIntentExt,
    ProfileConfigError,
    QueryIntent,
    RAGCitation,
    RAGEngineError,
    RAGFilterSpec,
    RAGNode,
    RAGQueryRequest,
    RAGResponse,
    RAGRetrievalResult,
    RAGStageError,
    RAGStore,
    RAGStreamEvent,
    ScopeStats,
    SourceSpan,
    StageUsage,
    StoreCapabilities,
    StoreCapabilityError,
    UnsupportedFilterError,
    UsageTotal,
    get_traits,
    has_trait,
    register_granularity,
)


# ── Error Hierarchy Tests ──────────────────────────────────────────────────────

def test_error_hierarchy():
    assert issubclass(ProfileConfigError, RAGEngineError)
    assert issubclass(StoreCapabilityError, RAGEngineError)
    assert issubclass(MissingRequiredScopeError, RAGEngineError)
    assert issubclass(UnsupportedFilterError, RAGEngineError)
    assert issubclass(EmbeddingProviderError, RAGEngineError)
    assert issubclass(EmbeddingSignatureMismatch, RAGEngineError)
    assert issubclass(IntentParseError, RAGEngineError)
    assert issubclass(GenerationError, RAGEngineError)
    assert issubclass(RAGStageError, RAGEngineError)
    assert issubclass(BudgetExceeded, RAGEngineError)
    assert issubclass(DeadlineExceeded, RAGEngineError)


def test_rag_engine_error_attributes():
    usage = [StageUsage(stage="intent", input_tokens=100, output_tokens=50)]
    usage_total = UsageTotal(stage="total", input_tokens=100, output_tokens=50)
    degs = [Degradation(id="DG-06", stage="rerank", reason="timeout", impact="Results not reranked")]

    err = RAGEngineError(
        "failure occurred",
        stage="retrieval",
        request_id="req-123",
        degradations=degs,
        usage=usage,
        usage_total=usage_total,
    )
    assert err.stage == "retrieval"
    assert err.request_id == "req-123"
    assert len(err.degradations) == 1
    assert err.degradations[0].id == "DG-06"
    assert len(err.usage) == 1
    assert err.usage_total.input_tokens == 100


def test_generation_error_carries_retrieval():
    dummy_retrieval = RAGRetrievalResult(strategy_used="dense")
    gen_err = GenerationError(
        "generation truncated",
        reason="truncated",
        retrieval=dummy_retrieval,
    )
    assert gen_err.reason == "truncated"
    assert gen_err.retrieval is dummy_retrieval


# ── Canonical Models Tests ─────────────────────────────────────────────────────

def test_source_span_and_node_models():
    span = SourceSpan(
        doc_id="doc-1",
        section_id="sec-1",
        section_seq=1,
        breadcrumb=["Chapter 1", "Section A"],
        char_start=0,
        char_end=500,
    )
    node = RAGNode(
        node_id="node-1",
        text="Sample text content for node",
        span=span,
        granularity="passage",
        score=0.92,
        score_kind="cosine",
        vector_score=0.92,
        metadata=DefaultNodeMeta(extra={"author": "Cody"}),
    )

    assert node.node_id == "node-1"
    assert node.span.char_end == 500
    assert node.metadata.extra["author"] == "Cody"
    assert node.score_kind == "cosine"

    # JSON serialization roundtrip
    dumped = node.model_dump_json()
    loaded = RAGNode.model_validate_json(dumped)
    assert loaded.node_id == node.node_id
    assert loaded.score == node.score


def test_rag_filter_spec():
    filters = RAGFilterSpec(
        doc_ids=["doc-1", "doc-2"],
        tenant_or_user_id="user-42",
        max_char_offset=12000,
        custom={"favorites_only": True},
    )
    assert filters.doc_ids == ["doc-1", "doc-2"]
    assert filters.tenant_or_user_id == "user-42"
    assert filters.custom["favorites_only"] is True


def test_query_intent_and_request():
    intent = QueryIntent(
        route="rag",
        scope="local_lookup",
        search_query="Who is the author?",
        is_comparison=False,
        is_followup=True,
        exact_terms=["Cody"],
    )
    assert intent.is_followup is True
    assert intent.exact_terms == ["Cody"]

    req = RAGQueryRequest(
        question="What is the story about?",
        request_id="q-1",
        cursor_offset=5000,
    )
    assert req.question == "What is the story about?"
    assert req.cursor_offset == 5000


def test_rag_response_and_citations():
    citation = RAGCitation(
        marker="[1]",
        node_id="n-1",
        span=SourceSpan(doc_id="doc-1", char_start=10, char_end=100),
        label="Chapter 1",
        verified=True,
    )
    response = RAGResponse(
        answer_id="ans-001",
        answer="The answer is grounded [1].",
        answer_final="The answer is grounded [1].",
        citations=[citation],
        sources=[citation],
        grounded=True,
        answer_found="yes",
        abstained=False,
        latency_ms=120.5,
    )
    assert response.answer_found == "yes"
    assert response.grounded is True
    assert len(response.citations) == 1
    assert response.citations[0].label == "Chapter 1"


# ── Granularity & Traits Tests ─────────────────────────────────────────────────

def test_granularity_registry():
    assert has_trait("passage", TRAIT_OFFSET_ADDRESSABLE)
    assert has_trait("passage", TRAIT_RERANKABLE)
    assert not has_trait("passage", TRAIT_ATOMIC)

    assert has_trait("article", TRAIT_ATOMIC)
    assert has_trait("article", TRAIT_RERANKABLE)

    assert has_trait("parent_summary", TRAIT_SUMMARY)
    assert has_trait("parent_summary", TRAIT_ORDERED)
    assert has_trait("parent_summary", TRAIT_ATOMIC)

    # Register custom granularity
    register_granularity("entity_fact", [TRAIT_ATOMIC, TRAIT_ORDERED])
    assert has_trait("entity_fact", TRAIT_ATOMIC)
    assert has_trait("entity_fact", TRAIT_ORDERED)
    assert not has_trait("entity_fact", TRAIT_RERANKABLE)


# ── Renderer Protocol & DefaultRenderer Tests ──────────────────────────────────

def test_default_renderer():
    renderer = DefaultRenderer()
    assert isinstance(renderer, NodeRenderer)

    span = SourceSpan(doc_id="doc-test", section_title="Introduction", breadcrumb=["Book", "Ch 1"])
    node = RAGNode(node_id="n-1", text="Hello world", span=span, metadata=DefaultNodeMeta())

    assert renderer.embed_text(node) == "Hello world"
    assert renderer.rerank_text(node) == "Hello world"
    assert renderer.prompt_text(node) == "Hello world"
    assert renderer.citation_label(node) == "Introduction"
    assert renderer.breadcrumb(node) == ["Book", "Ch 1"]
    assert renderer.format_meta(node.metadata) == ""
    assert renderer.features(node.metadata) == {}


# ── Store Port & Capabilities Tests ────────────────────────────────────────────

def test_store_capabilities_and_port():
    caps = StoreCapabilities(
        supports_vector_search=True,
        supports_keyword_fts=True,
        supports_native_hybrid_rpc=True,
        required_filter_keys=frozenset(["tenant_id"]),
    )
    assert caps.supports_vector_search is True
    assert caps.supports_keyword_fts is True
    assert "tenant_id" in caps.required_filter_keys

    # Direct subclass check
    class IncompleteStore(RAGStore):
        pass

    # Trying to instantiate without abstract methods vector_search, capabilities, embedding_space must fail
    with pytest.raises(TypeError):
        IncompleteStore()  # type: ignore[abstract]
