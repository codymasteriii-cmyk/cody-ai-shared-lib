"""Tests for NewsIntelligence migration (Phase P3):
  - AC-RAG-03: PassThroughSplitter covers one unit per node.
  - AC-RAG-17: TemporalResolver with injected Clock and applied_filters.
  - AC-RAG-18: Relaxation ladder touches only soft constraints.
  - AC-RAG-19: SecondaryQueryFanOut and fair interleaving for comparisons.
  - AC-RAG-35: RecencyBooster and ScoreFusionStage determinism under fixed clock.
  - AC-RAG-63: Snippets in keyword_only and prompt exclusion.
  - §8.10: Feed search sizing with 2,000 results.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from cody_ai_shared_lib.rag import (
    DefaultRenderer,
    FrozenClock,
    InMemoryRAGStore,
    ModelBindings,
    PassThroughSplitter,
    QueryIntent,
    RAGDocumentInput,
    RAGFilterSpec,
    RAGNode,
    RAGQueryRequest,
    RAGSectionInput,
    RAGEngine,
    RecencyBooster,
    ScoreFusionStage,
    SourceSpan,
    StoreCapabilities,
    news_chat,
    news_intelligence,
    news_search,
)
from cody_ai_shared_lib.rag.ports import EmbeddingSpace


def _setup_news_engine(clock=None, store_caps=None, relaxation=()):
    bindings = ModelBindings(
        generation=["gemini-3.5-flash-lite"],
        embedding="gemini-embedding-2",
        reranker="jina-reranker",
    )
    profile = news_intelligence(models=bindings).with_overrides(retrieval={"relaxation": list(relaxation)})
    caps = store_caps or StoreCapabilities(
        supports_vector_search=True,
        supports_keyword_fts=True,
        supports_native_hybrid_rpc=True,
        returns_raw_scores=True,
        returns_snippets=True,
    )
    caps = caps.model_copy(update={"supports_keyword_fts": True})
    store = InMemoryRAGStore(
        capabilities=caps,
        embedding_space=EmbeddingSpace(model="gemini-embedding-2", dims=768),
    )
    mock_gw = MagicMock()
    mock_gw.embed.side_effect = lambda texts, model, stage, usage_tracker, task_type="RETRIEVAL_DOCUMENT", **_: MagicMock(
        vectors=[[0.08] * 768 for _ in texts],
        usage=MagicMock(input_tokens=len(texts) * 10, cost_usd=0.0001, cached_tokens=0, price_table_version="v2026-11-08", estimated=False),
    )
    mock_gw.generate.return_value = MagicMock(text="News summary answer [1].")
    mock_gw.rerank.side_effect = lambda query, documents, model, stage, usage_tracker, degradation_tracker, top_n, **_: MagicMock(
        results=[MagicMock(index=i, relevance_score=0.9 - i * 0.05) for i in range(len(documents))],
        usage=MagicMock(cost_usd=0.0001, input_tokens=10),
    )

    engine = RAGEngine(profile=profile, store=store, gateway=mock_gw, clock=clock)
    return engine, store, mock_gw


def test_passthrough_splitter():
    """AC-RAG-03: PassThroughSplitter yields one node covering the whole unit."""
    splitter = PassThroughSplitter()
    text = "Whole article text with multiple sentences. Fact A. Fact B."
    chunks = splitter.split(text, base_offset=100)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].local_start == 0
    assert chunks[0].local_end == len(text)


def test_temporal_resolver_and_applied_filters():
    """AC-RAG-17: TemporalResolver with fixed clock resolves last week, reversed ranges repaired, applied_filters echoes."""
    # Fixed clock: Wednesday, Nov 11, 2026
    fixed_now = datetime(2026, 11, 11, 15, 30, 0, tzinfo=timezone.utc)
    clock = FrozenClock(fixed_now)
    engine, store, _ = _setup_news_engine(clock=clock)

    # 1. "last week" resolution
    req = RAGQueryRequest(question="What happened last week?")
    intent = engine.route(req)
    assert intent.has_time_expression is True
    assert intent.range_start_utc is not None
    assert intent.range_end_utc is not None
    # Start of last week is Monday Nov 2 00:00:00 UTC
    assert intent.range_start_utc.date() == datetime(2026, 11, 2).date()
    # End of last week is Sunday Nov 8 23:59:59 UTC
    assert intent.range_end_utc.date() == datetime(2026, 11, 8).date()

    # Retrieve echoes applied_filters
    ret = engine.retrieve(req, intent=intent)
    assert ret.applied_filters is not None
    assert ret.applied_filters.date_range_utc is not None
    assert ret.applied_filters.date_range_utc[0] == intent.range_start_utc
    assert ret.applied_filters.date_range_utc[1] == intent.range_end_utc

    # 2. Reversed range repair in TemporalResolver
    from cody_ai_shared_lib.rag.intent.temporal import TemporalResolver
    extractor = TemporalResolver(clock=clock)
    res = extractor.resolve_range(
        start=datetime(2026, 11, 10, tzinfo=timezone.utc),
        end=datetime(2026, 11, 5, tzinfo=timezone.utc),
    )
    # Start and end swapped
    assert res[0] < res[1]


def test_relaxation_ladder_soft_vs_hard():
    """AC-RAG-18: Relaxation ladder widens/drops soft intent window on empty results, never touches hard filters."""
    fixed_now = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)
    clock = FrozenClock(fixed_now)
    engine, store, _ = _setup_news_engine(clock=clock, relaxation=["widen_time_window(x2)", "drop_time_window"])

    # Ingest an article published 5 days ago (Nov 6)
    dt_article = fixed_now - timedelta(days=5)
    node = RAGNode(
        node_id="art-1",
        granularity="article",
        text="Semiconductor export controls announced.",
        span=SourceSpan(doc_id="art-1", char_start=0, char_end=41),
        metadata={"created_at": dt_article},
    )
    store.upsert_section_nodes("art-1", 0, [node])

    # Soft constraint: query for "yesterday" (Nov 10). Article from Nov 6 is outside 1-day window.
    # Initial search returns 0 -> relaxation ladder widens (x2) -> drops window -> finds article!
    req_soft = RAGQueryRequest(question="yesterday export controls")
    ret_soft = engine.retrieve(req_soft)
    assert len(ret_soft.nodes) > 0
    assert "drop_time_window" in ret_soft.relaxations or "widen_time_window(x2)" in ret_soft.relaxations

    # Hard constraint: caller explicitly passed hard date_range_utc in filters
    hard_filter = RAGFilterSpec(
        date_range_utc=(fixed_now - timedelta(days=1), fixed_now),
    )
    req_hard = RAGQueryRequest(question="export controls", filters=hard_filter)
    ret_hard = engine.retrieve(req_hard)
    # Hard filter cannot be relaxed; results remain empty
    assert len(ret_hard.nodes) == 0
    assert ret_hard.relaxations == []


def test_secondary_query_fanout_and_fair_interleave():
    """AC-RAG-19: Comparison query fans out to secondary queries and fairly interleaves both subjects."""
    engine, store, _ = _setup_news_engine()

    # Ingest articles about Apple and articles about Microsoft
    apple_node = RAGNode(
        node_id="apple-1",
        granularity="article",
        text="Apple reports record iPhone revenue.",
        embedding=[0.08] * 768,
        span=SourceSpan(doc_id="apple-1", char_start=0, char_end=37),
    )
    msft_node = RAGNode(
        node_id="msft-1",
        granularity="article",
        text="Microsoft expands Azure cloud infrastructure.",
        embedding=[0.08] * 768,
        span=SourceSpan(doc_id="msft-1", char_start=0, char_end=46),
    )
    store.upsert_section_nodes("apple-1", 0, [apple_node])
    store.upsert_section_nodes("msft-1", 0, [msft_node])

    # Comparison intent
    comparison_intent = QueryIntent(
        route="rag",
        scope="local_lookup",
        search_query="Apple financial results",
        secondary_queries=["Microsoft financial results"],
        is_comparison=True,
    )

    req = RAGQueryRequest(question="Compare Apple and Microsoft financials")
    ret = engine.retrieve(req, intent=comparison_intent)

    # Both subjects represented in candidate results
    ids = [n.node_id for n in ret.nodes]
    assert "apple-1" in ids
    assert "msft-1" in ids


def test_recency_booster_and_score_fusion():
    """AC-RAG-35: Recency booster and ScoreFusionStage are deterministic under fixed clock."""
    fixed_now = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)
    clock = FrozenClock(fixed_now)

    booster = RecencyBooster(half_life_days=7.0, weight=0.5, clock=clock)

    node_recent = RAGNode(
        node_id="n-rec",
        text="Recent news article",
        score=1.0,
        metadata={"created_at": fixed_now - timedelta(days=1)},
    )
    node_old = RAGNode(
        node_id="n-old",
        text="Old news article",
        score=1.0,
        metadata={"created_at": fixed_now - timedelta(days=21)},
    )

    boosted = booster.boost([node_old, node_recent])
    assert boosted[0].node_id == "n-rec"
    assert boosted[0].score > boosted[1].score

    # ScoreFusionStage
    fusion = ScoreFusionStage(weights={"semantic": 0.6, "keyword": 0.2, "recency": 0.1, "features": 0.1})
    node_recent.vector_score = 0.9
    node_recent.keyword_score = 5.0
    node_old.vector_score = 0.8
    node_old.keyword_score = 2.0

    fused = fusion.fuse([node_recent, node_old], clock=clock)
    assert all(n.score_kind == "fused" for n in fused)
    assert fused[0].node_id == "n-rec"


def test_snippet_return_and_prompt_exclusion():
    """AC-RAG-63: With returns_snippets=True, keyword_only returns snippet unchanged; prompt never contains snippet."""
    engine, store, _ = _setup_news_engine()

    node = RAGNode(
        node_id="doc-snip",
        granularity="article",
        text="The quick brown fox jumps over the lazy dog.",
        span=SourceSpan(doc_id="doc-snip", char_start=0, char_end=44),
    )
    store.upsert_section_nodes("doc-snip", 0, [node])

    req = RAGQueryRequest(question="quick brown fox", strategy_override="keyword_only")
    ret = engine.retrieve(req)

    assert len(ret.nodes) > 0
    assert ret.nodes[0].snippet is not None
    assert ret.nodes[0].snippet_kind == "match_excerpt"

    # Verify snippet is never included in prompt assembly
    assembled_sys, assembled_user, _ = engine.assembler.assemble(req, ret.nodes)
    assert "Excerpt:" not in assembled_sys
    assert "Excerpt:" not in assembled_user


def test_feed_search_sizing_and_2000_results():
    """§8.10: news_search variant supports up to 2,000 results and the truncated flag."""
    bindings = ModelBindings(generation=["gemini-3.5-flash-lite"], embedding="gemini-embedding-2")
    profile = news_search(models=bindings)
    assert profile.retrieval.retrieve_max_results == 2000

    caps = StoreCapabilities(
        supports_vector_search=True,
    )
    # A feed-search store declares how many rows it can return (1.2.0, review item M17): the engine
    # never asks a store for more than max_top_k, so news_search needs at least 2,001 here.
    caps = caps.model_copy(update={"supports_keyword_fts": True, "max_top_k": 2001})
    store = InMemoryRAGStore(
        capabilities=caps,
        embedding_space=EmbeddingSpace(model="gemini-embedding-2", dims=768),
    )
    mock_gw = MagicMock()
    mock_gw.embed.side_effect = lambda texts, model, stage, usage_tracker, task_type="RETRIEVAL_DOCUMENT", **_: MagicMock(
        vectors=[[0.05] * 768 for _ in texts],
        usage=MagicMock(input_tokens=len(texts) * 10, cost_usd=0.0001, cached_tokens=0, price_table_version="v2026-11-08", estimated=False),
    )

    engine = RAGEngine(profile=profile, store=store, gateway=mock_gw)

    # Populate store with 2,100 lightweight nodes
    total_items = 2100
    batch_nodes = [
        RAGNode(
            node_id=f"item-{i}",
            granularity="article",
            text=f"Headline {i} about economy.",
            embedding=[0.05] * 768,
            span=SourceSpan(doc_id=f"doc-{i}", char_start=0, char_end=25),
            metadata={"title": f"Title {i}", "category": "finance"},
        )
        for i in range(total_items)
    ]
    store.upsert_section_nodes("multi-doc", 0, batch_nodes)

    req = RAGQueryRequest(question="economy", strategy_override="dense_vector")
    ret = engine.retrieve(req)

    # Exactly 2,000 results returned
    assert len(ret.nodes) == 2000
    # More existed -> truncated flag is True
    assert ret.truncated is True
