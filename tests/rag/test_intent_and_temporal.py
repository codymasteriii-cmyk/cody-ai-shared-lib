"""Tests for query understanding, intent routers, and temporal extraction.

Covers:
  - AC-RAG-12: Passthrough router preserves query, zero model calls
  - AC-RAG-13: Heuristic router classifies overview, follow-up, and greeting (EN & ZH) without model calls
  - AC-RAG-14: Jev System One fast path and DG-03 fallback
  - AC-RAG-15 / AC-RAG-16: DG-05 value clamping, list capping, and supplied intent re-clamping
  - AC-RAG-17: Fixed clock calendar resolution (last week, yesterday, quarterly, range repair)
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest

from cody_ai_shared_lib.llm import LLMResult, LLMUsage
from cody_ai_shared_lib.rag import (
    HybridIntentRouter,
    CallContext,
    DegradationTracker,
    EngineProfile,
    FrozenClock,
    HeuristicIntentRouter,
    HeuristicRouter,
    IntentParseError,
    JevGateRouter,
    LLMIntentRouter,
    ModelBindings,
    ModelGateway,
    PassthroughRouter,
    ProfileConfigError,
    QueryIntent,
    RAGQueryRequest,
    RouterConfig,
    RouterContext,
    TemporalResolver,
    UsageTracker,
    clamp_intent,
    knowledge_base,
    long_form_reader,
)


# ── AC-RAG-17: Temporal Extractor Tests ────────────────────────────────────────

def test_temporal_extractor_last_week_calendar_math():
    """AC-RAG-17: 'last week' resolves to UTC calendar days (Monday to Sunday of previous week)."""
    # Fixed clock: Wednesday Nov 11, 2026 14:00:00 UTC (weekday 2)
    fixed_now = datetime(2026, 11, 11, 14, 0, 0, tzinfo=timezone.utc)
    clock = FrozenClock(fixed_now)
    extractor = TemporalResolver(clock=clock)

    res = extractor.extract("What happened in AI last week?")
    assert res["has_time_expression"] is True
    assert res["temporal_scope"] == "recent"
    assert res["date_window_days"] == 7

    # Previous Monday: Nov 2, 2026 00:00:00 UTC
    # Previous Sunday: Nov 8, 2026 23:59:59.999999 UTC
    start = res["range_start_utc"]
    end = res["range_end_utc"]
    assert start == datetime(2026, 11, 2, 0, 0, 0, tzinfo=timezone.utc)
    assert end.date() == datetime(2026, 11, 8, tzinfo=timezone.utc).date()
    assert end.hour == 23 and end.minute == 59


def test_temporal_extractor_yesterday_and_quarter():
    """AC-RAG-17: 'yesterday' and quarter patterns resolve to exact UTC bounds."""
    fixed_now = datetime(2026, 11, 11, 14, 0, 0, tzinfo=timezone.utc)
    clock = FrozenClock(fixed_now)
    extractor = TemporalResolver(clock=clock)

    # Yesterday
    y_res = extractor.extract("Show me reports from yesterday")
    assert y_res["has_time_expression"] is True
    assert y_res["range_start_utc"] == datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc)

    # Quarter: 2026-Q1
    q_res = extractor.extract("Summarize financial earnings for 2026-Q1")
    assert q_res["has_time_expression"] is True
    assert q_res["temporal_scope"] == "specific_range"
    assert q_res["range_start_utc"] == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert q_res["range_end_utc"].date() == datetime(2026, 3, 31, tzinfo=timezone.utc).date()


def test_temporal_range_repair():
    """AC-RAG-17: Reversed temporal ranges (start > end) are repaired."""
    start = datetime(2026, 11, 15, tzinfo=timezone.utc)
    end = datetime(2026, 11, 10, tzinfo=timezone.utc)
    repaired_start, repaired_end = TemporalResolver.repair_range(start, end)
    assert repaired_start == end
    assert repaired_end == start


# ── AC-RAG-13: Heuristic Router Tests ──────────────────────────────────────────

def test_heuristic_router_english_and_chinese():
    """AC-RAG-13: Overview, follow-up, and greeting prompts (EN & ZH) classified without model call."""
    router = HeuristicRouter()

    # Greetings (no_retrieval)
    int1 = router.classify("Hello!")
    assert int1 is not None and int1.route == "no_retrieval"

    int2 = router.classify("你好")
    assert int2 is not None and int2.route == "no_retrieval"

    int3 = router.classify("Thank you very much.")
    assert int3 is not None and int3.route == "no_retrieval"

    # Overview (scope=document_overview)
    int4 = router.classify("Can you give me a summary of the main characters?")
    assert int4 is not None and int4.scope == "document_overview"

    int5 = router.classify("请帮我总结一下全书梗概和主要人物")
    assert int5 is not None and int5.scope == "document_overview"

    # Follow-up (is_followup=True)
    int6 = router.classify("What about that other plan?")
    assert int6 is not None and int6.is_followup is True

    int7 = router.classify("那他后来怎么样了，详细说说")
    assert int7 is not None and int7.is_followup is True

    # Quoted exact terms
    int8 = router.classify('Where does the function "calculate_interest" get defined?')
    assert int8 is not None
    assert "calculate_interest" in int8.exact_terms


# ── AC-RAG-12: Passthrough Router Tests ────────────────────────────────────────

def test_passthrough_router_zero_calls():
    """AC-RAG-12: Passthrough router preserves original query text with zero model calls."""
    router = PassthroughRouter()
    tracker = UsageTracker("req-pt")
    degradations = DegradationTracker()
    models = ModelBindings(summary=["gemini-3.5-flash-lite"], generation=["gemini-3.8-flash"], embedding="gemini-embedding-2")
    profile = long_form_reader(models)
    ctx = RouterContext(usage_tracker=tracker, degradation_tracker=degradations, profile=profile)

    req = RAGQueryRequest(question="Specific text search")
    intent = router.route(req, None, ctx)

    assert intent.route == "rag"
    assert intent.scope == "local_lookup"
    assert intent.search_query == "Specific text search"
    assert len(tracker.stage_usages) == 0


# ── AC-RAG-15 & AC-RAG-16: Clamping & DG-05 Tests ─────────────────────────────

def test_clamp_intent_out_of_range_values():
    """AC-RAG-15 / DG-05: date_window_days=99999, suggested_top_k=-5, oversized lists clamped."""
    raw = {
        "route": "invalid_route",
        "scope": "unknown_scope",
        "search_query": "query",
        "suggested_top_k": -5,
        "date_window_days": 99999,
        "secondary_queries": ["q1", "q2", "q3", "q4", "q5", "q6"],
        "entity_terms": ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10"],
    }
    degradations = DegradationTracker()
    intent = clamp_intent(raw, degradations)

    assert intent.route == "rag"
    assert intent.scope == "local_lookup"
    assert intent.suggested_top_k == 1
    assert intent.date_window_days == 3650
    assert len(intent.secondary_queries) == 4
    assert len(intent.entity_terms) == 8

    # Recorded DG-05 degradations
    dg_ids = [d.id for d in degradations.degradations]
    assert "DG-05" in dg_ids


# ── AC-RAG-14: Jev Gate Tests ──────────────────────────────────────────────────

def test_jev_gate_missing_key_fails_construction():
    """AC-RAG-14: Router configured with enable_jev=True without API key raises at construction."""
    mock_gw = MagicMock()
    with pytest.raises(ProfileConfigError):
        JevGateRouter(gateway=mock_gw, api_key="")


def test_jev_gate_fast_path_and_dg03_fallback(monkeypatch):
    """AC-RAG-14: Jev fast path emits intent; Jev error continues on LLM path with DG-03."""
    mock_gw = MagicMock()
    mock_jev = MagicMock()
    mock_gw.jev_client = mock_jev

    router = JevGateRouter(gateway=mock_gw, api_key="test-key")
    tracker = UsageTracker("req-jev")
    degradations = DegradationTracker()

    # Fast path answer setup
    scope_mock = MagicMock(confidence=0.92, value="local_lookup")
    rewrite_mock = MagicMock(probability=0.15)
    time_mock = MagicMock(probability=0.10)
    probe_mock = MagicMock(probability=0.05)

    from cody_ai_shared_lib.jev import JevResult
    mock_jev.system_one.return_value = JevResult(
        model="typesafe-ai/jev",
        answers={
            "scope": scope_mock,
            "needs_query_rewrite": rewrite_mock,
            "has_time_expression": time_mock,
            "beyond_boundary_probe": probe_mock,
        },
        tokens_in=50,
        tokens_out=10,
        ms=45.0,
        cost_usd=0.000002,
    )

    intent = router.route("What color was the car?", tracker, degradations)
    assert intent is not None
    assert intent.router_source == "jev"
    assert intent.scope == "local_lookup"
    assert len(tracker.stage_usages) == 1

    # Now test DG-03 fallback on Jev error
    mock_jev.system_one.side_effect = RuntimeError("Jev timeout")
    intent_fallback = router.route("query", tracker, degradations)
    assert intent_fallback is None
    assert any(d.id == "DG-03" for d in degradations.degradations)
