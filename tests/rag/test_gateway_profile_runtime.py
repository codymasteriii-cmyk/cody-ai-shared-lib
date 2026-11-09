"""Tests for ModelGateway, EngineProfile, Runtime determinism, BudgetGuard, and Tracing.

Covers:
  - DG-02 Model list fallback (AC-RAG-44)
  - DG-09 Truncation retry (AC-RAG-44)
  - Safety filter blocking without retry (AC-RAG-44)
  - DG-11 BudgetGuard skip & BudgetExceeded raise (AC-RAG-46)
  - EngineProfile validation & Tier 0 context check (AC-RAG-49, AC-RAG-62)
  - EngineProfile sha256 fingerprint stability (AC-RAG-49)
  - Deadline enforcement with Clock (AC-RAG-48)
  - ObserverDispatcher DG-13 logging vs raise (AC-RAG-47)
  - UsageTracker aggregation (AC-RAG-70, RAG-OPS-1)
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest

from cody_ai_shared_lib.llm import LLMResult, LLMUsage, ModelCallError
from cody_ai_shared_lib.rag import (
    BudgetExceeded,
    CallContext,
    ComponentRegistry,
    Deadline,
    DeadlineExceeded,
    DegradationTracker,
    DeterministicRng,
    EngineProfile,
    FrozenClock,
    GenerationConfig,
    GenerationError,
    ModelBindings,
    ModelGateway,
    ObserverDispatcher,
    ProfileConfigError,
    RAGObserver,
    RAGTrace,
    RetrievalConfig,
    StageRecord,
    StageUsage,
    SystemClock,
    SystemRng,
    UsageTracker,
    create_default_registry,
    knowledge_base,
    long_form_reader,
    news_intelligence,
)


# ── ModelGateway Tests ─────────────────────────────────────────────────────────

def test_model_gateway_model_fallback_dg02():
    """AC-RAG-44 / DG-02: Model 1 failure triggers fallback to Model 2, recording DG-02."""
    mock_llm = MagicMock()

    # Model 1 fails, Model 2 succeeds
    usage_fail = LLMUsage(input_tokens=100, output_tokens=0, cost_usd=0.00015)
    mock_llm.generate_with_usage.side_effect = [
        ModelCallError("503 Service Unavailable", usage=usage_fail),
        LLMResult(
            text="Success from fallback model",
            usage=LLMUsage(input_tokens=120, output_tokens=25, cost_usd=0.00049),
            finish_reason="stop",
        ),
    ]

    gateway = ModelGateway(llm_client=mock_llm)
    tracker = UsageTracker(request_id="req-123")
    degradations = DegradationTracker()

    res = gateway.generate(
        system_prompt="sys",
        user_message="hello",
        models=["gemini-3.8-flash", "claude-sonnet-5"],
        stage="generation",
        usage_tracker=tracker,
        degradation_tracker=degradations,
    )

    assert res.text == "Success from fallback model"
    assert len(degradations.degradations) == 1
    assert degradations.degradations[0].id == "DG-02"
    assert degradations.degradations[0].fallback_used == "model:claude-sonnet-5"

    # Both usages recorded
    usages = tracker.stage_usages
    assert len(usages) == 2
    assert usages[0].outcome == "error"
    assert usages[1].outcome == "ok"
    assert usages[1].model == "claude-sonnet-5"


def test_model_gateway_all_models_fail():
    """AC-RAG-44: When all models in fallback chain fail, raise GenerationError."""
    mock_llm = MagicMock()
    mock_llm.generate_with_usage.side_effect = RuntimeError("Provider down")

    gateway = ModelGateway(llm_client=mock_llm)
    tracker = UsageTracker(request_id="req-fail")
    degradations = DegradationTracker()

    with pytest.raises(GenerationError) as exc_info:
        gateway.generate(
            system_prompt="sys",
            user_message="hello",
            models=["gemini-3.8-flash", "gpt-6.1-sol"],
            stage="generation",
            usage_tracker=tracker,
            degradation_tracker=degradations,
        )

    assert exc_info.value.reason == "all_models_failed"
    assert len(degradations.degradations) == 1  # DG-02 recorded between model 1 and 2


def test_model_gateway_truncation_retry_dg09():
    """AC-RAG-44 / DG-09: Output truncated with finish_reason='length' triggers retry."""
    mock_llm = MagicMock()
    mock_llm.generate_with_usage.side_effect = [
        LLMResult(
            text="Partial text...",
            usage=LLMUsage(input_tokens=100, output_tokens=50),
            finish_reason="length",
        ),
        LLMResult(
            text="Complete text completed successfully",
            usage=LLMUsage(input_tokens=100, output_tokens=90),
            finish_reason="stop",
        ),
    ]

    gateway = ModelGateway(llm_client=mock_llm)
    tracker = UsageTracker(request_id="req-trunc")
    degradations = DegradationTracker()

    res = gateway.generate(
        system_prompt="sys",
        user_message="hello",
        models=["gemini-3.8-flash"],
        stage="generation",
        usage_tracker=tracker,
        degradation_tracker=degradations,
        max_tokens=50,
    )

    assert res.text == "Complete text completed successfully"
    assert len(degradations.degradations) == 1
    assert degradations.degradations[0].id == "DG-09"
    # Second call had doubled max_tokens
    assert mock_llm.generate_with_usage.call_args_list[1].kwargs["max_tokens"] == 100


def test_model_gateway_safety_blocked_no_retry():
    """AC-RAG-44: Safety blocked generation raises GenerationError immediately."""
    mock_llm = MagicMock()
    mock_llm.generate_with_usage.return_value = LLMResult(
        text="",
        usage=LLMUsage(input_tokens=50, output_tokens=0),
        finish_reason="safety",
    )

    gateway = ModelGateway(llm_client=mock_llm)
    tracker = UsageTracker(request_id="req-safety")
    degradations = DegradationTracker()

    with pytest.raises(GenerationError) as exc_info:
        gateway.generate(
            system_prompt="sys",
            user_message="unsafe request",
            models=["gemini-3.8-flash", "gpt-6.1-sol"],
            stage="generation",
            usage_tracker=tracker,
            degradation_tracker=degradations,
        )

    assert exc_info.value.reason == "blocked"
    # No fallback model was called
    assert mock_llm.generate_with_usage.call_count == 1


# ── BudgetGuard Tests ──────────────────────────────────────────────────────────

def test_budget_guard_skip_dg11():
    """AC-RAG-46 / DG-11: BudgetGuard returning 'skip' skips stage and records DG-11."""
    mock_llm = MagicMock()
    mock_guard = MagicMock()
    mock_guard.before_call.return_value = "skip"

    gateway = ModelGateway(llm_client=mock_llm)
    tracker = UsageTracker(request_id="req-budget-skip")
    degradations = DegradationTracker()

    res = gateway.generate(
        system_prompt="sys",
        user_message="hello",
        models=["gemini-3.8-flash"],
        stage="summarizer",
        usage_tracker=tracker,
        degradation_tracker=degradations,
        budget_guard=mock_guard,
    )

    assert res.text == ""
    assert mock_llm.generate_with_usage.call_count == 0
    assert len(degradations.degradations) == 1
    assert degradations.degradations[0].id == "DG-11"


def test_budget_guard_exceeded_raise():
    """AC-RAG-46: BudgetGuard raising BudgetExceeded propagates cleanly."""
    mock_guard = MagicMock()
    mock_guard.before_call.side_effect = BudgetExceeded("Cost limit of $0.05 reached", limit_usd=0.05, spent_usd=0.051)

    gateway = ModelGateway()
    tracker = UsageTracker(request_id="req-budget-exceeded")
    degradations = DegradationTracker()

    with pytest.raises(BudgetExceeded) as exc_info:
        gateway.generate(
            system_prompt="sys",
            user_message="hello",
            models=["gemini-3.8-flash"],
            stage="generation",
            usage_tracker=tracker,
            degradation_tracker=degradations,
            budget_guard=mock_guard,
        )

    assert exc_info.value.limit_usd == 0.05


# ── EngineProfile & Validation Tests ───────────────────────────────────────────

def test_engine_profile_presets_valid():
    """AC-RAG-49: Built-in presets are valid against default registry."""
    registry = create_default_registry()
    models = ModelBindings(
        summary=["gemini-3.5-flash-lite"],
        generation=["gemini-3.8-flash", "gpt-6.1-sol"],
        intent=["gemini-3.5-flash-lite"],
        embedding="gemini-embedding-2",
        reranker="jina-reranker",
    )

    p1 = long_form_reader(models)
    p1.validate(registry)

    p2 = news_intelligence(models)
    p2.validate(registry)

    p3 = knowledge_base(models)
    p3.validate(registry)


def test_engine_profile_validation_empty_generation():
    """AC-RAG-49: Profile validation rejects empty generation models."""
    models = ModelBindings(generation=[], embedding="gemini-embedding-2")
    profile = long_form_reader(models)
    with pytest.raises(ProfileConfigError) as exc_info:
        profile.validate()
    assert "models.generation" in exc_info.value.missing


def test_engine_profile_validation_tier0_overflow():
    """AC-RAG-62: Rejects full_context_max_tokens + reserved zones > max_context_tokens."""
    models = ModelBindings(summary=["gemini-3.5-flash-lite"], generation=["gemini-3.8-flash"], embedding="gemini-embedding-2")
    profile = long_form_reader(
        models,
        retrieval=RetrievalConfig(full_context_max_tokens=155_000),
        # Default reserved: 4000 instruction + 4000 history = 8000 -> 155000 + 8000 = 163000 > 160000
    )
    with pytest.raises(ProfileConfigError) as exc_info:
        profile.validate()
    assert any("exceeds max_context_tokens" in p for p in exc_info.value.problems)


def test_engine_profile_validation_unknown_fallback():
    """AC-RAG-49: Rejects unknown fallback keys."""
    models = ModelBindings(summary=["gemini-3.5-flash-lite"], generation=["gemini-3.8-flash"], embedding="gemini-embedding-2")
    profile = long_form_reader(models, fallbacks={"invalid_stage": "raise"})
    with pytest.raises(ProfileConfigError) as exc_info:
        profile.validate()
    assert any("invalid_stage" in u for u in exc_info.value.unknown)


def test_engine_profile_fingerprint():
    """AC-RAG-49: SHA-256 fingerprint is deterministic and sensitive to configuration changes."""
    models = ModelBindings(summary=["gemini-3.5-flash-lite"], generation=["gemini-3.8-flash"], embedding="gemini-embedding-2")
    p1 = long_form_reader(models)
    p2 = long_form_reader(models)
    assert p1.fingerprint() == p2.fingerprint()

    p3 = long_form_reader(models, generation=GenerationConfig(temperature=0.7))
    assert p1.fingerprint() != p3.fingerprint()


# ── Runtime Determinism & Deadlines ────────────────────────────────────────────

def test_deadline_expired_handling():
    """AC-RAG-48: An expired deadline raises DeadlineExceeded for a required stage."""
    clock = FrozenClock(datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc), fixed_monotonic=100.0)
    dl = Deadline(deadline_seconds=5.0, clock=clock)

    # Initial state: 5.0 seconds remaining
    assert dl.remaining_seconds() == 5.0
    dl.check("generation")  # Should not raise

    # Advance time by 6 seconds -> expired
    clock.advance(6.0)
    assert dl.is_expired()

    # Required stage raises DeadlineExceeded
    with pytest.raises(DeadlineExceeded) as exc_info:
        dl.check("generation")
    assert exc_info.value.stage == "generation"


def test_deterministic_rng():
    """Seeded RNG produces identical byte streams and choices."""
    rng1 = DeterministicRng(seed=12345)
    rng2 = DeterministicRng(seed=12345)

    assert rng1.hex(8) == rng2.hex(8)
    items = ["a", "b", "c", "d", "e"]
    assert rng1.choice(items) == rng2.choice(items)


# ── ObserverDispatcher & Trace ─────────────────────────────────────────────────

def test_observer_dispatcher_dg13_fault_tolerance():
    """AC-RAG-47 / DG-13: Observer errors logged without raising when observer_errors='log'."""
    broken_observer = MagicMock()
    broken_observer.on_stage.side_effect = RuntimeError("Observer crashed")

    dispatcher = ObserverDispatcher(observers=[broken_observer], observer_errors="log")

    # Should not raise
    dispatcher.notify_stage(StageRecord(name="retrieval", duration_ms=12.5))
    assert broken_observer.on_stage.call_count == 1


def test_observer_dispatcher_raises_when_configured():
    """AC-RAG-47: Observer errors propagate when observer_errors='raise' (e.g. KnowledgeBase preset)."""
    broken_observer = MagicMock()
    broken_observer.on_stage.side_effect = RuntimeError("Fail loud")

    dispatcher = ObserverDispatcher(observers=[broken_observer], observer_errors="raise")

    with pytest.raises(RuntimeError):
        dispatcher.notify_stage(StageRecord(name="retrieval"))


# ── UsageTracker Aggregation ───────────────────────────────────────────────────

def test_usage_tracker_aggregation():
    """AC-RAG-70 / RAG-OPS-1: UsageTracker aggregates tokens, cost, calls, and latency."""
    tracker = UsageTracker(request_id="req-agg", tenant_or_user_id="user-42")

    tracker.record(
        StageUsage(
            stage="embedding",
            model="gemini-embedding-2",
            input_tokens=500,
            output_tokens=0,
            cost_usd=0.00010,
            latency_ms=25.0,
        )
    )
    tracker.record(
        StageUsage(
            stage="generation",
            model="gemini-3.8-flash",
            input_tokens=1500,
            output_tokens=400,
            cost_usd=0.00525,
            latency_ms=350.0,
        )
    )

    tot = tracker.total()
    assert tot.input_tokens == 2000
    assert tot.output_tokens == 400
    assert tot.calls == 2
    assert tot.latency_ms == 375.0
    assert tot.cost_usd == pytest.approx(0.00535, rel=1e-5)
    assert tot.cost_complete is True
    assert tot.unpriced_models == []
