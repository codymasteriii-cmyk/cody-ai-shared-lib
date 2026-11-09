"""Regression tests for the Phase 1-3 review's spec-contract findings (section B)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cody_ai_shared_lib.llm import LLMResult, LLMStreamChunk, LLMUsage
from cody_ai_shared_lib.rag.errors import (
    BudgetExceeded,
    DeadlineExceeded,
    GenerationError,
    MissingRequiredScopeError,
    ProfileConfigError,
    RetrievalError,
    StoreCapabilityError,
    UnsupportedFilterError,
)
from cody_ai_shared_lib.rag.gateway import ModelGateway
from cody_ai_shared_lib.rag.guards import SequentialBoundaryGuard
from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore
from cody_ai_shared_lib.rag.models import (
    DefaultNodeMeta,
    FilterPatch,
    QueryIntent,
    RAGFilterSpec,
    RAGNode,
    RAGQueryRequest,
    SourceSpan,
)
from cody_ai_shared_lib.rag.ports import StoreCapabilities
from cody_ai_shared_lib.rag.postgen.citations import CitationExtractor, OffsetCitationFormatter
from cody_ai_shared_lib.rag.postgen.quotes import QuoteValidator, compute_coverage
from cody_ai_shared_lib.rag.postgen.sanitizer import OutputSanitizer
from cody_ai_shared_lib.rag.postretrieval.threshold import ThresholdFilter
from cody_ai_shared_lib.rag.profile import (
    EngineProfile,
    GenerationConfig,
    IngestionConfig,
    ModelBindings,
    RetrievalConfig,
    RuntimeConfig,
    SummariesConfig,
)
from cody_ai_shared_lib.rag.registry import create_default_registry
from cody_ai_shared_lib.rag.runtime import FrozenClock
from cody_ai_shared_lib.rag.strategies.context import RetrievalStrategy
from cody_ai_shared_lib.rag.text.analyzer import CjkAwareAnalyzer, TokenGroup

from .test_blocker_regressions import (  # noqa: F401
    DIM,
    FailingReranker,
    FakeLLM,
    RecordingGuard,
    document,
    engine,
    profile,
    question,
    real_embedding_shape,
    seeded_store,
)

NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)


class StreamingLLM(FakeLLM):
    def __init__(self, chunks: list[LLMStreamChunk] | None = None, fail_models: tuple[str, ...] = ()):
        super().__init__()
        self.chunks = chunks if chunks is not None else [
            LLMStreamChunk(delta="[[answer_found=yes]]Answer [1]."),
            LLMStreamChunk(delta="", usage=LLMUsage(input_tokens=5, output_tokens=2), finish_reason="stop"),
        ]
        self.fail_models = fail_models

    def stream(self, **kwargs):
        if kwargs["model"] in self.fail_models:
            raise RuntimeError("provider down")
        yield from self.chunks


class RecordingObserver:
    def __init__(self):
        self.stages = []
        self.usage = []
        self.degradations = []
        self.completed = []

    def on_stage(self, record):
        self.stages.append(record)

    def on_usage(self, usage):
        self.usage.append(usage)

    def on_degrade(self, degradation):
        self.degradations.append(degradation)

    def on_complete(self, trace, outcome):
        self.completed.append((trace, outcome))


def native_profile(models: list[str] | None = None) -> EngineProfile:
    return EngineProfile(
        name="t",
        models=ModelBindings(generation=models or ["m1"], embedding="gemini-embedding-2"),
        generation=GenerationConfig(stream_mode="native"),
    )


# ── Item 11: pluggable components


class FixedRouter:
    name = "app_router"
    version = "2.0.0"

    def route(self, request, state, ctx):
        return QueryIntent(search_query="rewritten by the app", router_source="app")


def test_engine_uses_a_caller_supplied_router():
    response = engine(router=FixedRouter()).query(question())
    assert response.retrieved.intent.router_source == "app"
    assert response.trace.router_decision["source"] == "app"


def test_router_override_selects_a_registered_router():
    registry = create_default_registry()
    registry.register_router("app", lambda deps: FixedRouter())
    result = engine(registry=registry).retrieve(question(router_override="app"))
    assert result.intent.router_source == "app"


def test_registered_strategy_and_stage_are_used():
    class AppStrategy(RetrievalStrategy):
        name = "app_strategy"

        def retrieve(self, ctx):
            return [RAGNode(node_id="app", text="from the app", span=SourceSpan(doc_id="d", char_end=12))]

    class AppStage:
        name = "app_stage"
        version = "1.0.0"
        accepts_score_kinds = None

        def process(self, nodes, ctx):
            return [n.model_copy(update={"flags": ["seen"]}) for n in nodes]

    registry = create_default_registry()
    registry.register_strategy("app_strategy", AppStrategy)
    registry.register_stage("app_stage", lambda deps: AppStage())
    prof = profile(retrieval=RetrievalConfig(default_strategy="app_strategy"), post_stages=["app_stage"])

    result = engine(prof, registry=registry).retrieve(question())

    assert result.strategy_used == "app_strategy"
    assert result.nodes[0].flags == ["seen"]


def test_registered_guard_name_is_enforced():
    class DropEverything:
        name = "drop_all"
        hard = True
        forces_strict_grounding = False

        def to_filter(self):
            return FilterPatch()

        def verify(self, node):
            return None

        def verify_citation(self, citation):
            return True

        def prompt_rules(self):
            return []

    registry = create_default_registry()
    registry.register_guard("drop_all", lambda deps: DropEverything())
    result = engine(profile(guards=["drop_all"]), registry=registry).retrieve(question())
    assert result.nodes == []


def test_engine_level_guards_add_to_profile_guards():
    guard = SequentialBoundaryGuard(max_char_offset=5)
    eng = engine(profile(guards=["sequential_boundary"]), guards=[guard])
    active = eng.pipeline.turn_guards(question(), RAGFilterSpec(max_char_offset=9))
    assert len([g for g in active if isinstance(g, SequentialBoundaryGuard)]) == 2


def test_duplicate_registration_raises_unless_replaced():
    registry = create_default_registry()
    with pytest.raises(ProfileConfigError):
        registry.register_router("auto", lambda deps: None)
    registry.register_router("auto", lambda deps: None, replace=True)


def test_adaptive_uses_spec_rule_numbers_and_custom_rules():
    eng = engine(profile(retrieval=RetrievalConfig(default_strategy="adaptive")))
    assert eng.plan(question()).rule == 3

    custom = profile(
        retrieval=RetrievalConfig(default_strategy="adaptive", adaptive_rules=[("keyword_available", "keyword_only")])
    )
    plan = engine(custom).plan(question())
    assert (plan.strategy, plan.rule) == ("keyword_only", 1)


# ── Item 12: deadlines


def test_expired_deadline_raises_before_required_stage():
    clock = FrozenClock(NOW)
    prof = profile(runtime=RuntimeConfig(deadline_seconds=0.0))
    with pytest.raises(DeadlineExceeded) as caught:
        engine(prof, clock=clock).retrieve(question())
    assert caught.value.usage_total is not None


def rerank_profile(**runtime: object) -> EngineProfile:
    return EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], reranker="jina-reranker", embedding="gemini-embedding-2"),
        post_stages=["rerank"],
        runtime=RuntimeConfig(deadline_seconds=1.0, **runtime),
    )


def test_optional_stage_skipped_when_it_will_not_fit():
    gateway = ModelGateway(llm_client=FakeLLM(), rerank_client=FailingReranker())
    result = engine(rerank_profile(), gateway=gateway).retrieve(question())
    assert [d.id for d in result.degradations] == ["DG-10"]
    assert result.reranked is False


# ── Item 13: guards after every stage


def test_guards_run_after_a_stage_that_adds_text_past_the_boundary():
    class LeakyStage:
        name = "leaky"
        version = "1.0.0"
        accepts_score_kinds = None

        def process(self, nodes, ctx):
            return nodes + [RAGNode(node_id="spoiler", text="ending", span=SourceSpan(doc_id="d", char_start=500, char_end=506))]

    registry = create_default_registry()
    registry.register_stage("leaky", lambda deps: LeakyStage())
    prof = profile(post_stages=["leaky"], guards=["sequential_boundary"])

    result = engine(prof, registry=registry).retrieve(question(filters=RAGFilterSpec(max_char_offset=100)))

    assert "spoiler" not in [n.node_id for n in result.nodes]
    assert result.trace.guard_effects["sequential_boundary"]["dropped"] == 1


# ── Item 14: relaxation


def dated_store() -> InMemoryRAGStore:
    store = InMemoryRAGStore()
    node = RAGNode(
        node_id="old",
        granularity="article",
        text="An old article.",
        span=SourceSpan(doc_id="a", char_end=15),
        metadata=DefaultNodeMeta(extra={"created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}),
    )
    store.upsert_section_nodes("a", 0, [node], [[0.1] * DIM])
    return store


SOFT_WINDOW = QueryIntent(
    search_query="q",
    has_time_expression=True,
    range_start_utc=datetime(2026, 11, 1, tzinfo=timezone.utc),
    range_end_utc=NOW,
)


def test_relaxation_is_off_by_default():
    result = engine(store=dated_store()).retrieve(question(), intent=SOFT_WINDOW)
    assert result.nodes == []
    assert result.relaxations == []


def test_configured_relaxation_lifts_only_soft_constraints():
    prof = profile(retrieval=RetrievalConfig(default_strategy="dense_vector", relaxation=["drop_time_window"]))
    result = engine(prof, dated_store()).retrieve(question(), intent=SOFT_WINDOW)
    assert result.relaxations == ["drop_time_window"]
    assert [n.node_id for n in result.nodes] == ["old"]

    hard = RAGFilterSpec(date_range_utc=(datetime(2026, 11, 1, tzinfo=timezone.utc), NOW))
    assert engine(prof, dated_store()).retrieve(question(filters=hard)).relaxations == []


def test_lower_threshold_never_goes_below_the_callers_floor():
    from cody_ai_shared_lib.rag.strategies.relaxation import LowerThreshold, RelaxationState

    state = RelaxationState(RAGFilterSpec(min_similarity=0.70), QueryIntent(), False, 0.68)
    assert LowerThreshold(0.05).apply(state).filters.min_similarity == 0.68
    assert LowerThreshold(0.05).apply(RelaxationState(RAGFilterSpec(min_similarity=0.68), QueryIntent(), False, 0.68)) is None


# ── Items 15, 16: answer flag and result flags


def test_missing_flag_is_none_and_the_model_is_told_to_flag():
    class Capturing(FakeLLM):
        system = ""

        def generate_with_usage(self, **kwargs):
            Capturing.system = kwargs["system_prompt"]
            return super().generate_with_usage(**kwargs)

    response = engine(llm=Capturing("It cannot be found in the provided documents [1].")).query(question())
    assert response.answer_found is None
    assert response.abstained is False
    assert "[[answer_found=partial]]" in Capturing.system


def test_flag_with_no_nodes_is_still_no():
    response = engine(store=InMemoryRAGStore(), llm=FakeLLM("[[answer_found=yes]]Made up.")).query(question())
    assert response.answer_found == "no"
    assert response.citations == []
    assert response.abstained is True


def test_grounded_is_false_when_the_fallback_label_is_used():
    prof = profile(generation=GenerationConfig(grounding_mode="labelled_fallback"))
    text = "[[answer_found=no]]Not covered.\n\nNot from your sources: general knowledge."
    response = engine(prof, llm=FakeLLM(text)).query(question())
    assert response.grounded is False
    assert response.abstained is False


def test_open_grounding_mode_is_rejected():
    with pytest.raises(ValueError):
        GenerationConfig(grounding_mode="open")


def test_degraded_reflects_degradations():
    gateway = ModelGateway(llm_client=FakeLLM(), rerank_client=FailingReranker())
    prof = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], reranker="jina-reranker", embedding="gemini-embedding-2"),
        post_stages=["rerank"],
    )
    assert engine(prof, gateway=gateway).query(question()).degraded is True
    assert engine().query(question()).degraded is False


# ── Item 17: sentence cut at B


def test_store_stopped_text_gets_the_same_cut_as_a_passage():
    guard = SequentialBoundaryGuard(max_char_offset=20)
    text = "Alice met Bob. They "
    window = RAGNode(node_id="w", text=text, granularity="local_window", span=SourceSpan(doc_id="d", char_end=20))
    passage = RAGNode(node_id="p", text=text + "talked.", granularity="passage", span=SourceSpan(doc_id="d", char_end=27))

    assert guard.verify(window).text == guard.verify(passage).text == "Alice met Bob."
    assert guard.verify(guard.verify(window)).span.char_end == 14


# ── Item 18: threshold


def test_threshold_exempts_ordered_and_unscored_nodes():
    keep = [
        RAGNode(node_id="w", text="x", granularity="local_window", score_kind="cosine", score=0.0),
        RAGNode(node_id="r", text="x"),
        RAGNode(node_id="k", text="x", score=0.9, score_kind="bm25"),
    ]
    drop = RAGNode(node_id="v", text="x", vector_score=0.1, score_kind="rrf")
    assert [n.node_id for n in ThresholdFilter(min_similarity=0.5).filter([*keep, drop])] == ["w", "r", "k"]


def test_profile_floor_is_applied_store_side():
    prof = profile(retrieval=RetrievalConfig(min_similarity=0.6))
    assert engine(prof).retrieve(question()).applied_filters.min_similarity == 0.6
    caller = question(filters=RAGFilterSpec(min_similarity=0.8))
    assert engine(prof).retrieve(caller).applied_filters.min_similarity == 0.8


# ── Item 19: startup validation


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"post_stages": ["nope"]}, "stage:nope"),
        ({"transforms": ["nope"]}, "transform:nope"),
        ({"guards": ["nope"]}, "guard:nope"),
        ({"retrieval": RetrievalConfig(default_strategy="nope")}, "strategy:nope"),
        ({"retrieval": RetrievalConfig(relaxation=["nope"])}, "relaxation:nope"),
    ],
)
def test_unknown_component_names_fail_at_startup(changes: dict, expected: str):
    with pytest.raises(ProfileConfigError) as caught:
        engine(profile(**changes))
    assert expected in caught.value.unknown


def test_all_problems_are_listed_together():
    with pytest.raises(ProfileConfigError) as caught:
        engine(profile(post_stages=["a"], transforms=["b"], fallbacks={"c": "raise"}))
    assert len(caught.value.problems) == 3


def test_strategy_the_store_cannot_serve_fails_at_startup():
    store = InMemoryRAGStore(capabilities=StoreCapabilities(supports_vector_search=True))
    with pytest.raises(StoreCapabilityError):
        engine(profile(retrieval=RetrievalConfig(default_strategy="hybrid_rrf")), store)


def test_native_streaming_without_support_fails_at_startup():
    with pytest.raises(ProfileConfigError):
        engine(native_profile())


def test_validate_reports_without_raising(monkeypatch: pytest.MonkeyPatch):
    eng = engine()
    assert eng.validate().valid is True
    monkeypatch.setattr(eng.store, "embedding_space", lambda: eng.store._embedding_space.model_copy(update={"model": "x"}))
    report = eng.validate()
    assert report.valid is False
    assert report.problems


def test_summaries_enabled_without_a_model_fails():
    with pytest.raises(ProfileConfigError) as caught:
        engine(profile(ingestion=IngestionConfig(summaries=SummariesConfig(enabled=True))))
    assert "models.summary" in caught.value.missing


# ── Item 20: scope validation


def test_store_required_keys_and_custom_types_are_checked():
    caps = StoreCapabilities(
        supports_vector_search=True,
        required_filter_keys=frozenset(["tenant_id", "section_ids"]),
        supported_custom_filters={"active": bool},
    )
    eng = engine(store=InMemoryRAGStore(capabilities=caps))

    with pytest.raises(MissingRequiredScopeError) as caught:
        eng.retrieve(question())
    assert caught.value.missing_keys == ["section_ids", "tenant_id"]

    wrong_type = RAGFilterSpec(tenant_or_user_id="t", custom={"section_ids": ["1"], "active": "yes"})
    with pytest.raises(UnsupportedFilterError):
        eng.retrieve(question(filters=wrong_type))


# ── Item 21: budget guard coverage


def test_budget_guard_sees_routing_and_ingestion_calls():
    class IntentLLM(FakeLLM):
        def generate_with_usage(self, **kwargs):
            return LLMResult(
                text='{"search_query": "q"}' if "query analysis" in kwargs["system_prompt"] else "A short summary.",
                usage=LLMUsage(input_tokens=1, output_tokens=1),
                finish_reason="stop",
            )

    guard = RecordingGuard()
    prof = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], intent=["i1"], summary=["s1"], embedding="gemini-embedding-2"),
        ingestion=IngestionConfig(summaries=SummariesConfig(enabled=True)),
    )
    eng = engine(prof, InMemoryRAGStore(), llm=IntentLLM(), budget_guard=guard)
    eng.ingest_document(document("Sentence number one is here. " * 100))
    eng.query(question())

    assert {"summarizer", "ingest_embedding", "summary_embedding", "intent_llm", "retrieval_embedding", "generation"} <= set(guard.stages)


def test_budget_skip_on_summary_marks_it_skipped():
    class SkipSummary(RecordingGuard):
        def before_call(self, stage, model, est_input_tokens, optional, ctx):
            return "skip" if stage == "summarizer" else "allow"

    prof = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], summary=["s1"], embedding="gemini-embedding-2"),
        ingestion=IngestionConfig(summaries=SummariesConfig(enabled=True)),
    )
    store = InMemoryRAGStore()
    progress = engine(prof, store, budget_guard=SkipSummary()).ingest_document(document("A sentence here. " * 200))
    assert progress.summary_status == "skipped"
    assert [d.id for d in progress.degradations] == ["DG-11"]
    assert not any(n.is_summary for n in store._nodes.values())


def test_guard_can_stop_a_truncation_retry():
    class Truncating(FakeLLM):
        def generate_with_usage(self, **kwargs):
            return LLMResult(text="cut", usage=LLMUsage(input_tokens=1, output_tokens=9000), finish_reason="length")

    class StopRetries(RecordingGuard):
        def before_call(self, stage, model, est_input_tokens, optional, ctx):
            if stage.endswith("truncation_retry"):
                raise BudgetExceeded("no more")
            return "allow"

    with pytest.raises(BudgetExceeded):
        engine(llm=Truncating(), budget_guard=StopRetries()).query(question())


# ── Item 23: streaming


def test_stream_emits_degradation_before_the_first_delta():
    gateway = ModelGateway(llm_client=StreamingLLM(fail_models=("m1",)))
    events = list(engine(native_profile(["m1", "m2"]), gateway=gateway).stream_query(question()))
    names = [e.event for e in events]
    assert names.index("degradation") < names.index("delta")
    assert names[-2:] == ["citations", "done"]


def test_stream_records_a_failed_attempt_as_unknown_billing():
    gateway = ModelGateway(llm_client=StreamingLLM(fail_models=("m1",)))
    stream = engine(native_profile(["m1", "m2"]), gateway=gateway).stream_query(question())
    done = list(stream)[-1]
    failed = [u for u in done.data["usage"] if u["outcome"] == "error"]
    assert failed[0]["billing"] == "unknown"
    assert failed[0]["model"] == "m1"


def test_truncated_stream_raises():
    chunks = [LLMStreamChunk(delta="Partial"), LLMStreamChunk(delta="", usage=LLMUsage(input_tokens=1, output_tokens=1), finish_reason="length")]
    gateway = ModelGateway(llm_client=StreamingLLM(chunks))
    with pytest.raises(GenerationError) as caught:
        list(engine(native_profile(), gateway=gateway).stream_query(question()))
    assert caught.value.reason == "truncated"


def test_cancelled_stream_records_the_model_actually_used():
    chunks = [LLMStreamChunk(delta="One "), LLMStreamChunk(delta="two "), LLMStreamChunk(delta="three")]
    gateway = ModelGateway(llm_client=StreamingLLM(chunks, fail_models=("m1",)))
    stream = engine(native_profile(["m1", "m2"]), gateway=gateway).stream_query(question())
    for event in stream:
        if event.event == "delta":
            break
    stream.close()

    usage = stream._usage_tracker.stage_usages
    cancelled = [u for u in usage if u.outcome == "cancelled"]
    assert cancelled[0].model == "m2"
    assert cancelled[0].estimated is True


def test_stream_notifies_observers_like_query():
    observer = RecordingObserver()
    gateway = ModelGateway(llm_client=StreamingLLM())
    list(engine(native_profile(), gateway=gateway, observers=[observer]).stream_query(question()))
    assert len(observer.completed) == 1


# ── Item 24: observers, trace, cache breakpoints


def test_observers_receive_stage_usage_and_degradation_events():
    observer = RecordingObserver()
    gateway = ModelGateway(llm_client=FakeLLM(), rerank_client=FailingReranker())
    prof = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], reranker="jina-reranker", embedding="gemini-embedding-2"),
        post_stages=["rerank"],
    )
    engine(prof, gateway=gateway, observers=[observer]).query(question())

    assert {"dense_vector", "rerank", "stuff"} <= {s.name for s in observer.stages}
    assert observer.usage
    assert [d.id for d in observer.degradations] == ["DG-06"]


def test_failing_observer_records_dg13_and_the_request_continues():
    class Broken(RecordingObserver):
        def on_stage(self, record):
            raise RuntimeError("observer down")

    response = engine(observers=[Broken()]).query(question())
    assert "DG-13" in [d.id for d in response.degradations]


def test_trace_has_no_text_and_verbose_adds_snapshots():
    light = engine(llm=FakeLLM("SECRET ANSWER [1].")).query(question()).trace
    assert light.profile_fingerprint
    assert light.prompt_version
    assert light.tier_decision_reason is not None
    assert light.snapshots is None
    assert "SECRET" not in light.model_dump_json()

    verbose = engine().query(question(trace_level="verbose")).trace
    assert verbose.snapshots
    assert verbose.zone_tokens["question"] > 0
    assert verbose.model_calls


def test_cache_breakpoints_reach_the_llm():
    class Capturing(FakeLLM):
        seen = None

        def generate_with_usage(self, **kwargs):
            Capturing.seen = kwargs.get("cache_breakpoints")
            return super().generate_with_usage(**kwargs)

    engine(llm=Capturing()).query(question())
    assert Capturing.seen


def test_caller_request_id_is_echoed():
    assert engine().query(question(request_id="caller-1")).trace.request_id == "caller-1"


# ── Item 25: sanitizer


def test_sanitizer_uses_retrieved_sources_and_ignores_scheme_case():
    node = RAGNode(node_id="n", text="x", metadata=DefaultNodeMeta(extra={"url": "https://source.example/a"}))
    text = "[ok](https://source.example/b) [bad](HTTPS://evil.example/x) ![i](https://evil.example/p.png)"
    assert OutputSanitizer().sanitize(text, nodes=[node]) == "[ok](https://source.example/b) bad i"


def test_query_keeps_links_to_its_own_sources():
    store = InMemoryRAGStore()
    node = RAGNode(
        node_id="n1",
        text="Text.",
        span=SourceSpan(doc_id="d", char_end=5),
        metadata=DefaultNodeMeta(extra={"url": "https://source.example/a"}),
    )
    store.upsert_section_nodes("d", 0, [node], [[0.1] * DIM])
    response = engine(store=store, llm=FakeLLM("See [here](https://source.example/a) [1].")).query(question())
    assert "https://source.example/a" in response.answer_final


# ── Item 26: citations


def cited_node(node_id: str = "n1") -> RAGNode:
    return RAGNode(node_id=node_id, text="Some text.", span=SourceSpan(doc_id="d", char_end=10))


def test_bracketed_year_is_not_a_citation_and_markers_keep_their_number():
    nodes = [cited_node("a"), cited_node("b")]
    clean, citations = CitationExtractor().extract("Founded [2024] and cited [2] and again [2].", nodes)
    assert clean == "Founded [2024] and cited [2] and again [2]."
    assert [(c.index, c.node_id) for c in citations] == [(2, "b")]


def test_marker_for_a_node_outside_the_prompt_is_unverified():
    nodes = [cited_node("a"), cited_node("b")]
    _, citations = CitationExtractor().extract("Claim [2].", nodes, turn_map={"[1]": "a"})
    assert citations[0].verified is False


def test_offset_formatter_snaps_to_the_node():
    node = RAGNode(node_id="n", text="x" * 100, span=SourceSpan(doc_id="d", section_seq=3, char_start=100, char_end=200))
    _, citations = CitationExtractor(formatter=OffsetCitationFormatter()).extract("Claim [c3:150-250].", [node], turn_map={})
    assert citations[0].verified is True
    assert (citations[0].span.char_start, citations[0].span.char_end) == (150, 200)


def test_stale_revision_is_flagged():
    class Revised(InMemoryRAGStore):
        def current_revisions(self, node_ids):
            return {nid: "v2" for nid in node_ids}

    store = Revised()
    node = RAGNode(node_id="n1", text="Text.", revision="v1", span=SourceSpan(doc_id="d", char_end=5))
    store.upsert_section_nodes("d", 0, [node], [[0.1] * DIM])
    response = engine(store=store, llm=FakeLLM("Claim [1].")).query(question())
    assert response.citations[0].stale is True


def test_coverage_counts_next_sentence_and_ignores_short_ones():
    text = "This first long sentence has no marker of its own. The second long sentence is cited here [1]. Short one."
    citation = CitationExtractor().extract(text, [cited_node()])[1]
    coverage = compute_coverage(citation, text, CjkAwareAnalyzer())
    assert (coverage.sentences_cited, coverage.sentences_total) == (2, 2)


# ── Item 27: quotes


def test_scrambled_quote_fails_and_near_exact_quote_passes():
    source = ("the cat sat on the mat. " * 30) + "alice met bob in the old town and they talked for a very long time"
    scrambled = "".join(sorted(source[-60:]))
    near = "alice met bob in the old town and they spoke for a very long time"

    def check(quote: str) -> bool:
        citation = CitationExtractor().extract(f'He said "{quote}" [1].', [RAGNode(node_id="n", text=source)])
        verified, _ = QuoteValidator().verify(citation[1], [RAGNode(node_id="n", text=source)], citation[0])
        return verified[0].verified

    assert check(scrambled) is False
    assert check(near) is True


# ── Item 28: request fields


def test_request_variant_uses_the_named_profile():
    prof = profile(variants={"small": {"retrieval": {"top_k": 1}}})
    eng = engine(prof)
    assert eng.retrieve(question(variant="small")).nodes
    assert "small" in eng._variant_engines
    with pytest.raises(ProfileConfigError):
        eng.retrieve(question(variant="missing"))


def test_reranker_override_disables_the_stage():
    gateway = ModelGateway(llm_client=FakeLLM(), rerank_client=FailingReranker())
    prof = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], reranker="jina-reranker", embedding="gemini-embedding-2"),
        post_stages=["rerank"],
    )
    assert engine(prof, gateway=gateway).retrieve(question(enable_reranker_override=False)).degradations == []


# ── Items 29, 30: reference store, retry size


def overlapping_store() -> InMemoryRAGStore:
    store = InMemoryRAGStore()
    text = "abcdefghijklmnopqrst"
    nodes = [
        RAGNode(node_id="a", text=text[:12], span=SourceSpan(doc_id="d", char_end=12), metadata=DefaultNodeMeta(extra={"tenant_id": "t1"})),
        RAGNode(node_id="b", text=text[8:], span=SourceSpan(doc_id="d", char_start=8, char_end=20), metadata=DefaultNodeMeta(extra={"tenant_id": "t1"})),
    ]
    store.upsert_section_nodes("d", 0, nodes, [[0.1] * DIM, [0.1] * DIM])
    return store


def test_local_window_has_no_overlap_and_real_offsets():
    window = overlapping_store().fetch_local_window("d", 10, 100, RAGFilterSpec())[0]
    assert window.text == "abcdefghijklmnopqrst"
    assert (window.span.char_start, window.span.char_end) == (0, 20)


def test_local_window_honours_tenant():
    assert overlapping_store().fetch_local_window("d", 10, 5, RAGFilterSpec(tenant_or_user_id="other")) == []


def test_bm25_prefers_the_rarer_term_and_groups_filter():
    store = InMemoryRAGStore()
    texts = {"common": "apple apple banana", "rare": "apple zebra banana", "other": "apple banana cherry"}
    nodes = [RAGNode(node_id=k, text=v, span=SourceSpan(doc_id="d")) for k, v in texts.items()]
    store.upsert_section_nodes("d", 0, nodes, [[0.1] * DIM] * 3)

    assert store.keyword_search("apple zebra", RAGFilterSpec(), 5)[0].node_id == "rare"
    phrase = [TokenGroup(tokens=["zebra", "banana"], mode="phrase")]
    assert [n.node_id for n in store.keyword_search("x", RAGFilterSpec(), 5, analyzer_tokens=phrase)] == ["rare"]


def test_truncation_retry_asks_for_more_than_was_produced():
    class Truncating(FakeLLM):
        limits: list[int | None] = []

        def generate_with_usage(self, **kwargs):
            Truncating.limits.append(kwargs.get("max_tokens"))
            usage = LLMUsage(input_tokens=1, output_tokens=8000)
            return LLMResult(text="done", usage=usage, finish_reason="length" if len(Truncating.limits) == 1 else "stop")

    engine(llm=Truncating()).query(question())
    assert Truncating.limits == [None, 16000]


# ── Sections C and D


def test_state_holds_no_conversation_text_and_tracks_turns():
    first = engine(llm=FakeLLM("A private answer [1].")).query(question())
    assert "private answer" not in first.state.model_dump_json()
    assert first.state.turn_count == 1
    assert first.state.profile_fingerprint
    assert first.state.last_intent["route"] == "rag"
    assert first.state.prior_nodes[0].turn == 1


def test_prior_nodes_keep_only_recent_turns():
    from cody_ai_shared_lib.rag.context.conversation import ConversationManager
    from cody_ai_shared_lib.rag.models import ConversationState, NodeRef

    state = ConversationState(
        fence_id="f",
        turn_count=5,
        prior_nodes=[NodeRef(node_id=f"t{t}", turn=t) for t in range(1, 6)],
    )
    kept = ConversationManager(max_reused_turns=3).prune_state(state)
    assert [ref.node_id for ref in kept.prior_nodes] == ["t3", "t4", "t5"]


def test_oversized_state_prunes_node_refs_before_the_summary():
    from cody_ai_shared_lib.rag.context.conversation import ConversationManager
    from cody_ai_shared_lib.rag.models import ConversationState, NodeRef

    state = ConversationState(fence_id="f", rolling_summary="summary", prior_nodes=[NodeRef(node_id="n" * 100) for _ in range(5)])
    kept = ConversationManager(max_state_bytes=400).prune_state(state)
    assert len(kept.prior_nodes) < 5
    assert kept.rolling_summary == "summary"


def test_fuse_variants_keeps_every_variant_and_ranks_by_fused_score():
    from cody_ai_shared_lib.rag.transforms import fuse_variants

    def nodes(*ids: str) -> list[RAGNode]:
        return [RAGNode(node_id=i, text="x") for i in ids]

    fused = fuse_variants([nodes("a", "b", "c", "d"), nodes("x", "b")], [1.0, 0.8], top_k=4)
    assert fused[0].node_id == "b"
    assert "x" in [n.node_id for n in fused]
    assert len(fused) == 4


def test_identity_dedup_argument_collapses_syndicated_copies():
    store = InMemoryRAGStore()
    meta = DefaultNodeMeta(extra={"url": "https://news.example/a"})
    nodes = [RAGNode(node_id=i, text="same story", span=SourceSpan(doc_id=i, char_end=10), metadata=meta) for i in ("a", "b")]
    store.upsert_section_nodes("m", 0, nodes, [[0.1] * DIM] * 2)
    assert len(engine(profile(post_stages=["identity_dedup(url)"]), store).retrieve(question()).nodes) == 1


def test_news_and_knowledge_presets_match_the_spec():
    from cody_ai_shared_lib.rag.profile import knowledge_base, long_form_reader, news_intelligence

    bound = ModelBindings(generation=["m"], summary=["s"], embedding="gemini-embedding-2", reranker="jina-reranker")
    assert news_intelligence(bound).post_stages == ["identity_dedup(url)", "rerank", "recency", "top_k"]
    assert long_form_reader(bound).post_stages == ["offset_dedup", "rerank", "top_k"]

    kb = knowledge_base(bound)
    assert kb.retrieval.default_strategy == "adaptive"
    assert kb.retrieval.enable_keyword_first is True
    assert kb.citations.verify_quotes is True
    assert kb.generation.stream_mode == "native"
    assert "rerank" not in long_form_reader(ModelBindings(generation=["m"], summary=["s"], embedding="gemini-embedding-2")).post_stages


def test_embedding_binding_is_required_and_must_match_the_real_embedder():
    with pytest.raises(ValueError):
        ModelBindings(generation=["m"])

    binding = ModelBindings(generation=["m"], embedding="other-model")
    store = InMemoryRAGStore(embedding_space=seeded_store().embedding_space().model_copy(update={"model": "other-model"}))
    with pytest.raises(ProfileConfigError) as caught:
        engine(EngineProfile(name="t", models=binding), store)
    assert "other-model" in caught.value.problems[0]


def test_budget_guard_has_one_signature_and_its_type_errors_surface():
    class OneArgument:
        enforces_cost = False

        def before_call(self, ctx):
            return "allow"

    with pytest.raises(RetrievalError) as caught:
        engine(budget_guard=OneArgument()).query(question())
    assert isinstance(caught.value.cause, TypeError)


def test_app_renderer_receives_metadata_and_its_errors_surface():
    from cody_ai_shared_lib.rag.postretrieval.boosters import RecencyBooster
    from cody_ai_shared_lib.rag.render import DefaultRenderer

    class Broken(DefaultRenderer):
        def timestamp(self, meta):
            raise KeyError("published")

    with pytest.raises(KeyError):
        RecencyBooster(clock=FrozenClock(NOW)).boost([RAGNode(node_id="n", text="x")], renderer=Broken())


def test_removed_aliases_stay_removed():
    node = RAGNode(node_id="n", text="x", span=SourceSpan(doc_id="d"))
    for attribute in ("id", "document_id", "section_id", "order_index"):
        assert not hasattr(node, attribute)
    request = question()
    assert not hasattr(request, "query")
    assert not hasattr(request, "conversation")
