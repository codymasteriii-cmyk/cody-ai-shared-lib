"""Regression tests for the Phase 1-3 review blockers, using the real gateway and return shapes."""
from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime, timezone

import pytest

from cody_ai_shared_lib.embedding import EmbeddingResult, EmbeddingUsage
from cody_ai_shared_lib.llm import LLMResult, LLMUsage
from cody_ai_shared_lib.tokens import estimate_tokens
from cody_ai_shared_lib.rag import gateway as gateway_module
from cody_ai_shared_lib.rag.cache import InMemoryLRUCache
from cody_ai_shared_lib.rag.context.assembler import ContextAssembler
from cody_ai_shared_lib.rag.engine import RAGEngine
from cody_ai_shared_lib.rag.errors import GenerationError, RAGStageError, RetrievalError
from cody_ai_shared_lib.rag.gateway import ModelGateway
from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore
from cody_ai_shared_lib.rag.models import (
    DefaultNodeMeta,
    FilterPatch,
    QueryIntent,
    RAGDocumentInput,
    RAGFilterSpec,
    RAGNode,
    RAGQueryRequest,
    RAGSectionInput,
    SourceSpan,
)
from cody_ai_shared_lib.rag.ports import StoreCapabilities
from cody_ai_shared_lib.rag.profile import (
    EngineProfile,
    IngestionConfig,
    ModelBindings,
    RetrievalConfig,
    RuntimeConfig,
    SummariesConfig,
    long_form_reader,
)
from cody_ai_shared_lib.rag.render import DefaultRenderer

DIM = 768


class FakeLLM:
    def __init__(self, text: str = "Answer [1].", fail: bool = False):
        self.text = text
        self.fail = fail

    def generate_with_usage(self, **_: object) -> LLMResult:
        if self.fail:
            raise RuntimeError("provider down")
        return LLMResult(
            text=self.text,
            usage=LLMUsage(input_tokens=1, output_tokens=1, cost_usd=0.001),
            finish_reason="stop",
        )


@pytest.fixture(autouse=True)
def real_embedding_shape(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Same shape as generate_embedding: flat for a string, nested for a list. Records input counts."""
    calls: list[int] = []

    def fake(text: str | list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> EmbeddingResult:
        calls.append(len(text) if isinstance(text, list) else 1)
        vectors = [0.1] * DIM if isinstance(text, str) else [[0.1] * DIM for _ in text]
        return EmbeddingResult(vectors=vectors, usage=EmbeddingUsage(input_tokens=1, cost_usd=0.0))

    monkeypatch.setattr(gateway_module, "generate_embedding_with_usage", fake)
    return calls


def seeded_store(capabilities: StoreCapabilities | None = None) -> InMemoryRAGStore:
    store = InMemoryRAGStore(capabilities=capabilities)
    node = RAGNode(
        node_id="n1",
        text="Alice met Bob. They talked for hours.",
        span=SourceSpan(doc_id="d", char_start=0, char_end=37),
    )
    store.upsert_section_nodes("d", 0, [node], [[0.1] * DIM])
    return store


def summarizing_profile() -> EngineProfile:
    return EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], summary=["s1"], embedding="gemini-embedding-2"),
        ingestion=IngestionConfig(summaries=SummariesConfig(enabled=True)),
    )


def profile(**changes: object) -> EngineProfile:
    return EngineProfile(name="t", models=ModelBindings(generation=["m1"], embedding="gemini-embedding-2"), **changes)


def engine(prof: EngineProfile | None = None, store: InMemoryRAGStore | None = None, llm=None, **kwargs) -> RAGEngine:
    gateway = kwargs.pop("gateway", None) or ModelGateway(llm_client=llm or FakeLLM())
    return RAGEngine(prof or profile(), store or seeded_store(), gateway=gateway, **kwargs)


def question(**kwargs: object) -> RAGQueryRequest:
    return RAGQueryRequest(question="what happened", **kwargs)


def hybrid_retrieval(**changes: object) -> RetrievalConfig:
    return RetrievalConfig(default_strategy="hybrid_rrf", **changes)


# 1 ── embedding shape


@pytest.mark.parametrize("strategy", ["dense_vector", "hybrid_rrf"])
def test_query_works_with_real_embedding_return_shape(strategy: str):
    prof = profile(retrieval=RetrievalConfig(default_strategy=strategy))
    assert [n.node_id for n in engine(prof).retrieve(question()).nodes] == ["n1"]


def test_gateway_returns_list_of_vectors_for_string_on_miss_and_hit():
    cache = InMemoryLRUCache()
    gateway = ModelGateway(llm_client=FakeLLM(), cache=cache)
    from cody_ai_shared_lib.rag.usage import UsageTracker

    tracker = UsageTracker(request_id="r")
    for _ in range(2):
        result = gateway.embed("q", "gemini-embedding-2", "s", tracker, task_type="RETRIEVAL_QUERY")
        assert result.vectors == [[0.1] * DIM]


# 2 ── import without pytest


def test_rag_imports_without_pytest():
    code = "import sys; sys.modules['pytest'] = None; import cody_ai_shared_lib.rag"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


# 3 ── adaptive tier decision


def adaptive() -> EngineProfile:
    return profile(retrieval=RetrievalConfig(default_strategy="adaptive"))


def test_plan_with_small_doc_hint_still_measures_the_scope():
    plan = engine(adaptive()).plan(question(doc_total_chars=1000))
    assert plan.strategy == "full_context"
    assert plan.scope is not None


def test_doc_hint_rules_tier0_out_without_a_store_call(monkeypatch: pytest.MonkeyPatch):
    store = seeded_store()
    monkeypatch.setattr(store, "scope_stats", lambda _: pytest.fail("scope_stats must not be called"))
    plan = engine(adaptive(), store).plan(question(doc_total_chars=1_000_000))
    assert plan.strategy != "full_context"
    assert plan.scope is None


def test_doc_hint_uses_visible_text_under_a_boundary():
    filters = RAGFilterSpec(max_char_offset=100)
    assert engine(adaptive()).plan(question(doc_total_chars=1_000_000, filters=filters)).strategy == "full_context"


def test_plan_with_strategy_override_and_no_retrieval_route():
    eng = engine()
    assert eng.plan(question(strategy_override="dense_vector")).strategy == "dense_vector"
    assert eng.plan(question(), intent=QueryIntent(route="no_retrieval")).strategy == "none"


# 4 ── Tier 0 budget


def test_tier0_under_gate_reaches_the_prompt_whole_and_in_the_stable_block():
    prof = long_form_reader(ModelBindings(generation=["m1"], summary=["s1"], embedding="gemini-embedding-2"))
    nodes = [
        RAGNode(
            node_id=f"s{i}",
            text="word " * 8000,
            granularity="full_doc",
            span=SourceSpan(doc_id="d", section_seq=i, char_start=i * 40000, char_end=(i + 1) * 40000),
        )
        for i in range(10)
    ]
    assert sum(estimate_tokens(n.text) for n in nodes) <= prof.retrieval.full_context_max_tokens

    prompt = ContextAssembler(prof, DefaultRenderer()).assemble_prompt(question(), nodes)

    assert prompt.user_message.count("<node ref=") == 10
    assert len(prompt.cache_breakpoints) == 2


# 5 ── fan-out on a bounded pool


def test_fan_out_with_client_side_rrf_does_not_deadlock():
    caps = StoreCapabilities(supports_vector_search=True, supports_keyword_fts=True)
    prof = profile(retrieval=hybrid_retrieval(), runtime=RuntimeConfig(pool_size=2))
    eng = engine(prof, seeded_store(caps))
    intent = QueryIntent(search_query="a", secondary_queries=["b", "c"])
    result: dict[str, int] = {}

    worker = threading.Thread(
        target=lambda: result.update(count=len(eng.retrieve(question(), intent=intent).nodes)), daemon=True
    )
    worker.start()
    worker.join(10)

    assert not worker.is_alive(), "fan-out deadlocked on the bounded pool"
    assert result["count"] == 1


# 6 ── filters only narrow


def patched(base: RAGFilterSpec, **fields: object) -> RAGFilterSpec:
    return base.patch(FilterPatch(fields=fields))


def test_patch_narrows_categories_similarity_and_custom():
    result = patched(
        RAGFilterSpec(categories=["a", "b"], min_similarity=0.8, custom={"x": 1}),
        categories=["b", "c"],
        min_similarity=0.1,
        custom={"y": 2},
    )
    assert result.categories == ["b"]
    assert result.min_similarity == 0.8
    assert result.custom == {"x": 1, "y": 2}
    assert not result.is_empty_scope


@pytest.mark.parametrize(
    "base, fields",
    [
        (RAGFilterSpec(tenant_or_user_id="t1"), {"tenant_or_user_id": "t2"}),
        (RAGFilterSpec(domain_id="a"), {"domain_id": "b"}),
        (RAGFilterSpec(custom={"x": 1}), {"custom": {"x": 2}}),
        (RAGFilterSpec(categories=["a"]), {"categories": ["b"]}),
        (RAGFilterSpec(doc_ids=["a"]), {"doc_ids": ["b"]}),
        (
            RAGFilterSpec(date_range_utc=(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc))),
            {"date_range_utc": (datetime(2026, 2, 1, tzinfo=timezone.utc), datetime(2026, 2, 2, tzinfo=timezone.utc))},
        ),
    ],
)
def test_patch_conflict_yields_empty_scope(base: RAGFilterSpec, fields: dict):
    result = patched(base, **fields)
    assert result.is_empty_scope
    assert result.tenant_or_user_id == base.tenant_or_user_id


class ConflictingGuard:
    name = "conflict"
    hard = True
    forces_strict_grounding = False

    def to_filter(self) -> FilterPatch:
        return FilterPatch(fields={"tenant_or_user_id": "other"})

    def verify(self, node):
        return node

    def verify_citation(self, citation) -> bool:
        return True

    def prompt_rules(self) -> list[str]:
        return []


def test_conflicting_guard_returns_the_empty_retrieval_answer():
    request = question(filters=RAGFilterSpec(tenant_or_user_id="mine"), guards=[ConflictingGuard()])
    response = engine().query(request)
    assert response.retrieved.nodes == []
    assert response.retrieved.strategy_used == "none"
    assert response.answer_found == "no"


# 7 ── cache, budget guard and fallbacks reach the real gateway


class RecordingGuard:
    enforces_cost = False

    def __init__(self):
        self.stages: list[str] = []

    def before_call(self, stage, model, est_input_tokens, optional, ctx):
        self.stages.append(stage)
        return "allow"


def test_query_embedding_goes_through_cache_and_budget_guard(real_embedding_shape: list[int]):
    guard = RecordingGuard()
    cache = InMemoryLRUCache()
    eng = engine(budget_guard=guard, cache=cache)

    eng.query(question())
    eng.query(question())

    assert "retrieval_embedding" in guard.stages
    assert cache.current_bytes > 0
    assert real_embedding_shape == [1]


class FailingReranker:
    def rerank(self, *_: object, **__: object):
        raise RuntimeError("reranker down")


def rerank_profile(**changes: object) -> EngineProfile:
    return EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], reranker="jina-reranker", embedding="gemini-embedding-2"),
        post_stages=["rerank"],
        **changes,
    )


def test_rerank_raise_fallback_is_honoured():
    gateway = ModelGateway(llm_client=FakeLLM(), rerank_client=FailingReranker())
    eng = engine(rerank_profile(fallbacks={"rerank": "raise"}), gateway=gateway)
    with pytest.raises(RAGStageError) as caught:
        eng.retrieve(question())
    assert caught.value.stage == "post_retrieval_rerank"


def test_rerank_default_still_degrades():
    gateway = ModelGateway(llm_client=FakeLLM(), rerank_client=FailingReranker())
    result = engine(rerank_profile(), gateway=gateway).retrieve(question())
    assert [d.id for d in result.degradations] == ["DG-06"]


def test_genuine_store_type_error_is_not_retried(monkeypatch: pytest.MonkeyPatch):
    store = seeded_store()
    calls: list[int] = []

    def broken(*_: object, **__: object):
        calls.append(1)
        raise TypeError("bug inside the adapter")

    monkeypatch.setattr(store, "hybrid_search", broken)
    with pytest.raises(RetrievalError) as caught:
        engine(profile(retrieval=hybrid_retrieval()), store).retrieve(question())
    assert isinstance(caught.value.cause, TypeError)
    assert calls == [1]


# 8 ── failure context


def assert_failure_context(error: GenerationError) -> None:
    assert error.request_id
    assert error.usage
    assert error.usage_total is not None
    assert error.usage_total.calls >= 1
    assert error.retrieval is not None
    assert [n.node_id for n in error.retrieval.nodes] == ["n1"]


def test_query_failure_carries_usage_and_sources():
    with pytest.raises(GenerationError) as caught:
        engine(llm=FakeLLM(fail=True)).query(question())
    assert_failure_context(caught.value)


def test_stream_failure_carries_usage_and_sources():
    prof = profile(generation={"stream_mode": "buffered"})
    stream = engine(prof, llm=FakeLLM(fail=True)).stream_query(question())
    with pytest.raises(GenerationError) as caught:
        list(stream)
    assert_failure_context(caught.value)


def test_generate_from_nodes_failure_carries_usage_and_sources():
    node = RAGNode(node_id="n1", text="x", span=SourceSpan(doc_id="d", char_end=1))
    with pytest.raises(GenerationError) as caught:
        engine(llm=FakeLLM(fail=True)).generate_from_nodes(question(), [node])
    assert_failure_context(caught.value)


def test_ingestion_failure_carries_section_usage():
    prof = summarizing_profile()
    doc = RAGDocumentInput(
        doc_id="b",
        sections=[RAGSectionInput(section_id="s", section_seq=0, text="A sentence here. " * 200, metadata=DefaultNodeMeta())],
    )
    with pytest.raises(RAGStageError) as caught:
        engine(prof, InMemoryRAGStore(), llm=FakeLLM(fail=True)).ingest_document(doc)
    assert caught.value.usage
    assert caught.value.request_id == "ingest-b-s0"


# 9 ── typed fail-loud errors


def test_cache_raise_fallback_raises_a_typed_error():
    class BrokenCache:
        def get(self, key: str):
            raise OSError("cache down")

        def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
            raise OSError("cache down")

    eng = engine(profile(fallbacks={"cache": "raise"}), cache=BrokenCache())
    with pytest.raises(RAGStageError) as caught:
        eng.retrieve(question())
    assert caught.value.stage == "cache"


# 10 ── sync


class PrefixedRenderer(DefaultRenderer):
    def embed_text(self, node) -> str:
        return "Title: " + node.text


def document(text: str, doc_id: str = "b", extra_sections: int = 0) -> RAGDocumentInput:
    sections = [
        RAGSectionInput(section_id=f"s{i}", section_seq=i, text=text, metadata=DefaultNodeMeta())
        for i in range(1 + extra_sections)
    ]
    return RAGDocumentInput(doc_id=doc_id, sections=sections)


def summarizing_engine(store: InMemoryRAGStore, **kwargs) -> RAGEngine:
    return engine(summarizing_profile(), store, llm=FakeLLM("A short summary."), **kwargs)


def summary_ids(store: InMemoryRAGStore) -> set[str]:
    return set(store.get_fingerprints("b")) - {n.node_id for n in store._nodes.values() if not n.is_summary}


def test_sync_of_unchanged_document_embeds_nothing_with_custom_embed_text(real_embedding_shape: list[int]):
    store = InMemoryRAGStore()
    eng = engine(store=store, renderer=PrefixedRenderer())
    doc = document("Sentence number one is here. " * 200)
    eng.ingest_document(doc)
    real_embedding_shape.clear()

    progress = eng.sync_document(doc)

    assert real_embedding_shape == []
    assert progress.passages_skipped_unchanged > 0
    assert progress.passages_upserted == 0


def test_sync_keeps_summaries_of_unchanged_sections():
    store = InMemoryRAGStore()
    eng = summarizing_engine(store)
    doc = document("Sentence number one is here. " * 1000)
    eng.ingest_document(doc)
    before = summary_ids(store)
    assert len(before) >= 2

    eng.sync_document(doc)

    assert summary_ids(store) == before


def test_sync_refreshes_summaries_only_for_the_changed_section():
    store = InMemoryRAGStore()
    eng = summarizing_engine(store)
    long_text = "Sentence number one is here. " * 1000
    eng.ingest_document(document(long_text, extra_sections=1))

    changed = document(long_text, extra_sections=1)
    changed.sections[1].text = "A completely different chapter. " * 60
    progress = eng.sync_document(changed)

    ids = summary_ids(store)
    assert progress.summary_status == "generated"
    assert any("#s0_u" in i for i in ids), "unchanged section keeps its unit summaries"
    assert not any("#s1_u" in i for i in ids), "the shorter new text produces no unit summaries"
    assert "b#s1_sum" in ids


def test_sync_removes_summaries_of_a_removed_section():
    store = InMemoryRAGStore()
    eng = summarizing_engine(store)
    text = "Sentence number one is here. " * 100
    eng.ingest_document(document(text, extra_sections=1))
    assert "b#s1_sum" in store.get_fingerprints("b")

    eng.sync_document(document(text))

    fingerprints = store.get_fingerprints("b")
    assert "b#s1_sum" not in fingerprints
    assert "b#s0_sum" in fingerprints
    assert store.fetch_parent_summaries(RAGFilterSpec(doc_ids=["b"]))[0].node_id == "b#s0_sum"


def test_sync_writes_complete_nodes():
    store = InMemoryRAGStore()
    eng = engine(store=store)
    eng.sync_document(document("Sentence number one is here. " * 100))
    node = next(n for n in store._nodes.values() if n.node_id.startswith("b#s0p"))
    assert node.token_estimate
    assert node.search_tokens
    assert node.content_hash == store.get_fingerprints("b")[node.node_id].content_hash


def test_sync_without_delete_capability_raises():
    from cody_ai_shared_lib.rag.errors import StoreCapabilityError

    store = InMemoryRAGStore(capabilities=StoreCapabilities(supports_fingerprints=True, supports_delete=False))
    with pytest.raises(StoreCapabilityError):
        engine(store=store).sync_document(document("Some text here."))


def test_partial_upsert_keeps_unchanged_nodes_in_the_section():
    store = InMemoryRAGStore()
    eng = engine(store=store)
    text = "Sentence number one is here. " * 200
    eng.ingest_document(document(text))
    total = len(store._nodes)

    changed = document(text[:-29] + "A different last sentence. ")
    eng.sync_document(changed)

    assert store.delete_sections("b", 0) >= total
    assert not store._nodes
    assert store.get_fingerprints("b") == {}
