"""AudioReading integration feedback round 2 (RAG_SPEC 1.2.0): INT-01..INT-04 and the scoped-store additions."""
from __future__ import annotations

import pytest

from cody_ai_shared_lib.llm import LLMResult, LLMUsage
from cody_ai_shared_lib.rag import (
    BudgetExceeded,
    MissingRequiredScopeError,
    OwnerScope,
    ProfileConfigError,
    RAGStore,
    ScopeViolationError,
    SectionWrite,
    format_node_id,
    parse_node_id,
)
from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore
from cody_ai_shared_lib.rag.models import RAGFilterSpec, RAGNode, RAGQueryRequest, SourceSpan
from cody_ai_shared_lib.rag.profile import EngineProfile, IngestionConfig, ModelBindings, SummariesConfig

from .test_blocker_regressions import FakeLLM, document, engine, profile  # noqa: F401
from .test_blocker_regressions import real_embedding_shape  # noqa: F401  (autouse fixture)

UNIT_TEXT = "Sentence number one is here. " * 30          # about 870 chars: a summary and three units


def unit_profile(embed_summaries: bool = True, **changes: object) -> EngineProfile:
    return EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], summary=["s1"], embedding="gemini-embedding-2"),
        ingestion=IngestionConfig(
            summaries=SummariesConfig(
                enabled=True, min_chars_for_summary=1, summary_unit_chars=300, embed_summaries=embed_summaries
            )
        ),
        **changes,
    )


class RecordingStore(InMemoryRAGStore):
    """Overrides upsert_section like a SQL adapter, and never computes a fingerprint itself."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writes: list[SectionWrite] = []

    def upsert_section(self, write):
        self.writes.append(write)
        return super().upsert_section(write)

    def _fingerprint(self, node):
        raise AssertionError("the engine supplies fingerprints; the store must not compute them")


class LegacyStore(InMemoryRAGStore):
    """A store written before 1.2.0: it implements only upsert_section_nodes."""
    upsert_section = RAGStore.upsert_section

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.legacy_calls = 0

    def upsert_section_nodes(self, *args, **kwargs):
        self.legacy_calls += 1
        return super().upsert_section_nodes(*args, **kwargs)


def tenant_store(*keys: str, store_cls=InMemoryRAGStore):
    store = store_cls()
    store._capabilities = store.capabilities().model_copy(update={"required_filter_keys": frozenset(keys)})
    return store


class FailingNthLLM(FakeLLM):
    def __init__(self, fail_on: int):
        super().__init__("A short summary.")
        self.calls = 0
        self.fail_on = fail_on

    def generate_with_usage(self, **_: object) -> LLMResult:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("provider down")
        return LLMResult(text=self.text, usage=LLMUsage(input_tokens=1, output_tokens=1), finish_reason="stop")


# ── INT-01 · SectionWrite ─────────────────────────────────────────────────


def test_section_write_separates_passages_units_and_section_summary():
    store = RecordingStore()
    engine(unit_profile(), store, llm=FakeLLM("A short summary.")).ingest_document(document(UNIT_TEXT))

    write = store.writes[0]
    assert write.section_summary is not None and write.section_summary.unit_seq is None
    assert [u.unit_seq for u in write.unit_summaries] == [0, 1, 2]
    for unit in write.unit_summaries:
        assert unit.node.node_id == format_node_id("b", 0, "unit_summary", unit.unit_seq)
    assert all(not p.node.is_summary for p in write.passages)
    assert all(p.vector for p in write.passages)


def test_skipped_unit_leaves_a_gap_instead_of_shifting_later_units():
    store = RecordingStore()
    prof = unit_profile(fallbacks={"summarizer": "skip"})
    engine(prof, store, llm=FailingNthLLM(fail_on=3)).ingest_document(document(UNIT_TEXT))   # call 3 = unit 1

    write = store.writes[0]
    assert [u.unit_seq for u in write.unit_summaries] == [0, 2]
    assert [parse_node_id("b", u.node.node_id).number for u in write.unit_summaries] == [0, 2]


def test_legacy_store_still_ingests_through_the_default_delegate():
    store = LegacyStore()
    progress = engine(unit_profile(), store, llm=FakeLLM("A short summary.")).ingest_document(document(UNIT_TEXT))

    assert store.legacy_calls == 1
    units = sorted(n.node_id for n in store._nodes.values() if n.is_summary)
    assert units == [format_node_id("b", 0, "unit_summary", i) for i in range(3)]
    assert progress.passages_upserted > 0


# ── INT-02 · embed_summaries ──────────────────────────────────────────────


def test_embed_summaries_false_makes_no_summary_embedding_calls():
    store = RecordingStore()
    progress = engine(unit_profile(embed_summaries=False), store, llm=FakeLLM("A short summary.")).ingest_document(
        document(UNIT_TEXT)
    )

    assert "summary_embedding" not in {u.stage for u in progress.usage}
    write = store.writes[0]
    assert write.section_summary.vector is None
    assert all(u.vector is None for u in write.unit_summaries)


def test_embed_summaries_false_with_a_legacy_store_refuses_to_start():
    with pytest.raises(ProfileConfigError, match="overrides upsert_section"):
        engine(unit_profile(embed_summaries=False), LegacyStore(), llm=FakeLLM())


def test_summaries_are_embedded_in_one_batch(real_embedding_shape):
    engine(unit_profile(), RecordingStore(), llm=FakeLLM("A short summary.")).ingest_document(document(UNIT_TEXT))
    # one call for the passages, one for the section summary and its three units together
    assert real_embedding_shape[-1] == 4


# ── INT-03 · fingerprints come from the engine ───────────────────────────


def test_store_persists_engine_fingerprints_and_reindex_is_clean():
    store = RecordingStore()
    eng = engine(unit_profile(), store, llm=FakeLLM("A short summary."))
    eng.ingest_document(document(UNIT_TEXT))

    expected = "gemini-embedding-2|768|RETRIEVAL_DOCUMENT|1.0.0"
    fingerprints = store.get_fingerprints("b")
    assert fingerprints and {fp.embedding_signature for fp in fingerprints.values()} == {expected}
    passages = {nid: fp for nid, fp in fingerprints.items() if not nid.endswith("_sum")}
    summaries = {nid: fp for nid, fp in fingerprints.items() if nid.endswith("_sum")}
    assert {fp.fts_signature for fp in passages.values()} == {f"{eng.analyzer.name}|{eng.analyzer.version}"}
    assert summaries and {fp.fts_signature for fp in summaries.values()} == {None}   # never keyword-searched
    status = eng.needs_reindex("b")
    assert not status.embeddings_stale and not status.fts_stale


def test_reindex_flags_only_the_signature_that_changed():
    store = RecordingStore()
    eng = engine(unit_profile(), store, llm=FakeLLM("A short summary."))
    eng.ingest_document(document(UNIT_TEXT))

    store._embedding_space = store.embedding_space().model_copy(update={"splitter_version": "2.0.0"})
    status = eng.needs_reindex("b")
    assert status.embeddings_stale and not status.fts_stale

    store._embedding_space = store.embedding_space().model_copy(update={"splitter_version": "1.0.0"})
    eng.analyzer.version = "9.9.9"
    status = eng.needs_reindex("b")
    assert status.fts_stale and not status.embeddings_stale


def test_sync_writes_engine_fingerprints_too():
    store = RecordingStore()
    eng = engine(unit_profile(), store, llm=FakeLLM("A short summary."))
    eng.ingest_document(document(UNIT_TEXT))
    store.writes.clear()

    eng.sync_document(document(UNIT_TEXT + "A brand new closing sentence."))

    assert store.writes and all(p.fingerprint.embedding_signature for w in store.writes for p in w.passages)
    assert not eng.needs_reindex("b").embeddings_stale


def test_node_id_helpers_round_trip_and_need_the_doc_id():
    for kind, index in (("passage", 4), ("section_summary", None), ("unit_summary", 2)):
        node_id = format_node_id("book#7", 3, kind, index)
        parts = parse_node_id("book#7", node_id)
        assert (parts.section_seq, parts.kind, parts.number) == (3, kind, index)
    assert parse_node_id("book#7", "kb-entry-42") is None
    assert parse_node_id("other", format_node_id("book#7", 3, "passage", 0)) is None


# ── INT-04 · owner scope on ingestion ─────────────────────────────────────


def test_ingestion_without_required_owner_fails_before_any_paid_call(real_embedding_shape):
    store = tenant_store("tenant_or_user_id", "doc_ids")
    eng = engine(profile(), store)

    with pytest.raises(MissingRequiredScopeError) as caught:
        eng.ingest_document(document("Some text here. " * 10))
    assert caught.value.missing_keys == ["tenant_or_user_id"]          # doc_ids is the document itself
    assert real_embedding_shape == []


def test_every_ingestion_path_enforces_every_required_key():
    store = tenant_store("tenant_or_user_id", "domain_id")
    eng = engine(profile(), store)
    user_only = OwnerScope(tenant_or_user_id="u1")
    node = RAGNode(node_id="x", text="Some text.", span=SourceSpan(doc_id="b"))

    with pytest.raises(MissingRequiredScopeError, match="domain_id"):
        eng.ingest_document(document("Some text. " * 10), owner=user_only)
    with pytest.raises(MissingRequiredScopeError, match="domain_id"):
        eng.sync_document(document("Some text. " * 10), owner=user_only)
    with pytest.raises(MissingRequiredScopeError, match="domain_id"):
        eng.prepare_nodes([node], owner=user_only)

    full = OwnerScope(tenant_or_user_id="u1", domain_id="markets")
    assert eng.prepare_nodes([node], owner=full).prepared


def test_budget_guard_sees_the_owner_and_can_stop_ingestion_midway():
    class CountingGuard:
        enforces_cost = False

        def __init__(self):
            self.users = []

        def before_call(self, stage, model, est_input_tokens, optional, ctx):
            self.users.append(ctx.tenant_or_user_id)
            if len(self.users) == 3:
                raise BudgetExceeded("daily limit reached")
            return "allow"

    guard = CountingGuard()
    store = tenant_store("tenant_or_user_id", store_cls=RecordingStore)
    eng = engine(unit_profile(), store, llm=FakeLLM("A short summary."), budget_guard=guard)
    owner = OwnerScope(tenant_or_user_id="user-a")

    with pytest.raises(BudgetExceeded):
        eng.ingest_document(document(UNIT_TEXT), owner=owner)
    assert guard.users == ["user-a"] * 3
    assert store.writes == []


def test_owner_reaches_the_store_write():
    store = tenant_store("tenant_or_user_id", store_cls=RecordingStore)
    owner = OwnerScope(tenant_or_user_id="user-a")
    engine(profile(), store).ingest_document(document("Some text. " * 10), owner=owner)
    assert store.writes[0].owner == owner


def test_observers_receive_ingestion_usage():
    class UsageObserver:
        def __init__(self):
            self.usages = []

        def on_stage(self, record): ...
        def on_degrade(self, degradation): ...
        def on_complete(self, trace, outcome): ...

        def on_usage(self, usage):
            self.usages.append(usage)

    observer = UsageObserver()
    engine(profile(), InMemoryRAGStore(), observers=[observer]).ingest_document(document("Some text. " * 10))
    assert any(u.stage == "ingest_embedding" for u in observer.usages)


def test_prepare_nodes_calls_get_their_own_request_id():
    class IdGuard:
        enforces_cost = False

        def __init__(self):
            self.ids = []

        def before_call(self, stage, model, est_input_tokens, optional, ctx):
            self.ids.append(ctx.request_id)
            return "allow"

    guard = IdGuard()
    eng = engine(profile(), InMemoryRAGStore(), budget_guard=guard)
    node = RAGNode(node_id="x", text="Some text.", span=SourceSpan(doc_id="b"))
    eng.prepare_nodes([node])
    eng.prepare_nodes([node])
    assert len(set(guard.ids)) == 2 and all(i.startswith("prepare-") for i in guard.ids)


# ── Scoped stores (closing reply, request 1) ─────────────────────────────


def test_scope_violation_from_a_store_reaches_the_caller_typed():
    class ScopedStore(InMemoryRAGStore):
        def vector_search(self, query_vector, filters, top_k):
            raise ScopeViolationError("store is scoped to user-a", key="tenant_or_user_id")

        def hybrid_search(self, *args, **kwargs):
            raise ScopeViolationError("store is scoped to user-a", key="tenant_or_user_id")

    eng = engine(profile(), ScopedStore())
    with pytest.raises(ScopeViolationError):
        eng.retrieve(RAGQueryRequest(question="what happened", filters=RAGFilterSpec(doc_ids=["d"])))


def test_with_store_shares_pool_cache_and_gateway():
    eng = engine(profile(), InMemoryRAGStore())
    scoped = InMemoryRAGStore()
    sibling = eng.with_store(scoped)

    assert sibling.store is scoped
    assert sibling._pool is eng._pool and sibling.cache is eng.cache and sibling.gateway is eng.gateway
    assert sibling.profile is eng.profile


# ── INT-06 · scope_stats diagnostic ──────────────────────────────────────


def test_zero_scope_stats_with_a_text_hint_is_reported_but_not_trusted(caplog):
    from cody_ai_shared_lib.rag.models import QueryIntent
    from cody_ai_shared_lib.rag.profile import RetrievalConfig
    from cody_ai_shared_lib.rag.strategies.adaptive import AdaptiveRetrievalStrategy

    prof = profile(retrieval=RetrievalConfig(full_context_max_tokens=50_000))
    request = RAGQueryRequest(question="what happened", doc_total_chars=5_000)
    with caplog.at_level("WARNING", logger="shared-rag"):
        plan = AdaptiveRetrievalStrategy.decide_tier(
            request, QueryIntent(search_query="what happened"), RAGFilterSpec(doc_ids=["unindexed"]),
            InMemoryRAGStore(), prof,
        )

    assert plan.strategy != "full_context"                       # the hint never overrides the store
    assert "scope_stats reports 0 tokens" in plan.tier_decision_reason
    assert any("scope_stats" in r.message for r in caplog.records)


# ── INT-08 / INT-09 · streaming lifecycle and caller contract ────────────


def _streaming_engine(chunks: list[str]):
    from unittest.mock import MagicMock

    from cody_ai_shared_lib.llm import LLMStreamChunk
    from cody_ai_shared_lib.rag import ModelGateway, RAGDocumentInput, RAGEngine, RAGSectionInput, long_form_reader
    from cody_ai_shared_lib.rag.ports import EmbeddingSpace

    pulled: list[str] = []

    def stream(*args, **kwargs):
        for text in chunks:
            pulled.append(text)
            yield LLMStreamChunk(delta=text, usage=None, finish_reason=None)

    embedder = MagicMock()
    embedder.embed_with_usage.side_effect = lambda texts, task_type: MagicMock(
        vectors=[[0.05] * 768 for _ in texts],
        usage=MagicMock(input_tokens=1, cost_usd=0.0, price_table_version="v", estimated=True),
    )
    llm = MagicMock()
    llm.stream.side_effect = stream
    prof = long_form_reader(
        models=ModelBindings(generation=["gemini-3.5-flash-lite"], summary=["gemini-3.5-flash-lite"], embedding="gemini-embedding-2"),
        guards=[],   # not a spoiler test; the preset would otherwise require a boundary (M23)
    )
    store = InMemoryRAGStore(embedding_space=EmbeddingSpace(model="gemini-embedding-2", dims=768))
    eng = RAGEngine(profile=prof, store=store, gateway=ModelGateway(llm_client=llm, embedding_client=embedder))
    eng.ingest_document(RAGDocumentInput(doc_id="s1", sections=[RAGSectionInput(section_seq=0, text="Some text.")]))
    return eng, pulled


def test_leaving_a_with_block_closes_the_stream_and_records_the_cut_off_call():
    eng, pulled = _streaming_engine(["Chunk one ", "chunk two ", "chunk three"])

    with eng.stream_query(RAGQueryRequest(question="Query", document_id="s1")) as stream:
        for event in stream:
            if event.event == "delta":
                break
    pulled_at_exit = len(pulled)

    with pytest.raises(StopIteration):
        next(stream)
    assert len(pulled) == pulled_at_exit < 3                       # no further model output was read
    total = stream.usage_total
    assert total.output_tokens > 0                                 # the cancelled call is counted
    stream.close()                                                 # closing twice is harmless


def test_done_echoes_request_id_and_reports_a_missing_flag_as_none():
    eng, _ = _streaming_engine(["An answer with no flag [1]."])
    request = RAGQueryRequest(question="Query", document_id="s1", request_id="msg-42", metadata={"message_id": "m-7"})

    with eng.stream_query(request) as stream:
        done = [e for e in stream if e.event == "done"][0].data

    assert done["answer_found"] is None
    assert done["request_id"] == "msg-42"
    assert done["message_id"] == "m-7"
    assert "[[" not in done["answer_final"]
