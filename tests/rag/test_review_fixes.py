"""Fixes for the independent code review of 2026-11-11 (specs/RAG_CODE_REVIEW_2026-11-11.md), RAG_SPEC 1.2.0."""
from __future__ import annotations

import tomllib
from pathlib import Path

from cody_ai_shared_lib.rag import InMemoryRAGStore, RAGDocumentInput, RAGSectionInput
from cody_ai_shared_lib.rag.models import DefaultNodeMeta

from .test_blocker_regressions import FakeLLM, document, engine, profile  # noqa: F401
from .test_blocker_regressions import real_embedding_shape  # noqa: F401  (autouse fixture)
from .test_integration_round2 import UNIT_TEXT, FailingNthLLM, RecordingStore, unit_profile

ROOT = Path(__file__).resolve().parents[2]


# ── Platform (B1) ─────────────────────────────────────────────────────────


def test_package_requires_python_3_13():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["requires-python"] == ">=3.13"


# ── Index lifecycle (H1, H2, M6, M9, vectorless summaries) ───────────────


def _upgrade(store: InMemoryRAGStore, splitter_version: str = "2.0.0") -> None:
    """Simulate an index upgrade: the store now declares a new embedding signature."""
    store._embedding_space = store.embedding_space().model_copy(update={"splitter_version": splitter_version})


def test_sync_after_an_upgrade_refreshes_summaries_without_summarizing_again():
    store = InMemoryRAGStore()
    llm = FailingNthLLM(fail_on=10_000)
    eng = engine(unit_profile(), store, llm=llm)
    doc = document(UNIT_TEXT)
    eng.ingest_document(doc)
    calls_after_ingest = llm.calls

    _upgrade(store)
    assert eng.needs_reindex("b").embeddings_stale
    eng.sync_document(doc)

    assert not eng.needs_reindex("b").embeddings_stale
    assert all("2.0.0" in fp.embedding_signature for fp in store.get_fingerprints("b").values())
    assert llm.calls == calls_after_ingest                          # summaries re-embedded, not re-summarized


def test_sync_regenerates_stale_summaries_when_the_store_cannot_fetch_them():
    store = InMemoryRAGStore()
    store._capabilities = store.capabilities().model_copy(update={"supports_fetch_nodes": False})
    llm = FailingNthLLM(fail_on=10_000)
    eng = engine(unit_profile(), store, llm=llm)
    doc = document(UNIT_TEXT)
    eng.ingest_document(doc)
    calls_after_ingest = llm.calls

    _upgrade(store)
    eng.sync_document(doc)

    assert not eng.needs_reindex("b").embeddings_stale
    assert llm.calls > calls_after_ingest


def test_sync_rewrites_search_tokens_after_an_analyzer_change():
    store = InMemoryRAGStore()
    eng = engine(unit_profile(), store, llm=FakeLLM("A short summary."))
    doc = document(UNIT_TEXT)
    eng.ingest_document(doc)

    eng.analyzer.version = "9.9.9"
    assert eng.needs_reindex("b").fts_stale
    eng.sync_document(doc)
    assert not eng.needs_reindex("b").fts_stale


def test_a_splitter_version_bump_is_detected_on_a_default_store():
    store = InMemoryRAGStore()
    eng = engine(profile(), store)
    eng.ingest_document(document("Hello world. " * 40))
    assert store.embedding_space().splitter_version is None
    assert not eng.needs_reindex("b").embeddings_stale

    eng.splitter.version = "9.9.9"
    assert eng.needs_reindex("b").embeddings_stale


def test_summaries_without_vectors_carry_no_embedding_signature_and_never_go_stale():
    store = RecordingStore()
    eng = engine(unit_profile(embed_summaries=False), store, llm=FakeLLM("A short summary."))
    doc = document(UNIT_TEXT)
    eng.ingest_document(doc)

    write = store.writes[0]
    summaries = [write.section_summary, *write.unit_summaries]
    assert all(s.vector is None and s.fingerprint.embedding_signature is None for s in summaries)
    assert all(p.fingerprint.embedding_signature for p in write.passages)

    _upgrade(store)
    eng.sync_document(doc)
    assert not eng.needs_reindex("b").embeddings_stale
    fingerprints = store.get_fingerprints("b")
    assert all(fingerprints[s.node.node_id].embedding_signature is None for s in summaries)


def test_turning_summary_embedding_on_makes_vectorless_summaries_stale_and_sync_embeds_them():
    store = RecordingStore()
    engine(unit_profile(embed_summaries=False), store, llm=FakeLLM("A short summary.")).ingest_document(document(UNIT_TEXT))

    eng = engine(unit_profile(embed_summaries=True), store, llm=FakeLLM("A short summary."))
    assert eng.needs_reindex("b").embeddings_stale
    eng.sync_document(document(UNIT_TEXT))
    assert not eng.needs_reindex("b").embeddings_stale
    assert all(fp.embedding_signature for fp in store.get_fingerprints("b").values())


def test_every_ingested_node_carries_the_document_revision():
    store = InMemoryRAGStore()
    eng = engine(unit_profile(), store, llm=FakeLLM("A short summary."))
    section = RAGSectionInput(section_id="s0", section_seq=0, text=UNIT_TEXT, metadata=DefaultNodeMeta())
    eng.ingest_document(RAGDocumentInput(doc_id="b", sections=[section], revision="rev-7"))

    nodes = list(store._nodes.values()) + list(store._section_summaries.values())
    assert nodes and {n.revision for n in nodes} == {"rev-7"}
    assert store.current_revisions([nodes[0].node_id]) == {nodes[0].node_id: "rev-7"}


def test_the_embedder_uses_the_task_type_the_store_declares():
    store = InMemoryRAGStore()
    store._embedding_space = store.embedding_space().model_copy(update={"task_type_documents": "SEMANTIC_SIMILARITY"})
    eng = engine(profile(), store)
    assert eng.embedder.task_type == "SEMANTIC_SIMILARITY"


# ── Isolation and guards (H3, M25, M23, M24, M18) ────────────────────────

import pytest  # noqa: E402

from cody_ai_shared_lib.rag import (  # noqa: E402
    MissingRequiredScopeError,
    OwnerScope,
    RAGFilterSpec,
    RAGNode,
    RAGQueryRequest,
    ScopeViolationError,
    SourceSpan,
)
from cody_ai_shared_lib.rag.guards import TemporalGuard  # noqa: E402

from .test_integration_round2 import tenant_store  # noqa: E402


def test_a_scoped_store_cannot_overwrite_another_owners_document():
    store = tenant_store("tenant_or_user_id")
    a, b = OwnerScope(tenant_or_user_id="A"), OwnerScope(tenant_or_user_id="B")
    engine(profile(), store.scoped(a)).ingest_document(document("Secret of A. " * 30), owner=a)

    with pytest.raises(ScopeViolationError):
        engine(profile(), store.scoped(b)).ingest_document(document("Overwritten by B. " * 30), owner=b)
    assert store.doc_owner("b") == a
    assert all(n.text.startswith("Secret of A") for n in store._nodes.values())


def test_the_base_store_never_moves_a_document_to_another_owner():
    store = tenant_store("tenant_or_user_id")
    engine(profile(), store).ingest_document(document("Secret of A. " * 30), owner=OwnerScope(tenant_or_user_id="A"))
    with pytest.raises(ScopeViolationError):
        engine(profile(), store).ingest_document(document("By B. " * 30), owner=OwnerScope(tenant_or_user_id="B"))


def test_rows_without_an_owner_never_match_a_required_tenant():
    store = tenant_store("tenant_or_user_id")
    node = RAGNode(node_id="legacy", text="Legacy row", span=SourceSpan(doc_id="d"))
    store.upsert_section_nodes("d", 0, [node], [[0.1] * 768])          # written before owners existed
    assert store.vector_search([0.1] * 768, RAGFilterSpec(tenant_or_user_id="u1"), top_k=5) == []

    open_store = InMemoryRAGStore()                                     # no required tenant: unchanged
    open_store.upsert_section_nodes("d", 0, [node], [[0.1] * 768])
    assert len(open_store.vector_search([0.1] * 768, RAGFilterSpec(tenant_or_user_id="u1"), top_k=5)) == 1


def _reader():
    from cody_ai_shared_lib.rag import ModelBindings, long_form_reader
    from cody_ai_shared_lib.rag.profile import GenerationConfig

    return long_form_reader(
        ModelBindings(generation=["m1"], summary=["s1"], embedding="gemini-embedding-2"),
        generation=GenerationConfig(grounding_mode="strict", stream_mode="buffered"),   # the fake LLM cannot stream
    )


def test_a_profile_that_requires_the_spoiler_guard_rejects_a_request_without_a_boundary():
    eng = engine(_reader(), InMemoryRAGStore(), llm=FakeLLM("A short summary."))
    with pytest.raises(MissingRequiredScopeError, match="max_char_offset"):
        eng.retrieve(RAGQueryRequest(question="what happened", document_id="b"))


def test_spoiler_protection_off_is_the_document_length_not_a_missing_boundary():
    store = InMemoryRAGStore()
    eng = engine(_reader(), store, llm=FakeLLM("A short summary."))
    eng.ingest_document(document("Sentence number one is here. " * 40))
    request = RAGQueryRequest(question="what happened", document_id="b", filters=RAGFilterSpec(max_char_offset=10**9))
    assert eng.retrieve(request).nodes


def test_a_per_document_boundary_map_must_cover_every_document_in_scope():
    eng = engine(_reader(), InMemoryRAGStore(), llm=FakeLLM("A short summary."))
    partial = RAGFilterSpec(doc_ids=["a", "b"], max_char_offset_by_doc={"a": 100})
    with pytest.raises(MissingRequiredScopeError, match="max_char_offset_by_doc"):
        eng.retrieve(RAGQueryRequest(question="q", filters=partial))


def test_the_temporal_guard_reads_the_renderer_and_drops_undated_rows_the_store_did_not_filter():
    from datetime import datetime, timezone

    class StampRenderer:
        def timestamp(self, meta):
            return datetime(2026, 1, 5, tzinfo=timezone.utc) if meta.extra.get("dated") else None

    window = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 31, tzinfo=timezone.utc))
    from cody_ai_shared_lib.rag.models import DefaultNodeMeta as Meta

    dated = RAGNode(node_id="x", text="t", span=SourceSpan(doc_id="d"), metadata=Meta(extra={"dated": True}))
    undated = RAGNode(node_id="y", text="t", span=SourceSpan(doc_id="d"), metadata=Meta())

    guard = TemporalGuard(window, "published_on", renderer=StampRenderer(), store_applies_filter=False)
    assert guard.verify(dated) is dated
    assert guard.verify(undated) is None
    trusted = TemporalGuard(window, "published_on", renderer=StampRenderer(), store_applies_filter=True)
    assert trusted.verify(undated) is undated


def test_sibling_engines_share_the_fence_secret():
    eng = engine(profile(), InMemoryRAGStore())
    sibling = eng.with_store(InMemoryRAGStore())
    variant = eng.with_profile(profile())
    assert sibling.fencer is eng.fencer and variant.fencer is eng.fencer
    assert sibling.fencer.stable_id("user-a|doc-1") == eng.fencer.stable_id("user-a|doc-1")


# ── Retrieval, ranking and filters (H8-H13, M12-M17, M19-M21) ─────────────

from datetime import datetime, timedelta, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from cody_ai_shared_lib.rag import QueryIntent  # noqa: E402
from cody_ai_shared_lib.rag.granularity import FLAG_FULL_SCOPE, is_ranked  # noqa: E402
from cody_ai_shared_lib.rag.intent.temporal import TemporalResolver  # noqa: E402
from cody_ai_shared_lib.rag.postretrieval.boosters import RecencyBooster  # noqa: E402
from cody_ai_shared_lib.rag.postretrieval.stages import RerankStage  # noqa: E402
from cody_ai_shared_lib.rag.profile import RetrievalConfig  # noqa: E402
from cody_ai_shared_lib.rag.retrieval_pipeline import RetrievalPipeline  # noqa: E402
from cody_ai_shared_lib.rag.runtime import FrozenClock  # noqa: E402
from cody_ai_shared_lib.rag.strategies.full_context import FullContextStrategy  # noqa: E402
from cody_ai_shared_lib.rag.strategies.relaxation import DropTimeWindow, RelaxationState, WidenTimeWindow  # noqa: E402
from cody_ai_shared_lib.rag.transforms import QueryVariant  # noqa: E402

UTC = timezone.utc


def test_a_hard_filter_conflict_is_an_empty_answer_not_a_missing_key_error():
    store = tenant_store("tenant_or_user_id", "doc_ids")
    eng = engine(profile(), store)
    request = RAGQueryRequest(
        question="q", filters=RAGFilterSpec(tenant_or_user_id="u1", doc_ids=["a"]), document_id="b"
    )
    assert eng.retrieve(request).nodes == []
    plan = eng.plan(request)
    assert plan.strategy == "none" and "empty scope" in plan.tier_decision_reason
    with pytest.raises(MissingRequiredScopeError):            # leaving a key out is still an error
        eng.retrieve(RAGQueryRequest(question="q", filters=RAGFilterSpec(tenant_or_user_id="u1")))


def test_the_intent_window_intersects_the_callers_hard_range():
    hard = (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 6, 30, tzinfo=UTC))
    request = RAGQueryRequest(question="q", filters=RAGFilterSpec(date_range_utc=hard))
    week = QueryIntent(search_query="q", has_time_expression=True,
                       range_start_utc=datetime(2026, 6, 1, tzinfo=UTC), range_end_utc=datetime(2026, 6, 7, tzinfo=UTC))
    assert RetrievalPipeline.soft_date_window(request, week) == (week.range_start_utc, week.range_end_utc)

    outside = week.model_copy(update={"range_start_utc": datetime(2027, 1, 1, tzinfo=UTC),
                                      "range_end_utc": datetime(2027, 1, 7, tzinfo=UTC)})
    assert RetrievalPipeline.soft_date_window(request, outside) is None   # the hard range alone applies


def test_relaxation_never_goes_past_the_callers_hard_range():
    hard = (datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 6, 30, tzinfo=UTC))
    soft = (datetime(2026, 6, 1, tzinfo=UTC), datetime(2026, 6, 30, tzinfo=UTC))
    state = RelaxationState(RAGFilterSpec(date_range_utc=soft), QueryIntent(search_query="q"), True, None, hard)

    widened = WidenTimeWindow(factor=10).apply(state)
    assert widened.filters.date_range_utc[0] == hard[0]
    dropped = DropTimeWindow().apply(state)
    assert dropped.filters.date_range_utc == hard


def test_date_window_days_becomes_a_range_and_reversed_ranges_are_repaired():
    now = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
    resolver = TemporalResolver(clock=FrozenClock(now))
    recent = resolver.resolve(QueryIntent(search_query="q", has_time_expression=True, date_window_days=3))
    assert (recent.range_start_utc, recent.range_end_utc) == (now - timedelta(days=3), now)

    a, b = datetime(2026, 6, 7, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC)
    repaired = resolver.resolve(QueryIntent(search_query="q", has_time_expression=True, range_start_utc=a, range_end_utc=b))
    assert (repaired.range_start_utc, repaired.range_end_utc) == (b, a)


def test_ranked_searches_never_return_summaries():
    store = InMemoryRAGStore()
    eng = engine(unit_profile(retrieval=RetrievalConfig(default_strategy="dense_vector", top_k=3)), store,
                 llm=FakeLLM("A short summary."))
    eng.ingest_document(document(UNIT_TEXT))
    nodes = eng.retrieve(RAGQueryRequest(question="sentence")).nodes
    assert nodes and not any(n.is_summary for n in nodes)


def test_tier_3_keeps_the_section_summary_or_its_units_never_both():
    store = InMemoryRAGStore()
    engine(unit_profile(), store, llm=FakeLLM("A short summary.")).ingest_document(document(UNIT_TEXT))
    eng = engine(unit_profile(retrieval=RetrievalConfig(default_strategy="hierarchical")), store, llm=FakeLLM("x"))
    summaries = [n for n in eng.retrieve(RAGQueryRequest(question="sentence")).nodes if n.is_summary]
    assert [n.parent_kind for n in summaries] == ["section"]


def test_full_scope_text_has_no_overlap_and_stats_measure_the_same_text():
    store = InMemoryRAGStore()
    text = "Alpha beta gamma. " * 200
    engine(profile(), store).ingest_document(document(text))
    filters = RAGFilterSpec(doc_ids=["b"])
    nodes = store.fetch_full_scope(filters)
    assert all(prev.span.char_end <= nxt.span.char_start for prev, nxt in zip(nodes, nodes[1:]))
    assert store.scope_stats(filters).char_count == sum(len(n.text) for n in nodes)
    assert "".join(n.text for n in nodes) == text.rstrip()     # the whole text once (the splitter drops the trailing space)


def test_rerank_scores_each_comparison_subject_against_its_own_query():
    class WordReranker:
        def rerank(self, query, nodes, top_n, ctx):
            for n in nodes:
                n.rerank_score = 1.0 if query in n.text else 0.0
            return sorted(nodes, key=lambda n: -n.rerank_score)[:top_n]

    nodes = [RAGNode(node_id=f"{w}{i}", text=f"about {w}", span=SourceSpan(doc_id="d"))
             for w in ("alpha", "beta") for i in range(3)]
    ctx = SimpleNamespace(
        target_limit=lambda: 2, variants=[QueryVariant(text="alpha"), QueryVariant(text="beta", origin="secondary")],
        intent=QueryIntent(search_query="alpha"), request=RAGQueryRequest(question="alpha vs beta"), reranked=False,
    )
    kept = RerankStage(WordReranker()).process(nodes, ctx)
    assert {n.text for n in kept} == {"about alpha", "about beta"}


def test_boosters_never_move_ordered_layers():
    from cody_ai_shared_lib.rag.models import DefaultNodeMeta as Meta

    now = datetime(2026, 11, 11, tzinfo=UTC)
    summary = RAGNode(node_id="sum", text="s", span=SourceSpan(doc_id="d"), granularity="parent_summary",
                      is_summary=True, score=0.0)
    old = RAGNode(node_id="old", text="o", span=SourceSpan(doc_id="d"), score=0.9)
    new = RAGNode(node_id="new", text="n", span=SourceSpan(doc_id="d"), score=0.8,
                  metadata=Meta(extra={"created_at": now}))
    boosted = RecencyBooster(weight=1.0, clock=FrozenClock(now)).process([old, summary, new], SimpleNamespace(renderer=None))
    assert boosted[1] is summary and summary.score == 0.0
    assert [n.node_id for n in boosted if n is not summary] == ["new", "old"]


def test_tier_0_keeps_atomic_entries_whole_and_out_of_ranking():
    store = InMemoryRAGStore()
    entry = RAGNode(node_id="e1", text="An entry.", span=SourceSpan(doc_id="kb"), granularity="entry")
    store.upsert_section_nodes("kb", 0, [entry], [[0.1] * 768])
    ctx = SimpleNamespace(store=store, filters=RAGFilterSpec(doc_ids=["kb"]))
    nodes = FullContextStrategy().retrieve(ctx)
    assert nodes[0].granularity == "entry" and FLAG_FULL_SCOPE in nodes[0].flags and not is_ranked(nodes[0])
    assert FullContextStrategy.query_independent


def test_the_engine_never_asks_a_store_for_more_rows_than_it_declares():
    store = InMemoryRAGStore()
    store._capabilities = store.capabilities().model_copy(update={"max_top_k": 5})
    eng = engine(profile(retrieval=RetrievalConfig(default_strategy="dense_vector", top_k=50)), store)
    for i in range(10):
        store.upsert_section_nodes(f"d{i}", 0, [RAGNode(node_id=f"n{i}", text="t", span=SourceSpan(doc_id=f"d{i}"))],
                                   [[0.1] * 768])
    result = eng.retrieve(RAGQueryRequest(question="t"))
    assert len(result.nodes) <= 5 and result.truncated


def test_threshold_on_a_score_the_pipeline_cannot_produce_is_a_startup_error():
    from cody_ai_shared_lib.rag import ProfileConfigError

    with pytest.raises(ProfileConfigError, match=r"threshold\(rerank\)"):
        engine(profile(post_stages=["threshold(rerank)", "top_k"]), InMemoryRAGStore())


def test_summaries_that_do_not_fit_drop_the_oldest_and_keep_the_nearest():
    prof = profile().with_overrides(context={"max_context_tokens": 20_000, "reserved_history_tokens": 0,
                                             "summary_ratio": 0.10, "local_window_ratio": 0.0},
                                    retrieval={"full_context_max_tokens": 0})
    eng = engine(prof, InMemoryRAGStore())
    summaries = [
        RAGNode(node_id=f"b#s{seq}_sum", text=f"Summary of section {seq}. " + "word " * 400,
                span=SourceSpan(doc_id="b", section_seq=seq, char_start=seq * 1000, char_end=seq * 1000 + 900),
                granularity="parent_summary", is_summary=True, parent_kind="section")
        for seq in range(6)
    ]
    prompt = eng.assembler.assemble_prompt(RAGQueryRequest(question="q"), summaries)
    kept = sorted(n.span.section_seq for n in prompt.included_nodes if n.is_summary)
    assert kept and kept[-1] == 5 and len(kept) < 6     # some dropped, and never the one nearest the reader
