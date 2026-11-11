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
