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
