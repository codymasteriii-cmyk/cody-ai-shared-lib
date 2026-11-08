"""Tests for Ingestion Pipeline and Lifecycle (AC-RAG-05, AC-RAG-06, AC-RAG-07, AC-RAG-65, AC-RAG-66, AC-RAG-71)."""
from unittest.mock import MagicMock
import pytest

from cody_ai_shared_lib.rag import (
    BudgetExceeded,
    EmbeddingSignatureMismatch,
    EngineProfile,
    InMemoryRAGStore,
    ModelBindings,
    PreparedNode,
    ProfileConfigError,
    RAGDocumentInput,
    RAGNode,
    RAGQueryRequest,
    RAGSectionInput,
    RAGEngine,
    long_form_reader,
)
from cody_ai_shared_lib.rag.models import RAGFilterSpec, SourceSpan
from cody_ai_shared_lib.rag.ports import EmbeddingSpace, Fingerprint


def _make_engine_and_store():
    bindings = ModelBindings(
        generation=["gemini-3.5-flash-lite"],
        embedding="gemini-embedding-2",
        summary=["gemini-3.5-flash-lite"],
    )
    profile = long_form_reader(models=bindings)
    store = InMemoryRAGStore(
        embedding_space=EmbeddingSpace(
            model="gemini-embedding-2",
            dims=768,
            task_type_documents="RETRIEVAL_DOCUMENT",
        )
    )
    mock_gw = MagicMock()
    # Mock embedding: returns 768-dim mock vector for each text
    mock_gw.embed.side_effect = lambda texts, model, stage, usage_tracker, task_type="RETRIEVAL_DOCUMENT": MagicMock(
        vectors=[[0.05] * 768 for _ in texts],
        usage=MagicMock(input_tokens=len(texts) * 10, cost_usd=0.0001, cached_tokens=0, price_table_version="v2026-11-08", estimated=False),
    )
    mock_gw.generate.return_value = MagicMock(text="Section summary here.")

    engine = RAGEngine(profile=profile, store=store, gateway=mock_gw)
    return engine, store, mock_gw


def test_ingest_document_resumable_and_append_only():
    """AC-RAG-05: start_section_seq skips earlier sections; callback fires per section; append-only works."""
    engine, store, _ = _make_engine_and_store()

    sections = [
        RAGSectionInput(section_seq=0, text="Section 0 sentence one. Section 0 sentence two.", char_start=0),
        RAGSectionInput(section_seq=1, text="Section 1 sentence one. Section 1 sentence two.", char_start=100),
        RAGSectionInput(section_seq=2, text="Section 2 sentence one. Section 2 sentence two.", char_start=200),
        RAGSectionInput(section_seq=3, text="Section 3 sentence one. Section 3 sentence two.", char_start=300),
    ]
    doc = RAGDocumentInput(doc_id="doc-resumable", sections=sections)

    callbacks_fired = []

    # Ingest starting at section 2 (skips 0 and 1)
    progress = engine.ingest_document(
        doc=doc,
        start_section_seq=2,
        on_section_complete=lambda p: callbacks_fired.append(p.section_seq),
    )

    assert callbacks_fired == [2, 3]
    assert progress.sections_completed == 2
    assert progress.passages_upserted > 0

    # Earlier sections 0 and 1 were not ingested
    fps = store.get_fingerprints("doc-resumable")
    for nid in fps.keys():
        assert "s0" not in nid
        assert "s1" not in nid

    # Now append sections 4 and 5 in a second call (RAG-ING-6 append-only)
    doc_appended = RAGDocumentInput(
        doc_id="doc-resumable",
        sections=[
            RAGSectionInput(section_seq=4, text="Section 4 text sentence here.", char_start=400),
            RAGSectionInput(section_seq=5, text="Section 5 text sentence here.", char_start=500),
        ],
    )
    engine.ingest_document(doc=doc_appended)

    fps_after = store.get_fingerprints("doc-resumable")
    # Both earlier sections (2, 3) and appended sections (4, 5) exist
    assert any("s2" in nid for nid in fps_after.keys())
    assert any("s3" in nid for nid in fps_after.keys())
    assert any("s4" in nid for nid in fps_after.keys())
    assert any("s5" in nid for nid in fps_after.keys())


def test_sync_document_diffing():
    """AC-RAG-06: sync_document embeds changed, skips unchanged, deletes vanished, preserves node ids."""
    engine, store, mock_gw = _make_engine_and_store()

    doc_v1 = RAGDocumentInput(
        doc_id="doc-sync",
        sections=[
            RAGSectionInput(section_seq=0, text="Keep me unchanged sentence. Another sentence.", char_start=0),
            RAGSectionInput(section_seq=1, text="I will be deleted sentence. Another sentence.", char_start=100),
        ],
    )
    engine.ingest_document(doc_v1)
    fps_v1 = store.get_fingerprints("doc-sync")
    assert len(fps_v1) > 0

    # Reset mock call counts
    mock_gw.embed.reset_mock()

    # Doc v2: section 0 is unchanged, section 1 is removed, section 2 is added
    doc_v2 = RAGDocumentInput(
        doc_id="doc-sync",
        sections=[
            RAGSectionInput(section_seq=0, text="Keep me unchanged sentence. Another sentence.", char_start=0),
            RAGSectionInput(section_seq=2, text="Brand new section sentence. Another sentence.", char_start=200),
        ],
    )
    progress_v2 = engine.sync_document(doc_v2)

    assert progress_v2.passages_skipped_unchanged > 0
    assert progress_v2.passages_upserted > 0

    fps_v2 = store.get_fingerprints("doc-sync")
    # Vanished section 1 is gone
    assert not any("s1" in nid for nid in fps_v2.keys())
    # Section 0 nodes preserved
    for nid in fps_v1.keys():
        if "s0" in nid:
            assert nid in fps_v2


def test_embedding_space_mismatch():
    """AC-RAG-07: Store embedding space mismatch fails validate() and raises EmbeddingSignatureMismatch on query."""
    bindings = ModelBindings(
        generation=["gemini-3.5-flash-lite"],
        embedding="gemini-embedding-2",
    )
    profile = long_form_reader(models=bindings)
    # Store with different model
    mismatched_store = InMemoryRAGStore(
        embedding_space=EmbeddingSpace(
            model="text-embedding-004",  # Mismatch
            dims=768,
        )
    )
    engine = RAGEngine(profile=profile, store=mismatched_store)

    # validate() raises ProfileConfigError
    with pytest.raises(ProfileConfigError):
        engine.validate()

    # retrieve() raises EmbeddingSignatureMismatch
    with pytest.raises(EmbeddingSignatureMismatch):
        engine.retrieve(RAGQueryRequest(query="test query"))


def test_ingestion_budget_guard():
    """AC-RAG-65: BudgetGuard halts during ingestion: earlier sections stored, current section aborted, error propagates."""
    bindings = ModelBindings(
        generation=["gemini-3.5-flash-lite"],
        embedding="gemini-embedding-2",
    )
    profile = long_form_reader(models=bindings)
    store = InMemoryRAGStore(
        embedding_space=EmbeddingSpace(model="gemini-embedding-2", dims=768)
    )
    mock_gw = MagicMock()
    mock_gw.embed.return_value = MagicMock(
        vectors=[[0.05] * 768 for _ in range(10)],
        usage=MagicMock(input_tokens=10, cost_usd=0.001, cached_tokens=0, price_table_version="v2026-11-08", estimated=False),
    )

    budget_calls = 0

    class MockBudgetGuard:
        def before_call(self, ctx):
            nonlocal budget_calls
            budget_calls += 1
            if budget_calls == 3:  # Raise on third section
                raise BudgetExceeded("Daily dollar limit exceeded", limit_usd=5.0, spent_usd=5.01)
            return "proceed"

    engine = RAGEngine(
        profile=profile,
        store=store,
        gateway=mock_gw,
        budget_guard=MockBudgetGuard(),
    )

    doc = RAGDocumentInput(
        doc_id="doc-budget",
        sections=[
            RAGSectionInput(section_seq=0, text="Section 0 text sentence."),
            RAGSectionInput(section_seq=1, text="Section 1 text sentence."),
            RAGSectionInput(section_seq=2, text="Section 2 text sentence."),
            RAGSectionInput(section_seq=3, text="Section 3 text sentence."),
        ],
    )

    with pytest.raises(BudgetExceeded):
        engine.ingest_document(doc)

    # Check sections 0 and 1 are stored, section 2 was aborted
    fps = store.get_fingerprints("doc-budget")
    assert any("s0" in nid for nid in fps.keys())
    assert any("s1" in nid for nid in fps.keys())
    assert not any("s2" in nid for nid in fps.keys())


def test_prepare_nodes_parity():
    """AC-RAG-71: prepare_nodes computes vector, fingerprint, tokens without store write; failed returns []."""
    engine, store, mock_gw = _make_engine_and_store()

    nodes = [
        RAGNode(
            node_id="prep-1",
            doc_id="d1",
            text="Text for node one.",
            granularity="passage",
            span=SourceSpan(char_start=0, char_end=18),
        ),
        RAGNode(
            node_id="prep-2",
            doc_id="d1",
            text="Text for node two (failed).",
            granularity="passage",
            span=SourceSpan(char_start=19, char_end=45),
        ),
    ]

    # Mock embed: first succeeds, second returns zero vector
    mock_gw.embed.side_effect = None
    mock_gw.embed.return_value = MagicMock(
        vectors=[[0.05] * 768, [0.0] * 768],
        usage=MagicMock(input_tokens=20, cost_usd=0.0001, cached_tokens=0, price_table_version="v2026-11-08", estimated=False),
    )

    prepared = engine.prepare_nodes(nodes)
    assert len(prepared) == 2

    # Node 1 succeeded
    assert len(prepared[0].vector) == 768
    assert prepared[0].fingerprint.content_hash is not None
    assert prepared[0].fingerprint.embedding_signature is not None
    assert prepared[0].node.token_estimate is not None

    # Node 2 failed: vector is [] (never zero vector, AC-RAG-71)
    assert prepared[1].vector == []

    # Store was not written
    assert len(store.get_fingerprints("d1")) == 0
