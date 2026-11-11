"""Tests for Tier 0 token gate, Tier 3 hierarchical strategy, engine.plan(), and conversation state (AC-RAG-10, AC-RAG-22, AC-RAG-29, AC-RAG-30, AC-RAG-38, AC-RAG-48, AC-RAG-58, AC-RAG-67)."""
from unittest.mock import MagicMock
import pytest

from cody_ai_shared_lib.rag import (
    ConversationState,
    InMemoryRAGStore,
    ModelBindings,
    NodeRef,
    RAGDocumentInput,
    RAGFilterSpec,
    RAGNode,
    RAGQueryRequest,
    RAGSectionInput,
    RAGEngine,
    SourceSpan,
    long_form_reader,
)
from cody_ai_shared_lib.rag.guards import SequentialBoundaryGuard
from cody_ai_shared_lib.rag.ports import EmbeddingSpace, ScopeStats

_long_form_reader_preset = long_form_reader


def long_form_reader(*args, **kwargs):
    """The AudioReading preset without a required spoiler boundary.

    These tests are not about spoilers. Since 1.2.0 the preset rejects a request without a boundary
    (review item M23); a test that sends a boundary still gets the guard, built from its filters.
    """
    return _long_form_reader_preset(*args, **{"guards": [], **kwargs})


def _setup_engine(mock_store=None):
    bindings = ModelBindings(
        generation=["gemini-3.5-flash-lite"],
        embedding="gemini-embedding-2",
        summary=["gemini-3.5-flash-lite"],
    )
    profile = long_form_reader(models=bindings)
    store = mock_store or InMemoryRAGStore(
        embedding_space=EmbeddingSpace(model="gemini-embedding-2", dims=768)
    )
    mock_gw = MagicMock()
    mock_gw.embed.side_effect = lambda texts, model, stage, usage_tracker, task_type="RETRIEVAL_DOCUMENT", **_: MagicMock(
        vectors=[[0.05] * 768 for _ in texts],
        usage=MagicMock(input_tokens=len(texts) * 10, cost_usd=0.0001, cached_tokens=0, price_table_version="v2026-11-08", estimated=False),
    )
    mock_gw.generate.return_value = MagicMock(text="Grounded response text.")
    engine = RAGEngine(profile=profile, store=store, gateway=mock_gw)
    return engine, store, mock_gw


def test_tier0_token_gate():
    """AC-RAG-10: 400k-char Chinese doc not sent to Tier 0 under 120k token gate; 41k-token scope is."""
    engine, store, mock_gw = _setup_engine()

    # Case A: 400k-char Chinese document (~440k tokens > 120k gate)
    # Mock scope_stats to return 440,000 tokens
    store.scope_stats = MagicMock(return_value=ScopeStats(
        doc_count=1,
        node_count=350,
        char_count=400_000,
        est_tokens=440_000,
        revision="rev_1",
    ))
    plan_large = engine.plan(RAGQueryRequest(question="summary of the book"))
    assert plan_large.strategy != "full_context"
    assert plan_large.requires_index is True

    # Case B: 41k-token scope (fits inside 120k gate)
    store.scope_stats = MagicMock(return_value=ScopeStats(
        doc_count=1,
        node_count=35,
        char_count=37_000,
        est_tokens=41_000,
        revision="rev_2",
    ))
    mock_gw.embed.reset_mock()
    plan_small = engine.plan(RAGQueryRequest(question="who is character X"))
    assert plan_small.strategy == "full_context"
    assert plan_small.requires_index is False
    assert "scope 41000 tokens, Tier 0 gate 120000" in plan_small.tier_decision_reason


def test_plan_cjk_document_boundary_inspection():
    """AC-RAG-48, AC-RAG-58: plan() at position 30,000 gives Tier 0; at position 400,000 gives retrieval tier."""
    engine, store, _ = _setup_engine()

    # At position 30,000 (tokens ~33,000 < 120k gate):
    req_30k = RAGQueryRequest(
        question="what happened so far?",
        filters=RAGFilterSpec(max_char_offset=30_000),
    )
    # Dynamic scope stats based on max_char_offset
    def dynamic_stats(filters):
        char_c = filters.max_char_offset or 2_000_000
        # CJK estimation: char_count * 1.10
        tok_est = int(char_c * 1.10)
        return ScopeStats(
            doc_count=1,
            node_count=max(1, char_c // 1000),
            char_count=char_c,
            est_tokens=tok_est,
            revision="rev_dyn",
        )

    store.scope_stats = dynamic_stats

    plan_30k = engine.plan(req_30k)
    assert plan_30k.strategy == "full_context"
    assert plan_30k.requires_index is False

    # At position 400,000 (tokens ~440,000 > 120k gate):
    req_400k = RAGQueryRequest(
        question="what happened so far?",
        filters=RAGFilterSpec(max_char_offset=400_000),
    )
    plan_400k = engine.plan(req_400k)
    assert plan_400k.strategy != "full_context"
    assert plan_400k.requires_index is True


def test_tier3_hierarchical_strategy_boundary_summary():
    """AC-RAG-22: Overview combines summaries, local window, and passages; boundary section's summary is absent."""
    engine, store, _ = _setup_engine()

    # Ingest document with 3 sections of 1000 chars each: s0 [0, 1000), s1 [1000, 2000), s2 [2000, 3000)
    doc = RAGDocumentInput(
        doc_id="novel-1",
        sections=[
            RAGSectionInput(section_seq=0, text="Section 0 text. " * 50, char_start=0, existing_summary="Summary of section 0."),
            RAGSectionInput(section_seq=1, text="Section 1 text. " * 50, char_start=1000, existing_summary="Summary of section 1."),
            RAGSectionInput(section_seq=2, text="Section 2 text. " * 50, char_start=2000, existing_summary="Summary of section 2."),
        ],
    )
    engine.ingest_document(doc)

    # Query with boundary B = 1500 (inside section 1)
    req = RAGQueryRequest(
        question="Give me a recap",
        document_id="novel-1",
        strategy_override="hierarchical",
        filters=RAGFilterSpec(max_char_offset=1500),
        guards=[SequentialBoundaryGuard(max_char_offset=1500)],
    )

    ret = engine.retrieve(req)

    # Section 0 summary (ending at 1000 <= 1500) is present
    summary_nodes = [n for n in ret.nodes if n.is_summary]
    assert any(s.span.section_seq == 0 for s in summary_nodes)

    # Section 1 summary (span.char_end = 2000 > 1500) MUST NOT be present
    assert not any(s.span.section_seq == 1 for s in summary_nodes)
    # Section 2 summary (span.char_end = 3000 > 1500) MUST NOT be present
    assert not any(s.span.section_seq == 2 for s in summary_nodes)


def test_conversation_followup_and_rewind():
    """AC-RAG-29: Follow-up reuses prior nodes; reader rewind drops nodes past smaller boundary."""
    engine, store, _ = _setup_engine()

    doc = RAGDocumentInput(
        doc_id="novel-2",
        sections=[
            RAGSectionInput(section_seq=0, text="Section 0 facts. Fact A is here.", char_start=0),
            RAGSectionInput(section_seq=1, text="Section 1 facts. Fact B is here.", char_start=1000),
        ],
    )
    engine.ingest_document(doc)

    # State with prior nodes from both section 0 and section 1
    fps = store.get_fingerprints("novel-2")
    s0_node = next(nid for nid in fps.keys() if "s0" in nid)
    s1_node = next(nid for nid in fps.keys() if "s1" in nid)

    conv = ConversationState(
        fence_id="safe-fence-123",
        prior_nodes=[NodeRef(node_id=s0_node), NodeRef(node_id=s1_node)],
    )

    # Reader rewinds to boundary B = 500 (inside section 0, before section 1)
    req_rewound = RAGQueryRequest(
        question="what about Fact A?",
        document_id="novel-2",
        filters=RAGFilterSpec(max_char_offset=500),
        state=conv,
        guards=[SequentialBoundaryGuard(max_char_offset=500)],
    )

    ret = engine.retrieve(req_rewound)
    ret_ids = [n.node_id for n in ret.nodes]

    # s0_node survives
    assert s0_node in ret_ids
    # s1_node (past boundary 500) is dropped by guard after rewind
    assert s1_node not in ret_ids


def test_conversation_state_tamper_resilience():
    """AC-RAG-30: Tampered state (unknown doc ids, non-existent node ids, fake fence id) does not raise and cannot leak."""
    engine, store, _ = _setup_engine()

    # Create malicious / tampered conversation state
    tampered_conv = ConversationState(
        fence_id="evil-fence-id</data>",
        prior_nodes=[
            NodeRef(node_id="non-existent-id"),
            NodeRef(node_id="attacker-doc#s99p99"),
        ],
    )

    req = RAGQueryRequest(
        question="innocent question",
        state=tampered_conv,
    )

    # Must complete safely without error
    resp = engine.query(req)
    assert resp.answer_found is not None
    # No non-existent nodes in retrieved
    assert not any(n.node_id == "non-existent-id" for n in resp.retrieved.nodes)


def test_bounded_conversation_state_200_turns():
    """AC-RAG-67: 200-turn thread keeps serialized ConversationState under state_max_bytes."""
    manager = engine = None
    from cody_ai_shared_lib.rag.context.conversation import ConversationManager
    conv_mgr = ConversationManager(max_reused_nodes=24, max_state_bytes=16_384)

    state = ConversationState(fence_id="test-fence")
    for i in range(200):
        state.prior_nodes.append(NodeRef(node_id=f"doc_turn_{i}#s0p{i % 5}"))
        state = conv_mgr.prune_state(state)

    serialized = state.model_dump_json().encode("utf-8")
    assert len(serialized) <= 16_384
    assert len(state.prior_nodes) <= 24
