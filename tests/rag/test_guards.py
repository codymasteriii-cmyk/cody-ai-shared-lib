"""Tests for RAG guards: SequentialBoundaryGuard, TemporalGuard, TenantGuard, and ScopeGuard.

Covers:
  - AC-RAG-23: Canary token placed just after boundary never appears.
  - AC-RAG-24: Straddling passage cut/flagged; summary with char_end > B dropped.
  - AC-RAG-25: max_char_offset_by_doc applies separate boundary per document.
  - AC-RAG-73: Sentence boundary snapping with English, Chinese, quotes, and 3.14 avoidance.
"""
from __future__ import annotations

import pytest

from cody_ai_shared_lib.rag.guards import (
    ScopeGuard,
    SequentialBoundaryGuard,
    TemporalGuard,
    TenantGuard,
)
from cody_ai_shared_lib.rag.models import (
    DefaultNodeMeta,
    RAGCitation,
    RAGNode,
    SourceSpan,
)
from cody_ai_shared_lib.rag.text.analyzer import CjkAwareAnalyzer


def test_sequential_boundary_drops_post_boundary_nodes():
    guard = SequentialBoundaryGuard(max_char_offset=100)
    # Node entirely after boundary B=100
    n = RAGNode(
        node_id="n_after",
        text="Content after boundary",
        span=SourceSpan(doc_id="doc1", char_start=100, char_end=150),
        metadata=DefaultNodeMeta(),
    )
    assert guard.verify(n) is None


def test_sequential_boundary_drops_summary_past_boundary_ac_rag_24():
    guard = SequentialBoundaryGuard(max_char_offset=100)
    # Summary node whose span.char_end (120) > boundary (100) must be dropped
    sum_node = RAGNode(
        node_id="sum_1",
        text="Summary covering up to 120",
        span=SourceSpan(doc_id="doc1", char_start=0, char_end=120),
        granularity="parent_summary",
        metadata=DefaultNodeMeta(),
    )
    assert guard.verify(sum_node) is None

    # Summary ending at or before boundary is kept
    sum_ok = RAGNode(
        node_id="sum_2",
        text="Summary covering up to 90",
        span=SourceSpan(doc_id="doc1", char_start=0, char_end=90),
        granularity="parent_summary",
        metadata=DefaultNodeMeta(),
    )
    assert guard.verify(sum_ok) is not None


def test_sequential_boundary_cuts_passage_at_sentence_ac_rag_73():
    analyzer = CjkAwareAnalyzer()
    guard = SequentialBoundaryGuard(max_char_offset=50, max_snap_back_chars=300, analyzer=analyzer)

    # Text: "他说：“走吧。”后面的话不能出现CANARY"
    # Chinese sentence with closing quotes: "他说：“走吧。”"
    # char_start = 0, boundary B = 15 (which cuts into "后面的话")
    text = "他说：“走吧。”后面的话不能出现CANARY"
    sents = analyzer.sentences(text)
    first_sent_end = sents[0][1]  # length of "他说：“走吧。”" is 8

    # Set B to 12 (mid-sentence in "后面的话")
    guard = SequentialBoundaryGuard(max_char_offset=12, max_snap_back_chars=300, analyzer=analyzer)
    node = RAGNode(
        node_id="n_cut",
        text=text,
        span=SourceSpan(doc_id="doc1", char_start=0, char_end=len(text)),
        granularity="passage",
        metadata=DefaultNodeMeta(),
    )

    verified = guard.verify(node)
    assert verified is not None
    # Must end at the complete sentence "他说：“走吧。”"
    assert verified.text == "他说：“走吧。”"
    assert verified.span.char_end == first_sent_end
    assert verified.span.char_end <= 12
    assert "truncated_by_guard" in verified.flags

    # Re-applying guard changes nothing (idempotent)
    second_pass = guard.verify(verified)
    assert second_pass.text == verified.text
    assert second_pass.span.char_end == verified.span.char_end


def test_sentence_cut_ignores_3_14_and_snaps_back():
    analyzer = CjkAwareAnalyzer()
    text = "First sentence. Value is 3.14 approximately. CANARY after boundary"
    # Suppose B is at 35 (inside "approximately")
    guard = SequentialBoundaryGuard(max_char_offset=35, max_snap_back_chars=300, analyzer=analyzer)
    node = RAGNode(
        node_id="n_pi",
        text=text,
        span=SourceSpan(doc_id="doc1", char_start=0, char_end=len(text)),
        granularity="passage",
        metadata=DefaultNodeMeta(),
    )

    verified = guard.verify(node)
    assert verified is not None
    # 3.14 was NOT treated as sentence end, so it snapped back to "First sentence."
    assert verified.text == "First sentence."
    assert "CANARY" not in verified.text


def test_canary_token_never_leaks_ac_rag_23():
    canary = "SECRET_CANARY_TOKEN_999"
    text = f"Safe text before boundary. {canary} placed after boundary."
    boundary = len("Safe text before boundary. ")
    guard = SequentialBoundaryGuard(max_char_offset=boundary)

    node = RAGNode(
        node_id="n_canary",
        text=text,
        span=SourceSpan(doc_id="doc1", char_start=0, char_end=len(text)),
        granularity="passage",
        metadata=DefaultNodeMeta(),
    )

    verified = guard.verify(node)
    assert verified is not None
    assert canary not in verified.text


def test_per_document_boundary_ac_rag_25():
    # docA boundary is 50, docB boundary is 150
    guard = SequentialBoundaryGuard(
        max_char_offset=200,
        max_char_offset_by_doc={"docA": 50, "docB": 150},
    )

    nA = RAGNode(
        node_id="nA",
        text="Text for docA",
        span=SourceSpan(doc_id="docA", char_start=60, char_end=80),
        metadata=DefaultNodeMeta(),
    )
    nB = RAGNode(
        node_id="nB",
        text="Text for docB",
        span=SourceSpan(doc_id="docB", char_start=60, char_end=80),
        metadata=DefaultNodeMeta(),
    )

    # nA starts at 60 > 50 -> dropped
    assert guard.verify(nA) is None
    # nB starts at 60 < 150 -> kept
    assert guard.verify(nB) is not None


def test_citation_boundary_verification():
    guard = SequentialBoundaryGuard(max_char_offset=100)
    c_ok = RAGCitation(
        marker="[1]",
        node_id="n1",
        span=SourceSpan(doc_id="doc1", char_start=0, char_end=100),
        label="Ch 1",
    )
    c_bad = RAGCitation(
        marker="[2]",
        node_id="n2",
        span=SourceSpan(doc_id="doc1", char_start=90, char_end=110),
        label="Ch 2",
    )

    assert guard.verify_citation(c_ok) is True
    assert guard.verify_citation(c_bad) is False


def test_tenant_and_scope_guards():
    t_guard = TenantGuard(tenant_or_user_id="user1")
    n_match = RAGNode(node_id="n1", text="t", span=SourceSpan(doc_id="d"), metadata=DefaultNodeMeta(extra={"tenant_id": "user1"}))
    n_mismatch = RAGNode(node_id="n2", text="t", span=SourceSpan(doc_id="d"), metadata=DefaultNodeMeta(extra={"tenant_id": "user2"}))
    assert t_guard.verify(n_match) is not None
    assert t_guard.verify(n_mismatch) is None

    s_guard = ScopeGuard(doc_ids=["doc_allowed"])
    n_in_scope = RAGNode(node_id="n3", text="t", span=SourceSpan(doc_id="doc_allowed"), metadata=DefaultNodeMeta())
    n_out_scope = RAGNode(node_id="n4", text="t", span=SourceSpan(doc_id="doc_forbidden"), metadata=DefaultNodeMeta())
    assert s_guard.verify(n_in_scope) is not None
    assert s_guard.verify(n_out_scope) is None
