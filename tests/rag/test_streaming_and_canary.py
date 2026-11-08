"""Tests for streaming (stream_query, RAGStream, SSE, usage on cancel, flag buffer), canary testing, and reference SQL (AC-RAG-23, AC-RAG-45, AC-RAG-54, AC-RAG-67, AC-RAG-70, AC-RAG-72, AC-RAG-73)."""
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from cody_ai_shared_lib.llm import LLMStreamChunk, LLMUsage
from cody_ai_shared_lib.rag import (
    InMemoryRAGStore,
    ModelBindings,
    RAGDocumentInput,
    RAGFilterSpec,
    RAGNode,
    RAGQueryRequest,
    RAGSectionInput,
    RAGStream,
    RAGStreamEvent,
    RAGEngine,
    SourceSpan,
    format_event,
    long_form_reader,
)
from cody_ai_shared_lib.rag.generation.flag_buffer import FlagStreamBuffer
from cody_ai_shared_lib.rag.guards import SequentialBoundaryGuard
from cody_ai_shared_lib.rag.ports import EmbeddingSpace
from cody_ai_shared_lib.rag.testing.canary import (
    assert_canary_not_leaked,
    assert_no_text_past_boundary,
)


def _setup_streaming_engine():
    bindings = ModelBindings(
        generation=["gemini-3.5-flash-lite"],
        embedding="gemini-embedding-2",
        summary=["gemini-3.5-flash-lite"],
    )
    profile = long_form_reader(models=bindings)
    store = InMemoryRAGStore(
        embedding_space=EmbeddingSpace(model="gemini-embedding-2", dims=768)
    )
    mock_gw = MagicMock()
    mock_gw.embed.side_effect = lambda texts, model, stage, usage_tracker, task_type="RETRIEVAL_DOCUMENT": MagicMock(
        vectors=[[0.05] * 768 for _ in texts],
        usage=MagicMock(input_tokens=len(texts) * 10, cost_usd=0.0001, cached_tokens=0, price_table_version="v2026-11-08", estimated=False),
    )
    mock_gw.generate.return_value = MagicMock(text="Grounded answer [1].")

    # Mock stream to yield chunks with delta and final usage
    def _mock_stream(*args, **kwargs):
        yield LLMStreamChunk(delta="Here is ", usage=None, finish_reason=None)
        yield LLMStreamChunk(delta="the answer [1].", usage=LLMUsage(input_tokens=120, output_tokens=30, cost_usd=0.00015, price_table_version="v2026-11-08"), finish_reason="stop")

    mock_gw.stream.side_effect = _mock_stream
    mock_gw.llm_client = MagicMock()
    mock_gw.llm_client.stream.side_effect = _mock_stream

    engine = RAGEngine(profile=profile, store=store, gateway=mock_gw)
    return engine, store, mock_gw


def test_stream_query_event_order_and_payloads():
    """AC-RAG-45: stream_query yields intent -> status -> retrieval -> delta* -> citations -> done."""
    engine, store, mock_gw = _setup_streaming_engine()

    doc = RAGDocumentInput(
        doc_id="doc-stream-1",
        sections=[
            RAGSectionInput(section_seq=0, text="Knowledge text for streaming test.", char_start=0),
        ],
    )
    engine.ingest_document(doc)

    req = RAGQueryRequest(query="What is the text?", document_id="doc-stream-1")
    stream = engine.stream_query(req)

    events: list[RAGStreamEvent] = list(stream)

    event_names = [e.event for e in events]
    assert event_names[0] == "intent"
    assert "retrieval" in event_names
    assert "delta" in event_names
    assert "citations" in event_names
    assert event_names[-1] == "done"

    # Verify done payload
    done_data = events[-1].data
    assert "answer_final" in done_data
    assert "citations" in done_data
    assert "usage" in done_data
    assert "usage_total" in done_data
    assert done_data["grounded"] is True


def test_stream_cancel_records_usage():
    """AC-RAG-70, RAG-OPS-2: Cancelled stream records spend through RAGStream.usage_total."""
    engine, store, mock_gw = _setup_streaming_engine()

    doc = RAGDocumentInput(
        doc_id="doc-cancel-1",
        sections=[RAGSectionInput(section_seq=0, text="Some text.", char_start=0)],
    )
    engine.ingest_document(doc)

    def _infinite_stream(*args, **kwargs):
        yield LLMStreamChunk(delta="Chunk 1 ", usage=None, finish_reason=None)
        yield LLMStreamChunk(delta="Chunk 2 ", usage=None, finish_reason=None)
        yield LLMStreamChunk(delta="Chunk 3 ", usage=None, finish_reason=None)

    mock_gw.stream.side_effect = _infinite_stream
    mock_gw.llm_client.stream.side_effect = _infinite_stream

    req = RAGQueryRequest(query="Query", document_id="doc-cancel-1")
    stream = engine.stream_query(req)

    # Consume first events including two deltas
    consumed = []
    for ev in stream:
        consumed.append(ev)
        if len([e for e in consumed if e.event == "delta"]) >= 2:
            break

    # Consumer closes stream (client disconnect)
    stream.close()

    total = stream.usage_total
    assert total is not None
    assert total.input_tokens > 0
    assert total.output_tokens > 0
    assert total.cost_usd is not None


def test_flag_detection_streaming():
    """AC-RAG-72: [[answer_found=...]] flag detection and immediate release on mismatch."""
    # Case 1: FlagStreamBuffer with leading whitespace and valid flag
    buf = FlagStreamBuffer()
    deltas = buf.feed("\n[[answer_found=partial]]\nText starts here")
    assert buf.answer_found == "partial"
    assert "".join(deltas) == "Text starts here"

    # Case 2: Immediate release on mismatch at first char ('Based on...')
    buf2 = FlagStreamBuffer()
    deltas2 = buf2.feed("Based on your reading")
    assert buf2.answer_found is None
    assert "".join(deltas2) == "Based on your reading"

    # Case 3: Immediate release on second char ('[Source]')
    buf3 = FlagStreamBuffer()
    deltas3 = buf3.feed("[Source](url)")
    assert buf3.answer_found is None
    assert "".join(deltas3) == "[Source](url)"

    # Case 4: Invalid flag removed, never shown, logged warning
    buf4 = FlagStreamBuffer()
    deltas4 = buf4.feed("[[answer_found=maybe]]Remaining text")
    assert buf4.answer_found is None
    assert "".join(deltas4) == "Remaining text"

    # Case 5: Cap at 32 chars without closing ]]
    buf5 = FlagStreamBuffer()
    deltas5 = buf5.feed("[[answer_found=something_without_closing_bracket")
    assert buf5.answer_found is None
    assert len("".join(deltas5)) >= 32

    # Case 6: Stream ends while held
    buf6 = FlagStreamBuffer()
    deltas6 = buf6.feed("[[answ")
    deltas6 += buf6.finalize()
    assert "".join(deltas6) == "[[answ"
    assert buf6.answer_found is None


def test_sse_helper_format_event():
    """AC-RAG-54: sse.format_event produces exact SSE string event: <name>\ndata: <json>\n\n."""
    s1 = format_event("delta", {"text": "hello world"})
    assert s1 == 'event: delta\ndata: {"text": "hello world"}\n\n'

    s2 = format_event(RAGStreamEvent(event="status", data={"stage": "retrieving"}))
    assert s2 == 'event: status\ndata: {"stage": "retrieving"}\n\n'


def test_canary_sequential_boundary():
    """AC-RAG-23: Canary token placed past boundary appears nowhere in prompt, nodes, answer, citations."""
    engine, store, mock_gw = _setup_streaming_engine()

    canary = "CANARY_SPOILER_SECRET_999"
    # Text before B (0..1000) and text with canary past B (1000..2000)
    doc = RAGDocumentInput(
        doc_id="novel-canary",
        sections=[
            RAGSectionInput(section_seq=0, text="Visible text before boundary. Character was happy.", char_start=0),
            RAGSectionInput(section_seq=1, text=f"Secret future events. The villain was {canary} and died.", char_start=1000),
        ],
    )
    engine.ingest_document(doc)

    boundary = 800
    req = RAGQueryRequest(
        query="What happened?",
        document_id="novel-canary",
        filters=RAGFilterSpec(max_char_offset=boundary),
        guards=[SequentialBoundaryGuard(max_char_offset=boundary)],
    )

    ret = engine.retrieve(req)
    assert_no_text_past_boundary(ret.nodes, boundary=boundary, canary_token=canary)

    resp = engine.query(req)
    assert_canary_not_leaked(resp, canary_token=canary)


def test_reference_sql_templates():
    """AC-RAG-67: Reference SQL templates exist in specs/sql/ and contain required clauses (§4.7)."""
    sql_dir = Path(__file__).resolve().parents[2] / "specs" / "sql"
    assert sql_dir.exists(), f"Directory {sql_dir} must exist"

    expected_files = [
        "hybrid_search.sql",
        "vector_search.sql",
        "sequential_boundary.sql",
        "fingerprints.sql",
        "upsert_section.sql",
        "indexes.sql",
        "cjk_fts.sql",
    ]

    for filename in expected_files:
        fpath = sql_dir / filename
        assert fpath.exists(), f"Missing reference SQL template: {filename}"
        content = fpath.read_text(encoding="utf-8")
        assert "StoreCapabilities" in content or "CREATE INDEX" in content, f"Missing StoreCapabilities in {filename}"
        if filename.endswith("_search.sql") or filename in ("upsert_section.sql", "sequential_boundary.sql", "fingerprints.sql"):
            assert "REVOKE EXECUTE" in content, f"Missing REVOKE EXECUTE in {filename}"
            assert "active_predicate" in content or "is_active" in content, f"Missing active-row predicate in {filename}"
