"""Tests for splitters, BatchEmbedder, and SectionSummarizer (AC-RAG-01, AC-RAG-03, AC-RAG-04, AC-RAG-08, AC-RAG-59)."""
from unittest.mock import MagicMock
import pytest

from cody_ai_shared_lib.rag.degradation import DegradationTracker
from cody_ai_shared_lib.rag.ingestion.embedder import BatchEmbedder
from cody_ai_shared_lib.rag.errors import EmbeddingProviderError, RAGStageError
from cody_ai_shared_lib.rag.models import RAGNode, SourceSpan as RAGNodeSpan
from cody_ai_shared_lib.rag.ingestion.summarizer import SectionSummarizer
from cody_ai_shared_lib.rag.ingestion.splitter import PassThroughSplitter, SentenceSplitter
from cody_ai_shared_lib.rag.usage import UsageTracker


def test_sentence_splitter_english_and_cjk_offsets():
    """AC-RAG-01: SentenceSplitter preserves exact character offsets section.text[start:end] == chunk.text."""
    splitter = SentenceSplitter(target_size=100, overlap=20)

    # Mixed English and Chinese text
    raw_text = (
        "This is sentence one. Sentence two is slightly longer! "
        "Here is sentence three with numbers like 3.14 which should not split. "
        "第一句话在这里。第二句话比较长，包含一些中文内容！"
        "第三句话也在这里？这是第四句话……他说：“走吧。”"
    )

    chunks = splitter.split(raw_text)
    assert len(chunks) > 1

    for chunk in chunks:
        # Strict RAG-ING-1 offset check
        assert raw_text[chunk.local_start:chunk.local_end] == chunk.text
        assert len(chunk.text) > 0


def test_passthrough_splitter():
    """AC-RAG-03: PassThroughSplitter yields exactly one node spanning the entire unit."""
    splitter = PassThroughSplitter()
    text = "Full article content that should not be split."
    chunks = splitter.split(text)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].local_start == 0
    assert chunks[0].local_end == len(text)


def test_batch_embedder_zero_vector_guard():
    """AC-RAG-04: Single zero vector node is skipped with DG-08; all-zero raises EmbeddingProviderError."""
    mock_gw = MagicMock()
    embedder = BatchEmbedder(gateway=mock_gw, model="gemini-embedding-2", batch_size=2)
    tracker = UsageTracker("req-test")
    deg_tracker = DegradationTracker()

    nodes = [
        RAGNode(node_id="n1", text="Text 1", granularity="passage", span=RAGNodeSpan(char_start=0, char_end=6, doc_id="d1")),
        RAGNode(node_id="n2", text="Text 2", granularity="passage", span=RAGNodeSpan(char_start=7, char_end=13, doc_id="d1")),
    ]

    # Scenario 1: One normal vector, one zero vector
    emb_res_mock = MagicMock()
    emb_res_mock.vectors = [[0.1, 0.2], [0.0, 0.0]]
    mock_gw.embed.return_value = emb_res_mock

    pairs = embedder.embed_nodes(nodes, tracker, deg_tracker)
    assert len(pairs) == 1
    assert pairs[0][0].node_id == "n1"
    assert pairs[0][1] == [0.1, 0.2]
    # Check DG-08 recorded
    assert any(d.id == "DG-08" for d in deg_tracker.degradations)

    # Scenario 2: All nodes zero in non-empty section -> raises EmbeddingProviderError
    emb_res_all_zero = MagicMock()
    emb_res_all_zero.vectors = [[0.0, 0.0], [0.0, 0.0]]
    mock_gw.embed.return_value = emb_res_all_zero

    with pytest.raises(EmbeddingProviderError):
        embedder.embed_nodes(nodes, tracker, deg_tracker)


def test_summarizer_failure_and_skip():
    """AC-RAG-08: Summarizer failure raises RAGStageError by default; with skip fallback, records DG-14."""
    mock_gw = MagicMock()
    mock_gw.generate.side_effect = RuntimeError("Summarizer LLM timeout")

    # Default strict mode
    summarizer_strict = SectionSummarizer(
        gateway=mock_gw,
        summary_models=["gemini-3.5-flash-lite"],
        fallback_action="raise",
        min_chars_for_summary=50,
    )
    u_tracker = UsageTracker("req-test")
    deg_tracker = DegradationTracker()

    with pytest.raises(RAGStageError):
        summarizer_strict.summarize_section(
            doc_id="d1",
            section_seq=0,
            section_id="s0",
            text="A" * 100,
            char_start=0,
            metadata={},
            existing_summary=None,
            usage_tracker=u_tracker,
            degradation_tracker=deg_tracker,
        )

    # Opt-in skip fallback (DG-14)
    summarizer_skip = SectionSummarizer(
        gateway=mock_gw,
        summary_models=["gemini-3.5-flash-lite"],
        fallback_action="skip",
        min_chars_for_summary=50,
    )
    res = summarizer_skip.summarize_section(
        doc_id="d1",
        section_seq=0,
        section_id="s0",
        text="A" * 100,
        char_start=0,
        metadata={},
        existing_summary=None,
        usage_tracker=u_tracker,
        degradation_tracker=deg_tracker,
    )
    assert res.summary_status == "failed"
    assert res.section_summary is None
    assert any(d.id == "DG-14" for d in deg_tracker.degradations)


def test_summarizer_units_generation():
    """AC-RAG-59: Section > summary_unit_chars generates unit summaries with level=0 and parent_kind='section_unit'."""
    mock_gw = MagicMock()
    gen_res = MagicMock()
    gen_res.text = "Unit summary text"
    mock_gw.generate.return_value = gen_res

    summarizer = SectionSummarizer(
        gateway=mock_gw,
        summary_models=["gemini-3.5-flash-lite"],
        summary_unit_chars=100,  # small for testing
        min_chars_for_summary=50,
    )
    u_tracker = UsageTracker("req-test")
    deg_tracker = DegradationTracker()

    # Create text with several sentences totaling ~300 chars
    long_text = "Sentence one is here. " * 15

    res = summarizer.summarize_section(
        doc_id="d1",
        section_seq=0,
        section_id="s0",
        text=long_text,
        char_start=0,
        metadata={},
        existing_summary=None,
        usage_tracker=u_tracker,
        degradation_tracker=deg_tracker,
    )

    assert res.summary_status == "generated"
    assert res.section_summary is not None
    assert res.section_summary.level == 1
    assert res.section_summary.parent_kind == "section"

    assert len(res.unit_summaries) > 1
    for unit in res.unit_summaries:
        assert unit.level == 0
        assert unit.parent_kind == "section_unit"
        assert unit.parent_id == "s0"
        assert unit.text == "Unit summary text"
