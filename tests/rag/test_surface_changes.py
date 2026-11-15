"""OI-6: stream coverage detail, quote only when quoted, canonical set-like profile fields (RAG_SPEC 1.2.10)."""
from __future__ import annotations

from cody_ai_shared_lib.rag.profile import CitationConfig, EngineProfile, ModelBindings
from tests.rag.test_blocker_regressions import FakeLLM, engine, profile, question, real_embedding_shape  # noqa: F401

TEXT = "Alice met Bob. They talked for hours."


def _profile(verify_quotes: bool = False) -> EngineProfile:
    return profile(citations=CitationConfig(verify_quotes=verify_quotes))


def test_citation_quote_is_none_when_the_answer_quoted_nothing():
    response = engine(_profile(), llm=FakeLLM("Alice met Bob [1].")).query(question())
    assert response.citations and all(c.quote is None for c in response.citations)


def test_citation_quote_is_what_the_answer_quoted():
    llm = FakeLLM('The text says "They talked for hours" [1].')
    response = engine(_profile(verify_quotes=True), llm=llm).query(question())
    assert response.citations[0].quote == "They talked for hours"


def test_citations_event_carries_the_full_coverage_and_keeps_the_number():
    events = list(engine(_profile(), llm=FakeLLM("Alice met Bob and they talked for hours [1].")).stream_query(question()))
    data = next(e for e in events if e.event == "citations").data
    detail = data["citation_coverage_detail"]
    assert data["citation_coverage"] == detail["ratio"]
    assert set(detail) == {
        "sentences_total", "sentences_cited", "ratio", "uncited_sentences", "total_citations", "verified_citations",
    }


def test_fingerprint_ignores_the_order_and_repeats_of_set_like_fields():
    a = profile(allowed_link_hosts=["b.example", "a.example"])
    b = profile(allowed_link_hosts=["a.example", "b.example", "a.example"])
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != profile(allowed_link_hosts=["a.example"]).fingerprint()
    assert b.allowed_link_hosts == ["a.example", "b.example", "a.example"]      # the field itself is not rewritten
