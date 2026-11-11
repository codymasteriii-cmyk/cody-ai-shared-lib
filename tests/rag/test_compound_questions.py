"""Compound questions: the engine, not the app, splits a question into searches (§7.2, §7.8, AC-RAG-110).

"Who is Jonathan and what does he do in his job?" may need passages from two places. The app passes the
question whole; the intent router returns one extra search per other part in `secondary_queries`, and
the fan-out searches each. `sub_questions` was parsed but never searched, and is removed in 1.2.1.
"""
from __future__ import annotations

import json

from cody_ai_shared_lib.llm import LLMResult, LLMUsage
from cody_ai_shared_lib.rag import (
    DegradationTracker,
    EngineProfile,
    InMemoryRAGStore,
    ModelBindings,
    ModelGateway,
    QueryIntent,
    RAGEngine,
    RAGNode,
    RAGQueryRequest,
    RetrievalConfig,
    SourceSpan,
    clamp_intent,
)
from cody_ai_shared_lib.rag.intent.llm_rewriter import INTENT_SYSTEM_PROMPT

from .test_blocker_regressions import real_embedding_shape  # noqa: F401  (autouse fixture)

DIM = 768
QUESTION = "Who is Jonathan and what does he do in his job?"
WHO = "Jonathan is the eldest son of the miller family in the valley."
JOB = "He works as an accountant at the harbour office every weekday."


class ScriptedLLM:
    """Answers the intent call with a fixed JSON intent and every other call with a plain answer."""

    def __init__(self, intent: dict):
        self.intent = intent
        self.prompts: list[dict] = []

    def generate_with_usage(self, **kwargs) -> LLMResult:
        self.prompts.append(kwargs)
        is_intent = "query analysis assistant" in (kwargs.get("system_prompt") or "")
        text = json.dumps(self.intent) if is_intent else "[[answer_found=yes]]Answer [1]."
        return LLMResult(text=text, usage=LLMUsage(input_tokens=1, output_tokens=1, cost_usd=0.0), finish_reason="stop")

    def stream(self, **kwargs):
        yield from ()


def two_place_store() -> InMemoryRAGStore:
    """The two halves of the answer live in different sections, and only the first names Jonathan."""
    store = InMemoryRAGStore()
    for seq, text in enumerate([WHO, JOB]):
        span = SourceSpan(doc_id="d", section_seq=seq, char_start=seq * 100, char_end=seq * 100 + len(text))
        store.upsert_section_nodes("d", seq, [RAGNode(node_id=f"p{seq}", text=text, span=span)], [[0.1] * DIM])
    return store


def keyword_engine(llm: ScriptedLLM) -> RAGEngine:
    profile = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], intent=["i1"], embedding="gemini-embedding-2"),
        retrieval=RetrievalConfig(default_strategy="keyword_only"),
    )
    return RAGEngine(profile, two_place_store(), gateway=ModelGateway(llm_client=llm))


def retrieved_ids(intent_json: dict) -> list[str]:
    llm = ScriptedLLM(intent_json)
    result = keyword_engine(llm).retrieve(RAGQueryRequest(question=QUESTION))
    return sorted(n.node_id for n in result.nodes)


def test_each_part_of_a_compound_question_is_searched():
    """The router's extra search for the second part reaches the passage that never names Jonathan."""
    ids = retrieved_ids({"search_query": "Jonathan", "secondary_queries": ["accountant job work"]})
    assert ids == ["p0", "p1"]


def test_a_legacy_sub_questions_key_from_the_model_is_still_searched():
    """A model that still answers with the old key is not silently ignored (AC-RAG-110)."""
    ids = retrieved_ids({"search_query": "Jonathan", "sub_questions": ["accountant job work"]})
    assert ids == ["p0", "p1"]


def test_a_single_search_finds_only_the_first_part():
    """The control: without an extra search, the second half of the answer is never retrieved."""
    assert retrieved_ids({"search_query": "Jonathan"}) == ["p0"]


def test_clamp_folds_sub_questions_into_secondary_queries_without_duplicates_and_caps_the_total():
    tracker = DegradationTracker()
    intent = clamp_intent(
        {"search_query": "q", "secondary_queries": ["a", "b", "c"], "sub_questions": ["b", "d", "e"]}, tracker
    )
    assert intent.secondary_queries == ["a", "b", "c", "d"]
    assert "DG-05" in [d.id for d in tracker.degradations]


def test_sub_questions_is_gone_from_the_intent_and_the_router_prompt():
    assert "sub_questions" not in QueryIntent.model_fields
    assert "sub_questions" not in INTENT_SYSTEM_PROMPT
    assert "part" in INTENT_SYSTEM_PROMPT.lower()            # the prompt says when to split a question
    # An app that still builds an intent with the old field gets no error (unknown fields are ignored)
    assert QueryIntent(search_query="q", sub_questions=["x"]).secondary_queries == []


def test_a_supplied_intent_is_searched_as_given():
    """An app that splits the question itself passes secondary_queries; the engine does not split again."""
    llm = ScriptedLLM({"search_query": "unused"})
    intent = QueryIntent(search_query="Jonathan", secondary_queries=["accountant job work"])
    result = keyword_engine(llm).retrieve(RAGQueryRequest(question=QUESTION), intent)
    assert sorted(n.node_id for n in result.nodes) == ["p0", "p1"]
    assert not any("query analysis assistant" in (p.get("system_prompt") or "") for p in llm.prompts)
