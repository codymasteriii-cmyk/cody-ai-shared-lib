"""Several required names on the token path (RAG_SPEC 7.9 "Several names", AudioReading's A1)."""
from __future__ import annotations

import re

import pytest

from cody_ai_shared_lib.embedding import EmbeddingResult, EmbeddingUsage
from cody_ai_shared_lib.llm import LLMResult, LLMUsage
from cody_ai_shared_lib.rag import gateway as gateway_module
from cody_ai_shared_lib.rag.engine import RAGEngine
from cody_ai_shared_lib.rag.gateway import ModelGateway
from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore
from cody_ai_shared_lib.rag.models import QueryIntent, RAGNode, RAGQueryRequest, SourceSpan
from cody_ai_shared_lib.rag.profile import EngineProfile, ModelBindings, RetrievalConfig
from cody_ai_shared_lib.rag.text.analyzer import (
    CjkAwareAnalyzer,
    TokenGroup,
    build_tsquery,
    tokenize_query_groups,
)
from cody_ai_shared_lib.rag.text import analyzer as analyzer_module

DIM = 768
ANALYZER = CjkAwareAnalyzer()


class _LLM:
    def generate_with_usage(self, **_: object) -> LLMResult:
        return LLMResult(text="x", usage=LLMUsage(input_tokens=1, output_tokens=1, cost_usd=0.0), finish_reason="stop")


@pytest.fixture(autouse=True)
def _embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(text, task_type="RETRIEVAL_DOCUMENT"):
        vectors = [0.1] * DIM if isinstance(text, str) else [[0.1] * DIM for _ in text]
        return EmbeddingResult(vectors=vectors, usage=EmbeddingUsage(input_tokens=1, cost_usd=0.0))

    monkeypatch.setattr(gateway_module, "generate_embedding_with_usage", fake)


class SpyStore(InMemoryRAGStore):
    """Records every keyword search: its text and token groups."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, list[TokenGroup]]] = []

    def keyword_search(self, query_text, filters, top_k, analyzer_tokens=None):
        self.calls.append((query_text, list(analyzer_tokens or [])))
        return super().keyword_search(query_text, filters, top_k, analyzer_tokens)


def _store(texts: list[str]) -> SpyStore:
    store = SpyStore()
    for i, text in enumerate(texts):
        node = RAGNode(node_id=f"n{i}", text=text, span=SourceSpan(doc_id="d", char_start=i * 1000, char_end=i * 1000 + len(text)))
        store.upsert_section_nodes("d", i, [node], [[0.1] * DIM])
    return store


def _engine(store: SpyStore) -> RAGEngine:
    profile = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m"], embedding="gemini-embedding-2"),
        retrieval=RetrievalConfig(default_strategy="keyword_only"),
    )
    return RAGEngine(profile, store, gateway=ModelGateway(llm_client=_LLM()))


def _retrieve(store: SpyStore, intent: QueryIntent, top_k: int) -> list[str]:
    request = RAGQueryRequest(question=intent.search_query, top_k_override=top_k)
    return [n.node_id for n in _engine(store).retrieve(request, intent).nodes]


def _entity_groups(groups: list[TokenGroup]) -> list[TokenGroup]:
    return [g for g in groups if g.origin == "entity"]


# ── groups and rendering ───────────────────────────────────────────────────


def test_names_missing_from_the_search_text_are_not_required():
    groups = tokenize_query_groups(ANALYZER, "李小明和王五的关系", None, ["李小明", "王五", "赵六"], "require")
    entities = _entity_groups(groups)
    assert [g.mode for g in entities] == ["phrase", "phrase"]            # 赵六 is not in this text
    assert build_tsquery(entities) == "(李小 <-> 小明) & 王五"


def test_no_name_in_the_text_gives_one_any_of_group_with_safe_syntax():
    groups = tokenize_query_groups(ANALYZER, "what happened next", None, ["Sam Altman", "Dario"], "require")
    (group,) = _entity_groups(groups)
    assert group.mode == "any_of"
    rendered = build_tsquery([group])
    assert rendered == "((sam <-> altman) | dario)"
    assert not re.search(r"[&!:*'\"]", rendered)                          # only | <-> and parentheses


def test_single_name_is_unchanged_and_never_widens():
    groups = tokenize_query_groups(ANALYZER, "Sam Altman funding", None, ["Sam Altman"], "require")
    assert build_tsquery(groups) == "(sam | altman | funding) & (sam <-> altman)"
    assert analyzer_module.widen_name_groups(groups) is None


def test_widen_turns_the_required_names_into_one_any_of_group():
    groups = tokenize_query_groups(ANALYZER, "Alice and Bob", ["quarrel"], ["Alice", "Bob"], "require")
    wide = analyzer_module.widen_name_groups(groups)
    assert wide is not None
    assert build_tsquery(wide) == "(alice | and | bob) & quarrel & (alice | bob)"
    assert analyzer_module.widen_name_groups(wide) is None


def test_names_with_one_present_does_not_widen():
    groups = tokenize_query_groups(ANALYZER, "only Alice here", None, ["Alice", "Bob"], "require")
    assert analyzer_module.widen_name_groups(groups) is None


# ── Audio's measured cases, through the engine ─────────────────────────────


def test_names_that_never_share_a_passage_no_longer_return_nothing():
    store = _store(["Alice walked home alone.", "Bob stayed at the mill.", "Nothing relevant here."])
    intent = QueryIntent(search_query="Alice and Bob", entity_terms=["Alice", "Bob"], entity_mode="require")
    assert sorted(_retrieve(store, intent, top_k=5)) == ["n0", "n1"]
    assert len(store.calls) == 2                                          # narrow, then widened


def test_a_relationship_search_keeps_both_names_and_does_not_widen_when_full():
    store = _store(
        ["Alice and Bob argued.", "Alice and Bob made up.", "Alice alone.", "Alice again.", "Bob alone."]
    )
    intent = QueryIntent(search_query="Alice and Bob", entity_terms=["Alice", "Bob"], entity_mode="require")
    assert sorted(_retrieve(store, intent, top_k=2)) == ["n0", "n1"]
    assert len(store.calls) == 1


def test_a_comparison_search_for_one_subject_requires_only_that_subject():
    store = _store(["Alice sailed.", "Alice repaired the boat.", "Bob sailed."])
    intent = QueryIntent(
        search_query="Alice and Bob",
        secondary_queries=["Alice"],
        is_comparison=True,
        entity_terms=["Alice", "Bob"],
        entity_mode="require",
    )
    _retrieve(store, intent, top_k=1)
    by_text = {text: groups for text, groups in store.calls if text == "Alice"}
    names = [g.tokens for g in _entity_groups(by_text["Alice"])]
    assert names == [["alice"]]                                           # Bob is not required in Alice's search


def test_widened_passages_follow_the_narrow_ones_without_duplicates():
    store = _store(["Bob alone one.", "Alice alone one.", "Alice and Bob together.", "Alice alone two."])
    intent = QueryIntent(search_query="Alice and Bob", entity_terms=["Alice", "Bob"], entity_mode="require")
    ids = _retrieve(store, intent, top_k=4)
    assert ids[0] == "n2"                                                 # the passage naming both stays first
    assert sorted(ids) == ["n0", "n1", "n2", "n3"]
    assert len(ids) == len(set(ids))


def test_a_single_name_search_makes_one_call_even_when_thin():
    store = _store(["Alice walked home.", "Nothing here."])
    intent = QueryIntent(search_query="Alice", entity_terms=["Alice"], entity_mode="require")
    assert _retrieve(store, intent, top_k=5) == ["n0"]
    assert len(store.calls) == 1


def test_exact_terms_are_unchanged():
    groups = tokenize_query_groups(ANALYZER, "find it", ["GPT-4o"], ["Alice", "Bob"], "require")
    assert [g.origin for g in groups if g.origin == "exact_term"] == ["exact_term"]
    assert all(g.mode == "phrase" for g in groups if g.origin == "exact_term")


def test_hybrid_path_b_widens_the_keyword_leg_too():
    """A store without a native hybrid call: the engine fuses the legs itself, and the keyword leg widens."""
    store = _store(["Alice walked home alone.", "Bob stayed at the mill.", "Nothing relevant here."])
    store._capabilities = store.capabilities().model_copy(update={"supports_native_hybrid_rpc": False})
    profile = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m"], embedding="gemini-embedding-2"),
        retrieval=RetrievalConfig(default_strategy="hybrid_rrf"),
    )
    engine = RAGEngine(profile, store, gateway=ModelGateway(llm_client=_LLM()))
    intent = QueryIntent(search_query="Alice and Bob", entity_terms=["Alice", "Bob"], entity_mode="require")
    result = engine.retrieve(RAGQueryRequest(question="Alice and Bob", top_k_override=5), intent)
    keyword_hits = {n.node_id for n in result.nodes if n.keyword_rank is not None}
    assert {"n0", "n1"} <= keyword_hits
    assert len(store.calls) == 2
