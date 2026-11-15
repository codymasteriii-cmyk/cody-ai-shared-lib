"""Text rules (RAG_SPEC 7.9, 1.2.11): the rule record, delivery to every store, the engine's checked store call."""
from __future__ import annotations

import pytest

from cody_ai_shared_lib.rag import models as models_module
from cody_ai_shared_lib.rag.engine import RAGEngine
from cody_ai_shared_lib.rag.gateway import ModelGateway
from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore
from cody_ai_shared_lib.rag.models import QueryIntent, RAGFilterSpec, RAGNode, RAGQueryRequest, SourceSpan
from cody_ai_shared_lib.rag.ports import StoreCapabilities
from cody_ai_shared_lib.rag.profile import EngineProfile, ModelBindings, RetrievalConfig
from tests.rag.test_several_names import DIM, _LLM, _embeddings  # noqa: F401  (autouse embedding stub)


def rule(**kw):
    """A TextRule, imported late so a missing class fails the test, not the whole module."""
    return models_module.TextRule(**kw)


class NewsLikeStore(InMemoryRAGStore):
    """No analyzer tokens, no SQL rule support: the engine has to do the work. Records every ranked call."""

    def __init__(self) -> None:
        caps = StoreCapabilities(
            supports_vector_search=True,
            supports_keyword_fts=False,
            supports_full_scope_fetch=True,
            supports_parent_summaries=True,
            uses_analyzer_tokens=False,
            max_top_k=200,
        )
        super().__init__(capabilities=caps)
        self.vector_calls: list[tuple[int, RAGFilterSpec]] = []
        self.scope_filters: list[RAGFilterSpec] = []

    def vector_search(self, query_vector, filters, top_k):
        self.vector_calls.append((top_k, filters))
        return super().vector_search(query_vector, filters, top_k)

    def fetch_full_scope(self, filters):
        self.scope_filters.append(filters)
        return super().fetch_full_scope(filters)


def _store(texts: list[str]) -> NewsLikeStore:
    store = NewsLikeStore()
    for i, text in enumerate(texts):
        node = RAGNode(node_id=f"n{i:02d}", text=text, span=SourceSpan(doc_id="d", char_start=i * 1000, char_end=i * 1000 + len(text)))
        store.upsert_section_nodes("d", i, [node], [[0.1] * DIM])
    return store


def _engine(store, **profile_extra) -> RAGEngine:
    profile = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m"], embedding="gemini-embedding-2"),
        retrieval=RetrievalConfig(default_strategy="dense_vector"),
        **profile_extra,
    )
    return RAGEngine(profile, store, gateway=ModelGateway(llm_client=_LLM()))


def _ids(engine, intent, top_k=3, **request_extra) -> list[str]:
    request = RAGQueryRequest(question=intent.search_query or "q", top_k_override=top_k, **request_extra)
    return [n.node_id for n in engine.retrieve(request, intent).nodes]


FILLER = [f"Filler passage number {i} about weather." for i in range(10)]


# ── the script-aware text check ────────────────────────────────────────────


@pytest.mark.parametrize(
    "form, text, expected",
    [
        ("AI", "He said so.", False),                       # not inside a word
        ("AI", "The AI industry grew.", True),
        ("GPT-6", "GPT-60 is rumoured.", False),            # digits on the edge
        ("GPT-4o", "We tested GPT-4o today.", True),        # a hyphenated name is one unit
        ("BM25", "bm25 is a ranking function", True),       # case-insensitive
        ("李小明", "这里有李小明同学", True),                  # CJK: substring
        ("小明", "这里有李小明同学", True),
        ("AI", "我用AI做研究", True),                         # a Latin form next to CJK text is still a word
    ],
)
def test_default_text_check_by_script(form, text, expected):
    from cody_ai_shared_lib.rag.text import rules

    assert rules.mentions(text, rule(text=form, kind="must_mention")) is expected


def test_alternatives_satisfy_a_rule():
    from cody_ai_shared_lib.rag.text import rules

    r = rule(text="BM25", kind="must_mention", alternatives=["Okapi BM25"])
    assert rules.mentions("Okapi BM25 scoring", r)


# ── merge ───────────────────────────────────────────────────────────────────


def test_filters_merge_rules_with_the_stricter_strength():
    base = RAGFilterSpec(text_rules=[rule(text="Acme", kind="must_mention", strength="prefer", droppable=True)])
    merged = base.patch({"text_rules": [rule(text="acme", kind="must_mention", strength="filter", droppable=False)]})
    (only,) = merged.text_rules
    assert only.strength == "filter" and only.droppable is False


# ── delivery: the engine enforces a filter rule for a store that cannot ────


def test_a_filter_rule_is_enforced_by_the_engine_and_the_store_is_over_fetched():
    store = _store(FILLER + ["Acme announced a deal.", "Acme hired a CEO."])
    intent = QueryIntent(search_query="news", rules=[rule(text="Acme", kind="must_mention")])
    ids = _ids(_engine(store), intent, top_k=3)
    assert sorted(ids) == ["n10", "n11"]
    asked = store.vector_calls[0][0]
    assert asked >= 4 * 3, "an unfiltered store is asked for about four times the candidates"


def test_must_not_filter_excludes():
    store = _store(["Apple shipped a phone.", "Meta shipped a headset.", "Google shipped a pixel."])
    intent = QueryIntent(search_query="shipped", rules=[rule(text="Apple", kind="must_not")])
    assert sorted(_ids(_engine(store), intent, top_k=5)) == ["n01", "n02"]


def test_exact_terms_are_a_filter_for_a_store_without_analyzer_tokens():
    store = _store(FILLER + ["GPT-60 is rumoured.", "We tested GPT-4o today."])
    intent = QueryIntent(search_query="tested", exact_terms=["GPT-4o"])
    assert _ids(_engine(store), intent, top_k=3) == ["n11"]


def test_names_are_only_preferred_by_default_so_nothing_is_excluded():
    store = _store(["Alice walked.", "Nothing here.", "Still nothing."])
    intent = QueryIntent(search_query="walked", entity_terms=["Alice"], entity_mode="require")
    assert sorted(_ids(_engine(store), intent, top_k=5)) == ["n00", "n01", "n02"]


def test_a_profile_can_make_names_a_filter_and_several_names_do_not_return_nothing():
    """News's setting. Names that never share a passage must still find both (AudioReading's A1, now for every store)."""
    store = _store(FILLER + ["Alice walked home alone.", "Bob stayed at the mill."])
    intent = QueryIntent(search_query="Alice and Bob", entity_terms=["Alice", "Bob"], entity_mode="require")
    engine = _engine(store, text_rules={"names": "filter"})
    assert sorted(_ids(engine, intent, top_k=3)) == ["n10", "n11"]


def test_widening_counts_rows_after_the_engine_check():
    """The store returns plenty of unfiltered rows; only the checked rows count towards "too few"."""
    store = _store(FILLER + ["Alice and Bob argued.", "Alice walked alone.", "Bob stayed home."])
    intent = QueryIntent(search_query="Alice and Bob", entity_terms=["Alice", "Bob"], entity_mode="require")
    engine = _engine(store, text_rules={"names": "filter"})
    ids = _ids(engine, intent, top_k=3)
    assert ids[0] == "n10", "the passage naming both stays first"
    assert sorted(ids) == ["n10", "n11", "n12"]


# ── rules do not touch summaries, windows or Tier 0 ────────────────────────


def test_rules_never_reach_a_non_ranked_store_call():
    store = _store(["Acme announced a deal.", "Nothing here."])
    engine = _engine(store)
    intent = QueryIntent(search_query="deal", rules=[rule(text="Acme", kind="must_mention")])
    engine.retrieve(RAGQueryRequest(question="deal", top_k_override=3), intent)
    store.fetch_full_scope(RAGFilterSpec())            # direct call: the helper below checks what the engine passes
    engine_scope_calls = [f for f in store.scope_filters if f.text_rules]
    assert engine_scope_calls == []


# ── router none adds no rules ──────────────────────────────────────────────


def test_router_none_adds_no_rules_and_the_store_gets_raw_filters():
    store = _store(["Acme announced a deal.", "Nothing here."])
    engine = _engine(store, router={"name": "none"})
    result = engine.retrieve(RAGQueryRequest(question='"Acme" and "Bob"', top_k_override=3))
    assert [n.node_id for n in result.nodes] == ["n00", "n01"]
    assert all(not filters.text_rules for _, filters in store.vector_calls)
    assert result.applied_filters.text_rules == []


# ── prefer: ranks higher, never excludes, never reaches a store ─────────────


def test_prefer_moves_matching_passages_up_without_removing_any():
    texts = ["Weather today.", "More weather.", "Acme made news.", "Even more weather."]
    store = _store(texts)
    intent = QueryIntent(search_query="news", rules=[rule(text="Acme", kind="must_mention", strength="prefer")])
    engine = _engine(store, text_rules={"prefer_boost": 10.0})
    ids = _ids(engine, intent, top_k=4)
    assert ids[0] == "n02", "the passage that satisfies the rule moves to the front"
    assert sorted(ids) == ["n00", "n01", "n02", "n03"], "nothing is excluded"


def test_a_prefer_rule_is_never_sent_to_the_store():
    store = _store(["Acme made news.", "Weather today."])
    intent = QueryIntent(search_query="news", rules=[rule(text="Acme", kind="must_mention", strength="prefer")])
    _ids(_engine(store), intent, top_k=3)
    assert all(not filters.text_rules for _, filters in store.vector_calls)


def test_the_boost_can_be_turned_off():
    store = _store(["Weather today.", "Acme made news."])
    intent = QueryIntent(search_query="news", rules=[rule(text="Acme", kind="must_mention", strength="prefer")])
    assert _ids(_engine(store, text_rules={"prefer_boost": 1.0}), intent, top_k=3) == ["n00", "n01"]


def test_a_prefer_must_not_demotes_instead_of_excluding():
    store = _store(["Apple shipped a phone.", "Meta shipped a headset.", "Google shipped a pixel."])
    intent = QueryIntent(search_query="shipped", rules=[rule(text="Apple", kind="must_not", strength="prefer")])
    ids = _ids(_engine(store, text_rules={"prefer_boost": 10.0}), intent, top_k=5)
    assert ids[-1] == "n00" and sorted(ids) == ["n00", "n01", "n02"]


# ── a store that applies rules itself: no over-fetch, only filter rules ─────


def test_a_store_that_applies_the_kind_is_not_over_fetched():
    store = InMemoryRAGStore()                                   # declares all three kinds
    for i, text in enumerate(FILLER + ["Acme announced a deal."]):
        node = RAGNode(node_id=f"n{i:02d}", text=text, span=SourceSpan(doc_id="d", char_start=i * 1000, char_end=i * 1000 + len(text)))
        store.upsert_section_nodes("d", i, [node], [[0.1] * DIM])
    seen: list[int] = []
    original = store.vector_search
    store.vector_search = lambda query_vector, filters, top_k: (seen.append(top_k), original(query_vector, filters, top_k))[1]
    intent = QueryIntent(search_query="news", rules=[rule(text="Acme", kind="must_mention")])
    assert _ids(_engine(store), intent, top_k=3) == ["n10"]
    assert seen == [4]                                           # candidate_limit only: the store did the filtering


def test_over_fetch_is_capped_by_the_store_maximum():
    store = _store(FILLER + ["Acme announced a deal."])
    store._capabilities = store.capabilities().model_copy(update={"max_top_k": 6})
    intent = QueryIntent(search_query="news", rules=[rule(text="Acme", kind="must_mention")])
    _ids(_engine(store), intent, top_k=3)
    assert max(asked for asked, _ in store.vector_calls) <= 6


# ── trace ───────────────────────────────────────────────────────────────────


def test_the_trace_says_the_engine_checked_a_rule():
    store = _store(["Acme announced a deal.", "Weather today."])
    intent = QueryIntent(search_query="deal", rules=[rule(text="Acme", kind="must_mention")])
    engine = _engine(store)
    request = RAGQueryRequest(question="deal", top_k_override=3)
    engine.retrieve(request, intent)
    result = engine.query(request, intent)
    assert "must_mention:Acme" in result.trace.text_rules_checked


# ── relaxation drops only droppable rules ───────────────────────────────────


def _relaxing_engine(store, **extra) -> RAGEngine:
    profile = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m"], embedding="gemini-embedding-2"),
        retrieval=RetrievalConfig(default_strategy="dense_vector", relaxation=["drop_required_entities"], min_results_before_relax=3),
        **extra,
    )
    return RAGEngine(profile, store, gateway=ModelGateway(llm_client=_LLM()))


def test_relaxation_drops_a_droppable_rule_and_keeps_a_non_droppable_one():
    store = _store(["Weather one.", "Weather two.", "Weather three.", "Acme news."])
    droppable = rule(text="Acme", kind="must_mention", strength="filter", droppable=True)
    fixed = rule(text="Weather", kind="must_mention", strength="filter", droppable=False)
    intent = QueryIntent(search_query="news", rules=[droppable, fixed])
    result = _relaxing_engine(store).retrieve(RAGQueryRequest(question="news", top_k_override=5), intent)
    assert result.relaxations == ["drop_required_entities"]
    assert sorted(n.node_id for n in result.nodes) == ["n00", "n01", "n02"]      # Weather still required, Acme dropped


def test_relaxation_has_nothing_to_drop_when_no_rule_is_droppable():
    store = _store(["Weather one.", "Acme news."])
    intent = QueryIntent(search_query="news", rules=[rule(text="Acme", kind="must_mention")])
    result = _relaxing_engine(store).retrieve(RAGQueryRequest(question="news", top_k_override=5), intent)
    assert result.relaxations == [] and [n.node_id for n in result.nodes] == ["n01"]


# ── the router's must_not ───────────────────────────────────────────────────


def test_clamp_keeps_rules_the_caller_set():
    from cody_ai_shared_lib.rag.intent.llm_rewriter import clamp_intent

    intent = clamp_intent({"search_query": "q", "rules": [rule(text="Acme", kind="must_mention").model_dump()]})
    assert [r.text for r in intent.rules] == ["Acme"]


def test_router_must_not_is_kept_only_when_the_question_says_it():
    from cody_ai_shared_lib.rag.intent.llm_rewriter import _must_not_rules

    rules = _must_not_rules({"must_not": ["Apple", "Microsoft"]}, "Tell me about phones, not about Apple", None)
    assert [(r.text, r.kind, r.strength, r.droppable, r.origin) for r in rules] == [("Apple", "must_not", "prefer", True, "router")]


def test_a_profile_can_make_the_routers_must_not_a_filter():
    store = _store(["Apple shipped a phone.", "Meta shipped a headset."])
    router_rule = rule(text="Apple", kind="must_not", strength="prefer", droppable=True, origin="router")
    intent = QueryIntent(search_query="shipped", rules=[router_rule])
    assert _ids(_engine(store, text_rules={"must_not": "filter"}), intent, top_k=5) == ["n01"]


# ── the boost needs candidates to promote ───────────────────────────────────


def _acme_at_rank_five() -> NewsLikeStore:
    texts = [f"Weather report {i}." for i in range(4)] + ["Acme made news."] + [f"Weather report {i}." for i in range(5, 12)]
    return _store(texts)


def test_prefer_fetches_a_wider_pool_without_a_reranker_so_the_boost_can_reach_rank_five():
    intent = QueryIntent(search_query="news", rules=[rule(text="Acme", kind="must_mention", strength="prefer")])
    ids = _ids(_engine(_acme_at_rank_five(), text_rules={"prefer_boost": 2.0}), intent, top_k=3)
    assert "n04" in ids, "a passage naming the subject at rank 5 fills one of the top 3"
    assert len(ids) == 3


def test_without_prefer_rules_the_pool_stays_one_more_than_top_k():
    store = _acme_at_rank_five()
    _ids(_engine(store), QueryIntent(search_query="news"), top_k=3)
    assert [asked for asked, _ in store.vector_calls] == [4]


def test_a_turned_off_boost_does_not_widen_the_pool():
    store = _acme_at_rank_five()
    intent = QueryIntent(search_query="news", rules=[rule(text="Acme", kind="must_mention", strength="prefer")])
    _ids(_engine(store, text_rules={"prefer_boost": 1.0}), intent, top_k=3)
    assert [asked for asked, _ in store.vector_calls] == [4]


def test_prefer_boost_defaults_to_two():
    from cody_ai_shared_lib.rag.profile import TextRulesConfig

    assert TextRulesConfig().prefer_boost == 2.0
