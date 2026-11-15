"""Release review fixes (RAG_SPEC 1.2.18): caller rule grouping, typed store errors in query_batched, Jev fast path."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cody_ai_shared_lib.rag.degradation import DegradationTracker
from cody_ai_shared_lib.rag.errors import RetrievalError
from cody_ai_shared_lib.rag.intent.router import HybridIntentRouter, RouterContext
from cody_ai_shared_lib.rag.models import ChatTurn, ConversationState, QueryIntent, RAGQueryRequest, TextRule
from cody_ai_shared_lib.rag.profile import RouterConfig, TextRulesConfig
from cody_ai_shared_lib.rag.text.rules import derive_rules, resolve_group
from cody_ai_shared_lib.rag.usage import UsageTracker
from tests.rag.test_blocker_regressions import profile, real_embedding_shape  # noqa: F401  (autouse)
from tests.rag.test_lists import ListLLM, _batched, _docs, _intent, _store

# ── 1. a caller's grouping beats the router's guess ────────────────────────


def test_caller_any_group_survives_the_router_finding_the_same_names():
    intent = QueryIntent(search_query="Apple and Microsoft news", entity_terms=["Apple", "Microsoft"], entity_mode="require")
    caller = [
        TextRule(text="Apple", kind="must_mention", group="any", origin="caller"),
        TextRule(text="Microsoft", kind="must_mention", group="any", origin="caller"),
    ]
    rules = derive_rules(intent, caller, TextRulesConfig())
    assert [(r.text, r.group, r.strength, r.origin) for r in rules] == [
        ("Apple", "any", "filter", "caller"),
        ("Microsoft", "any", "filter", "caller"),
    ]
    narrow, wide = resolve_group(rules, intent.search_query)
    assert [r.group for r in narrow] == ["any", "any"] and wide is None


def test_router_rule_does_not_override_an_app_rule_in_either_order():
    from cody_ai_shared_lib.rag.models import merge_text_rules

    app = TextRule(text="BM25", group="any", origin="app")
    router = TextRule(text="BM25", group="auto", origin="router", strength="prefer", droppable=True)
    for merged in (merge_text_rules([app], [router]), merge_text_rules([router], [app])):
        assert [(r.group, r.origin, r.strength, r.droppable) for r in merged] == [("any", "app", "filter", False)]


# ── 2. a store failure while query_batched reads is a typed error ──────────


def test_a_store_failure_during_a_batched_read_is_a_retrieval_error():
    store = _store(_docs(2))

    def broken(filters):
        raise RuntimeError("connection reset")

    store.fetch_full_scope = broken
    eng = _batched(store, ListLLM())
    with pytest.raises(RetrievalError) as caught:
        eng.query_batched(RAGQueryRequest(question="list every Seedance"), _intent(search_query="Seedance"), doc_ids=["d1"])
    assert isinstance(caught.value.cause, RuntimeError)


# ── 3. the Jev fast path decides only what it can describe ─────────────────


def _router_and_context(**profile_changes):
    jev = MagicMock()
    jev.route.return_value = QueryIntent(search_query="q", router_source="jev")
    router = HybridIntentRouter(jev_gate=jev)
    ctx = RouterContext(
        usage_tracker=UsageTracker("req"), degradation_tracker=DegradationTracker(),
        profile=profile(router=RouterConfig(enable_jev=True), **profile_changes),
    )
    return router, jev, ctx


def test_jev_decides_a_standalone_question():
    router, jev, ctx = _router_and_context()
    intent = router.route(RAGQueryRequest(question="What color was the car?"), None, ctx)
    assert jev.route.called and intent.router_source == "jev"


@pytest.mark.parametrize(
    "request_, state",
    [
        (RAGQueryRequest(question="what did he do next?", history=[ChatTurn(role="user", content="Tell me about Li Wei")]), None),
        (RAGQueryRequest(question="what did he do next?"), ConversationState(turn_count=2)),
        (RAGQueryRequest(question="list every model mentioned this year"), None),       # a heuristic list cue
        (RAGQueryRequest(question='what does "Model-X7" mean?'), None),                 # a heuristic exact term
    ],
    ids=["history", "state_with_turns", "list_cue", "exact_term"],
)
def test_jev_is_skipped_when_its_intent_would_lose_something(request_, state):
    router, jev, ctx = _router_and_context()
    intent = router.route(request_, state, ctx)
    assert not jev.route.called and intent.router_source != "jev"


def test_jev_is_skipped_when_the_profile_filters_on_names():
    router, jev, ctx = _router_and_context(text_rules=TextRulesConfig(names="filter"))
    router.route(RAGQueryRequest(question="What did Bill Gates do in 1975?"), None, ctx)
    assert not jev.route.called


def test_an_empty_first_turn_state_still_lets_jev_decide():
    router, jev, ctx = _router_and_context()
    router.route(RAGQueryRequest(question="What color was the car?"), ConversationState(), ctx)
    assert jev.route.called


# ── 4. the merge only groups: its wording is shown only if it is an item's own ─


def test_merged_text_that_is_not_an_item_falls_back_to_the_first_item():
    from cody_ai_shared_lib.rag.batching import group_text

    assert group_text("Seedance 1.5 (the best)", ["Seedance 1.5", "Seedance-1.5"]) == "Seedance 1.5"
    assert group_text("seedance-1.5", ["Seedance 1.5", "Seedance-1.5"]) == "seedance-1.5"


# ── 5. a clarifying question needs two options ─────────────────────────────


def test_max_options_below_two_is_a_profile_error():
    from pydantic import ValidationError

    from cody_ai_shared_lib.rag.profile import ClarificationConfig

    with pytest.raises(ValidationError):
        ClarificationConfig(max_options=1)


# ── 6. a failed clarification lookup degrades instead of failing the answer ─


def test_a_failed_clarification_lookup_is_dg16_and_the_question_is_answered(monkeypatch):
    import cody_ai_shared_lib.rag.engine as engine_module
    from tests.rag.test_clarification import HE, MEN, CountingLLM, _engine, _request
    from tests.rag.test_clarification import _store as _book

    def broken(**kw):
        raise RetrievalError("Store operation 'fetch_nodes' failed: RuntimeError", stage="store")

    monkeypatch.setattr(engine_module, "find_clarification", broken)
    llm = CountingLLM()
    response = _engine(_book(MEN), llm).query(_request(clarify="ask"), HE)
    assert response.clarification is None and len(llm.prompts) == 1
    assert any(d.id == "DG-16" for d in response.degradations)


def test_an_app_candidate_key_error_still_propagates():
    from tests.rag.test_clarification import MEN, CountingLLM
    from tests.rag.test_clarification import _store as _book
    from tests.rag.test_blocker_regressions import engine as make_engine

    def bad_key(node):
        raise ValueError("app bug")

    eng = make_engine(profile(), store=_book(MEN), llm=CountingLLM(), candidate_key=bad_key)
    with pytest.raises(ValueError):
        eng.query(RAGQueryRequest(question="what did Bob do"), QueryIntent(search_query="what did Bob do"))


# ── 7. low items ───────────────────────────────────────────────────────────


def test_the_llm_router_cannot_name_its_own_source():
    from cody_ai_shared_lib.rag.intent import llm_rewriter

    source = open(llm_rewriter.__file__, encoding="utf-8").read()
    assert 'parsed.pop("router_source", None)' in source


def test_the_clarifying_question_is_sanitized():
    from tests.rag.test_clarification import HE, CountingLLM, _engine, _request
    from tests.rag.test_clarification import _store as _book

    names = ("[Bob](https://evil.example)", "Carl")
    store = _book(["[Bob](https://evil.example) went to the market.", "Carl went to the harbor."])
    response = _engine(store, CountingLLM()).query(_request(clarify="ask", names=names), HE)
    assert response.clarification is not None and response.clarification.asked
    assert "Bob" in response.answer_final and "https://evil.example" not in response.answer_final
