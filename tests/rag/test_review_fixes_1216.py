"""Pre-production review fixes (RAG_SPEC 1.2.16): query_batched, completeness "searched", clarification, rules."""
from __future__ import annotations

import dataclasses
import logging

import pytest
from pydantic import ValidationError

from cody_ai_shared_lib.llm import LLMResult, LLMUsage
from cody_ai_shared_lib.rag.models import ConversationState, QueryIntent, RAGFilterSpec, RAGQueryRequest, TextRule
from cody_ai_shared_lib.rag.profile import GenerationConfig, RetrievalConfig, TextRulesConfig
from tests.rag.test_blocker_regressions import FakeLLM, engine, profile, real_embedding_shape  # noqa: F401
from tests.rag.test_lists import ListLLM, _ask, _batched, _docs, _intent, _list_engine, _read, _store


class SpyLLM(ListLLM):
    """ListLLM that also keeps every system prompt."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.systems: list[str] = []

    def generate_with_usage(self, **kw) -> LLMResult:
        self.systems.append(kw.get("system_prompt", ""))
        return super().generate_with_usage(**kw)


def _drop_last_node(eng):
    """Make every assembled prompt leave its last node out, as a tight prompt budget would; record the notes."""
    original = eng.assembler.assemble_prompt
    notes: list[list[str]] = []

    def assemble(request, nodes, fence_id=None, guards=None, service_notes=None, **kw):
        notes.append(list(service_notes or []))
        prompt = original(request, nodes, fence_id, guards, service_notes=service_notes, **kw)
        if len(prompt.included_nodes) > 1:
            prompt = dataclasses.replace(prompt, included_nodes=prompt.included_nodes[:-1])
        elif prompt.included_nodes:                       # a single node: shortened instead of dropped
            only = prompt.included_nodes[0].model_copy(deep=True)
            only.flags.append("truncated_by_budget")
            prompt = dataclasses.replace(prompt, included_nodes=[only])
        return prompt

    eng.assembler.assemble_prompt = assemble
    return notes


# ── query_batched is always strict ─────────────────────────────────────────


def test_query_batched_map_prompts_are_strict_whatever_the_profile_says():
    llm = SpyLLM()
    eng = engine(
        profile(
            retrieval=RetrievalConfig(default_strategy="adaptive", full_context_max_tokens=70),
            generation=GenerationConfig(grounding_mode="labelled_fallback"),
        ),
        store=_store(_docs(2)),
        llm=llm,
    )
    _ask(eng)
    maps = [s for s in llm.systems if "Merge these list items" not in s]
    assert maps and all("GROUNDING RULE (STRICT)" in s for s in maps)


def test_an_empty_batched_list_abstains_because_query_batched_is_always_strict():
    eng = engine(
        profile(
            retrieval=RetrievalConfig(default_strategy="adaptive", full_context_max_tokens=70),
            generation=GenerationConfig(grounding_mode="labelled_fallback"),
        ),
        store=_store({"d1": [False, False]}),
        llm=ListLLM(),
    )
    response = eng.query_batched(RAGQueryRequest(question="list every Seedance"), _intent(search_query="Seedance"), doc_ids=["d1"])
    assert response.answer_found == "no" and response.abstained is True


# ── query_batched keeps the conversation state ─────────────────────────────


def test_query_batched_updates_the_state_like_query():
    state = ConversationState(turn_count=2, clarification_asked=True)
    request = RAGQueryRequest(question="list every Seedance", filters=RAGFilterSpec(min_similarity=0.5), state=state)
    response = _batched(_store(_docs(2)), ListLLM()).query_batched(request, _intent(search_query="Seedance"))
    assert response.state.turn_count == 3
    assert response.state.clarification_asked is False, "the never-ask-twice guard lasts one turn"
    assert response.state.profile_fingerprint
    assert {ref.node_id for ref in response.state.prior_nodes} >= {"d1-1", "d2-1"}
    assert response.trace.state_size_bytes > 0


def test_query_batched_without_a_state_starts_one():
    response = _ask(_batched(_store(_docs(1)), ListLLM()))
    assert response.state.turn_count == 1


# ── query_batched reports and logs like query ──────────────────────────────


def test_query_batched_writes_the_summary_line_and_keeps_the_applied_filters(caplog):
    with caplog.at_level(logging.INFO, logger="shared-rag"):
        response = _ask(_batched(_store(_docs(2)), ListLLM()))
    assert any("strategy=query_batched" in r.getMessage() for r in caplog.records)
    assert response.retrieved.applied_filters.min_similarity == 0.5


def test_text_the_map_prompt_left_out_makes_the_batched_list_partial():
    eng = _batched(_store(_docs(1)), ListLLM())
    _drop_last_node(eng)
    response = _ask(eng)
    assert response.completeness == "partial" and response.answer_found == "partial"
    assert "may be incomplete" in response.answer_final


def test_an_empty_scope_reads_nothing_from_the_store():
    store = _store(_docs(2))
    calls: list[str] = []
    original = store.fetch_full_scope
    store.fetch_full_scope = lambda f: calls.append("fetch") or original(f)
    request = RAGQueryRequest(question="list every Seedance", filters=RAGFilterSpec(doc_ids=["elsewhere"]))
    response = _batched(store, ListLLM()).query_batched(request, _intent(search_query="Seedance"), doc_ids=["d1"])
    assert calls == [] and response.batches.documents_found == 0 and response.answer_found == "no"


# ── completeness: whole_scope, searched, partial ───────────────────────────


def test_a_list_gathered_by_search_and_read_in_full_is_searched_and_stays_yes():
    response = _ask(_batched(_store(_docs(3)), ListLLM()))
    assert response.completeness == "searched" and response.answer_found == "yes"
    assert "may be incomplete" not in response.answer_final


def test_find_then_read_around_every_hit_is_searched():
    result = _read(_list_engine(_store({"docA": [False, True, False]})))
    assert result.strategy_used == "find_then_read" and result.completeness == "searched"


def test_find_then_read_that_falls_back_to_the_bare_hit_for_budget_is_partial():
    store = _store({"docA": [False, True, False]}, summaries=False)
    result = _read(_list_engine(store, local_window_radius_chars=60, list_read_radius_chars=300))
    assert result.strategy_used == "find_then_read" and result.completeness == "partial"


def test_a_plain_top_k_search_stays_partial():
    prof = profile(retrieval=RetrievalConfig(default_strategy="dense_vector"))
    response = engine(prof, llm=FakeLLM("[[answer_found=yes]] A [1].")).query(RAGQueryRequest(question="list all"), _intent())
    assert response.completeness == "partial" and response.answer_found == "partial"


def test_a_searched_list_gets_no_incomplete_note():
    seen: list[str] = []

    class Spy(FakeLLM):
        def generate_with_usage(self, **kw):
            seen.append(str(kw))
            return super().generate_with_usage(**kw)

    eng = _list_engine(_store({"docA": [False, True, False]}))
    eng.gateway.llm_client = Spy("[[answer_found=yes]] ok [1].")
    request = RAGQueryRequest(question="list every Seedance", filters=RAGFilterSpec(min_similarity=0.5))
    response = eng.query(request, _intent(search_query="Seedance"))
    assert response.completeness == "searched" and response.answer_found == "yes"
    assert seen and not any("may be incomplete" in s for s in seen)


def test_a_tier0_list_cut_by_the_prompt_budget_tells_the_model():
    prof = profile(retrieval=RetrievalConfig(default_strategy="adaptive"))
    eng = engine(prof, llm=FakeLLM("[[answer_found=yes]] Alice met Bob [1]."))
    notes = _drop_last_node(eng)
    response = eng.query(RAGQueryRequest(question="list all"), _intent())
    assert response.completeness == "partial" and response.answer_found == "partial"
    assert any("may be incomplete" in note for note in notes[-1]), "the model is told, not only the response"


# ── clarification ──────────────────────────────────────────────────────────


def test_after_a_several_matches_ask_the_next_turn_gets_the_best_guess_note():
    from tests.rag.test_clarification import FAMILY, NAMED, CountingLLM, _hooked
    from tests.rag.test_clarification import _store as family_store

    llm = CountingLLM()
    response = _hooked(family_store(FAMILY), llm).query(
        RAGQueryRequest(question="Li Ming", clarify="ask", state=ConversationState(clarification_asked=True)), NAMED
    )
    assert response.clarification.reason == "several_matches" and response.clarification.asked is False
    assert "most likely" in llm.prompts[0]


def test_a_name_only_in_the_rendered_prompt_text_is_remembered():
    from cody_ai_shared_lib.rag.render import DefaultRenderer
    from tests.rag.test_clarification import CountingLLM
    from tests.rag.test_clarification import _store as plain_store

    class SpeakerRenderer(DefaultRenderer):
        def prompt_text(self, node):
            return f"Speaker: Dana\n{node.text}"

    eng = engine(profile(), store=plain_store(["She went to the market."]), llm=CountingLLM(), renderer=SpeakerRenderer())
    intent = QueryIntent(search_query="Dana", entity_terms=["Dana"])
    response = eng.query(RAGQueryRequest(question="what did Dana do"), intent)
    assert response.state.recent_entities == ["Dana"]


def test_the_referent_lookup_makes_no_store_call_on_an_empty_scope():
    from tests.rag.test_clarification import HE, MEN, CountingLLM, _engine, _request
    from tests.rag.test_clarification import _store as men_store

    store = men_store(MEN)
    state = ConversationState(recent_entities=["Bob", "Carl"])
    state.prior_nodes = []
    from cody_ai_shared_lib.rag.models import NodeRef

    state.prior_nodes = [NodeRef(node_id="n0", turn=1), NodeRef(node_id="n1", turn=1)]
    request = _request(clarify="ask", filters=RAGFilterSpec(doc_ids=["book"])).model_copy(update={"state": state})
    request = request.model_copy(update={"filters": request.filters.patch({"doc_ids": ["other"]})})
    _engine(store, CountingLLM()).query(request, HE)
    assert store.fetched == []


# ── router rules, prefer_boost, CJK word boundaries ────────────────────────


def test_rules_in_the_llm_routers_own_json_are_ignored():
    from cody_ai_shared_lib.rag.degradation import DegradationTracker
    from cody_ai_shared_lib.rag.gateway import ModelGateway
    from cody_ai_shared_lib.rag.intent.llm_rewriter import LLMIntentRouter
    from cody_ai_shared_lib.rag.usage import UsageTracker

    class RouterLLM:
        def generate_with_usage(self, **_):
            text = (
                '{"search_query": "phones", "rules": [{"text": "Apple", "kind": "must_not", "strength": "filter", '
                '"origin": "caller"}], "must_not": ["Apple"]}'
            )
            return LLMResult(text=text, usage=LLMUsage(input_tokens=1, output_tokens=1), finish_reason="stop")

    router = LLMIntentRouter(ModelGateway(llm_client=RouterLLM()), ["m"])
    intent = router.route("Tell me about phones, not about Apple", UsageTracker("r"), DegradationTracker())
    assert [(r.origin, r.strength) for r in intent.rules] == [("router", "prefer")], "only the checked must_not survives"


def test_an_app_supplied_intent_keeps_its_rules():
    from cody_ai_shared_lib.rag.intent.llm_rewriter import clamp_intent

    rule = TextRule(text="Apple", kind="must_not", strength="filter", origin="app")
    assert clamp_intent({"search_query": "q", "rules": [rule.model_dump()]}).rules == [rule]


@pytest.mark.parametrize("bad", [0.0, -1.0, 0.5, 0.99])
def test_prefer_boost_below_one_is_rejected(bad):
    with pytest.raises(ValidationError):
        TextRulesConfig(prefer_boost=bad)


def test_prefer_boost_of_one_turns_it_off_and_above_one_is_allowed():
    assert TextRulesConfig(prefer_boost=1.0).prefer_boost == 1.0
    assert TextRulesConfig(prefer_boost=3.0).prefer_boost == 3.0


def test_a_latin_name_next_to_any_cjk_character_is_a_whole_word():
    from cody_ai_shared_lib.rag.text.rules import mentions

    assert mentions("见到Bob\U00020000了", TextRule(text="Bob"))       # CJK Extension B, as tokens.is_cjk_char counts it
    assert not mentions("Bobby left", TextRule(text="Bob"))
