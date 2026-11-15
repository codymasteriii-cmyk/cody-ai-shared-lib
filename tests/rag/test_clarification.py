"""Asking the user to clarify (RAG_SPEC section 7.10, 1.2.14, OI-5)."""
from __future__ import annotations

import pytest

from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore
from cody_ai_shared_lib.rag.models import ConversationState, QueryIntent, RAGFilterSpec, RAGNode, RAGQueryRequest, SourceSpan
from cody_ai_shared_lib.rag.profile import RetrievalConfig
from tests.rag.test_blocker_regressions import DIM, FakeLLM, engine, profile, real_embedding_shape  # noqa: F401


class CountingLLM(FakeLLM):
    def __init__(self, text: str = "[[answer_found=yes]] He went to the market."):
        super().__init__(text)
        self.prompts: list[str] = []

    def generate_with_usage(self, **kw):
        self.prompts.append(str(kw.get("system_prompt", "")) + str(kw.get("user_message", "")))
        return super().generate_with_usage(**kw)


class SpyStore(InMemoryRAGStore):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fetched: list[list[str]] = []

    def fetch_nodes(self, node_ids, filters):
        self.fetched.append(list(node_ids))
        return super().fetch_nodes(node_ids, filters)


def _store(texts: list[str], starts: list[int] | None = None) -> SpyStore:
    store = SpyStore()
    for i, text in enumerate(texts):
        start = (starts or [i * 100 for i in range(len(texts))])[i]
        node = RAGNode(node_id=f"n{i}", text=text, span=SourceSpan(doc_id="book", section_seq=i, char_start=start, char_end=start + len(text)))
        store.upsert_section_nodes("book", i, [node], [[0.1] * DIM])
    return store


MEN = ["Bob went to the market on Monday.", "Carl went to the harbor on Tuesday."]
HE = QueryIntent(search_query="what did he do the following week", is_followup=True, unresolved_referent=True)


def _request(question="what did he do the following week", clarify=None, names=("Bob", "Carl"), asked=False, state=True, **kw):
    st = ConversationState(recent_entities=list(names), clarification_asked=asked) if state else None
    return RAGQueryRequest(question=question, clarify=clarify, state=st, **kw)


def _engine(store, llm, **extra):
    return engine(profile(**extra), store=store, llm=llm)


# ── your driving case ───────────────────────────────────────────────────────


def test_an_unresolved_he_between_two_known_men_asks_and_makes_no_generation_call():
    llm = CountingLLM()
    response = _engine(_store(MEN), llm).query(_request(clarify="ask"), HE)
    assert llm.prompts == [], "asking first costs no generation call"
    c = response.clarification
    assert c.asked is True and c.reason == "unresolved_referent"
    assert {o.label for o in c.options} == {"Bob", "Carl"}
    assert "Bob" in response.answer_final and "Carl" in response.answer_final
    assert response.answer == response.answer_final


def test_the_ask_response_cannot_be_mistaken_for_an_answer():
    response = _engine(_store(MEN), CountingLLM()).query(_request(clarify="ask"), HE)
    assert response.answer_found is None, "never 'no': KB's absence-prover reads answer_found"
    assert response.citations == [] and response.sources == []
    assert response.grounded is False and response.abstained is False and response.completeness is None
    assert response.state.clarification_asked is True and response.state.turn_count == 1


# ── nothing changes by default ──────────────────────────────────────────────


def test_default_mode_answers_and_only_marks_with_no_new_prompt_text():
    llm = CountingLLM()
    response = _engine(_store(MEN), llm).query(_request(), HE)
    assert len(llm.prompts) == 1
    assert response.clarification is not None and response.clarification.asked is False
    assert "most likely" not in llm.prompts[0]


def test_retrieve_only_marks():
    result = _engine(_store(MEN), CountingLLM()).retrieve(_request(clarify="ask"), HE)
    assert result.clarification is not None and result.clarification.asked is False


# ── evidence, never suspicion ───────────────────────────────────────────────


def test_the_router_flag_without_two_confirmed_candidates_does_not_ask():
    llm = CountingLLM()
    response = _engine(_store(MEN), llm).query(_request(clarify="ask", names=("Bob", "Zed")), HE)
    assert response.clarification is None and len(llm.prompts) == 1


def test_a_candidate_outside_the_visible_scope_is_never_offered():
    store = _store(MEN, starts=[0, 1000])                              # Carl's passage lies past the reader's position
    request = _request(clarify="ask", filters=RAGFilterSpec(max_char_offset=500))
    response = _engine(store, CountingLLM()).query(request, HE)
    assert response.clarification is None


def test_a_tampered_state_cannot_reveal_a_name_that_was_never_shown():
    response = _engine(_store(MEN), CountingLLM()).query(_request(clarify="ask", names=("Bob", "Dracula")), HE)
    assert response.clarification is None
    shown = _engine(_store(MEN), CountingLLM()).query(_request(clarify="ask", names=("Bob", "Carl", "Dracula")), HE)
    assert {o.label for o in shown.clarification.options} == {"Bob", "Carl"}
    assert "Dracula" not in shown.answer_final


def test_entity_names_in_the_state_are_capped_and_trimmed_on_read():
    from cody_ai_shared_lib.rag.clarification import clean_entities

    cleaned = clean_entities([f"Name{i}" + "x" * 200 for i in range(100)] + ["Bob", "bob", "  Bob  ", 7, ""])
    assert len(cleaned) <= 20 and all(len(n) <= 60 for n in cleaned) and "Bob" not in cleaned[:0]
    assert [n.casefold() for n in clean_entities(["Bob", "bob", " BOB "])] == ["bob"]


def test_an_old_state_without_the_new_fields_still_works():
    state = ConversationState.model_validate({"version": 1, "fence_id": "x", "turn_count": 2})
    assert state.recent_entities == [] and state.clarification_asked is False


# ── never twice, and the state round trip ───────────────────────────────────


def test_after_an_ask_the_next_turn_answers_with_a_hint_and_never_asks_again():
    llm = CountingLLM()
    response = _engine(_store(MEN), llm).query(_request(clarify="ask", asked=True), HE)
    assert len(llm.prompts) == 1, "the user is never asked twice in a row"
    assert response.clarification is not None and response.clarification.asked is False
    assert response.state.clarification_asked is False
    assert "most likely" in llm.prompts[0], "right after an ask the model is told to answer its best guess and say which"


def test_without_a_state_the_engine_does_not_ask():
    llm = CountingLLM()
    response = _engine(_store(MEN), llm).query(_request(clarify="ask", state=False), HE)
    assert len(llm.prompts) == 1 and response.clarification is None, "the referent case draws its candidates from the state"


def test_without_a_state_a_hook_case_is_marked_but_not_asked():
    llm = CountingLLM()
    response = _hooked(_store(FAMILY), llm).query(RAGQueryRequest(question="Li Ming", clarify="ask"), NAMED)
    assert len(llm.prompts) == 1
    assert response.clarification.reason == "several_matches" and response.clarification.asked is False


def test_the_profile_default_can_set_ask_and_the_request_overrides_it():
    llm = CountingLLM()
    asked = _engine(_store(MEN), llm, clarification={"mode": "ask"}).query(_request(), HE)
    assert asked.clarification.asked is True and llm.prompts == []
    answered = _engine(_store(MEN), llm, clarification={"mode": "ask"}).query(_request(clarify="answer"), HE)
    assert answered.clarification.asked is False and len(llm.prompts) == 1


def test_the_hedge_note_is_opt_in():
    off, on = CountingLLM(), CountingLLM()
    _engine(_store(MEN), off).query(_request(), HE)
    _engine(_store(MEN), on, clarification={"hedge_note": True}).query(_request(), HE)
    assert "most likely" not in off.prompts[0] and "most likely" in on.prompts[0]


# ── the state remembers names that were actually shown ──────────────────────


def test_a_normal_answer_remembers_names_that_appeared_in_shown_passages():
    intent = QueryIntent(search_query="Bob and Zed", entity_terms=["Bob", "Zed"], entity_mode="ignore")
    response = _engine(_store(MEN), CountingLLM()).query(RAGQueryRequest(question="Bob and Zed"), intent)
    assert response.state.recent_entities == ["Bob"], "Zed was never in a shown passage"
    assert response.state.clarification_asked is False


# ── candidate hook: only apps that can tell referents apart get several_matches / too_broad ──


def _key(node):
    for key in ("father", "son"):
        if key in node.text:
            return key, f"Li Ming the {key}"
    return None


FAMILY = ["Li Ming the father farmed rice.", "Li Ming the father sold the farm.", "Li Ming the son studied law.", "Li Ming the son left home."]
NAMED = QueryIntent(search_query="Li Ming", entity_terms=["Li Ming"], entity_mode="ignore")


def _hooked(store, llm, **extra):
    from cody_ai_shared_lib.rag.engine import RAGEngine
    from cody_ai_shared_lib.rag.gateway import ModelGateway

    return RAGEngine(profile(**extra), store, gateway=ModelGateway(llm_client=llm), candidate_key=_key)


def test_with_the_hook_two_referents_of_one_name_ask():
    llm = CountingLLM()
    response = _hooked(_store(FAMILY), llm).query(RAGQueryRequest(question="Li Ming", clarify="ask", state=ConversationState()), NAMED)
    assert response.clarification.reason == "several_matches" and response.clarification.asked is True
    assert {o.label for o in response.clarification.options} == {"Li Ming the father", "Li Ming the son"}
    assert llm.prompts == []


def test_without_the_hook_several_documents_never_ask():
    """Six articles about Apple are not six meanings."""
    store = SpyStore()
    for i in range(6):
        node = RAGNode(node_id=f"a{i}", text=f"Apple announced product {i}.", span=SourceSpan(doc_id=f"article{i}", char_start=0, char_end=30))
        store.upsert_section_nodes(f"article{i}", 0, [node], [[0.1] * DIM])
    intent = QueryIntent(search_query="Apple", entity_terms=["Apple"], entity_mode="ignore")
    llm = CountingLLM()
    response = _engine(store, llm).query(RAGQueryRequest(question="Apple", clarify="ask", state=ConversationState()), intent)
    assert response.clarification is None and len(llm.prompts) == 1


def test_too_broad_is_only_ever_a_mark():
    llm = CountingLLM()
    intent = QueryIntent(search_query="tell me about the family")
    response = _hooked(_store(FAMILY), llm).query(RAGQueryRequest(question="family", clarify="ask", state=ConversationState()), intent)
    assert response.clarification is not None and response.clarification.reason == "too_broad"
    assert response.clarification.asked is False and len(llm.prompts) == 1


def test_one_dominant_group_does_not_ask():
    texts = ["Li Ming the father farmed rice."] * 1 + ["Li Ming the son studied law.", "Li Ming the son left home.", "Li Ming the son won a prize.", "Li Ming the son moved."]
    response = _hooked(_store(texts), CountingLLM()).query(
        RAGQueryRequest(question="Li Ming", clarify="ask", state=ConversationState()), NAMED
    )
    assert response.clarification is None


# ── no check without a reason ───────────────────────────────────────────────


def test_an_ordinary_question_makes_no_extra_store_call():
    store = _store(MEN)
    _engine(store, CountingLLM()).query(RAGQueryRequest(question="Bob", state=ConversationState(recent_entities=["Bob", "Carl"])), QueryIntent(search_query="Bob"))
    assert store.fetched == []


# ── wording ─────────────────────────────────────────────────────────────────


def test_a_chinese_question_gets_the_chinese_template():
    texts = ["鲍勃周一去了市场。", "卡尔周二去了港口。"]
    intent = QueryIntent(search_query="他下周做了什么", unresolved_referent=True, is_followup=True)
    response = _engine(_store(texts), CountingLLM()).query(
        _request(question="他下周做了什么", clarify="ask", names=("鲍勃", "卡尔")), intent
    )
    assert "你指的是" in response.answer_final and "鲍勃" in response.answer_final and "卡尔" in response.answer_final


def test_router_json_keeps_the_unresolved_referent_flag():
    from cody_ai_shared_lib.rag.intent.llm_rewriter import clamp_intent

    assert clamp_intent({"search_query": "q", "unresolved_referent": True}).unresolved_referent is True
    assert clamp_intent({"search_query": "q"}).unresolved_referent is False


def test_config_limits():
    from pydantic import ValidationError

    from cody_ai_shared_lib.rag.profile import ClarificationConfig

    for bad in ({"max_options": 0}, {"min_group_hits": 0}, {"dominance": 0}, {"dominance": 1.5}):
        with pytest.raises(ValidationError):
            ClarificationConfig(**bad)


# ── streaming ───────────────────────────────────────────────────────────────


def _kinds(events):
    return [e.event for e in events if e.event != "status"]


def test_the_stream_for_an_ask_sends_the_question_as_a_delta_then_a_clarification_event():
    events = list(_engine(_store(MEN), CountingLLM()).stream_query(_request(clarify="ask"), HE))
    assert _kinds(events) == ["intent", "retrieval", "delta", "clarification", "citations", "done"]
    delta = next(e for e in events if e.event == "delta").data["text"]
    done = next(e for e in events if e.event == "done").data
    assert "Bob" in delta and done["clarification"]["asked"] is True and done["answer_found"] is None
    citations = next(e for e in events if e.event == "citations").data
    assert citations["citations"] == []


def test_the_default_stream_order_is_unchanged():
    events = list(_engine(_store(MEN), CountingLLM()).stream_query(_request(), HE))
    kinds = _kinds(events)
    assert kinds[0] == "intent" and kinds[1] == "retrieval" and kinds[-2:] == ["citations", "done"]
    assert "clarification" not in kinds
