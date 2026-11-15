"""Lists across documents (RAG_SPEC 7.9 boundaries / OI-4, 1.2.12): wants_list, completeness, find-then-read, query_batched."""
from __future__ import annotations

import pytest

from cody_ai_shared_lib.rag.intent.heuristic import HeuristicRouter
from cody_ai_shared_lib.rag.intent.llm_rewriter import clamp_intent
from cody_ai_shared_lib.rag.models import QueryIntent, RAGQueryRequest
from cody_ai_shared_lib.rag.profile import RetrievalConfig
from tests.rag.test_blocker_regressions import FakeLLM, engine, profile, real_embedding_shape  # noqa: F401


def _intent(**changes) -> QueryIntent:
    changes.setdefault("search_query", "every model mentioned")
    return QueryIntent(wants_list=True, **changes)


# ── wants_list ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question, expected",
    [
        ("List every Seedance model mentioned this year", True),
        ("Enumerate all the characters who appear in chapter 3", True),
        ("列出所有出现过的角色", True),
        ("Alice, Bob and Carol: what does each of them do?", False),     # per-subject, the several-names rule covers it
        ("What does each character want?", False),
        ("Who is Alice?", False),
    ],
)
def test_heuristic_wants_list_means_an_enumeration_the_question_does_not_name(question, expected):
    intent = HeuristicRouter().classify(question)
    assert bool(intent and intent.wants_list) is expected


def test_clamp_keeps_wants_list_and_defaults_to_false():
    assert clamp_intent({"search_query": "q", "wants_list": True}).wants_list is True
    assert clamp_intent({"search_query": "q"}).wants_list is False


# ── completeness ────────────────────────────────────────────────────────────


def _answer(text="[[answer_found=yes]] Alice met Bob [1]."):
    return FakeLLM(text)


def test_whole_scope_read_is_complete():
    prof = profile(retrieval=RetrievalConfig(default_strategy="adaptive"))
    response = engine(prof, llm=_answer()).query(RAGQueryRequest(question="list all"), _intent())
    assert response.retrieved.strategy_used == "full_context"
    assert response.completeness == "whole_scope"
    assert response.answer_found == "yes"


def test_a_ranked_search_is_never_complete_and_a_yes_becomes_partial():
    prof = profile(retrieval=RetrievalConfig(default_strategy="dense_vector"))
    response = engine(prof, llm=_answer()).query(RAGQueryRequest(question="list all"), _intent())
    assert response.completeness == "partial"
    assert response.answer_found == "partial"
    assert response.abstained is False


def test_a_no_stays_no_when_the_list_is_partial():
    prof = profile(retrieval=RetrievalConfig(default_strategy="dense_vector"))
    response = engine(prof, llm=_answer("[[answer_found=no]] Nothing.")).query(RAGQueryRequest(question="list all"), _intent())
    assert response.completeness == "partial" and response.answer_found == "no"


def test_no_completeness_for_a_question_that_is_not_a_list():
    response = engine(llm=_answer()).query(RAGQueryRequest(question="who is Alice"), QueryIntent(search_query="Alice"))
    assert response.completeness is None and response.answer_found == "yes"


def test_the_prompt_tells_the_model_a_partial_list_may_be_incomplete():
    seen: list[str] = []

    class Spy(FakeLLM):
        def generate_with_usage(self, **kw):
            seen.append(str(kw))
            return super().generate_with_usage(**kw)

    prof = profile(retrieval=RetrievalConfig(default_strategy="dense_vector"))
    engine(prof, llm=Spy("[[answer_found=yes]] A [1].")).query(RAGQueryRequest(question="list all"), _intent())
    assert any("may be incomplete" in s for s in seen)


# ── find-then-read, widened to the section ──────────────────────────────────

from cody_ai_shared_lib.rag.granularity import is_ranked  # noqa: E402
from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore  # noqa: E402
from cody_ai_shared_lib.rag.models import RAGFilterSpec, RAGNode, SourceSpan  # noqa: E402
from tests.rag.test_blocker_regressions import DIM  # noqa: E402

ORTHOGONAL = [0.0] * (DIM - 1) + [1.0]      # far from the fake query vector: only keyword hits come back


def _section_text(doc: str, seq: int, hit: bool) -> str:
    base = f"{doc} section {seq} ordinary words about weather and roads. Nothing else of note here."
    text = base.replace("ordinary", "Seedance mentioned among") if hit else base
    return text.ljust(100)               # sections are contiguous: [k*100, k*100 + 100)


def _store(docs: dict[str, list[bool]], summaries: bool = True) -> InMemoryRAGStore:
    """doc id -> per section, whether it mentions Seedance. Section k spans [k*100, k*100 + 100)."""
    store = InMemoryRAGStore()
    for doc, hits in docs.items():
        for seq, hit in enumerate(hits):
            text = _section_text(doc, seq, hit)
            node = RAGNode(node_id=f"{doc}-{seq}", text=text, span=SourceSpan(doc_id=doc, section_seq=seq, char_start=seq * 100, char_end=seq * 100 + 100))
            summary = None
            if summaries:
                summary = RAGNode(
                    node_id=f"{doc}-s{seq}", text="Summary.", is_summary=True, granularity="parent_summary", parent_kind="section", level=1,
                    span=SourceSpan(doc_id=doc, section_seq=seq, char_start=seq * 100, char_end=seq * 100 + 100),
                )
            store.upsert_section_nodes(doc, seq, [node], [ORTHOGONAL], summary=summary)
    return store


def _list_engine(store, gate=40, **retrieval):
    """A section is about 24 tokens: with the default gate of 40, one section fits and two do not."""
    prof = profile(retrieval=RetrievalConfig(default_strategy="adaptive", list_strategy="find_then_read", full_context_max_tokens=gate, **retrieval))
    return engine(prof, store=store, llm=FakeLLM("[[answer_found=yes]] ok [1]."))


def _read(eng, store=None, **filters):
    request = RAGQueryRequest(question="list every Seedance", filters=RAGFilterSpec(min_similarity=0.5, **filters))
    return eng.retrieve(request, _intent(search_query="Seedance"))


def test_find_then_read_reads_the_section_of_each_hit_not_the_whole_document():
    store = _store({"docA": [False, True, False]})
    result = _read(_list_engine(store))
    assert result.strategy_used == "find_then_read"
    assert [n.node_id for n in result.nodes] == ["docA-1"], "only the section holding the hit, not sections 0 and 2"
    assert all(not is_ranked(n) for n in result.nodes), "read text is Tier 0 style, never ranked candidates"
    # A ranked finder never proves the list complete, but every hit was read in full: "searched" (1.2.16)
    assert result.completeness == "searched"


def test_hits_in_one_section_become_one_read_and_two_sections_are_both_read():
    store = _store({"docA": [True, False, True]})
    assert [n.node_id for n in _read(_list_engine(store, gate=60)).nodes] == ["docA-0", "docA-2"]


def test_a_document_that_fits_is_read_whole():
    store = _store({"docA": [True]})
    result = _read(_list_engine(store))
    assert [n.node_id for n in result.nodes] == ["docA-0"]
    assert result.strategy_used in ("full_context", "find_then_read")


def test_the_sequential_boundary_is_kept_inside_a_section():
    store = _store({"docA": [False, True, False]})
    result = _read(_list_engine(store), max_char_offset=150)
    assert result.nodes and all(n.span.char_end <= 150 for n in result.nodes), "no text past the reader's position"


def test_without_summaries_the_engine_reads_a_window_around_the_hit():
    store = _store({"docA": [False, True, False]}, summaries=False)
    result = _read(_list_engine(store, local_window_radius_chars=60, list_read_radius_chars=60))
    assert result.nodes, "a window was read"
    assert any("docA section 1" in n.text for n in result.nodes)


def test_the_document_cap_stops_reading_and_marks_the_result_partial():
    store = _store({"d1": [False, True], "d2": [False, True], "d3": [False, True]})
    result = _read(_list_engine(store, gate=120, list_max_documents=2))
    assert {n.span.doc_id for n in result.nodes} == {"d1", "d2"}
    assert result.truncated is True and result.completeness == "partial"


def test_the_token_budget_stops_reading_and_marks_the_result_partial():
    store = _store({"d1": [False, True], "d2": [False, True]})
    result = _read(_list_engine(store))                      # 40-token gate: one section fits, two do not
    assert [n.node_id for n in result.nodes] == ["d1-1"] and result.truncated is True


def test_find_then_read_is_off_unless_the_profile_opts_in_and_the_question_is_a_list():
    store = _store({"docA": [False, True, False]})
    prof = profile(retrieval=RetrievalConfig(default_strategy="adaptive", full_context_max_tokens=40))
    result = engine(prof, store=store, llm=FakeLLM()).retrieve(
        RAGQueryRequest(question="x", filters=RAGFilterSpec(min_similarity=0.5)), _intent(search_query="Seedance")
    )
    assert result.strategy_used != "find_then_read"
    opted = _list_engine(store)
    not_a_list = opted.retrieve(RAGQueryRequest(question="x", filters=RAGFilterSpec(min_similarity=0.5)), QueryIntent(search_query="Seedance"))
    assert not_a_list.strategy_used != "find_then_read"


# ── query_batched ───────────────────────────────────────────────────────────

import re  # noqa: E402
import threading  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from cody_ai_shared_lib.llm import LLMResult, LLMUsage  # noqa: E402
from cody_ai_shared_lib.rag.errors import GenerationError, RequestCancelled  # noqa: E402
from cody_ai_shared_lib.rag.profile import GenerationConfig  # noqa: E402
from cody_ai_shared_lib.rag.runtime import CancelToken, FrozenClock  # noqa: E402

_NODE = re.compile(r'<node ref="(\[\d+\])"[^>]*>\n(.*?)\n</node>', re.S)


class ListLLM:
    """Scripted model. Map calls list one bullet per passage that mentions Seedance, citing its marker.
    Merge calls (their instructions start "Merge these list items") answer with `merge_reply(items)`."""

    def __init__(self, fail_on: str | None = None, merge_reply=None, on_call=None):
        self.calls: list[str] = []
        self.fail_on = fail_on
        self.merge_reply = merge_reply
        self.on_call = on_call
        self._lock = threading.Lock()

    def generate_with_usage(self, **kw) -> LLMResult:
        system, user = kw.get("system_prompt", ""), kw.get("user_message", "")
        is_merge = "Merge these list items" in system
        with self._lock:
            self.calls.append("merge" if is_merge else "map")
        if self.on_call:
            self.on_call(is_merge)
        if is_merge:
            items = re.findall(r"^I(\d+): (.*)$", user, re.M)
            text = self.merge_reply(items) if self.merge_reply else "\n".join(f"{i} | {t}" for i, t in items)
        else:
            if self.fail_on and self.fail_on in user:
                raise RuntimeError("provider down")
            bullets = [
                f"- Seedance in {t.split()[0]} {t.split()[1]} {t.split()[2]} {ref}"
                for ref, t in _NODE.findall(user)
                if "Seedance" in t
            ]
            text = "[[answer_found=yes]]\n" + "\n".join(bullets)
        return LLMResult(text=text, usage=LLMUsage(input_tokens=10, output_tokens=5, cost_usd=0.001), finish_reason="stop")


def _batched(store, llm, gate=70, max_batches=4, max_docs=40, on_failure="raise", **extra):
    prof = profile(
        retrieval=RetrievalConfig(default_strategy="adaptive", full_context_max_tokens=gate),
        generation=GenerationConfig(list_max_batches=max_batches, list_max_documents=max_docs, list_on_batch_failure=on_failure),
    )
    return engine(prof, store=store, llm=llm, **extra)


def _docs(n: int) -> dict[str, list[bool]]:
    return {f"d{i}": [False, True] for i in range(1, n + 1)}      # each document: 2 sections, 56 tokens, one hit


def _ask(eng, **kw):
    request = RAGQueryRequest(question="list every Seedance", filters=RAGFilterSpec(min_similarity=0.5))
    return eng.query_batched(request, _intent(search_query="Seedance"), **kw)


def test_query_batched_returns_a_normal_response_with_a_report_of_what_was_read():
    llm = ListLLM()
    response = _ask(_batched(_store(_docs(3)), llm))
    assert response.answer_found == "yes" and response.completeness == "searched"     # found by search, nothing lost (1.2.16)
    assert response.batches.documents_found == 3 and response.batches.documents_read == 3
    assert response.batches.batches_read == 3 and response.batches.capped_by is None
    assert llm.calls.count("map") == 3 and llm.calls.count("merge") == 1


def test_a_citation_survives_from_a_batch_item_into_the_merged_answer():
    response = _ask(_batched(_store(_docs(3)), ListLLM()))
    by_marker = {c.marker: c for c in response.citations}
    seen = 0
    for line in response.answer_final.splitlines():
        match = re.match(r"- Seedance in (d\d) section 1 (\[\d+\])$", line.strip())
        if match:
            doc, marker = match.groups()
            assert by_marker[marker].node_id == f"{doc}-1" and by_marker[marker].verified is True
            seen += 1
    assert seen == 3
    assert {c.node_id for c in response.citations} == {"d1-1", "d2-1", "d3-1"}
    assert len({c.marker for c in response.citations}) == 3


def test_one_batch_needs_no_merge_call():
    llm = ListLLM()
    response = _ask(_batched(_store(_docs(1)), llm))
    assert llm.calls == ["map"] and response.answer_found == "yes"


def test_the_batch_cap_stops_the_work_and_the_answer_says_it_may_be_incomplete():
    llm = ListLLM()
    response = _ask(_batched(_store(_docs(6)), llm, max_batches=2))
    assert response.batches.capped_by == "batches" and response.batches.documents_read == 2
    assert response.batches.documents_found == 6
    assert response.answer_found == "partial" and response.completeness == "partial"
    assert "may be incomplete" in response.answer_final


def test_the_document_cap_stops_the_work():
    response = _ask(_batched(_store(_docs(4)), ListLLM(), max_docs=2))
    assert response.batches.capped_by == "documents" and response.batches.documents_read == 2
    assert response.answer_found == "partial"


def test_the_worst_case_is_the_batch_cap_plus_one_merge():
    llm = ListLLM()
    _ask(_batched(_store(_docs(8)), llm, max_batches=4))
    assert len(llm.calls) == 5


def test_a_failed_batch_raises_by_default():
    with pytest.raises(GenerationError):
        _ask(_batched(_store(_docs(3)), ListLLM(fail_on="d2 section")))


def test_a_failed_batch_can_degrade_but_is_never_silent():
    response = _ask(_batched(_store(_docs(3)), ListLLM(fail_on="d2 section"), on_failure="degrade"))
    assert response.batches.batches_failed and response.answer_found == "partial"
    assert any(d.id == "DG-15" for d in response.degradations)
    assert {c.node_id for c in response.citations} == {"d1-1", "d3-1"}


def test_a_malformed_merge_reply_falls_back_to_the_unique_items_with_a_degradation():
    response = _ask(_batched(_store(_docs(3)), ListLLM(merge_reply=lambda items: "no list here, sorry")))
    assert {c.node_id for c in response.citations} == {"d1-1", "d2-1", "d3-1"}
    assert any(d.id == "DG-15" for d in response.degradations)


def test_items_the_merge_forgets_are_kept():
    response = _ask(_batched(_store(_docs(3)), ListLLM(merge_reply=lambda items: f"{items[0][0]} | {items[0][1]}")))
    assert {c.node_id for c in response.citations} == {"d1-1", "d2-1", "d3-1"}


def test_the_app_can_supply_the_documents_and_a_complete_read_is_whole_scope():
    store = _store(_docs(3))
    llm = ListLLM()
    request = RAGQueryRequest(question="list every Seedance")
    response = _batched(store, llm).query_batched(request, _intent(search_query="Seedance"), doc_ids=["d1", "d2"])
    assert response.batches.documents_found == 2 and response.batches.documents_read == 2
    assert response.completeness == "whole_scope" and response.answer_found == "yes"
    assert {c.node_id for c in response.citations} == {"d1-1", "d2-1"}


def test_every_model_call_goes_through_the_budget_guard():
    class Guard:
        enforces_cost = False

        def __init__(self):
            self.stages: list[str] = []

        def before_call(self, stage, model, est_input_tokens, optional, ctx):
            self.stages.append(stage)
            return "allow"

    guard, llm = Guard(), ListLLM()
    _ask(_batched(_store(_docs(3)), llm, budget_guard=guard))
    assert len([s for s in guard.stages if s.startswith("list_")]) == len(llm.calls) == 4


def test_a_cancelled_request_makes_no_model_call():
    token, llm = CancelToken(), ListLLM()
    token.cancel("user left")
    with pytest.raises(RequestCancelled):
        _ask(_batched(_store(_docs(3)), llm), cancel=token)
    assert llm.calls == []


def test_a_merge_skipped_for_time_keeps_every_item_and_is_not_partial():
    """No items are missing, only de-duplication was skipped: a degradation says so, the answer is not partial."""
    ticks = {"now": 0.0}
    clock = FrozenClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    clock.monotonic = lambda: ticks["now"]
    llm = ListLLM(on_call=lambda is_merge: ticks.__setitem__("now", ticks["now"] + 100.0))   # each call "takes" 100 s of the 30 s deadline
    response = _ask(_batched(_store(_docs(3)), llm, clock=clock))
    assert "merge" not in llm.calls
    assert response.batches.merge_skipped is True and response.batches.capped_by is None
    assert response.answer_found == "yes", "every item was found: a skipped merge is not a partial answer"
    assert "may be incomplete" not in response.answer_final
    assert {c.node_id for c in response.citations} == {"d1-1", "d2-1", "d3-1"}
    notes = [d for d in response.degradations if d.id == "DG-10" and "duplicates" in d.impact]
    assert notes, "a degradation says the list may contain duplicates"


def test_unread_documents_and_failed_batches_stay_partial_even_when_the_merge_runs():
    response = _ask(_batched(_store(_docs(6)), ListLLM(), max_batches=2))
    assert response.batches.merge_skipped is False and response.answer_found == "partial"


def test_list_caps_must_be_at_least_one():
    from pydantic import ValidationError

    for bad in ({"list_max_batches": 0}, {"list_max_documents": 0}):
        with pytest.raises(ValidationError):
            GenerationConfig(**bad)
    with pytest.raises(ValidationError):
        RetrievalConfig(list_max_documents=0)
