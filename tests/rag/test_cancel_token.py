"""A caller can stop a request from another thread, including while it routes or retrieves (§13.4, AC-RAG-113).

Closing a stream already stops generation. Before the first event the engine runs routing and retrieval in one
blocking step, and a generator cannot be closed from another thread while it runs, so a thread-safe CancelToken
is checked at the engine's stage boundaries instead. It cannot interrupt a provider call already in flight.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from cody_ai_shared_lib.llm import LLMResult, LLMStreamChunk, LLMUsage
from cody_ai_shared_lib.rag import (
    CancelToken,
    EngineProfile,
    InMemoryRAGStore,
    ModelBindings,
    ModelGateway,
    RAGEngine,
    RAGNode,
    RAGQueryRequest,
    RequestCancelled,
    RetrievalConfig,
    SourceSpan,
)
from cody_ai_shared_lib.rag.errors import NEVER_DEGRADABLE
from cody_ai_shared_lib.rag.profile import GenerationConfig

from .test_blocker_regressions import real_embedding_shape  # noqa: F401  (autouse fixture)

DIM = 768
QUESTION = "Who is Jonathan?"
INTENT = {"search_query": "Jonathan"}


class ScriptedLLM:
    """Answers the intent call with a fixed intent; `on_intent` runs first, and `delay` makes it slow."""

    def __init__(self, on_intent=None, delay: float = 0.0):
        self.calls: list[str] = []
        self.on_intent = on_intent
        self.delay = delay

    def generate_with_usage(self, **kwargs) -> LLMResult:
        is_intent = "query analysis assistant" in (kwargs.get("system_prompt") or "")
        self.calls.append("intent" if is_intent else "generation")
        if is_intent:
            if self.delay:
                time.sleep(self.delay)
            if self.on_intent:
                self.on_intent()
            text = json.dumps(INTENT)
        else:
            text = "[[answer_found=yes]]Jonathan is the miller's son [1]."
        return LLMResult(text=text, usage=LLMUsage(input_tokens=7, output_tokens=3, cost_usd=0.0), finish_reason="stop")

    def stream(self, **kwargs):
        self.calls.append("generation")
        yield LLMStreamChunk(delta="[[answer_found=yes]]Jonathan is ")
        yield LLMStreamChunk(delta="the miller's son [1].")
        yield LLMStreamChunk(delta="", usage=LLMUsage(input_tokens=5, output_tokens=2), finish_reason="stop")


class Observer:
    def __init__(self):
        self.completed = []

    def on_stage(self, record): ...
    def on_usage(self, usage): ...
    def on_degrade(self, degradation): ...

    def on_complete(self, trace, outcome):
        self.completed.append(outcome)


class HookedStore(InMemoryRAGStore):
    """Runs `hook` when the engine searches by keyword, to cancel in the middle of retrieval."""
    hook = None

    def keyword_search(self, *args, **kwargs):
        if self.hook:
            self.hook()
        return super().keyword_search(*args, **kwargs)


def make_engine(llm: ScriptedLLM, stream: bool = False, store: InMemoryRAGStore | None = None, observers=None) -> RAGEngine:
    store = store or HookedStore()
    text = "Jonathan is the eldest son of the miller family in the valley."
    span = SourceSpan(doc_id="d", section_seq=0, char_start=0, char_end=len(text))
    store.upsert_section_nodes("d", 0, [RAGNode(node_id="p0", text=text, span=span)], [[0.1] * DIM])
    profile = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], intent=["i1"], embedding="gemini-embedding-2"),
        retrieval=RetrievalConfig(default_strategy="keyword_only"),
        generation=GenerationConfig(stream_mode="native" if stream else "buffered"),
    )
    return RAGEngine(profile, store, gateway=ModelGateway(llm_client=llm), observers=observers)


def request() -> RAGQueryRequest:
    return RAGQueryRequest(question=QUESTION)


def test_a_token_that_is_never_set_changes_nothing():
    llm = ScriptedLLM()
    response = make_engine(llm).query(request(), cancel=CancelToken())
    assert "miller" in response.answer_final
    assert llm.calls == ["intent", "generation"]


def test_cancelled_before_the_call_makes_no_model_call_at_all():
    token = CancelToken()
    token.cancel("user pressed stop")
    llm = ScriptedLLM()
    with pytest.raises(RequestCancelled, match="user pressed stop"):
        make_engine(llm).query(request(), cancel=token)
    assert llm.calls == []


def test_cancelled_while_routing_stops_before_retrieval_and_generation():
    """The router call already in flight finishes, and nothing after it runs."""
    token = CancelToken()
    llm = ScriptedLLM(on_intent=token.cancel)
    with pytest.raises(RequestCancelled) as err:
        make_engine(llm).query(request(), cancel=token)
    assert llm.calls == ["intent"]
    assert err.value.stage == "routing"
    assert err.value.usage_total.input_tokens == 7          # the router call was paid for, and is reported


def test_cancelled_from_another_thread_during_a_slow_router():
    token = CancelToken()
    llm = ScriptedLLM(delay=0.4)
    threading.Timer(0.05, token.cancel).start()
    started = time.monotonic()
    with pytest.raises(RequestCancelled):
        make_engine(llm).query(request(), cancel=token)
    assert time.monotonic() - started < 2.0
    assert llm.calls == ["intent"]


def test_cancelled_during_retrieval_stops_before_generation():
    token = CancelToken()
    store = HookedStore()
    store.hook = token.cancel
    llm = ScriptedLLM()
    with pytest.raises(RequestCancelled):
        make_engine(llm, store=store).query(request(), cancel=token)
    assert llm.calls == ["intent"]


def test_retrieve_honours_the_token_too():
    token = CancelToken()
    llm = ScriptedLLM(on_intent=token.cancel)
    with pytest.raises(RequestCancelled):
        make_engine(llm).retrieve(request(), cancel=token)


def test_cancelling_a_stream_mid_generation_ends_it_with_an_error_event_and_reports_spend():
    token = CancelToken()
    llm = ScriptedLLM()
    stream = make_engine(llm, stream=True).stream_query(request(), cancel=token)
    events, raised = [], None
    try:
        for event in stream:
            events.append(event)
            if event.event == "delta" and not token.is_cancelled:
                token.cancel()
    except RequestCancelled as exc:
        raised = exc
    assert raised is not None
    kinds = [e.event for e in events]
    assert kinds.count("delta") == 1 and "done" not in kinds     # no further text, and no turn to save
    assert kinds[-1] == "error" and events[-1].data["type"] == "RequestCancelled"
    assert events[-1].data["usage_total"]["input_tokens"] >= 7   # spend so far is on the event
    assert stream.usage_total.input_tokens >= 7


def test_observers_see_a_cancelled_request_with_no_answer():
    watcher = Observer()
    token = CancelToken()
    with pytest.raises(RequestCancelled):
        make_engine(ScriptedLLM(on_intent=token.cancel), observers=[watcher]).query(request(), cancel=token)
    assert watcher.completed == [None]


def test_a_cancel_is_never_absorbed_by_a_degradation():
    assert RequestCancelled in NEVER_DEGRADABLE


def test_cancel_is_idempotent_and_keeps_the_first_reason():
    token = CancelToken()
    token.cancel("first")
    token.cancel("second")
    assert token.is_cancelled and token.reason == "first"
