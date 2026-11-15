"""JevClient gives every question kind one answer shape, and the RAG Jev gate reads it (1.2.17, spec §7.4).

The SDK's answers differ by kind: a Noul answer has only `.noul` (the probability of yes); a Choice or Score answer has
`.choice` / `.score` and a `.confidence`. `JevClient.ask` wraps each in a `JevAnswer(kind, value, probability, raw)`, so callers
never guess attribute names. These tests build the answers from the REAL `typesafe_sdk` classes, because fakes with invented
attribute names are how the gate once read fields that did not exist.
"""
from unittest.mock import MagicMock

import pytest
from typesafe_sdk import ChoiceAnswer, Noul
from typesafe_sdk import NoulAnswer as SdkNoulAnswer
from typesafe_sdk import ScoreAnswer

from cody_ai_shared_lib.jev import JevAnswer, JevClient
from cody_ai_shared_lib.rag.degradation import DegradationTracker
from cody_ai_shared_lib.rag.intent.jev_gate import JevGateRouter
from cody_ai_shared_lib.rag.usage import UsageTracker


def client_returning(answers: dict) -> JevClient:
    """A JevClient whose SDK returns `answers`, so no network is used."""
    response = MagicMock()
    response.answers = answers
    response.usage = MagicMock(input_tokens=100, output_tokens=10)
    sdk = MagicMock()
    sdk.system_one.return_value = response
    client = JevClient(api_key="k")
    client._client = sdk
    return client


def choice(value: str, confidence: float) -> ChoiceAnswer:
    return ChoiceAnswer(choice=value, confidence=confidence, probabilities={value: confidence})


# ── the unified answer ─────────────────────────────────────────────────────

def test_a_noul_answer_is_a_probability_and_a_bool():
    sdk_answer = SdkNoulAnswer(noul=0.9)
    answer = client_returning({"q": sdk_answer}).ask("state", {"q": Noul(instructions="x")}).answers["q"]
    assert (answer.kind, answer.value, answer.probability) == ("noul", True, 0.9)
    assert answer.raw is sdk_answer
    low = client_returning({"q": SdkNoulAnswer(noul=0.2)}).ask("state", {"q": Noul(instructions="x")}).answers["q"]
    assert (low.value, low.probability) == (False, 0.2)


def test_a_choice_answer_carries_its_choice_and_its_confidence():
    answer = client_returning({"scope": choice("document_overview", 0.91)}).ask("s", {"scope": MagicMock()}).answers["scope"]
    assert (answer.kind, answer.value, answer.probability) == ("choice", "document_overview", 0.91)


def test_a_score_answer_carries_its_score_and_its_confidence():
    sdk_answer = ScoreAnswer(score=4, confidence=0.7, legend={}, probabilities={})
    answer = client_returning({"rating": sdk_answer}).ask("s", {"rating": MagicMock()}).answers["rating"]
    assert (answer.kind, answer.value, answer.probability) == ("score", 4.0, 0.7)


def test_a_question_jev_did_not_answer_is_absent():
    result = client_returning({}).ask("s", {"q": MagicMock()})
    assert "q" not in result.answers


def test_an_answer_of_an_unknown_type_is_refused_not_guessed():
    with pytest.raises(TypeError, match="q"):
        client_returning({"q": object()}).ask("s", {"q": MagicMock()})


def test_the_old_raw_interface_is_gone():
    assert not hasattr(JevClient, "system_one")


def test_usage_and_cost_are_still_reported():
    result = client_returning({"q": SdkNoulAnswer(noul=0.5)}).ask("s", {"q": MagicMock()})
    assert (result.tokens_in, result.tokens_out) == (100, 10)
    assert result.cost_usd == pytest.approx(100 * 0.04 / 1_000_000, rel=1e-5)


# ── ask_noul_batch ─────────────────────────────────────────────────────────

def test_ask_noul_batch_returns_unified_answers():
    answers = client_returning({"a": SdkNoulAnswer(noul=0.98), "b": SdkNoulAnswer(noul=0.02)}).ask_noul_batch(
        "state", {"a": "is it a?", "b": "is it b?"}
    )
    assert answers["a"].probability == 0.98 and answers["a"].value is True
    assert answers["b"].probability == 0.02 and answers["b"].value is False


def test_ask_noul_batch_strict_names_the_unanswered_questions():
    client = client_returning({"a": SdkNoulAnswer(noul=0.9)})
    with pytest.raises(ValueError, match="b"):
        client.ask_noul_batch("state", {"a": "x", "b": "y"}, strict=True)


def test_ask_noul_batch_default_reports_an_unanswered_question_as_zero():
    answers = client_returning({}).ask_noul_batch("state", {"b": "y"})
    assert answers["b"].probability == 0.0 and answers["b"].value is False


# ── the RAG gate reads the unified answer ──────────────────────────────────

def gate_for(answers: dict) -> JevGateRouter:
    gateway = MagicMock()
    gateway.jev_client = client_returning(answers)
    return JevGateRouter(gateway=gateway, api_key="k")


def sure_answers(scope="local_lookup", rewrite=0.10, time=0.05, probe=0.20) -> dict:
    return {
        "scope": choice(scope, 0.93),
        "needs_query_rewrite": SdkNoulAnswer(noul=rewrite),
        "has_time_expression": SdkNoulAnswer(noul=time),
        "beyond_boundary_probe": SdkNoulAnswer(noul=probe),
    }


def test_the_gate_takes_the_fast_path_on_real_sdk_answer_shapes():
    intent = gate_for(sure_answers()).route("What color was the car?", UsageTracker("r"), DegradationTracker())
    assert intent is not None and intent.router_source == "jev"
    assert intent.scope == "local_lookup"
    assert intent.beyond_boundary_risk == pytest.approx(0.20)


def test_the_gate_uses_the_scope_jev_chose():
    intent = gate_for(sure_answers(scope="document_overview")).route("what is it about", UsageTracker("r"), DegradationTracker())
    assert intent is not None and intent.scope == "document_overview"


def test_the_gate_declines_when_jev_says_a_rewrite_is_needed():
    assert gate_for(sure_answers(rewrite=0.60)).route("q", UsageTracker("r"), DegradationTracker()) is None


def test_the_gate_declines_when_jev_sees_a_time_expression():
    assert gate_for(sure_answers(time=0.50)).route("q", UsageTracker("r"), DegradationTracker()) is None


def test_the_gate_declines_when_a_question_is_unanswered():
    answers = sure_answers()
    del answers["needs_query_rewrite"]                      # a missing answer is "not sure", never "sure"
    assert gate_for(answers).route("q", UsageTracker("r"), DegradationTracker()) is None
