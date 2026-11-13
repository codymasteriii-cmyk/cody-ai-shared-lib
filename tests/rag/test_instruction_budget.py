"""The instructions zone is the measured system prompt, not a guess (§10.1, AC-RAG-112).

The persona, custom instruction and guard rules the app supplies are part of the system prompt. The budget
used to subtract a fixed 4,000 tokens whatever their size, so a long preamble overran the window unnoticed.
"""
from __future__ import annotations

import pytest

from cody_ai_shared_lib.rag import (
    DefaultRenderer,
    EngineProfile,
    ModelBindings,
    ProfileConfigError,
    RAGNode,
    RAGQueryRequest,
    RetrievalConfig,
    SourceSpan,
)
from cody_ai_shared_lib.rag.context.assembler import ContextAssembler
from cody_ai_shared_lib.rag.context.budgeter import TokenBudgeter
from cody_ai_shared_lib.rag.profile import ContextConfig, GenerationConfig
from cody_ai_shared_lib.rag.text.analyzer import estimate_tokens

CONTEXT = ContextConfig(max_context_tokens=20_000, reserved_instruction_tokens=1000, question_reserve_tokens=500,
                        answer_reserve_tokens=500)


def test_instructions_under_the_reserve_change_nothing():
    budgeter = TokenBudgeter(CONTEXT)
    assert budgeter.allocate(instruction_tokens=300) == budgeter.allocate()


def test_longer_instructions_take_their_room_from_the_passages():
    budgeter = TokenBudgeter(CONTEXT)
    base, long = budgeter.allocate(), budgeter.allocate(instruction_tokens=4000)
    assert long.usable_budget == base.usable_budget - 3000
    assert long.retrieved_budget < base.retrieved_budget


def test_instructions_that_are_most_of_the_window_are_refused():
    with pytest.raises(ProfileConfigError, match="instructions"):
        TokenBudgeter(CONTEXT).allocate(instruction_tokens=10_001)


def assembler(preamble: str) -> ContextAssembler:
    profile = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m1"], embedding="gemini-embedding-2"),
        context=CONTEXT,
        generation=GenerationConfig(system_preamble=preamble),
        retrieval=RetrievalConfig(full_context_max_tokens=0),
    )
    return ContextAssembler(profile, DefaultRenderer())


def total_tokens(prompt) -> int:
    return estimate_tokens(prompt.system_prompt) + estimate_tokens(prompt.user_message)


def test_a_long_persona_still_fits_the_window_because_the_passages_shrink():
    big = RAGNode(node_id="a", text="word " * 60_000, span=SourceSpan(doc_id="d"))
    persona = "You are a careful reading companion. " * 400            # well over the 1,000-token reserve
    assert estimate_tokens(persona) > 3000

    short = assembler("Be brief.").assemble_prompt(RAGQueryRequest(question="q"), [big])
    long = assembler(persona).assemble_prompt(RAGQueryRequest(question="q"), [big.model_copy(deep=True)])

    assert long.zone_tokens["instructions"] > short.zone_tokens["instructions"]
    assert long.zone_tokens["ranked"] < short.zone_tokens["ranked"]     # the persona's room came from the passages
    assert total_tokens(long) <= CONTEXT.max_context_tokens - CONTEXT.answer_reserve_tokens


def test_the_no_retrieval_route_measures_its_instructions_too():
    with pytest.raises(ProfileConfigError):
        assembler("word " * 12_000).assemble_conversational(RAGQueryRequest(question="q"))
