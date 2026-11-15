"""Generation strategy port and the single-call `stuff` strategy (§10.6, AC-RAG-39).

A generation strategy turns an assembled prompt into model text. The only built-in one, `stuff`, puts the whole prompt
in one call ("stuffs" it) and lets the gateway handle models, fallbacks, budgets and usage. A different strategy (say, one
that answers in two steps) would implement the same two methods and be registered under "generator".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterator, Protocol, runtime_checkable

from ...llm import LLMResult
from ..context.assembler import AssembledPrompt
from ..degradation import DegradationTracker
from ..gateway import ModelGateway
from ..profile import EngineProfile
from ..usage import BudgetGuard, UsageTracker


@dataclass
class GenerationContext:
    """Per-request collaborators for one generation call: where to record usage and degradations, the budget guard, the stage name."""
    usage_tracker: UsageTracker
    degradation_tracker: DegradationTracker
    budget_guard: BudgetGuard | None = None
    stage: str = "generation"


@runtime_checkable
class GenerationStrategy(Protocol):
    """The contract for generation: `generate` returns the whole text, `stream` yields it in chunks."""
    name: str
    version: str

    def generate(self, prompt: AssembledPrompt[Any], ctx: GenerationContext) -> LLMResult:
        ...

    def stream(self, prompt: AssembledPrompt[Any], ctx: GenerationContext) -> Iterator[Any]:
        ...


class StuffStrategy:
    """Passes the full assembled context to the generation models in a single call."""
    name: ClassVar[str] = "stuff"
    version: ClassVar[str] = "1.0.0"

    def __init__(self, gateway: ModelGateway, profile: EngineProfile):
        self.gateway = gateway
        self.profile = profile

    def _call_args(self, prompt: AssembledPrompt[Any], ctx: GenerationContext) -> dict[str, Any]:
        """The arguments of one gateway call. `generate` and `stream` share them, so they cannot drift apart."""
        return {
            "system_prompt": prompt.system_prompt,
            "user_message": prompt.user_message,
            "models": self.profile.models.generation,
            "stage": ctx.stage,
            "usage_tracker": ctx.usage_tracker,
            "degradation_tracker": ctx.degradation_tracker,
            "budget_guard": ctx.budget_guard,
            "temperature": self.profile.generation.temperature,
            "cache_breakpoints": prompt.cache_breakpoints,
            "user_blocks": prompt.user_blocks or None,
        }

    def generate(self, prompt: AssembledPrompt[Any], ctx: GenerationContext) -> LLMResult:
        return self.gateway.generate(**self._call_args(prompt, ctx))

    def stream(self, prompt: AssembledPrompt[Any], ctx: GenerationContext) -> Iterator[Any]:
        return self.gateway.stream(**self._call_args(prompt, ctx))
