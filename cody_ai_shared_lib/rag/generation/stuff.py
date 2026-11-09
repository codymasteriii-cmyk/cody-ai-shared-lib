"""Generation strategy port and the single-call `stuff` strategy (§10.6, AC-RAG-39)."""
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
    """Per-request collaborators for one generation call."""
    usage_tracker: UsageTracker
    degradation_tracker: DegradationTracker
    budget_guard: BudgetGuard | None = None
    stage: str = "generation"


@runtime_checkable
class GenerationStrategy(Protocol):
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
        }

    def generate(self, prompt: AssembledPrompt[Any], ctx: GenerationContext) -> LLMResult:
        return self.gateway.generate(**self._call_args(prompt, ctx))

    def stream(self, prompt: AssembledPrompt[Any], ctx: GenerationContext) -> Iterator[Any]:
        return self.gateway.stream(**self._call_args(prompt, ctx))
