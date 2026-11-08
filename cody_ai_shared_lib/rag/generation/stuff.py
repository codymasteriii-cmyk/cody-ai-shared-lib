"""Direct single-turn stuff generation strategy (§11.1, AC-RAG-39)."""
from __future__ import annotations

import re
from typing import Any
from ..degradation import DegradationTracker
from ..gateway import ModelGateway
from ...llm import LLMResult
from ..profile import EngineProfile
from ..usage import BudgetGuard, UsageTracker


class StuffStrategy:
    """Passes full assembled context to generation model in a single call."""

    def __init__(self, gateway: ModelGateway, profile: EngineProfile):
        self.gateway = gateway
        self.profile = profile

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
        budget_guard: BudgetGuard | None = None,
        request_id: str = "",
        tenant_or_user_id: str | None = None,
    ) -> LLMResult:
        """Call generation models in profile fallback order."""
        models = self.profile.models.generation
        temperature = self.profile.generation.temperature

        return self.gateway.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            models=models,
            stage="generation",
            usage_tracker=usage_tracker,
            degradation_tracker=degradation_tracker,
            budget_guard=budget_guard,
            temperature=temperature,
            request_id=request_id,
            tenant_or_user_id=tenant_or_user_id,
        )

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
        budget_guard: BudgetGuard | None = None,
        cache_breakpoints: list[int] | None = None,
        request_id: str = "",
        tenant_or_user_id: str | None = None,
    ) -> Any:
        """Stream generation through ordered model fallback list."""
        models = self.profile.models.generation
        temperature = self.profile.generation.temperature

        return self.gateway.stream(
            system_prompt=system_prompt,
            user_message=user_message,
            models=models,
            stage="generation",
            usage_tracker=usage_tracker,
            degradation_tracker=degradation_tracker,
            budget_guard=budget_guard,
            temperature=temperature,
            cache_breakpoints=cache_breakpoints,
            request_id=request_id,
            tenant_or_user_id=tenant_or_user_id,
        )
