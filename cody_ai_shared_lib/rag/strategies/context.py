"""RetrievalContext and the RetrievalStrategy contract (§8.1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal
from pydantic import BaseModel, Field

from ..degradation import DegradationTracker
from ..models import MetaT, QueryIntent, RAGFilterSpec, RAGNode, RAGQueryRequest
from ..profile import EngineProfile
from ..runtime import Deadline
from ..transforms.base import QueryVariant
from ..usage import UsageTracker


class RetrievalContext(BaseModel, Generic[MetaT]):
    """Unified context passed to retrieval strategies (§8.1)."""
    model_config = {"arbitrary_types_allowed": True}

    request: RAGQueryRequest
    intent: QueryIntent
    variants: list[QueryVariant] = Field(default_factory=list)
    filters: RAGFilterSpec
    store: Any
    profile: EngineProfile
    analyzer: Any
    renderer: Any
    gateway: Any
    deadline: Deadline
    usage_tracker: UsageTracker
    degradation_tracker: DegradationTracker
    mode: Literal["query", "retrieve"] = "query"
    budget_guard: Any = None
    pool: Any = None
    cache: Any = None
    last_strategy_used: str = ""
    last_tier_reason: str = ""
    last_rule: int = 0
    registry: Any = None
    rerank_active: bool = False
    reranked: bool = False
    truncated: bool = False
    scope_revision: str | None = None

    def target_limit(self) -> int:
        """Compute effective target candidate count based on mode (§7.1, §8.10)."""
        if self.mode == "retrieve":
            return (
                self.request.top_k_override
                or self.profile.retrieval.retrieve_max_results
                or self.profile.retrieval.top_k
            )
        return (
            self.request.top_k_override
            or self.intent.suggested_top_k
            or self.profile.retrieval.top_k
        )


    def candidate_limit(self) -> int:
        """How many ranked candidates to fetch: one extra to detect truncation, or a reranker's pool (§9.3)."""
        floor = self.target_limit() + 1
        if not self.rerank_active:
            return floor
        pool = self.profile.retrieval.rerank_candidates or self.target_limit() * 3
        return max(floor, min(pool, max(floor, self.store.capabilities().max_top_k)))


class RetrievalStrategy(ABC, Generic[MetaT]):
    """Abstract retrieval strategy interface (§8.1)."""
    name: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset()
    experimental: ClassVar[bool] = False

    @classmethod
    def is_runnable(cls, capabilities: Any) -> bool:
        """True when the store declares everything this strategy needs."""
        return all(getattr(capabilities, requirement, False) for requirement in cls.requires)

    @abstractmethod
    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        """Execute retrieval and return candidate nodes."""
        ...
