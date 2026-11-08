"""RetrievalContext, QueryVariant, and RetrievalStrategy protocol (§8.1, §7.8, AC-RAG-19)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Literal
from pydantic import BaseModel, Field

from ..degradation import DegradationTracker
from ..gateway import ModelGateway
from ..models import MetaT, QueryIntent, RAGFilterSpec, RAGNode, RAGQueryRequest
from ..ports import NodeRenderer, RAGStore, StoreCapabilities
from ..profile import EngineProfile
from ..runtime import Deadline
from ..text.analyzer import TextAnalyzer
from ..usage import UsageTracker


class QueryVariant(BaseModel):
    """Query variant produced by transforms or fan-out (§7.8)."""
    text: str
    weight: float = 1.0
    origin: str = "primary"  # "primary" | "secondary" | "lexicon" | "hyde" | app name
    embed_as: Literal["query", "document"] = "query"


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


class RetrievalStrategy(ABC, Generic[MetaT]):
    """Abstract retrieval strategy interface (§8.1)."""
    name: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset()
    experimental: ClassVar[bool] = False

    @abstractmethod
    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        """Execute retrieval and return candidate nodes."""
        ...
