"""Query variants and the QueryTransform contract (§7.8)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import QueryIntent, RAGQueryRequest


class QueryVariant(BaseModel):
    """One phrasing of the query to search with (§7.8)."""
    text: str
    weight: float = 1.0
    origin: str = "primary"  # "primary" | "secondary" | "lexicon" | "hyde" | app name
    embed_as: Literal["query", "document"] = "query"


@dataclass
class TransformContext:
    """What a transform may use. A transform that calls a model reports its spend through these."""
    profile: Any
    analyzer: Any
    gateway: Any
    usage_tracker: Any
    degradation_tracker: Any
    deadline: Any
    budget_guard: Any = None


@runtime_checkable
class QueryTransform(Protocol):
    """Decides what to search for; a retrieval strategy decides how to search (§7.8)."""
    name: str
    version: str

    def transform(self, request: RAGQueryRequest, intent: QueryIntent, ctx: TransformContext) -> list[QueryVariant]:
        ...
