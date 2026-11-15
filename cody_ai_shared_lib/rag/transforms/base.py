"""Query variants and the QueryTransform contract (§7.8).

A question sometimes needs more than one search: "compare A and B" has two subjects, and one search for the whole sentence may
find only the better-matching one. A transform turns one question into several `QueryVariant`s (the primary one plus extras),
each searched separately and then fused (`transforms/fusion.py`). The only built-in transform is the secondary-query fan-out; an app
may add others (a keyword-expansion lexicon, a hypothetical-answer search) by registering them as "transform".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import QueryIntent, RAGQueryRequest


class QueryVariant(BaseModel):
    """One phrasing of the query to search with (§7.8): its text, how much it counts in fusion, and where it came from."""
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
