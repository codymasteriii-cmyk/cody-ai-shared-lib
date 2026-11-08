"""Ports for post-retrieval stages and rerankers (§9, §9.3)."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models import RAGNode, ScoreKind


@runtime_checkable
class PostRetrievalStage(Protocol):
    """One named step applied to candidate nodes, in the order `profile.post_stages` lists."""
    name: str
    version: str
    accepts_score_kinds: frozenset[ScoreKind] | None

    def process(self, nodes: list[RAGNode[Any]], ctx: Any) -> list[RAGNode[Any]]:
        ...


@runtime_checkable
class Reranker(Protocol):
    name: str
    version: str

    def rerank(self, query: str, nodes: list[RAGNode[Any]], top_n: int, ctx: Any) -> list[RAGNode[Any]]:
        ...
