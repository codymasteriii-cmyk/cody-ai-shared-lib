"""Ports for post-retrieval stages and rerankers (§9, §9.3).

After a strategy has found candidates, the pipeline runs the stages the profile lists in `post_stages`, in order.
Each takes the node list and returns a new one: drop some (threshold, dedup), reorder (rerank, boost), or cut (top_k).

    strategy -> [threshold] -> [dedup] -> [rerank] -> [boost] -> [top_k] -> nodes for the prompt

Every stage must leave "ordered layers" (Tier 0 text, summaries, windows) alone: only ranked passages compete on
score. `granularity.apply_to_ranked` and `is_ranked` are the helpers that enforce that.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models import RAGNode, ScoreKind


@runtime_checkable
class PostRetrievalStage(Protocol):
    """One named step applied to candidate nodes, in the order `profile.post_stages` lists.

    To add a stage: write a class with these attributes, register it under "stage" in the component registry, and
    name it in the profile. `accepts_score_kinds` says which score kinds it understands (None: all).
    """
    name: str
    version: str
    accepts_score_kinds: frozenset[ScoreKind] | None

    def process(self, nodes: list[RAGNode[Any]], ctx: Any) -> list[RAGNode[Any]]:
        ...


@runtime_checkable
class Reranker(Protocol):
    """A model that re-scores candidates against the query: slower and more exact than the first search."""
    name: str
    version: str

    def rerank(self, query: str, nodes: list[RAGNode[Any]], top_n: int, ctx: Any) -> list[RAGNode[Any]]:
        ...
