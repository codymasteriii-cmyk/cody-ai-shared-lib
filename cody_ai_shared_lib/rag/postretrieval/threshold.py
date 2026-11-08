"""Score threshold filter (§9.1, AC-RAG-31, AC-RAG-60).

Drops candidate nodes whose raw similarity score is strictly below min_similarity.
"""
from __future__ import annotations

from typing import Generic
from ..models import MetaT, RAGNode


class ThresholdFilter(Generic[MetaT]):
    """Filters candidate nodes below a minimum similarity threshold."""

    def __init__(self, min_similarity: float):
        self.min_similarity = min_similarity

    def filter(self, nodes: list[RAGNode[MetaT]]) -> list[RAGNode[MetaT]]:
        """Keep only nodes with score is None (unscored) or score >= min_similarity."""
        passed: list[RAGNode[MetaT]] = []
        for n in nodes:
            if n.score is None or n.score >= self.min_similarity:
                passed.append(n)
        return passed
