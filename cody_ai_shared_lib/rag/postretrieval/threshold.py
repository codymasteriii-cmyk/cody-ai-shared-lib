"""Score threshold filter (§9.1, AC-RAG-31, AC-RAG-60).

Drops candidate nodes whose raw similarity score is strictly below min_similarity.
"""
from __future__ import annotations

from typing import Generic
from ..errors import ProfileConfigError
from ..models import MetaT, RAGNode

VALID_THRESHOLD_KINDS = frozenset(["cosine", "rerank", "bm25", "score"])


class ThresholdFilter(Generic[MetaT]):
    """Filters candidate nodes below a minimum similarity threshold."""

    def __init__(self, min_similarity: float, kind: str = "cosine"):
        if kind not in VALID_THRESHOLD_KINDS:
            raise ProfileConfigError(
                f"Invalid threshold kind '{kind}'. Must be one of: {sorted(VALID_THRESHOLD_KINDS)}"
            )
        self.min_similarity = min_similarity
        self.kind = kind

    def filter(self, nodes: list[RAGNode[MetaT]]) -> list[RAGNode[MetaT]]:
        """Keep only nodes with score is None (unscored) or score >= min_similarity.

        A threshold of kind="cosine" reads vector_score, never the fused score.
        A node with no value for the declared signal passes (§9.1, AC-RAG-31).
        """
        passed: list[RAGNode[MetaT]] = []
        for n in nodes:
            if self.kind == "cosine":
                if n.vector_score is not None:
                    val = n.vector_score
                elif n.score_kind in ("rrf", "bm25"):
                    # Keyword-only hit in hybrid/FTS has no cosine signal (§9.1, AC-RAG-31)
                    val = None
                else:
                    val = n.score
            elif self.kind == "rerank":
                val = n.rerank_score
            elif self.kind == "bm25":
                val = n.keyword_score
            else:
                val = n.score

            if val is None or val >= self.min_similarity:
                passed.append(n)
        return passed
