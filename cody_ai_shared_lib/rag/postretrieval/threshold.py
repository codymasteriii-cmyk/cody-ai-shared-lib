"""Score threshold filter (§9.1, AC-RAG-31, AC-RAG-60).

Drops candidate nodes whose raw similarity score is strictly below min_similarity.
"""
from __future__ import annotations

from typing import Any, ClassVar, Generic
from ..errors import ProfileConfigError
from ..granularity import TRAIT_ORDERED, has_trait
from ..models import MetaT, RAGNode

VALID_THRESHOLD_KINDS = frozenset(["cosine", "rerank", "bm25", "score"])


class ThresholdFilter(Generic[MetaT]):
    """Filters candidate nodes below a minimum similarity threshold."""
    name: ClassVar[str] = "threshold"
    version: ClassVar[str] = "1.0.0"
    accepts_score_kinds: ClassVar[frozenset[str] | None] = None

    def __init__(self, min_similarity: float, kind: str = "cosine"):
        if kind not in VALID_THRESHOLD_KINDS:
            raise ProfileConfigError(
                f"Invalid threshold kind '{kind}'. Must be one of: {sorted(VALID_THRESHOLD_KINDS)}"
            )
        self.min_similarity = min_similarity
        self.kind = kind

    def process(self, nodes: list[RAGNode[MetaT]], ctx: Any) -> list[RAGNode[MetaT]]:
        return self.filter(nodes)

    def filter(self, nodes: list[RAGNode[MetaT]]) -> list[RAGNode[MetaT]]:
        """Keep ordered nodes, nodes without the declared signal, and nodes meeting the minimum (§9.1).

        A threshold of kind="cosine" reads vector_score, never the fused score (AC-RAG-31).
        """
        passed: list[RAGNode[MetaT]] = []
        for n in nodes:
            if has_trait(n.granularity, TRAIT_ORDERED):
                # Tier 0 text and local windows are not ranked by similarity (§9.1)
                passed.append(n)
                continue
            if self.kind == "cosine":
                if n.vector_score is not None:
                    val = n.vector_score
                else:
                    val = n.score if n.score_kind == "cosine" else None
            elif self.kind == "rerank":
                val = n.rerank_score
            elif self.kind == "bm25":
                val = n.keyword_score
            else:
                val = n.score

            if val is None or val >= self.min_similarity:
                passed.append(n)
        return passed
