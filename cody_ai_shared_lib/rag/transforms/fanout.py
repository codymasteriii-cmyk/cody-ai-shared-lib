"""Secondary-query fan-out for comparisons (§7.8, AC-RAG-19).

Dummy example: "compare Seedance 1.0 and Kling 2.1". The router returns search_query "Seedance 1.0 vs Kling 2.1" and
secondary_queries ["Kling 2.1"] (and perhaps others). The primary search finds mostly Seedance; the secondary search makes sure
Kling passages are in the pool too, so the answer can compare both."""
from __future__ import annotations

from ..models import QueryIntent, RAGQueryRequest
from .base import QueryVariant, TransformContext


class SecondaryQueryFanOut:
    """One variant per secondary query, so the second subject of a comparison is searched too.

    Secondary variants weigh 0.8 against the primary's 1.0 in fusion: they widen the pool without outranking the main question.
    """
    name = "secondary_fanout"
    version = "1.0.0"

    def transform(self, request: RAGQueryRequest, intent: QueryIntent, ctx: TransformContext) -> list[QueryVariant]:
        """One variant per non-blank secondary query the router produced; none when the intent has none (no extra searches, no cost).

        >>> intent = QueryIntent(secondary_queries=["Kling 2.1", "  "])
        >>> [(v.text, v.weight, v.origin) for v in SecondaryQueryFanOut().transform(RAGQueryRequest(question="q"), intent, None)]
        [('Kling 2.1', 0.8, 'secondary')]
        """
        return [
            QueryVariant(text=secondary.strip(), weight=0.8, origin="secondary")
            for secondary in intent.secondary_queries
            if secondary.strip()
        ]
