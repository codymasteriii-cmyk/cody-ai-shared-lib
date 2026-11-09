"""Secondary-query fan-out for comparisons (§7.8, AC-RAG-19)."""
from __future__ import annotations

from ..models import QueryIntent, RAGQueryRequest
from .base import QueryVariant, TransformContext


class SecondaryQueryFanOut:
    """One variant per secondary query, so the second subject of a comparison is searched too."""
    name = "secondary_fanout"
    version = "1.0.0"

    def transform(self, request: RAGQueryRequest, intent: QueryIntent, ctx: TransformContext) -> list[QueryVariant]:
        return [
            QueryVariant(text=secondary.strip(), weight=0.8, origin="secondary")
            for secondary in intent.secondary_queries
            if secondary.strip()
        ]
