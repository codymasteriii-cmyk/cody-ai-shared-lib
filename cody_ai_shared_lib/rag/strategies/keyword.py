"""Keyword retrieval strategies (§8.3, AC-RAG-21)."""
from __future__ import annotations

from typing import ClassVar, Generic
from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy


class KeywordOnlyStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "keyword_only"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_keyword_fts"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        caps = ctx.store.capabilities() if callable(ctx.store.capabilities) else ctx.store.capabilities
        if not caps.supports_keyword_fts:
            raise StoreCapabilityError(
                "KeywordOnlyStrategy requires store capability 'supports_keyword_fts'"
            )
        query_text = ctx.intent.search_query or ctx.request.query
        target_limit = (
            ctx.request.top_k_override
            or ctx.intent.suggested_top_k
            or ctx.profile.retrieval.retrieve_max_results
            or ctx.profile.retrieval.top_k
        )
        return ctx.store.keyword_search(query_text, top_k=target_limit + 1, filters=ctx.filters)


class KeywordFirstStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "keyword_first"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_keyword_fts"])

    def __init__(self, min_hits: int = 3):
        self.min_hits = min_hits

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        query_text = ctx.intent.search_query or ctx.request.query
        top_k = ctx.intent.suggested_top_k or ctx.profile.retrieval.top_k
        hits = ctx.store.keyword_search(query_text, top_k=top_k, filters=ctx.filters)
        if len(hits) >= self.min_hits:
            return hits

        # Fallback to HybridRRF
        from .hybrid_rrf import HybridRRFStrategy
        return HybridRRFStrategy().retrieve(ctx)
