"""Keyword retrieval strategies (§8.3, AC-RAG-21)."""
from __future__ import annotations

from typing import ClassVar, Generic

from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy
from .hybrid_rrf import HybridRRFStrategy
from .shared import query_text, query_token_groups


def _keyword_hits(ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
    return ctx.store.keyword_search(
        query_text(ctx),
        top_k=ctx.target_limit() + 1,
        filters=ctx.filters,
        analyzer_tokens=query_token_groups(ctx),
    )


class KeywordOnlyStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "keyword_only"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_keyword_fts"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        if not self.is_runnable(ctx.store.capabilities()):
            raise StoreCapabilityError("KeywordOnlyStrategy requires store capability 'supports_keyword_fts'")
        return _keyword_hits(ctx)


class KeywordFirstStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "keyword_first"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_keyword_fts"])

    def __init__(self, min_hits: int = 3):
        self.min_hits = min_hits

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        caps = ctx.store.capabilities()
        if not self.is_runnable(caps):
            raise StoreCapabilityError("KeywordFirstStrategy requires store capability 'supports_keyword_fts'")

        hits = _keyword_hits(ctx)
        if len(hits) >= self.min_hits or not HybridRRFStrategy.is_runnable(caps):
            return hits
        return HybridRRFStrategy().retrieve(ctx)
