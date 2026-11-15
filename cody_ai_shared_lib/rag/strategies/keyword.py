"""Keyword retrieval strategies (§8.3, AC-RAG-21)."""
from __future__ import annotations

from typing import ClassVar, Generic

from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy
from .hybrid_rrf import HybridRRFStrategy
from .shared import query_text, ranked_store_call


def _keyword_hits(ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
    return ranked_store_call(
        ctx,
        ctx.candidate_limit(),
        lambda filters, groups, k: ctx.store.keyword_search(
            query_text(ctx), top_k=k, filters=filters, analyzer_tokens=groups
        ),
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

    def __init__(self, min_hits: int | None = None):
        self.min_hits = min_hits                  # None: profile.retrieval.keyword_first_min_hits

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        caps = ctx.store.capabilities()
        if not self.is_runnable(caps):
            raise StoreCapabilityError("KeywordFirstStrategy requires store capability 'supports_keyword_fts'")

        min_hits = self.min_hits if self.min_hits is not None else ctx.profile.retrieval.keyword_first_min_hits
        hits = _keyword_hits(ctx)
        if len(hits) >= min_hits or not HybridRRFStrategy.is_runnable(caps):
            return hits
        reason = f"keyword_first: {len(hits)} keyword hits < min_hits {min_hits}; fell back to hybrid_rrf"
        ctx.last_tier_reason = f"{ctx.last_tier_reason}; {reason}" if ctx.last_tier_reason else reason
        return HybridRRFStrategy().retrieve(ctx)
