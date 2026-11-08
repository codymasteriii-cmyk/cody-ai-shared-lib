"""Tier 1 DenseVectorStrategy (§8.4)."""
from __future__ import annotations

from typing import ClassVar, Generic

from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from ..vectors import is_zero_vector
from .context import RetrievalContext, RetrievalStrategy
from .shared import embed_query, keyword_fallback, with_local_window


class DenseVectorStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "dense_vector"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_vector_search"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        if not self.is_runnable(ctx.store.capabilities()):
            raise StoreCapabilityError("DenseVectorStrategy requires store capability 'supports_vector_search'")

        top_k = ctx.target_limit() + 1
        query_vector = embed_query(ctx)
        if is_zero_vector(query_vector):
            return keyword_fallback(ctx, top_k)

        results = ctx.store.vector_search(query_vector=query_vector, top_k=top_k, filters=ctx.filters)
        return with_local_window(ctx, results)
