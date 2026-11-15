"""Tier 1 DenseVectorStrategy (§8.4): embed the question, then search the store's vectors for the closest passages."""
from __future__ import annotations

from typing import ClassVar, Generic

from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from ..vectors import is_zero_vector
from .context import RetrievalContext, RetrievalStrategy
from .shared import embed_query, keyword_fallback, ranked_store_call, with_local_window


class DenseVectorStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    """Vector search only. Good for meaning ("a trip abroad"), weaker for exact names; see `hybrid_rrf` for both."""
    name: ClassVar[str] = "dense_vector"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_vector_search"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        if not self.is_runnable(ctx.store.capabilities()):
            raise StoreCapabilityError("DenseVectorStrategy requires store capability 'supports_vector_search'")

        top_k = ctx.candidate_limit()
        query_vector = embed_query(ctx)
        # An empty or all-zero vector means the embedding failed: fall back to keyword search if the app allowed it (DG-07)
        if is_zero_vector(query_vector):
            return keyword_fallback(ctx, top_k)

        results = ranked_store_call(
            ctx, top_k, lambda filters, _groups, k: ctx.store.vector_search(query_vector=query_vector, top_k=k, filters=filters)
        )
        return with_local_window(ctx, results)
