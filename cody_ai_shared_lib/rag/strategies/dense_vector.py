"""Tier 1 DenseVectorStrategy (§8.4).

Embeds query using gateway embedder and performs exact or ANN cosine vector search.
"""
from __future__ import annotations

from typing import ClassVar, Generic
from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy


class DenseVectorStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "dense_vector"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_vector_search"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        caps = ctx.store.capabilities() if callable(ctx.store.capabilities) else ctx.store.capabilities
        if not caps.supports_vector_search:
            raise StoreCapabilityError(
                "DenseVectorStrategy requires store capability 'supports_vector_search'"
            )

        query_text = ctx.intent.search_query or ctx.request.query
        emb_res = ctx.gateway.embed(
            texts=query_text,
            model=ctx.profile.models.embedding,
            stage="retrieval_embedding",
            usage_tracker=ctx.usage_tracker,
            task_type="RETRIEVAL_QUERY",
        )

        top_k = ctx.intent.suggested_top_k or ctx.profile.retrieval.top_k
        return ctx.store.vector_search(
            query_vector=emb_res.vectors[0],
            top_k=top_k,
            filters=ctx.filters,
        )
