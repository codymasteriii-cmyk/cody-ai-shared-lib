"""Tier 2 HybridRRFStrategy (§8.5, AC-RAG-20).

Combines dense vector search and keyword full-text search with Reciprocal Rank Fusion.
"""
from __future__ import annotations

from typing import ClassVar, Generic
from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy


class HybridRRFStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "hybrid_rrf"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_vector_search", "supports_keyword_fts"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        query_text = ctx.intent.search_query or ctx.request.query
        target_limit = (
            ctx.request.top_k_override
            or ctx.intent.suggested_top_k
            or ctx.profile.retrieval.retrieve_max_results
            or ctx.profile.retrieval.top_k
        )
        top_k = target_limit + 1
        rrf_k = ctx.profile.retrieval.rrf_k

        # 1. Embed query
        emb_res = ctx.gateway.embed(
            texts=query_text,
            model=ctx.profile.models.embedding,
            stage="retrieval_embedding",
            usage_tracker=ctx.usage_tracker,
            task_type="RETRIEVAL_QUERY",
        )
        query_vector = emb_res.vectors[0]

        caps = ctx.store.capabilities() if callable(ctx.store.capabilities) else ctx.store.capabilities
        # 2. Native hybrid if supported
        if caps.supports_native_hybrid:
            return ctx.store.hybrid_search(
                query_text=query_text,
                query_vector=query_vector,
                top_k=top_k,
                filters=ctx.filters,
                rrf_k=rrf_k,
            )

        # 3. Client-side RRF fusion
        vec_nodes = ctx.store.vector_search(query_vector, top_k=top_k * 2, filters=ctx.filters)
        text_nodes = ctx.store.keyword_search(query_text, top_k=top_k * 2, filters=ctx.filters)

        rrf_scores: dict[str, float] = {}
        node_map: dict[str, RAGNode[MetaT]] = {}

        for rank, node in enumerate(vec_nodes):
            rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + (1.0 / (rrf_k + rank + 1))
            node_map[node.id] = node

        for rank, node in enumerate(text_nodes):
            rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + (1.0 / (rrf_k + rank + 1))
            if node.id not in node_map:
                node_map[node.id] = node

        # Sort by merged RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda nid: rrf_scores[nid], reverse=True)
        results: list[RAGNode[MetaT]] = []
        for nid in sorted_ids[:top_k]:
            n = node_map[nid]
            n.score = rrf_scores[nid]
            n.score_kind = "hybrid_rrf"
            results.append(n)

        return results
