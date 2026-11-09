"""Tier 2 HybridRRFStrategy (§8.5, AC-RAG-20).

Combines dense vector and keyword search with Reciprocal Rank Fusion, through the store's native
hybrid search (Path A) or by fusing the two legs in the engine (Path B).
"""
from __future__ import annotations

from typing import Any, ClassVar, Generic

from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from ..vectors import is_zero_vector
from .context import RetrievalContext, RetrievalStrategy
from .shared import embed_query, keyword_fallback, query_text, query_token_groups, with_local_window


class HybridRRFStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "hybrid_rrf"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_vector_search", "supports_keyword_fts"])

    @classmethod
    def is_runnable(cls, capabilities: Any) -> bool:
        return capabilities.supports_vector_search and (
            capabilities.supports_keyword_fts or capabilities.supports_native_hybrid_rpc
        )

    @staticmethod
    def _fusion_weights(ctx: RetrievalContext[MetaT]) -> tuple[float, float]:
        """The first intent facet with a table row wins (§8.5)."""
        weights = ctx.profile.retrieval.fusion_weights
        if ctx.intent.exact_terms:
            return weights.get("exact_terms", (0.7, 1.3))
        if ctx.intent.negation:
            return weights.get("negation", (0.7, 1.3))
        if ctx.intent.entity_terms:
            return weights.get("entity_terms", (0.9, 1.1))
        return weights.get("default", (1.0, 1.0))

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        caps = ctx.store.capabilities()
        if not self.is_runnable(caps):
            raise StoreCapabilityError(
                "HybridRRFStrategy requires 'supports_vector_search' and keyword or native hybrid search"
            )

        top_k = ctx.candidate_limit()
        query_vector = embed_query(ctx)
        if is_zero_vector(query_vector):
            return keyword_fallback(ctx, top_k)

        if caps.supports_native_hybrid_rpc:
            w_vec, w_fts = self._fusion_weights(ctx)
            results = ctx.store.hybrid_search(
                query_text=query_text(ctx),
                query_vector=query_vector,
                top_k=top_k,
                filters=ctx.filters,
                rrf_k=ctx.profile.retrieval.rrf_k,
                w_vec=w_vec,
                w_fts=w_fts,
                analyzer_tokens=query_token_groups(ctx),
            )
        else:
            results = self._fuse_in_engine(ctx, query_vector, top_k)
        return with_local_window(ctx, results)

    def _fuse_in_engine(
        self,
        ctx: RetrievalContext[MetaT],
        query_vector: list[float],
        top_k: int,
    ) -> list[RAGNode[MetaT]]:
        """Path B: run both legs, in parallel when a pool is available, then apply RRF."""
        def vector_leg() -> list[RAGNode[MetaT]]:
            return ctx.store.vector_search(query_vector, top_k=top_k * 2, filters=ctx.filters)

        def keyword_leg() -> list[RAGNode[MetaT]]:
            return ctx.store.keyword_search(
                query_text(ctx),
                top_k=top_k * 2,
                filters=ctx.filters,
                analyzer_tokens=query_token_groups(ctx),
            )

        if ctx.pool is not None:
            vector_future = ctx.pool.submit(vector_leg)
            keyword_future = ctx.pool.submit(keyword_leg)
            vector_nodes, keyword_nodes = vector_future.result(), keyword_future.result()
        else:
            vector_nodes, keyword_nodes = vector_leg(), keyword_leg()

        rrf_k = ctx.profile.retrieval.rrf_k
        w_vec, w_fts = self._fusion_weights(ctx)
        scores: dict[str, float] = {}
        nodes: dict[str, RAGNode[MetaT]] = {}

        for rank, node in enumerate(vector_nodes, start=1):
            scores[node.node_id] = scores.get(node.node_id, 0.0) + w_vec / (rrf_k + rank)
            nodes[node.node_id] = node
        for rank, node in enumerate(keyword_nodes, start=1):
            scores[node.node_id] = scores.get(node.node_id, 0.0) + w_fts / (rrf_k + rank)
            if node.node_id not in nodes:
                nodes[node.node_id] = node
            else:
                nodes[node.node_id].keyword_score = node.keyword_score
                nodes[node.node_id].keyword_rank = node.keyword_rank

        ranked = sorted(scores, key=lambda node_id: (-scores[node_id], node_id))
        results = []
        for node_id in ranked[:top_k]:
            node = nodes[node_id]
            node.score = scores[node_id]
            node.score_kind = "rrf"
            results.append(node)
        return results
