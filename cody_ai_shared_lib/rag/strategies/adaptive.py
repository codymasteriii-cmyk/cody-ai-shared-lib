"""Tier ladder AdaptiveStrategy (§8.9, AC-RAG-10, AC-RAG-20)."""
from __future__ import annotations

from typing import ClassVar, Generic
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy
from .dense_vector import DenseVectorStrategy
from .full_context import FullContextStrategy
from .hierarchical import HierarchicalStrategy
from .hybrid_rrf import HybridRRFStrategy
from .keyword import KeywordFirstStrategy, KeywordOnlyStrategy


class AdaptiveStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    """Selects optimal retrieval tier based on scope statistics, query intent, and store capabilities."""
    name: ClassVar[str] = "adaptive"
    version: ClassVar[str] = "1.0.0"

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        # Rule 1: has_strategy_override (§8.9)
        if ctx.request.strategy_override:
            override = ctx.request.strategy_override
            ctx.request.metadata = dict(ctx.request.metadata or {})
            ctx.request.metadata["strategy_used"] = override
            ctx.request.metadata["tier_decision_reason"] = f"override: {override}"
            if override == "hierarchical":
                return HierarchicalStrategy().retrieve(ctx)
            if override == "hybrid_rrf":
                return HybridRRFStrategy().retrieve(ctx)
            if override == "full_context":
                return FullContextStrategy().retrieve(ctx)
            if override == "dense_vector":
                return DenseVectorStrategy().retrieve(ctx)
            if override == "keyword_first":
                return KeywordFirstStrategy().retrieve(ctx)
            if override == "keyword_only":
                return KeywordOnlyStrategy().retrieve(ctx)

        # Rule 2: route_no_retrieval (§8.9)
        if ctx.intent.route == "direct":
            ctx.request.metadata = dict(ctx.request.metadata or {})
            ctx.request.metadata["strategy_used"] = "none"
            ctx.request.metadata["tier_decision_reason"] = "route_no_retrieval: direct route"
            return []

        stats = ctx.store.scope_stats(ctx.filters)
        max_t0 = ctx.profile.retrieval.full_context_max_tokens
        caps = ctx.store.capabilities() if callable(ctx.store.capabilities) else ctx.store.capabilities

        # Rule 3: Tier 0 Full Context Gate (AC-RAG-10, AC-RAG-11)
        if (
            max_t0 > 0
            and stats.total_tokens > 0
            and stats.total_tokens <= max_t0
            and caps.supports_full_scope_fetch
        ):
            ctx.request.metadata = dict(ctx.request.metadata or {})
            ctx.request.metadata["strategy_used"] = "full_context"
            ctx.request.metadata["tier_decision_reason"] = (
                f"full_context: {stats.total_tokens} <= {max_t0} tokens"
            )
            return FullContextStrategy().retrieve(ctx)

        # Rule 2: Tier 3 Hierarchical Strategy (AC-RAG-22)
        if (
            (ctx.intent.scope == "document_overview" or ctx.profile.retrieval.always_include_summaries)
            and caps.supports_parent_summaries
        ):
            ctx.request.metadata = dict(ctx.request.metadata or {})
            ctx.request.metadata["strategy_used"] = "hierarchical"
            ctx.request.metadata["tier_decision_reason"] = "hierarchical: overview query or always_include_summaries"
            return HierarchicalStrategy().retrieve(ctx)

        # Rule 3: Exact Terms -> Keyword First
        if ctx.intent.exact_terms and caps.supports_keyword_fts:
            ctx.request.metadata = dict(ctx.request.metadata or {})
            ctx.request.metadata["strategy_used"] = "keyword_first"
            ctx.request.metadata["tier_decision_reason"] = "keyword_first: exact terms in query"
            return KeywordFirstStrategy().retrieve(ctx)

        # Rule 4: Hybrid RRF (Tier 2)
        if caps.supports_vector_search and caps.supports_keyword_fts:
            ctx.request.metadata = dict(ctx.request.metadata or {})
            ctx.request.metadata["strategy_used"] = "hybrid_rrf"
            ctx.request.metadata["tier_decision_reason"] = (
                f"hybrid_rrf: scope {stats.total_tokens} tokens exceeded Tier 0 gate {max_t0}"
            )
            return HybridRRFStrategy().retrieve(ctx)

        # Rule 5: Dense Vector (Tier 1)
        if caps.supports_vector_search:
            ctx.request.metadata = dict(ctx.request.metadata or {})
            ctx.request.metadata["strategy_used"] = "dense_vector"
            ctx.request.metadata["tier_decision_reason"] = "dense_vector: store supports vector search"
            return DenseVectorStrategy().retrieve(ctx)

        # Rule 6: Keyword Only
        if caps.supports_keyword_fts:
            ctx.request.metadata = dict(ctx.request.metadata or {})
            ctx.request.metadata["strategy_used"] = "keyword_only"
            ctx.request.metadata["tier_decision_reason"] = "keyword_only: keyword store available"
            return KeywordOnlyStrategy().retrieve(ctx)

        # Default fallback: DenseVector
        ctx.request.metadata = dict(ctx.request.metadata or {})
        ctx.request.metadata["strategy_used"] = "dense_vector"
        return DenseVectorStrategy().retrieve(ctx)
