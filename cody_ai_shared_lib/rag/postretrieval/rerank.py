"""Cross-encoder reranking post-retrieval stage (§9.3, AC-RAG-34).

Only reranks nodes whose granularity carries TRAIT_RERANKABLE, preserving
non-rerankable (e.g., ordered local window or parent summary) nodes in place.
Sets node.rerank_score, node.score, and node.score_kind = 'rerank'.
"""
from __future__ import annotations

from typing import Any, ClassVar, Generic
from ..degradation import DegradationTracker
from ..gateway import ModelGateway
from ..granularity import TRAIT_RERANKABLE, has_trait
from ..models import MetaT, RAGNode
from ..ports import NodeRenderer
from ..render import DefaultRenderer
from ..usage import BudgetGuard, UsageTracker


class JinaCrossEncoderReranker(Generic[MetaT]):
    """Reranks candidate nodes with Jina Rerank API with DG-06 degradation on failure."""
    name: ClassVar[str] = "jina"
    version: ClassVar[str] = "1.0.0"

    def __init__(
        self,
        gateway: ModelGateway,
        model: str = "jina-reranker",
        renderer: NodeRenderer[MetaT] | None = None,
        fallback_action: str = "keep_order",
    ):
        self.gateway = gateway
        self.model = model
        self.renderer = renderer or DefaultRenderer()
        self.fallback_action = fallback_action

    def rerank(self, query: str, nodes: list[RAGNode[MetaT]], top_n: int, ctx: Any) -> list[RAGNode[MetaT]]:
        """Reranker port (§9.3)."""
        return self.rerank_nodes(
            query,
            nodes,
            ctx.usage_tracker,
            ctx.degradation_tracker,
            top_n=None if len(nodes) <= top_n else top_n,
            budget_guard=ctx.budget_guard,
            request_id=ctx.usage_tracker.request_id,
            tenant_or_user_id=ctx.usage_tracker.tenant_or_user_id,
        )

    def rerank_nodes(
        self,
        query: str,
        nodes: list[RAGNode[MetaT]],
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
        top_n: int | None = None,
        budget_guard: BudgetGuard | None = None,
        request_id: str | None = None,
        tenant_or_user_id: str | None = None,
    ) -> list[RAGNode[MetaT]]:
        if not nodes:
            return []

        # Partition into rerankable vs non-rerankable (e.g., ordered local window / parent summaries)
        rerankable: list[RAGNode[MetaT]] = []
        non_rerankable: list[RAGNode[MetaT]] = []
        for n in nodes:
            if has_trait(n.granularity, TRAIT_RERANKABLE):
                rerankable.append(n)
            else:
                non_rerankable.append(n)

        if not rerankable:
            return nodes[:top_n] if top_n else nodes

        doc_texts = [
            self.renderer.rerank_text(n) if hasattr(self.renderer, "rerank_text") else n.text
            for n in rerankable
        ]
        res = self.gateway.rerank(
            query=query,
            documents=doc_texts,
            model=self.model,
            stage="post_retrieval_rerank",
            usage_tracker=usage_tracker,
            degradation_tracker=degradation_tracker,
            budget_guard=budget_guard,
            fallbacks={"rerank": self.fallback_action},
            request_id=request_id or "",
            tenant_or_user_id=tenant_or_user_id,
            top_n=top_n or len(rerankable),
        )

        if res is None:
            # DG-06 fallback: keep incoming order
            if not any(d.id == "DG-06" for d in degradation_tracker.degradations):
                degradation_tracker.record(
                    dg_id="DG-06",
                    stage="post_retrieval_rerank",
                    reason="Reranker failed or unavailable",
                    fallback_used="keep_incoming_order",
                    impact="Candidate nodes were not reranked",
                )
            return nodes[:top_n] if top_n else nodes

        # Map reranked results back to rerankable nodes
        reranked_nodes: list[RAGNode[MetaT]] = []
        for item in res.results:
            idx = item.index
            if 0 <= idx < len(rerankable):
                node = rerankable[idx]
                node.rerank_score = item.relevance_score
                node.score = item.relevance_score
                node.score_kind = "rerank"
                reranked_nodes.append(node)

        combined = non_rerankable + reranked_nodes
        return combined[:top_n] if top_n else combined
