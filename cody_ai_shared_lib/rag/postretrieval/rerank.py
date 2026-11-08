"""Cross-encoder reranking post-retrieval stage (§9.4, AC-RAG-34)."""
from __future__ import annotations

from typing import Generic
from ..degradation import DegradationTracker
from ..gateway import ModelGateway
from ..models import MetaT, RAGNode
from ..usage import UsageTracker


class JinaCrossEncoderReranker(Generic[MetaT]):
    """Reranks candidate nodes with Jina Rerank API with DG-06 degradation on failure."""

    def __init__(self, gateway: ModelGateway, model: str = "jina-reranker"):
        self.gateway = gateway
        self.model = model

    def rerank(
        self,
        query: str,
        nodes: list[RAGNode[MetaT]],
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
        top_n: int | None = None,
    ) -> list[RAGNode[MetaT]]:
        if not nodes:
            return []

        doc_texts = [n.text for n in nodes]
        try:
            res = self.gateway.rerank(
                query=query,
                documents=doc_texts,
                model=self.model,
                stage="post_retrieval_rerank",
                usage_tracker=usage_tracker,
                degradation_tracker=degradation_tracker,
                top_n=top_n or len(nodes),
            )
        except Exception:
            res = None

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

        # Map reranked results back to nodes
        reranked_nodes: list[RAGNode[MetaT]] = []
        for item in res.results:
            idx = item.index
            if 0 <= idx < len(nodes):
                node = nodes[idx]
                node.score = item.relevance_score
                node.score_kind = "cross_encoder"
                reranked_nodes.append(node)

        return reranked_nodes
