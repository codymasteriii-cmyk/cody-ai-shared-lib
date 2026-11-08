"""BatchEmbedder for embedding RAG nodes and text through ModelGateway (§6.4).

Implements:
  - Batching and token usage tracking via ModelGateway.embed
  - RAG-ING-4: Zero-vector guard (skips failed nodes with ERROR log and DG-08;
    raises EmbeddingProviderError if all nodes in a non-empty set fail).
"""
from __future__ import annotations

import logging
from typing import Generic, TypeVar

from ..degradation import DegradationTracker
from ..errors import EmbeddingProviderError
from ..gateway import ModelGateway
from ..models import MetaT, RAGNode
from ..render import DefaultRenderer, NodeRenderer
from ..usage import BudgetGuard, UsageTracker
from ..vectors import is_zero_vector

logger = logging.getLogger("shared-rag")

T = TypeVar("T")


class BatchEmbedder(Generic[MetaT]):
    """Embeds nodes or raw texts with batching, usage tracking, and zero-vector safety."""

    def __init__(
        self,
        gateway: ModelGateway,
        model: str,
        batch_size: int = 50,
        renderer: NodeRenderer | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        self.gateway = gateway
        self.model = model
        self.batch_size = batch_size
        self.renderer = renderer or DefaultRenderer()
        self.task_type = task_type

    def embed_nodes(
        self,
        nodes: list[RAGNode[MetaT]],
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker | None = None,
        stage: str = "ingest_embedding",
        budget_guard: BudgetGuard | None = None,
    ) -> list[tuple[RAGNode[MetaT], list[float]]]:
        """Embed nodes and return valid (node, vector) pairs.

        Guarantees that zero vectors are never returned (RAG-ING-4).
        If every node in a non-empty set has a zero vector, raises EmbeddingProviderError.
        If some nodes are zero, skips them, records DG-08, and logs ERROR.
        """
        if not nodes:
            return []

        embed_texts = [self.renderer.embed_text(n) for n in nodes]
        all_vectors: list[list[float]] = []

        for i in range(0, len(embed_texts), self.batch_size):
            batch = embed_texts[i : i + self.batch_size]
            emb_res = self.gateway.embed(
                texts=batch,
                model=self.model,
                stage=stage,
                usage_tracker=usage_tracker,
                degradation_tracker=degradation_tracker,
                budget_guard=budget_guard,
                task_type=self.task_type,
            )
            all_vectors.extend(emb_res.vectors)

        zero_indices: list[int] = []
        valid_pairs: list[tuple[RAGNode[MetaT], list[float]]] = []

        for idx, (node, vec) in enumerate(zip(nodes, all_vectors)):
            if is_zero_vector(vec):
                zero_indices.append(idx)
            else:
                valid_pairs.append((node, vec))

        if len(zero_indices) == len(nodes):
            raise EmbeddingProviderError(
                f"All {len(nodes)} embedding vectors were zero from model '{self.model}'",
                stage=stage,
            )

        if zero_indices:
            logger.error(
                f"[BatchEmbedder] {len(zero_indices)} of {len(nodes)} nodes had zero vectors and were skipped [DG-08]."
            )
            if degradation_tracker is not None:
                degradation_tracker.record(
                    dg_id="DG-08",
                    stage=stage,
                    reason=f"{len(zero_indices)} nodes returned zero vectors from embedder",
                    fallback_used="skipped_zero_vectors",
                    impact=f"Skipped {len(zero_indices)} unindexed nodes; continuing ingestion",
                )

        return valid_pairs
