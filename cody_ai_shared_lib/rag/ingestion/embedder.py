"""BatchEmbedder: embeds nodes through ModelGateway with a zero-vector guard (§6.4, RAG-ING-4)."""
from __future__ import annotations

from typing import Generic

from ..degradation import DegradationTracker
from ..errors import EmbeddingProviderError
from ..gateway import ModelGateway
from ..models import MetaT, RAGNode
from ..render import DefaultRenderer, NodeRenderer
from ..usage import BudgetGuard, UsageTracker
from ..vectors import is_zero_vector


class BatchEmbedder(Generic[MetaT]):
    """Embeds nodes in batches. A zero or missing vector is a failure and is never returned."""

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

    def embed_each(
        self,
        nodes: list[RAGNode[MetaT]],
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker | None = None,
        stage: str = "ingest_embedding",
        budget_guard: BudgetGuard | None = None,
    ) -> tuple[list[tuple[RAGNode[MetaT], list[float]]], list[RAGNode[MetaT]]]:
        """Return (embedded pairs, failed nodes), judging every node on its own (§6.4.1)."""
        vectors: list[list[float]] = []
        texts = [self.renderer.embed_text(node) for node in nodes]
        for start in range(0, len(texts), self.batch_size):
            result = self.gateway.embed(
                texts=texts[start : start + self.batch_size],
                model=self.model,
                stage=stage,
                usage_tracker=usage_tracker,
                degradation_tracker=degradation_tracker,
                budget_guard=budget_guard,
                task_type=self.task_type,
            )
            vectors.extend(result.vectors)

        embedded: list[tuple[RAGNode[MetaT], list[float]]] = []
        failed: list[RAGNode[MetaT]] = []
        for index, node in enumerate(nodes):
            vector = vectors[index] if index < len(vectors) else []
            if is_zero_vector(vector):
                failed.append(node)
            else:
                embedded.append((node, vector))
        return embedded, failed

    def embed_nodes(
        self,
        nodes: list[RAGNode[MetaT]],
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker | None = None,
        stage: str = "ingest_embedding",
        budget_guard: BudgetGuard | None = None,
    ) -> list[tuple[RAGNode[MetaT], list[float]]]:
        """Embed one unit all-or-nothing: any failure raises before anything is written (RAG-ING-4)."""
        embedded, failed = self.embed_each(nodes, usage_tracker, degradation_tracker, stage, budget_guard)
        if failed:
            raise EmbeddingProviderError(
                f"{len(failed)} of {len(nodes)} nodes failed to embed with model '{self.model}'",
                failed_node_ids=[node.node_id for node in failed],
                stage=stage,
            )
        return embedded
