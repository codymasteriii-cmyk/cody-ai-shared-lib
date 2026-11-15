"""BatchEmbedder: embeds nodes through ModelGateway with a zero-vector guard (§6.4, RAG-ING-4).

Two ways to embed, for two needs:

    embed_each / embed_each_with_reasons   judge every node on its own: return the good ones and list the failures
                                           (prepare_nodes uses this: an app wants the good rows even if one node failed)
    embed_nodes                            all or nothing: any failure raises before anything is written
                                           (ingest_document and sync use this: a half-written section must not exist)

A provider can return an empty or all-zero vector for text it could not embed. Such a vector would match everything or
nothing, so it is treated as a failed node, never stored.
"""
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
    """Embeds nodes in batches. A zero or missing vector is a failure and is never returned.

    Nodes are embedded from the renderer's `embed_text` (not the raw text), so an app controls what is embedded, e.g. a
    title prefixed to each passage. Batches are sized to stay within provider limits (`batch_size`).
    """

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
        embedded, failures = self.embed_each_with_reasons(nodes, usage_tracker, degradation_tracker, stage, budget_guard)
        return embedded, [node for node, _ in failures]

    def embed_each_with_reasons(
        self,
        nodes: list[RAGNode[MetaT]],
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker | None = None,
        stage: str = "ingest_embedding",
        budget_guard: BudgetGuard | None = None,
    ) -> tuple[list[tuple[RAGNode[MetaT], list[float]]], list[tuple[RAGNode[MetaT], str]]]:
        """Like embed_each, with the reason each failed node failed (§6.4.1).

        Vectors come back in input order, so a node's vector is found by position. If a whole batch call fails the gateway
        raises (`EmbeddingProviderError`); only a returned-but-unusable vector is a per-node failure here.
        """
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
        failures: list[tuple[RAGNode[MetaT], str]] = []
        for index, node in enumerate(nodes):
            vector = vectors[index] if index < len(vectors) else []
            if not vector:
                failures.append((node, "the provider returned no vector"))
            elif is_zero_vector(vector):
                failures.append((node, "the provider returned a zero vector"))
            else:
                embedded.append((node, vector))
        return embedded, failures

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
