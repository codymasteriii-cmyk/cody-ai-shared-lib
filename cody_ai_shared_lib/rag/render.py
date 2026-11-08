"""NodeRenderer protocol and default implementation (§3.4).

Provides customized views of a node for embedding, reranking, prompt assembly,
and citations without altering node storage.
"""
from __future__ import annotations

from datetime import datetime
from typing import Generic, Protocol, runtime_checkable

from .models import MetaT, RAGNode


@runtime_checkable
class NodeRenderer(Protocol[MetaT]):
    """App-supplied views of a node."""

    def embed_text(self, node: RAGNode[MetaT]) -> str:
        """Text presented to embedding model."""
        ...

    def rerank_text(self, node: RAGNode[MetaT]) -> str:
        """Text presented to cross-encoder reranker."""
        ...

    def prompt_text(self, node: RAGNode[MetaT]) -> str:
        """Text inserted into context fence for model generation."""
        ...

    def citation_label(self, node: RAGNode[MetaT]) -> str:
        """Display label used for citations (e.g. section title or doc id)."""
        ...

    def breadcrumb(self, node: RAGNode[MetaT]) -> list[str]:
        """Heading hierarchy path for context assembly."""
        ...

    def format_meta(self, meta: MetaT) -> str:
        """Formatted metadata string rendered alongside node text."""
        ...

    def timestamp(self, meta: MetaT) -> datetime | None:
        """Publication or creation timestamp for recency scoring."""
        ...

    def features(self, meta: MetaT) -> dict[str, float]:
        """Numeric feature weights for score fusion stages."""
        ...

    def flags(self, meta: MetaT) -> list[str]:
        """Domain flags (e.g. 'requires_editorial_review')."""
        ...


class DefaultRenderer(Generic[MetaT]):
    """Standard fallback renderer using default node attributes."""

    def embed_text(self, node: RAGNode[MetaT]) -> str:
        return node.text

    def rerank_text(self, node: RAGNode[MetaT]) -> str:
        return node.text

    def prompt_text(self, node: RAGNode[MetaT]) -> str:
        return node.text

    def citation_label(self, node: RAGNode[MetaT]) -> str:
        return node.span.section_title or node.span.doc_id

    def breadcrumb(self, node: RAGNode[MetaT]) -> list[str]:
        return list(node.span.breadcrumb)

    def format_meta(self, meta: MetaT) -> str:
        return ""

    def timestamp(self, meta: MetaT) -> datetime | None:
        return None

    def features(self, meta: MetaT) -> dict[str, float]:
        return {}

    def flags(self, meta: MetaT) -> list[str]:
        return []
