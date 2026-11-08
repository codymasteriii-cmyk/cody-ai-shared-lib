"""NodeRenderer protocol and default implementation (§3.4).

Provides customized views of a node for embedding, reranking, prompt assembly,
and citations without altering node storage.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Protocol, runtime_checkable

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


def _metadata_value(meta: Any, field: str) -> Any:
    if isinstance(meta, dict):
        return meta.get(field)
    value = getattr(meta, field, None)
    if value is None:
        value = (getattr(meta, "extra", None) or {}).get(field)
    return value


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
        """The first date-like metadata field, for apps that do not supply a renderer."""
        for field in ("created_at", "timestamp", "published_at", "date"):
            value = _metadata_value(meta, field)
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    continue
        return None

    def features(self, meta: MetaT) -> dict[str, float]:
        """Common numeric metadata fields, for apps that do not supply a renderer."""
        for field in ("confidence", "priority_score", "weight"):
            value = _metadata_value(meta, field)
            if isinstance(value, (int, float)):
                return {field: float(value)}
        return {}

    def flags(self, meta: MetaT) -> list[str]:
        return []

