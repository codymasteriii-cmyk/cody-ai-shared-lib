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
    """App-supplied views of a node.

    Extend `DefaultRenderer` and override only what you need. Do not subclass this protocol: the
    method bodies below are empty stubs, so any method you leave out would return None. The engine
    refuses such a renderer at startup (`renderer_problems`, §3.4, AC-RAG-111).
    """

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


RENDERER_METHODS = (
    "embed_text", "rerank_text", "prompt_text", "citation_label", "breadcrumb",
    "format_meta", "timestamp", "features", "flags",
)


def inherited_protocol_stubs(obj: Any, protocol: type, names: tuple[str, ...]) -> list[str]:
    """Names of `names` that `obj` gets from `protocol`'s own empty bodies rather than implementing.

    A class that subclasses a runtime-checkable protocol passes `isinstance` even when it implements
    nothing, and each inherited `...` body returns None. Comparing the function objects finds exactly
    those methods; a duck-typed class or a `DefaultRenderer` subclass never matches.
    """
    cls = type(obj)
    return [name for name in names if name in vars(protocol) and getattr(cls, name, None) is vars(protocol)[name]]


def renderer_problems(renderer: Any) -> list[str]:
    """Startup problems for an app-supplied renderer (§3.4, AC-RAG-111)."""
    stubs = inherited_protocol_stubs(renderer, NodeRenderer, RENDERER_METHODS)
    if not stubs:
        return []
    return [
        f"Renderer {type(renderer).__name__} inherits the empty NodeRenderer stub for "
        f"{', '.join(stubs)}, which would return None; extend DefaultRenderer and override only "
        "what you need (§3.4)"
    ]


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

