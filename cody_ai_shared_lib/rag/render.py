"""NodeRenderer protocol and default implementation (§3.4).

Provides customized views of a node for embedding, reranking, prompt assembly,
and citations without altering node storage.

The renderer is how an app teaches the library about its own data without the library knowing the schema. The same node
can be presented differently to each consumer:

    embed_text      what the embedding model reads      e.g. the title prefixed to the body, for better vectors
    rerank_text     what the reranker reads
    prompt_text     what the answering model reads      e.g. "Published 2026-03-02, Reuters: ..."
    citation_label  the name shown for a source         e.g. the article headline
    timestamp / features / flags   dates and numbers for recency and fusion stages

An app extends `DefaultRenderer` and overrides only what it needs. The engine checks a custom renderer at startup and refuses one
that left a method as an empty protocol stub, because that method would silently return None.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Generic, Protocol, runtime_checkable

from .models import MetaT, RAGNode


@runtime_checkable
class NodeRenderer(Protocol[MetaT]):
    """App-supplied views of a node.

    Extend `DefaultRenderer` and override only what you need. Do not subclass this protocol: the
    method bodies below are empty stubs, so any method you leave out would return None. The engine
    refuses such a renderer at startup (`renderer_problems`, §3.4, AC-RAG-111).

    Dummy example of a news renderer: `prompt_text` returns f"{published_date} {source}: {node.text}" and `timestamp` returns the
    article's date; every other method is inherited from `DefaultRenderer`.
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
    """Startup problems for an app-supplied renderer (§3.4, AC-RAG-111).

    >>> renderer_problems(DefaultRenderer())
    []
    """
    stubs = inherited_protocol_stubs(renderer, NodeRenderer, RENDERER_METHODS)
    if not stubs:
        return []
    return [
        f"Renderer {type(renderer).__name__} inherits the empty NodeRenderer stub for "
        f"{', '.join(stubs)}, which would return None; extend DefaultRenderer and override only "
        "what you need (§3.4)"
    ]


def metadata_value(meta: Any, field: str) -> Any:
    """A metadata field from a dict, an attribute or `extra`. Only None counts as missing, never False, 0 or "".

    >>> metadata_value({"active": False}, "active")
    False
    """
    if isinstance(meta, dict):
        return meta.get(field)
    value = getattr(meta, field, None)
    if value is None:
        extra = getattr(meta, "extra", None)
        value = extra.get(field) if isinstance(extra, dict) else None
    return value


_metadata_value = metadata_value       # the older private name, kept so existing imports keep working


def parse_timestamp(value: Any) -> datetime | None:
    """A datetime from a datetime, a date (midnight) or an ISO 8601 string (a trailing "Z" is UTC); else None.

    >>> parse_timestamp("2026-01-02T03:04:05Z").isoformat()
    '2026-01-02T03:04:05+00:00'
    >>> parse_timestamp("not a date") is None
    True
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


class DefaultRenderer(Generic[MetaT]):
    """Standard fallback renderer using default node attributes.

    Text methods return the node's own text; the label is the section title or the document id; `timestamp` and `features`
    look for common metadata names (created_at, published_at, confidence...) so simple apps need no renderer at all.
    """

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
            parsed = parse_timestamp(_metadata_value(meta, field))
            if parsed is not None:
                return parsed
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

