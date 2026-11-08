"""Storage port and Guard protocol contracts (§4, §5).

The engine reaches persistent storage only through RAGStore[MetaT].
The library contains no SQL, table name, or database client (D-01).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, get_origin, runtime_checkable
from pydantic import BaseModel, Field

from .errors import MissingRequiredScopeError, UnsupportedFilterError
from .models import FilterPatch, MetaT, RAGFilterSpec, RAGNode
from .render import NodeRenderer
from .text.analyzer import TokenGroup


# ── Storage Capabilities & Signatures (§4.1, §4.4) ─────────────────────────────

class EmbeddingSpace(BaseModel):
    """Configuration signature of the vector embedding space."""
    model: str
    dims: int
    task_type_documents: str = "RETRIEVAL_DOCUMENT"
    splitter_version: str | None = "1.0.0"


class ScopeStats(BaseModel):
    """Aggregate statistics for documents and nodes matching a filter specification."""
    doc_count: int
    node_count: int
    char_count: int
    est_tokens: int | None = None
    revision: str
    has_summaries: bool = False

    @property
    def total_tokens(self) -> int:
        return self.est_tokens if self.est_tokens is not None else self.char_count


class StoreCapabilities(BaseModel):
    """Declared capabilities and query features supported by a RAGStore adapter."""
    supports_vector_search: bool = True
    supports_keyword_fts: bool = False
    supports_native_hybrid_rpc: bool = False
    returns_raw_scores: bool = False
    uses_analyzer_tokens: bool = False
    returns_snippets: bool = False
    supports_parent_summaries: bool = False
    supports_summary_levels: bool = False
    supports_local_window: bool = False
    supports_full_scope_fetch: bool = False
    supports_sequential_boundary: bool = False
    supports_hydration: bool = False
    supports_fetch_nodes: bool = False
    supports_fingerprints: bool = False
    supports_delete: bool = False
    supports_revisions: bool = False
    supports_related: bool = False
    supports_links: bool = False
    supports_graph_traversal: bool = False
    supports_embedding_return: bool = False

    required_filter_keys: frozenset[str] = frozenset()
    supported_custom_filters: dict[str, Any] = Field(default_factory=dict)
    supported_date_fields: frozenset[str] = frozenset()
    time_basis: str | None = None
    max_top_k: int = 200


class Fingerprint(BaseModel):
    """Ingestion fingerprint for node change detection and cache reconciliation."""
    content_hash: str
    embedding_signature: str
    fts_signature: str | None = None


_FILTER_KEY_ALIASES = {"tenant_id": "tenant_or_user_id", "document_ids": "doc_ids", "doc_id": "doc_ids"}


def validate_scope(filters: RAGFilterSpec, capabilities: StoreCapabilities) -> None:
    """Fail closed on missing required keys and on unknown or wrongly typed custom filters (§3.6, §5.6)."""
    missing = []
    for key in capabilities.required_filter_keys:
        value = getattr(filters, _FILTER_KEY_ALIASES.get(key, key), None)
        if value is None:
            value = filters.custom.get(key)
        if value is None or value in ("", [], {}):
            missing.append(key)
    if missing:
        raise MissingRequiredScopeError(
            f"Required filter keys are missing or empty: {sorted(missing)}", missing_keys=sorted(missing)
        )

    for key, value in filters.custom.items():
        if key not in capabilities.supported_custom_filters:
            raise UnsupportedFilterError(f"Unsupported custom filter key: {key!r}", key=key)
        expected = capabilities.supported_custom_filters[key]
        expected_type = get_origin(expected) or expected
        if not isinstance(value, expected_type):
            raise UnsupportedFilterError(
                f"Custom filter '{key}' expected {expected_type.__name__}, got {type(value).__name__}", key=key
            )


# ── RAGStore Port (§4.2, §4.3, §4.4) ──────────────────────────────────────────

class RAGStore(ABC, Generic[MetaT]):
    """Synchronous storage port for node retrieval, hydration, and lifecycle operations."""

    @abstractmethod
    def capabilities(self) -> StoreCapabilities:
        """Return the declared capabilities of this store."""
        ...

    @abstractmethod
    def embedding_space(self) -> EmbeddingSpace:
        """Return the expected vector space signature (model and dimension)."""
        ...

    @abstractmethod
    def vector_search(
        self,
        query_vector: list[float],
        filters: RAGFilterSpec,
        top_k: int,
    ) -> list[RAGNode[MetaT]]:
        """Perform dense vector similarity search returning up to top_k nodes."""
        ...

    def keyword_search(
        self,
        query_text: str,
        filters: RAGFilterSpec,
        top_k: int,
        analyzer_tokens: list[TokenGroup] | None = None,
    ) -> list[RAGNode[MetaT]]:
        """Perform full-text lexical search."""
        raise NotImplementedError("keyword_search not supported by this store")

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        filters: RAGFilterSpec,
        top_k: int,
        rrf_k: int = 60,
        w_vec: float = 1.0,
        w_fts: float = 1.0,
        analyzer_tokens: list[TokenGroup] | None = None,
    ) -> list[RAGNode[MetaT]]:
        """Perform native hybrid search with Reciprocal Rank Fusion."""
        raise NotImplementedError("hybrid_search not supported by this store")

    def fetch_parent_summaries(
        self,
        filters: RAGFilterSpec,
        level: int | None = None,
    ) -> list[RAGNode[MetaT]]:
        """Fetch summary nodes in document order."""
        raise NotImplementedError("fetch_parent_summaries not supported by this store")

    def fetch_local_window(
        self,
        doc_id: str,
        center_offset: int,
        radius_chars: int,
        filters: RAGFilterSpec,
    ) -> list[RAGNode[MetaT]]:
        """Fetch contiguous text window around center offset."""
        raise NotImplementedError("fetch_local_window not supported by this store")

    def scope_stats(self, filters: RAGFilterSpec) -> ScopeStats:
        """Calculate aggregate scope statistics and current revision."""
        raise NotImplementedError("scope_stats not supported by this store")

    def fetch_full_scope(self, filters: RAGFilterSpec) -> list[RAGNode[MetaT]]:
        """Fetch all content in scope in document order for Tier 0 context."""
        raise NotImplementedError("fetch_full_scope not supported by this store")

    def fetch_nodes(self, node_ids: list[str], filters: RAGFilterSpec) -> list[RAGNode[MetaT]]:
        """Fetch specific nodes by IDs, enforcing current filters."""
        raise NotImplementedError("fetch_nodes not supported by this store")

    def hydrate(self, node_ids: list[str], projection: str) -> dict[str, MetaT]:
        """Hydrate full metadata for previously returned light nodes."""
        raise NotImplementedError("hydrate not supported by this store")

    def upsert_section_nodes(
        self,
        doc_id: str,
        section_seq: int,
        nodes: list[RAGNode[MetaT]],
        embeddings: list[list[float]],
        summary: RAGNode[MetaT] | None = None,
    ) -> int:
        """Atomically upsert passages and optional summary for a section. Returns passages written."""
        raise NotImplementedError("upsert_section_nodes not supported by this store")

    def get_fingerprints(self, doc_id: str) -> dict[str, Fingerprint]:
        """Return map of node_id -> Fingerprint for change detection."""
        raise NotImplementedError("get_fingerprints not supported by this store")

    def delete_nodes(self, node_ids: list[str]) -> int:
        """Delete specific nodes by IDs. Returns count of nodes deleted."""
        raise NotImplementedError("delete_nodes not supported by this store")

    def delete_document(self, doc_id: str) -> int:
        """Delete all nodes for a document. Returns count of nodes deleted."""
        raise NotImplementedError("delete_document not supported by this store")

    def delete_sections(self, doc_id: str, from_seq: int) -> int:
        """Delete document sections starting at from_seq. Returns count of nodes deleted."""
        raise NotImplementedError("delete_sections not supported by this store")

    def list_unindexed(self, limit: int) -> list[str]:
        """List document IDs needing initial indexing or sync."""
        raise NotImplementedError("list_unindexed not supported by this store")

    def current_revisions(self, node_ids: list[str]) -> dict[str, str]:
        """Fetch current content revision hashes for stale citation check."""
        raise NotImplementedError("current_revisions not supported by this store")

    def fetch_related(
        self,
        node_ids: list[str],
        relation: str,
        limit: int,
        filters: RAGFilterSpec,
    ) -> list[RAGNode[MetaT]]:
        """Fetch related nodes across structure hierarchy."""
        raise NotImplementedError("fetch_related not supported by this store")

    def fetch_linked(
        self,
        node_ids: list[str],
        limit: int,
        filters: RAGFilterSpec,
    ) -> list[RAGNode[MetaT]]:
        """Fetch linked nodes across entity/concept relationships."""
        raise NotImplementedError("fetch_linked not supported by this store")


# ── Guard Protocol (§5) ────────────────────────────────────────────────────────

@runtime_checkable
class Guard(Protocol):
    """Enforces boundaries, tenant isolation, or custom visibility rules (§5.1).

    Guards are evaluated in two phases:
      1. to_filter(): translates guard into FilterPatch for store-side filtering.
      2. verify(): verifies, truncates (with flag 'truncated_by_guard') or drops node in-memory.
    """
    name: str
    hard: bool = True
    forces_strict_grounding: bool = False

    def to_filter(self) -> FilterPatch:
        """Narrow filter predicates before query execution. MUST NOT widen filters."""
        ...

    def verify(self, node: RAGNode[Any]) -> RAGNode[Any] | None:
        """Return the node (possibly truncated) or None to drop."""
        ...

    def verify_citation(self, citation: Any) -> bool:
        """Return True if citation is permitted under guard."""
        ...

    def prompt_rules(self) -> list[str]:
        """Directives appended to system prompt (e.g. spoiler warning)."""
        ...
