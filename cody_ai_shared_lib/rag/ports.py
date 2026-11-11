"""Storage port and Guard protocol contracts (§4, §5).

The engine reaches persistent storage only through RAGStore[MetaT].
The library contains no SQL, table name, or database client (D-01).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Protocol, TypeVar, get_origin, runtime_checkable
from pydantic import BaseModel, Field

from .errors import (
    MissingRequiredScopeError,
    RAGEngineError,
    RetrievalError,
    StoreCapabilityError,
    UnsupportedFilterError,
)
from .models import FilterPatch, MetaT, PreparedNode, RAGFilterSpec, RAGNode
from .render import NodeRenderer
from .text.analyzer import TokenGroup

T = TypeVar("T")


# ── Storage Capabilities & Signatures (§4.1, §4.4) ─────────────────────────────

class EmbeddingSpace(BaseModel):
    """Configuration signature of the vector embedding space."""
    model: str
    dims: int
    task_type_documents: str = "RETRIEVAL_DOCUMENT"
    splitter_version: str | None = None      # None: the engine uses its splitter's own version


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
    """How a stored row was made, so the engine can tell whether it is still current (§6.6).

    `embedding_signature` is None for a row stored without a vector by design (a summary when
    embed_summaries is False); such a row is never stale for embeddings.
    """
    content_hash: str
    embedding_signature: str | None
    fts_signature: str | None = None


class OwnerScope(BaseModel):
    """Who an ingestion runs for and which partition it writes into (§6.4.2, INT-04).

    Key names match RAGFilterSpec, so a store's `required_filter_keys` apply to writes as they do to
    reads. A key that is not a field here is looked up in `custom`. The same user both pays for the
    calls (BudgetGuard reads `tenant_or_user_id`) and owns the rows written.
    """
    model_config = {"frozen": True, "extra": "forbid"}
    tenant_or_user_id: str | None = None
    domain_id: str | None = None
    custom: dict[str, Any] = Field(default_factory=dict)

    def value(self, key: str) -> Any:
        """The value this owner gives a required filter key, or None."""
        name = _FILTER_KEY_ALIASES.get(key, key)
        if name in ("tenant_or_user_id", "domain_id"):
            return getattr(self, name)
        return self.custom.get(key)

    def as_scope(self) -> dict[str, Any]:
        """The non-empty keys, for CallContext.scope and logs."""
        scope: dict[str, Any] = dict(self.custom)
        if self.tenant_or_user_id is not None:
            scope["tenant_or_user_id"] = self.tenant_or_user_id
        if self.domain_id is not None:
            scope["domain_id"] = self.domain_id
        return scope


class SummaryWrite(BaseModel, Generic[MetaT]):
    """One summary to persist: the node, its fingerprint and, when summaries are embedded, its vector."""
    node: RAGNode[MetaT]
    vector: list[float] | None = None      # None when profile.ingestion.summaries.embed_summaries is False
    fingerprint: Fingerprint
    unit_seq: int | None = None            # 0-based unit index (gaps allowed); None for the section summary


class SectionWrite(BaseModel, Generic[MetaT]):
    """Everything one `upsert_section` call persists, in one structure (§4.4, INT-01).

    An upsert never deletes: during sync a write may hold only the changed nodes of a section.
    Passages always carry a vector. Fields may be added in later versions; adapters read what they need.
    """
    doc_id: str
    section_seq: int
    owner: OwnerScope = Field(default_factory=OwnerScope)
    passages: list[PreparedNode[MetaT]] = Field(default_factory=list)   # document order
    unit_summaries: list[SummaryWrite[MetaT]] = Field(default_factory=list)
    section_summary: SummaryWrite[MetaT] | None = None


_FILTER_KEY_ALIASES = {"tenant_id": "tenant_or_user_id", "document_ids": "doc_ids", "doc_id": "doc_ids"}


def filter_value(filters: RAGFilterSpec, key: str) -> Any:
    """The value a store's required filter key has in these filters, or None."""
    value = getattr(filters, _FILTER_KEY_ALIASES.get(key, key), None)
    return value if value is not None else filters.custom.get(key)


def store_overrides(store: Any, method: str) -> bool:
    """True when the store's class replaces the RAGStore default for `method`."""
    return getattr(type(store), method, None) is not getattr(RAGStore, method)


def validate_owner(owner: OwnerScope | None, capabilities: StoreCapabilities) -> None:
    """Fail closed on a write that does not state the store's required scope (§5.6, INT-04).

    "What you must state to read, you must state to write." `doc_ids` is satisfied by the
    document being written.
    """
    missing = []
    for key in capabilities.required_filter_keys:
        if _FILTER_KEY_ALIASES.get(key, key) == "doc_ids":
            continue
        value = owner.value(key) if owner is not None else None
        if value is None or value in ("", [], {}):
            missing.append(key)
    if missing:
        raise MissingRequiredScopeError(
            f"Ingestion owner is missing required scope keys: {sorted(missing)}",
            missing_keys=sorted(missing),
            stage="ingestion",
        )


def call_store(operation: str, call: Callable[[], T]) -> T:
    """Typed engine errors pass through; anything else from a store becomes RetrievalError (§13.3)."""
    try:
        return call()
    except RAGEngineError:
        raise
    except Exception as exc:
        raise RetrievalError(
            f"Store operation '{operation}' failed: {type(exc).__name__}",
            stage="store",
            cause=exc,
        ) from exc


def validate_scope(filters: RAGFilterSpec, capabilities: StoreCapabilities) -> None:
    """Fail closed on missing required keys and on unknown or wrongly typed custom filters (§3.6, §5.6)."""
    missing = []
    for key in capabilities.required_filter_keys:
        value = filter_value(filters, key)
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
    """Synchronous storage port. Retrieval methods return complete rows: everything guards, the
    renderer and stages need. How a row is produced (one query or several) is the adapter's concern."""

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
        """Size and revision of everything the filters allow (Tier 0 decision, cache key).

        MUST describe exactly the content `fetch_full_scope(filters)` would return: the source text,
        not indexed rows. A store that serves Tier 0 from text that is not embedded yet counts that
        text here, or Tier 0 is never chosen for it (§4.2, INT-06).
        """
        raise NotImplementedError("scope_stats not supported by this store")

    def fetch_full_scope(self, filters: RAGFilterSpec) -> list[RAGNode[MetaT]]:
        """All content in scope, for Tier 0 (§8.2).

        Nodes come in document order with real doc_id, section_seq, char_start, char_end and title.
        They do not overlap, and no node crosses a section boundary: one node per section or finer
        (e.g. passages). Each node becomes its own citation, so a single node for the whole scope
        would give one citation that points at offset 0 (INT-07).
        """
        raise NotImplementedError("fetch_full_scope not supported by this store")

    def fetch_nodes(self, node_ids: list[str], filters: RAGFilterSpec) -> list[RAGNode[MetaT]]:
        """Fetch specific nodes by IDs, enforcing current filters."""
        raise NotImplementedError("fetch_nodes not supported by this store")

    def upsert_section_nodes(
        self,
        doc_id: str,
        section_seq: int,
        nodes: list[RAGNode[MetaT]],
        embeddings: list[list[float]],
        summary: RAGNode[MetaT] | None = None,
    ) -> int:
        """Atomically upsert passages and optional summary for a section. Returns passages written.

        Legacy shape: unit summaries ride in `nodes`, and the store gets no fingerprints. The engine
        calls `upsert_section`, whose default delegates here; SQL adapters SHOULD override that instead.
        """
        raise NotImplementedError("upsert_section_nodes not supported by this store")

    def upsert_section(self, write: SectionWrite[MetaT]) -> int:
        """Persist one section write atomically for readers (RAG-ING-10). Returns passages written.

        Idempotent on node_id, never deletes. Persist each item's `fingerprint` as given: the engine
        computes it, the store never does (§6.6). `node_id` is an opaque key: delete by it, never parse it.

        The default delegates to `upsert_section_nodes` with its legacy argument shapes, so stores
        written before 1.2.0 keep working. It needs a vector on every unit summary, which is why
        `embed_summaries=False` requires a store that overrides this method.
        """
        nodes: list[RAGNode[MetaT]] = [p.node for p in write.passages]
        vectors: list[list[float]] = [p.vector for p in write.passages]
        for unit in write.unit_summaries:
            if unit.vector is None:
                raise StoreCapabilityError(
                    "A unit summary without a vector needs a store that overrides upsert_section",
                    stage="ingestion",
                )
            nodes.append(unit.node)
            vectors.append(unit.vector)
        summary = write.section_summary.node if write.section_summary is not None else None
        return self.upsert_section_nodes(
            doc_id=write.doc_id,
            section_seq=write.section_seq,
            nodes=nodes,
            embeddings=vectors,
            summary=summary,
        )

    def get_fingerprints(self, doc_id: str) -> dict[str, Fingerprint]:
        """Return map of node_id -> Fingerprint for change detection."""
        raise NotImplementedError("get_fingerprints not supported by this store")

    def delete_nodes(self, node_ids: list[str]) -> int:
        """Delete specific nodes by IDs (the stored node_id key; never parse it). Returns count deleted."""
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


# ── FilterInferencer Protocol (§5.5) ──────────────────────────────────────────

@runtime_checkable
class FilterInferencer(Protocol):
    """App-supplied: proposes extra filters from the question (KB-13).

    The engine intersects the proposal with the request's filters (`RAGFilterSpec.patch`), so it can
    only narrow the search; a conflict gives an empty scope and the not-found answer, never a wider one.
    """

    def infer(self, request: Any, intent: Any) -> dict[str, Any] | FilterPatch | None:
        ...


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
