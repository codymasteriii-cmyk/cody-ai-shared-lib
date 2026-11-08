"""Canonical RAG domain data models.

All data exchanged between the caller, the engine, and the store MUST use these
Pydantic v2 models per §3.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar
from pydantic import BaseModel, Field, model_validator

# ── Generic Parameters (§3.1) ──────────────────────────────────────────────────

class DefaultNodeMeta(BaseModel):
    """Default metadata container for nodes without specialized schemas."""
    model_config = {"extra": "allow"}
    extra: dict[str, Any] = Field(default_factory=dict)


class NoIntentExt(BaseModel):
    """Default (empty) intent extension."""
    pass


class FilterPatch(BaseModel):
    """A partial RAGFilterSpec. Patches combine by narrowing only (§5.5)."""
    fields: dict[str, Any] = Field(default_factory=dict)


MetaT = TypeVar("MetaT", bound=BaseModel, default=DefaultNodeMeta)
ExtT = TypeVar("ExtT", bound=BaseModel, default=NoIntentExt)

ScoreKind = Literal["cosine", "bm25", "rrf", "rerank", "fused", "none"]


# ── Source Coordinates & Granularity (§3.2, §3.3) ──────────────────────────────

class SourceSpan(BaseModel):
    """Coordinates identifying a slice of a source document."""
    doc_id: str = ""
    corpus: str | None = None
    section_id: str | None = None
    section_seq: int = 0
    section_title: str | None = None
    breadcrumb: list[str] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    position_ratio: float = 0.0


RAGNodeSpan = SourceSpan


class RAGNode(BaseModel, Generic[MetaT]):
    """A retrievable, traceable unit of content with provenance and scores."""
    node_id: str
    text: str
    span: SourceSpan = Field(default_factory=SourceSpan)
    granularity: str = "passage"
    revision: str | None = None
    content_hash: str | None = None
    token_estimate: int | None = None

    # Parent summary fields
    parent_id: str | None = None
    parent_kind: str | None = None
    level: int | None = None
    is_summary: bool = False

    # Scoring
    score: float = 0.0
    score_kind: ScoreKind = "none"
    vector_score: float | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None

    embedding: list[float] | None = None
    flags: list[str] = Field(default_factory=list)
    hydrated: bool = True
    search_tokens: list[str] | None = None
    snippet: str | None = None
    snippet_kind: Literal["match_excerpt", "summary"] | None = None
    metadata: MetaT = Field(default_factory=DefaultNodeMeta)  # type: ignore[assignment]

    @model_validator(mode="before")
    @classmethod
    def _populate_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "node_id" not in data and "id" in data:
                data["node_id"] = data["id"]
            if "span" not in data or data["span"] is None:
                doc_id = data.get("document_id", "")
                sec_id = data.get("section_id", "")
                seq = data.get("order_index", 0)
                data["span"] = SourceSpan(doc_id=doc_id, section_id=sec_id, section_seq=seq)
            elif isinstance(data["span"], dict):
                if "doc_id" not in data["span"] and "document_id" in data:
                    data["span"]["doc_id"] = data["document_id"]
            elif hasattr(data["span"], "doc_id") and not data["span"].doc_id and "document_id" in data:
                data["span"] = data["span"].model_copy(update={"doc_id": data["document_id"]})
        return data

    @property
    def id(self) -> str:
        return self.node_id

    @property
    def document_id(self) -> str:
        return self.span.doc_id

    @property
    def section_id(self) -> str | None:
        return self.span.section_id

    @property
    def order_index(self) -> int:
        return self.span.section_seq


# ── Filters (§3.6) ─────────────────────────────────────────────────────────────

class RAGFilterSpec(BaseModel):
    """Store-level query filter specifications built from query request, guards, and intent."""
    doc_ids: list[str] | None = None
    tenant_or_user_id: str | None = None
    domain_id: str | None = None
    categories: list[str] | None = None
    max_char_offset: int | None = None
    max_char_offset_by_doc: dict[str, int] | None = None
    min_char_offset: int | None = None
    date_range_utc: tuple[datetime | None, datetime | None] | None = None
    date_field: str | None = None
    min_similarity: float | None = None
    custom: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _populate_filter_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "tenant_or_user_id" not in data and "tenant_id" in data:
                data["tenant_or_user_id"] = data["tenant_id"]
            if "doc_ids" not in data and "document_ids" in data:
                data["doc_ids"] = data["document_ids"]
        return data

    @property
    def tenant_id(self) -> str | None:
        return self.tenant_or_user_id

    @tenant_id.setter
    def tenant_id(self, val: str | None) -> None:
        self.tenant_or_user_id = val

    @property
    def document_ids(self) -> list[str] | None:
        return self.doc_ids

    @document_ids.setter
    def document_ids(self, val: list[str] | None) -> None:
        self.doc_ids = val

    def patch(self, patch: Any) -> RAGFilterSpec:
        """Combine patches by narrowing only (§5.5)."""
        fields = patch.fields if hasattr(patch, "fields") else patch
        if not isinstance(fields, dict):
            return self
        copied = self.model_copy(deep=True)
        for k, v in fields.items():
            if v is not None:
                if k in ("tenant_id", "tenant_or_user_id"):
                    copied.tenant_or_user_id = v
                elif k in ("doc_ids", "document_ids"):
                    if copied.doc_ids is not None:
                        copied.doc_ids = [d for d in copied.doc_ids if d in v]
                    else:
                        copied.doc_ids = list(v)
                elif k == "max_char_offset":
                    if copied.max_char_offset is not None:
                        copied.max_char_offset = min(copied.max_char_offset, v)
                    else:
                        copied.max_char_offset = v
                elif k == "max_char_offset_by_doc":
                    if copied.max_char_offset_by_doc is not None:
                        for doc, off in v.items():
                            cur = copied.max_char_offset_by_doc.get(doc)
                            copied.max_char_offset_by_doc[doc] = min(cur, off) if cur is not None else off
                    else:
                        copied.max_char_offset_by_doc = dict(v)
                elif k == "min_char_offset":
                    if copied.min_char_offset is not None:
                        copied.min_char_offset = max(copied.min_char_offset, v)
                    else:
                        copied.min_char_offset = v
                elif k == "date_range_utc":
                    copied.date_range_utc = v
                else:
                    if hasattr(copied, k):
                        setattr(copied, k, v)
                    else:
                        copied.custom[k] = v
        return copied


# ── Conversation State & History (§3.7, §7.7) ─────────────────────────────────

class ChatTurn(BaseModel):
    """One user or assistant turn in conversation history."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime | None = None


class NodeRef(BaseModel):
    """Lightweight reference to a node retrieved in a prior turn."""
    node_id: str
    revision: str | None = None


class ConversationState(BaseModel):
    """Opaque, client-roundtripped conversation memory state."""
    version: int = 1
    fence_id: str = ""
    prior_nodes: list[NodeRef] = Field(default_factory=list)
    last_intent: dict[str, Any] | None = None
    rolling_summary: str | None = None
    turn_count: int = 0
    profile_fingerprint: str | None = None


# ── Citations (§3.8) ───────────────────────────────────────────────────────────

class RAGCitation(BaseModel):
    """Citation verified against retrieved sources."""
    marker: str = ""
    label: str = ""
    index: int = 1
    node_id: str
    span: SourceSpan = Field(default_factory=SourceSpan)
    document_id: str = ""
    section_id: str | None = None
    revision: str | None = None
    quote: str | None = None
    verified: bool = True
    verification_score: float = 1.0
    stale: bool = False

    @model_validator(mode="before")
    @classmethod
    def _populate_citation_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            idx = data.get("index", 1)
            if not data.get("marker"):
                data["marker"] = f"[{idx}]"
            if not data.get("label"):
                data["label"] = f"Source {idx}"
        return data


# ── Usage, Degradation, Trace Summaries (§3.9, §13) ───────────────────────────

class StageUsage(BaseModel):
    """Resource consumption and latency for a single stage or model call."""
    stage: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    price_table_version: str | None = None
    estimated: bool = False
    billing: Literal["billed", "unknown"] = "billed"
    outcome: Literal["ok", "error", "cancelled"] = "ok"


class UsageTotal(StageUsage):
    """Aggregated resource consumption across an entire RAG request."""
    stage: str = "total"
    cost_complete: bool = True
    unpriced_models: list[str] = Field(default_factory=list)


class Degradation(BaseModel):
    """Record of an agreed degradation that fired during execution (§13.3)."""
    id: str
    stage: str
    reason: str
    error_type: str | None = None
    message: str | None = None
    attempts: int = 1
    fallback_used: str = ""
    impact: str = ""


class CitationCoverage(BaseModel):
    """Sentence-level grounding coverage verification."""
    sentences_total: int = 0
    sentences_cited: int = 0
    ratio: float = 1.0
    uncited_sentences: list[str] = Field(default_factory=list)
    total_citations: int = 0
    verified_citations: int = 0

    @model_validator(mode="before")
    @classmethod
    def _populate_coverage_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "total_citations" in data and "sentences_total" not in data:
                data["sentences_total"] = data["total_citations"]
            if "verified_citations" in data and "sentences_cited" not in data:
                data["sentences_cited"] = data["verified_citations"]
            if "coverage_ratio" in data and "ratio" not in data:
                data["ratio"] = data["coverage_ratio"]
        return data

    @property
    def coverage_ratio(self) -> float:
        return self.ratio


# ── Query Intent (§7.1) ────────────────────────────────────────────────────────

class QueryIntent(BaseModel, Generic[ExtT]):
    """Structured query analysis and facet classification."""
    route: Literal["rag", "no_retrieval"] = "rag"
    scope: Literal["local_lookup", "document_overview"] = "local_lookup"
    search_query: str = ""
    secondary_queries: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    is_comparison: bool = False
    is_followup: bool = False

    # Temporal facets
    has_time_expression: bool = False
    temporal_scope: Literal["none", "recent", "historical", "specific_range"] = "none"
    date_window_days: int | None = None
    range_start_utc: datetime | None = None
    range_end_utc: datetime | None = None

    # Lexical signals
    entity_terms: list[str] = Field(default_factory=list)
    entity_mode: Literal["boost", "require", "ignore"] = "boost"
    exact_terms: list[str] = Field(default_factory=list)
    negation: bool = False

    suggested_top_k: int | None = None
    beyond_boundary_risk: float = 0.0
    router_source: str = "passthrough"
    ext: ExtT = Field(default_factory=NoIntentExt)  # type: ignore[assignment]


# ── Requests and Responses (§3.7, §3.10) ───────────────────────────────────────

class RAGQueryRequest(BaseModel):
    """Incoming user question and execution constraints."""
    question: str
    request_id: str | None = None
    filters: RAGFilterSpec = Field(default_factory=RAGFilterSpec)
    guards: list[Any] = Field(default_factory=list)
    history: list[ChatTurn] = Field(default_factory=list)
    state: ConversationState | None = None
    cursor_offset: int | None = None
    doc_total_chars: int | None = None

    tenant_id: str | None = None
    document_id: str | None = None
    custom_instruction: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    router_override: str | None = None
    strategy_override: str | None = None
    top_k_override: int | None = None
    enable_reranker_override: bool | None = None
    variant: str | None = None
    trace_level: Literal["light", "verbose"] | None = None
    system_prompt_variables: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _populate_question(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "question" not in data and "query" in data:
                data["question"] = data["query"]
        return data

    @property
    def query(self) -> str:
        return self.question

    @property
    def conversation(self) -> ConversationState | None:
        return self.state


class RAGRetrievalResult(BaseModel, Generic[MetaT]):
    """Result of candidate retrieval, filtering, and scoring before generation."""
    nodes: list[RAGNode[MetaT]] = Field(default_factory=list)
    intent: QueryIntent | None = None
    strategy_used: str = ""
    tier_decision_reason: str = ""
    applied_filters: RAGFilterSpec = Field(default_factory=RAGFilterSpec)
    relaxations: list[str] = Field(default_factory=list)
    reranked: bool = False
    truncated: bool = False
    degradations: list[Degradation] = Field(default_factory=list)
    usage: list[StageUsage] = Field(default_factory=list)
    trace: Any | None = None


class RAGResponse(BaseModel, Generic[MetaT]):
    """Final grounded answer, verified citations, provenance, and spend accounting."""
    answer_id: str
    answer: str
    answer_final: str
    citations: list[RAGCitation] = Field(default_factory=list)
    sources: list[RAGCitation] = Field(default_factory=list)
    retrieved: RAGRetrievalResult[MetaT] = Field(default_factory=RAGRetrievalResult)
    grounded: bool = True
    answer_found: Literal["yes", "no", "partial"] | None = None
    abstained: bool = False
    degraded: bool = False
    citation_coverage: CitationCoverage | None = None
    state: ConversationState = Field(default_factory=ConversationState)
    usage: list[StageUsage] = Field(default_factory=list)
    usage_total: UsageTotal = Field(default_factory=lambda: UsageTotal(stage="total"))
    degradations: list[Degradation] = Field(default_factory=list)
    latency_ms: float = 0.0
    trace: Any | None = None

    @property
    def retrieval(self) -> RAGRetrievalResult[MetaT]:
        """Convenience alias for self.retrieved."""
        return self.retrieved


class RAGStreamEvent(BaseModel):
    """Standardized event emitted during streaming execution (§3.11, §10.8)."""
    event: Literal["intent", "status", "retrieval", "delta", "citations", "degradation", "done", "error"]
    data: dict[str, Any]


# ── Ingestion & Reindex Status (§6) ───────────────────────────────────────────

class RAGSectionInput(BaseModel, Generic[MetaT]):
    """Input representation of a document section for ingestion (§6.6)."""
    section_id: str | None = None
    section_seq: int = 0
    title: str | None = None
    breadcrumb: list[str] = Field(default_factory=list)
    text: str
    char_start: int = 0
    existing_summary: str | None = None
    metadata: MetaT = Field(default_factory=dict)  # type: ignore


class RAGDocumentInput(BaseModel, Generic[MetaT]):
    """Input representation of a full document with ordered sections (§6.6)."""
    doc_id: str
    title: str | None = None
    sections: list[RAGSectionInput[MetaT]]
    total_char_length: int | None = None
    revision: str | None = None


class PreparedNode(BaseModel, Generic[MetaT]):
    """Result of engine.prepare_nodes() for app-side insertion (§6.4.1, RAG-ING-13)."""
    node: RAGNode[MetaT]
    vector: list[float]
    fingerprint: Any
    usage: StageUsage | None = None


class IngestionProgress(BaseModel):
    """Progress snapshot emitted after completing an ingestion section."""
    doc_id: str
    section_seq: int
    sections_completed: int
    sections_total: int
    passages_upserted: int
    passages_skipped_unchanged: int = 0
    passages_failed: int = 0
    summary_status: Literal["generated", "provided", "skipped", "failed"] = "skipped"
    usage: list[StageUsage] = Field(default_factory=list)
    degradations: list[Degradation] = Field(default_factory=list)


class ReindexStatus(BaseModel):
    """Index staleness status for a document."""
    embeddings_stale: bool = False
    fts_stale: bool = False


class QueryPlan(BaseModel):
    """Evaluated execution plan without retrieval or generation (§8.12)."""
    strategy: str
    tier_decision_reason: str
    rule: int = 1
    requires_index: bool = False
    reads: frozenset[str] = Field(default_factory=frozenset)
    applied_filters: RAGFilterSpec = Field(default_factory=RAGFilterSpec)
    scope: Any | None = None
