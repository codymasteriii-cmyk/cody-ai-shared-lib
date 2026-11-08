"""EngineProfile, ModelBindings, validation, and domain presets (§12, AC-RAG-49, AC-RAG-62)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from pydantic import BaseModel, Field

from .errors import ProfileConfigError
from .registry import ComponentRegistry


class ModelBindings(BaseModel):
    """Model identifiers bound to engine stages (§12.1).

    Must be explicitly provided by the app (no default models in library, D-12).
    """
    model_config = {"frozen": True, "extra": "forbid"}
    generation: list[str]
    intent: list[str] | None = None
    summary: list[str] | None = None
    embedding: str = "gemini-embedding-2"
    reranker: str | None = None
    judge: list[str] | None = None


class RouterConfig(BaseModel):
    model_config = {"frozen": True}
    name: str = "auto"
    enable_jev: bool = False
    enable_temporal_extraction: bool = False
    jev_timeout: float = 2.0


class RetrievalConfig(BaseModel):
    model_config = {"frozen": True}
    default_strategy: str = "dense_vector"
    top_k: int = 8
    full_context_max_tokens: int = 120_000
    local_window_radius_chars: int = 6000
    always_include_summaries: bool = False
    rrf_k: int = 60
    min_similarity: float | None = None


class ContextConfig(BaseModel):
    model_config = {"frozen": True}
    max_context_tokens: int = 160_000
    reserved_instruction_tokens: int = 4000
    reserved_history_tokens: int = 4000


class GenerationConfig(BaseModel):
    model_config = {"frozen": True}
    strategy: str = "stuff"
    temperature: float | None = None
    grounding_mode: Literal["strict", "labelled_fallback", "open"] = "strict"
    fallback_label: str = "Based on retrieved context"
    stream_mode: Literal["native", "buffered"] = "buffered"


class CitationConfig(BaseModel):
    model_config = {"frozen": True}
    formatter: str = "ref"
    verify_quotes: bool = False
    policy: str = "filter"


class IngestionConfig(BaseModel):
    model_config = {"frozen": True}
    splitter: str = "sentence"
    target_size: int = 1200
    overlap: int = 150


class RuntimeConfig(BaseModel):
    model_config = {"frozen": True}
    deadline_seconds: float = 30.0
    pool_size: int = 4


VALID_FALLBACK_KEYS = {
    "rerank", "jev", "intent", "generation", "summarizer",
    "query_embedding", "ingest_embedding", "cache", "observer", "deadline",
}


class EngineProfile(BaseModel):
    """Complete, immutable, fingerprinted engine profile configuration."""
    model_config = {"frozen": True, "extra": "forbid"}

    name: str
    version: str = "1.0.0"
    models: ModelBindings
    router: RouterConfig = Field(default_factory=RouterConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    transforms: list[str] = Field(default_factory=list)
    post_stages: list[str] = Field(default_factory=list)
    context: ContextConfig = Field(default_factory=ContextConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    citations: CitationConfig = Field(default_factory=CitationConfig)
    post_generation: list[str] = Field(default_factory=list)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    guards: list[str] = Field(default_factory=list)
    renderer: str = "default"
    analyzer: str = "cjk_aware"
    fallbacks: dict[str, str] = Field(default_factory=dict)
    observer_errors: Literal["log", "raise"] = "log"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    trace_level: Literal["light", "verbose"] = "light"

    def fingerprint(self) -> str:
        """Stable sha256 fingerprint of resolved configuration."""
        dumped = self.model_dump_json()
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def validate(self, registry: ComponentRegistry | None = None) -> None:
        """Validate configuration and bindings; raises ProfileConfigError with all issues."""
        problems: list[str] = []
        missing: list[str] = []
        unknown: list[str] = []

        if not self.models.generation:
            missing.append("models.generation")
            problems.append("ModelBindings.generation cannot be empty")

        if "rerank" in self.post_stages and not self.models.reranker:
            missing.append("models.reranker")
            problems.append("Stage 'rerank' requires models.reranker to be configured")

        # AC-RAG-62: Tier 0 gate plus reserved zones cannot exceed max_context_tokens
        reserved_sum = (
            self.retrieval.full_context_max_tokens
            + self.context.reserved_instruction_tokens
            + self.context.reserved_history_tokens
        )
        if reserved_sum > self.context.max_context_tokens:
            problems.append(
                f"full_context_max_tokens ({self.retrieval.full_context_max_tokens}) + "
                f"reserved zones exceeds max_context_tokens ({self.context.max_context_tokens})"
            )

        # Check fallback keys
        for k in self.fallbacks:
            if k not in VALID_FALLBACK_KEYS:
                unknown.append(f"fallback:{k}")
                problems.append(f"Unknown fallback key '{k}' in profile.fallbacks")

        if registry:
            if not registry.has("analyzer", self.analyzer):
                unknown.append(f"analyzer:{self.analyzer}")
                problems.append(f"Unknown analyzer '{self.analyzer}'")
            if not registry.has("renderer", self.renderer):
                unknown.append(f"renderer:{self.renderer}")
                problems.append(f"Unknown renderer '{self.renderer}'")

        if problems:
            raise ProfileConfigError(
                f"Profile validation failed for '{self.name}': {'; '.join(problems)}",
                problems=problems,
                missing=missing,
                unknown=unknown,
            )


# ── Preset Profiles (§12.2) ────────────────────────────────────────────────────

def long_form_reader(models: ModelBindings, **overrides: Any) -> EngineProfile:
    """Preset profile for AudioReading (long-form fiction, spoiler guard, Tier 0 enabled)."""
    cfg: dict[str, Any] = {
        "name": "long_form_reader",
        "models": models,
        "retrieval": RetrievalConfig(
            default_strategy="adaptive",
            top_k=8,
            full_context_max_tokens=120_000,
            local_window_radius_chars=6000,
            always_include_summaries=True,
        ),
        "post_stages": ["offset_dedup", "top_k"],
        "context": ContextConfig(max_context_tokens=160_000),
        "generation": GenerationConfig(grounding_mode="strict"),
        "guards": ["sequential_boundary"],
    }
    cfg.update(overrides)
    return EngineProfile(**cfg)


def news_intelligence(models: ModelBindings, **overrides: Any) -> EngineProfile:
    """Preset profile for NewsIntelligence (dated news corpus, hybrid search, recency)."""
    cfg: dict[str, Any] = {
        "name": "news_intelligence",
        "models": models,
        "router": RouterConfig(enable_temporal_extraction=True),
        "retrieval": RetrievalConfig(
            default_strategy="hybrid_rrf",
            top_k=12,
            full_context_max_tokens=0,
            local_window_radius_chars=0,
            always_include_summaries=False,
            min_similarity=0.60,
        ),
        "post_stages": ["identity_dedup", "recency", "top_k"],
        "context": ContextConfig(max_context_tokens=32_000),
        "generation": GenerationConfig(grounding_mode="labelled_fallback"),
    }
    cfg.update(overrides)
    return EngineProfile(**cfg)


def knowledge_base(models: ModelBindings, **overrides: Any) -> EngineProfile:
    """Preset profile for KnowledgeBase (fail-loud ADRs, wiki entries, markdown structural)."""
    cfg: dict[str, Any] = {
        "name": "knowledge_base",
        "models": models,
        "retrieval": RetrievalConfig(
            default_strategy="dense_vector",
            top_k=10,
            full_context_max_tokens=100_000,
            local_window_radius_chars=0,
            always_include_summaries=False,
        ),
        "post_stages": ["identity_dedup", "top_k"],
        "context": ContextConfig(max_context_tokens=160_000),
        "generation": GenerationConfig(grounding_mode="strict"),
        "observer_errors": "raise",
        "fallbacks": {
            "rerank": "raise",
            "jev": "raise",
            "deadline": "raise",
            "cache": "raise",
            "ingest_embedding": "raise",
        },
    }
    cfg.update(overrides)
    return EngineProfile(**cfg)
