"""EngineProfile, ModelBindings, validation, and domain presets (§12, AC-RAG-49, AC-RAG-62)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

from .errors import ProfileConfigError
from .registry import ComponentRegistry, split_component_name


class ModelBindings(BaseModel):
    """Model identifiers bound to engine stages (§12.1).

    Must be explicitly provided by the app (no default models in library, D-12).
    """
    model_config = {"frozen": True, "extra": "forbid"}
    generation: list[str]
    intent: list[str] | None = None
    summary: list[str] | None = None
    embedding: str
    reranker: str | None = None
    judge: list[str] | None = None


class RouterConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    name: str = "auto"
    enable_jev: bool = False
    enable_temporal_extraction: bool = False
    jev_timeout: float = 2.5


DEFAULT_FUSION_WEIGHTS: dict[str, tuple[float, float]] = {
    "exact_terms": (0.7, 1.3),
    "negation": (0.7, 1.3),
    "entity_terms": (0.9, 1.1),
    "default": (1.0, 1.0),
}


class RetrievalConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    default_strategy: str = "dense_vector"
    top_k: int = 8
    retrieve_max_results: int = 100
    full_context_max_tokens: int = 120_000
    keyword_first_min_hits: int = 3      # keyword_first falls back to hybrid below this many hits (§8.3)
    local_window_radius_chars: int = 6000
    always_include_summaries: bool = False
    rrf_k: int = 60
    min_similarity: float | None = None
    min_results_before_relax: int = 3
    rerank_candidates: int | None = None  # how many candidates a rerank stage receives; default top_k x 3
    relaxation: list[str] = Field(default_factory=list)
    adaptive_rules: list[tuple[str, str]] | None = None
    enable_keyword_first: bool = False
    fusion_weights: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: dict(DEFAULT_FUSION_WEIGHTS)
    )


class ContextConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    max_context_tokens: int = 160_000
    reserved_instruction_tokens: int = 4000
    reserved_history_tokens: int = 8000  # hard cap on the history zone
    question_reserve_tokens: int = 1000
    answer_reserve_tokens: int = 4000
    history_ratio: float = 0.15
    summary_ratio: float = 0.25
    local_window_ratio: float = 0.25
    retrieved_ratio: float | None = None   # deprecated, ignored: ranked nodes take the remainder (§10.1)
    min_recent_turns_kept: int = 2
    history_decay_lambda: float = 0.35
    history_overlap_alpha: float = 0.40
    state_max_bytes: int = 16_384
    reused_turns: int = 3
    max_reused_nodes: int = 24
    rolling_summary_max_tokens: int = 400


class GenerationConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    strategy: str = "stuff"
    system_preamble: str = ""  # persona; {variables} come from request.system_prompt_variables
    temperature: float | None = None
    grounding_mode: Literal["strict", "labelled_fallback"] = "strict"
    fallback_label: str = "Not from your sources:"
    stream_mode: Literal["native", "buffered"] = "buffered"


class CitationConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    formatter: str = "ref"
    verify_quotes: bool = False
    policy: str = "filter"
    on_unverified: Literal["filter", "flag"] = "filter"
    min_citation_coverage: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _sync_policy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "on_unverified" in data and "policy" not in data:
                data["policy"] = data["on_unverified"]
            elif "policy" in data and "on_unverified" not in data and data["policy"] in ("filter", "flag"):
                data["on_unverified"] = data["policy"]
        return data


class SummariesConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    enabled: bool = False
    min_chars_for_summary: int = 1500
    summary_unit_chars: int = 8000
    # False skips the summary_embedding stage; the store then gets summaries without vectors and must
    # override upsert_section (§6.3, INT-02). Keep True for a store that searches summaries by vector.
    embed_summaries: bool = True


class IngestionConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    splitter: str = "sentence"
    granularity: str = "passage"         # what ingested chunks are: "article"/"entry" are atomic (§3.3)
    target_size: int = 1200
    overlap: int = 150
    summaries: SummariesConfig = Field(default_factory=SummariesConfig)


class RuntimeConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    deadline_seconds: float = 30.0
    pool_size: int = 4
    stage_budgets: dict[str, float] = Field(
        default_factory=lambda: {"rerank": 4.0, "secondary_queries": 3.0, "relaxation": 2.0}
    )


class CacheConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    enabled: bool = True
    max_bytes: int = 32 * 1024 * 1024
    query_embedding_ttl_seconds: int = 3600
    enable_embedding_cache: bool = True
    enable_result_cache: bool = False


class ValidationReport(BaseModel):
    """Report returned by engine.validate() / profile.validate() (§13.3)."""
    valid: bool = True
    profile_fingerprint: str = ""
    problems: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)


VALID_FALLBACK_KEYS = {"rerank", "jev", "summarizer", "query_embedding", "cache", "deadline"}
# §13.3: opt-in rows take their enabling value; any row can be set to "raise"
VALID_FALLBACK_VALUES: dict[str, set[str]] = {
    "rerank": {"raise", "keep_order"},
    "jev": {"raise"},
    "summarizer": {"raise", "skip"},
    "query_embedding": {"raise", "keyword"},
    "cache": {"raise"},
    "deadline": {"raise"},
}
# Citation policies this version implements; repair/raise and coverage minimums come with §11.4-11.5 (P4)
IMPLEMENTED_CITATION_POLICIES = {"filter"}


class EngineProfile(BaseModel):
    """Complete, immutable, fingerprinted engine profile configuration."""
    model_config = {"frozen": True, "extra": "forbid"}

    name: str
    version: str = "1.0.0"
    models: ModelBindings
    router: RouterConfig = Field(default_factory=RouterConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    transforms: list[str] = Field(default_factory=lambda: ["secondary_fanout"])
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
    cache: CacheConfig = Field(default_factory=CacheConfig)
    allowed_link_hosts: list[str] = Field(default_factory=list)
    trace_level: Literal["light", "verbose"] = "light"
    variants: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable sha256 fingerprint of resolved configuration, independent of dict insertion order."""
        dumped = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def with_overrides(self, **changes: Any) -> "EngineProfile":
        """Return a new profile with deep-merged changes (§12.1)."""
        data = self.model_dump()
        for k, v in changes.items():
            if isinstance(v, dict) and isinstance(data.get(k), dict):
                data[k].update(v)
            else:
                data[k] = v
        return EngineProfile(**data)

    def variant(self, name: str, **overrides: Any) -> "EngineProfile":
        """Return named variant of this profile (§12.1)."""
        var_data = dict(self.variants.get(name, {}))
        var_data.update(overrides)
        profile = self.with_overrides(**var_data)
        if not profile.name.endswith(name):
            data = profile.model_dump()
            data["name"] = f"{self.name}_{name}"
            return EngineProfile(**data)
        return profile

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EngineProfile":
        """Construct EngineProfile from mapping dictionary (§12.1)."""
        return cls(**data)

    def _threshold_problems(self) -> list[str]:
        """A threshold on a score kind this pipeline can never produce would silently keep everything (§9.1)."""
        problems: list[str] = []
        stages = [split_stage_name(name) for name in self.post_stages]
        strategy = self.retrieval.default_strategy
        for index, (name, kind) in enumerate(stages):
            if name != "threshold":
                continue
            kind = kind or "cosine"
            if kind == "rerank" and "rerank" not in [n for n, _ in stages[:index]]:
                problems.append("threshold(rerank) needs a rerank stage before it in post_stages")
            elif kind == "bm25" and strategy == "dense_vector":
                problems.append("threshold(bm25) cannot apply: strategy dense_vector produces no keyword scores")
            elif kind == "cosine" and strategy == "keyword_only":
                problems.append("threshold(cosine) cannot apply: strategy keyword_only produces no vector scores")
        return problems

    def check(self, registry: ComponentRegistry | None = None) -> ValidationReport:
        """Return every configuration problem without raising."""
        problems: list[str] = []
        missing: list[str] = []
        unknown: list[str] = []

        def require_binding(present: Any, binding: str, reason: str) -> None:
            if not present:
                missing.append(binding)
                problems.append(f"{reason} requires {binding} to be configured")

        require_binding(self.models.generation, "models.generation", "Generation")
        stage_names = [split_stage_name(name)[0] for name in self.post_stages]
        require_binding(self.models.reranker or "rerank" not in stage_names, "models.reranker", "Stage 'rerank'")
        require_binding(self.models.intent or self.router.name != "llm", "models.intent", "Router 'llm'")
        require_binding(
            self.models.summary or not self.ingestion.summaries.enabled, "models.summary", "Enabled summaries"
        )

        # AC-RAG-62: Tier 0 gate plus reserved zones cannot exceed max_context_tokens
        reserved_sum = (
            self.retrieval.full_context_max_tokens
            + self.context.reserved_instruction_tokens
            + self.context.reserved_history_tokens
            + self.context.question_reserve_tokens
        )
        if reserved_sum > self.context.max_context_tokens:
            problems.append(
                f"full_context_max_tokens ({self.retrieval.full_context_max_tokens}) + "
                f"reserved zones exceeds max_context_tokens ({self.context.max_context_tokens})"
            )

        ratio_sum = self.context.history_ratio + self.context.summary_ratio + self.context.local_window_ratio
        if ratio_sum > 1.0 + 1e-9:
            problems.append(f"Context zone ratios (history, summary, local window) sum to {ratio_sum:.2f}, above 1")

        problems.extend(self._threshold_problems())

        for k, v in self.fallbacks.items():
            if k not in VALID_FALLBACK_KEYS:
                unknown.append(f"fallback:{k}")
                problems.append(f"Unknown fallback key '{k}' in profile.fallbacks")
            elif v not in VALID_FALLBACK_VALUES[k]:
                problems.append(f"fallbacks['{k}'] = {v!r}: allowed values are {sorted(VALID_FALLBACK_VALUES[k])}")
        if self.citations.policy not in IMPLEMENTED_CITATION_POLICIES:
            problems.append(f"citations.policy {self.citations.policy!r} is not implemented in this version (§11.4)")
        if self.citations.min_citation_coverage:
            problems.append("citations.min_citation_coverage is not enforced in this version (§11.4); leave it at 0")

        if registry is not None:
            for category, name in self._component_names():
                if not registry.has(category, name):
                    unknown.append(f"{category}:{name}")
                    problems.append(f"Unknown {category} '{name}'")

        if self.post_generation:
            problems.append("post_generation stages are a seam and are not run by this version; leave the list empty")

        for name in self.variants:
            try:
                nested = self.variant(name).model_copy(update={"variants": {}}).check(registry)
            except ValueError as exc:
                problems.append(f"Variant '{name}' is invalid: {exc}")
                continue
            problems.extend(f"Variant '{name}': {problem}" for problem in nested.problems)
            missing.extend(nested.missing)
            unknown.extend(nested.unknown)

        return ValidationReport(
            valid=not problems,
            profile_fingerprint=self.fingerprint(),
            problems=problems,
            missing=missing,
            unknown=unknown,
        )

    def _component_names(self) -> list[tuple[str, str]]:
        """Every (registry category, name) this profile refers to."""
        names = [
            ("analyzer", self.analyzer),
            ("renderer", self.renderer),
            ("router", self.router.name),
            ("strategy", self.retrieval.default_strategy),
            ("splitter", self.ingestion.splitter),
            ("generator", self.generation.strategy),
            ("formatter", self.citations.formatter),
        ]
        names += [("transform", name) for name in self.transforms]
        names += [("guard", name) for name in self.guards]
        names += [("post_generation", name) for name in self.post_generation]
        names += [
            ("stage", split_component_name(name)[0])
            for name in self.post_stages
            if name != "guard_enforcement"
        ]
        names += [("relaxation", split_component_name(step)[0]) for step in self.retrieval.relaxation]
        for predicate, strategy in self.retrieval.adaptive_rules or ():
            names.append(("predicate", predicate))
            if not strategy.startswith("$") and strategy != "none":
                names.append(("strategy", strategy))
        return names

    def validate(self, registry: ComponentRegistry | None = None) -> ValidationReport:
        """Raise ProfileConfigError listing every configuration problem at once (§12.1)."""
        report = self.check(registry)
        if not report.valid:
            raise ProfileConfigError(
                f"Profile validation failed for '{self.name}': {'; '.join(report.problems)}",
                problems=report.problems,
                missing=report.missing,
                unknown=report.unknown,
            )
        return report


def _rerank_if_bound(models: ModelBindings) -> list[str]:
    """The rerank stage needs a bound model; a preset without one simply has no such stage."""
    return ["rerank"] if models.reranker else []


# ── Preset Profiles (§12.2) ────────────────────────────────────────────────────

def long_form_reader(models: ModelBindings, **overrides: Any) -> EngineProfile:
    """Preset profile for AudioReading (long-form fiction, spoiler guard, Tier 0 enabled)."""
    cfg: dict[str, Any] = {
        "name": "long_form_reader",
        "models": models,
        "ingestion": IngestionConfig(
            summaries=SummariesConfig(enabled=True),
        ),
        "retrieval": RetrievalConfig(
            default_strategy="adaptive",
            top_k=8,
            full_context_max_tokens=120_000,
            local_window_radius_chars=6000,
            always_include_summaries=True,
        ),
        "post_stages": ["offset_dedup", *_rerank_if_bound(models), "top_k"],
        "context": ContextConfig(max_context_tokens=160_000),
        "generation": GenerationConfig(grounding_mode="strict", stream_mode="native"),
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
            retrieve_max_results=100,
            full_context_max_tokens=0,
            local_window_radius_chars=0,
            always_include_summaries=False,
            min_similarity=0.60,
        ),
        "post_stages": ["identity_dedup(url)", *_rerank_if_bound(models), "recency", "top_k"],
        "context": ContextConfig(max_context_tokens=32_000),
        "generation": GenerationConfig(grounding_mode="labelled_fallback", stream_mode="buffered"),
        "ingestion": IngestionConfig(
            splitter="pass_through",
            summaries=SummariesConfig(enabled=False),
        ),
        "variants": {
            "news_chat": {
                "retrieval": {"top_k": 12, "min_similarity": 0.60, "retrieve_max_results": 100},
                "post_stages": ["identity_dedup(url)", *_rerank_if_bound(models), "recency", "top_k"],
            },
            "news_search": {
                "router": {"name": "none"},
                "retrieval": {"top_k": 2000, "min_similarity": 0.68, "retrieve_max_results": 2000},
                "post_stages": ["identity_dedup(url)", "recency", "top_k"],
            },
        },
    }
    cfg.update(overrides)
    return EngineProfile(**cfg)


def news_chat(models: ModelBindings, **overrides: Any) -> EngineProfile:
    """Preset variant for News chat (§12.2, §16.2)."""
    base = news_intelligence(models=models)
    return base.variant("news_chat", **overrides)


def news_search(models: ModelBindings, **overrides: Any) -> EngineProfile:
    """Preset variant for News search (supports up to 2000 results, §8.10, §12.2)."""
    base = news_intelligence(models=models)
    return base.variant("news_search", **overrides)


def knowledge_base(models: ModelBindings, **overrides: Any) -> EngineProfile:
    """Preset profile for KnowledgeBase (fail-loud ADRs, wiki entries, markdown structural)."""
    cfg: dict[str, Any] = {
        "name": "knowledge_base",
        "models": models,
        "retrieval": RetrievalConfig(
            default_strategy="adaptive",
            top_k=10,
            full_context_max_tokens=100_000,
            local_window_radius_chars=0,
            always_include_summaries=False,
            enable_keyword_first=True,
        ),
        "post_stages": ["identity_dedup", *_rerank_if_bound(models), "score_fusion(confidence)", "top_k"],
        "context": ContextConfig(max_context_tokens=160_000),
        "generation": GenerationConfig(grounding_mode="strict", stream_mode="native"),
        "citations": CitationConfig(verify_quotes=True),
        "observer_errors": "raise",
        "fallbacks": {
            "rerank": "raise",
            "jev": "raise",
            "deadline": "raise",
            "cache": "raise",
        },
    }
    cfg.update(overrides)
    return EngineProfile(**cfg)


def split_stage_name(name: str) -> tuple[str, str | None]:
    """'threshold(rerank)' -> ('threshold', 'rerank'); 'top_k' -> ('top_k', None)."""
    if name.endswith(")") and "(" in name:
        base, argument = name[:-1].split("(", 1)
        return base.strip(), argument.strip() or None
    return name.strip(), None
