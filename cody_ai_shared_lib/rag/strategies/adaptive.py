"""Tier ladder AdaptiveStrategy driven by ordered (predicate, strategy) rules (§8.9, §8.12)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic

from ..errors import StoreCapabilityError
from ..models import MetaT, QueryIntent, QueryPlan, RAGFilterSpec, RAGNode, RAGQueryRequest
from ..ports import ScopeStats, StoreCapabilities
from ..profile import EngineProfile
from ..registry import ComponentRegistry, create_default_registry
from .context import RetrievalContext, RetrievalStrategy

# The lowest rate estimate_tokens can produce (non-CJK text), so a document ruled out is truly too large.
_MIN_TOKENS_PER_CHAR = 0.25

OVERRIDE_STRATEGY = "$override"
NO_STRATEGY = "none"

DEFAULT_ADAPTIVE_RULES: tuple[tuple[str, str], ...] = (
    ("has_strategy_override", OVERRIDE_STRATEGY),
    ("route_no_retrieval", NO_STRATEGY),
    ("fits_full_context", "full_context"),
    ("wants_overview", "hierarchical"),
    ("exact_terms_only", "keyword_first"),
    ("hybrid_available", "hybrid_rrf"),
    ("vector_available", "dense_vector"),
    ("keyword_available", "keyword_only"),
)


@dataclass
class RuleContext:
    """What a rule predicate may inspect. `stats` is filled when Tier 0 is measured."""
    request: RAGQueryRequest
    intent: QueryIntent
    filters: RAGFilterSpec
    store: Any
    caps: StoreCapabilities
    profile: EngineProfile
    stats: ScopeStats | None = field(default=None)


def _hint_exceeds_gate(rc: RuleContext, gate: int) -> bool:
    """True when doc_total_chars proves the visible text cannot fit, avoiding a store call (§8.2)."""
    chars = rc.request.doc_total_chars
    if chars is None:
        return False
    if rc.filters.max_char_offset is not None:
        chars = min(chars, rc.filters.max_char_offset)
    if rc.filters.max_char_offset_by_doc:
        chars = min(chars, max(rc.filters.max_char_offset_by_doc.values()))
    return chars * _MIN_TOKENS_PER_CHAR > gate


def fits_full_context(rc: RuleContext) -> bool:
    gate = rc.profile.retrieval.full_context_max_tokens
    if gate <= 0 or not rc.caps.supports_full_scope_fetch or _hint_exceeds_gate(rc, gate):
        return False
    rc.stats = rc.store.scope_stats(rc.filters)
    return 0 < rc.stats.total_tokens <= gate


BUILTIN_PREDICATES = {
    "has_strategy_override": lambda rc: bool(rc.request.strategy_override)
    and rc.request.strategy_override != AdaptiveStrategy.name,
    "route_no_retrieval": lambda rc: rc.intent.route == "no_retrieval",
    "fits_full_context": fits_full_context,
    "wants_overview": lambda rc: rc.intent.scope == "document_overview"
    or rc.profile.retrieval.always_include_summaries,
    "exact_terms_only": lambda rc: bool(rc.intent.exact_terms) and rc.profile.retrieval.enable_keyword_first,
    "hybrid_available": lambda rc: rc.caps.supports_vector_search
    and (rc.caps.supports_keyword_fts or rc.caps.supports_native_hybrid_rpc),
    "vector_available": lambda rc: rc.caps.supports_vector_search,
    "keyword_available": lambda rc: rc.caps.supports_keyword_fts,
}


class AdaptiveStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    """Runs the first rule that is true and whose strategy the store can serve (§8.9)."""
    name: ClassVar[str] = "adaptive"
    version: ClassVar[str] = "1.0.0"

    @staticmethod
    def decide_tier(
        request: RAGQueryRequest,
        intent: QueryIntent,
        filters: RAGFilterSpec,
        store: Any,
        profile: EngineProfile,
        registry: ComponentRegistry | None = None,
    ) -> QueryPlan:
        """Evaluate the adaptive rules and return a QueryPlan without retrieving anything (§8.12)."""
        components = registry or create_default_registry()
        caps = store.capabilities()
        rc = RuleContext(request, intent, filters, store, caps, profile)
        rules = profile.retrieval.adaptive_rules or DEFAULT_ADAPTIVE_RULES

        for number, (predicate, configured) in enumerate(rules, start=1):
            if not components.get("predicate", predicate)(rc):
                continue
            is_override = configured == OVERRIDE_STRATEGY
            strategy = request.strategy_override if is_override else configured
            cls = components.get("strategy", strategy)
            if strategy != NO_STRATEGY and not is_override and not cls.is_runnable(caps):
                continue

            measured = (
                f"scope {rc.stats.total_tokens} tokens, Tier 0 gate {profile.retrieval.full_context_max_tokens}"
                if rc.stats is not None
                else "Tier 0 not evaluated"
            )
            return QueryPlan(
                strategy=strategy,
                tier_decision_reason=f"{strategy}: rule {number} {predicate}; {measured}",
                rule=number,
                requires_index=strategy not in ("full_context", NO_STRATEGY),
                reads=cls.requires if cls is not None else frozenset(),
                applied_filters=filters,
                scope=rc.stats,
            )

        raise StoreCapabilityError("No adaptive rule matched a strategy this store can run", stage="retrieval")

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        registry = ctx.registry or create_default_registry()
        plan = self.decide_tier(ctx.request, ctx.intent, ctx.filters, ctx.store, ctx.profile, registry)
        ctx.last_strategy_used = plan.strategy
        ctx.last_tier_reason = plan.tier_decision_reason
        ctx.last_rule = plan.rule
        ctx.scope_revision = plan.scope.revision if plan.scope is not None else None

        if plan.strategy == NO_STRATEGY:
            return []
        return registry.get("strategy", plan.strategy)().retrieve(ctx)
