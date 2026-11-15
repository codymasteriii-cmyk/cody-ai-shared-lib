"""Tier ladder AdaptiveRetrievalStrategy driven by ordered (predicate, strategy) rules (§8.9, §8.12).

How the choice is made: `DEFAULT_ADAPTIVE_RULES` is an ordered list of (predicate, strategy) pairs. The first
predicate that is true, whose strategy the store can run, decides. A profile may replace the list.

Dummy example: a 12-page document in a store with vectors and keywords, the question "list every mention of tea".
    rule 1 has_strategy_override   no override on the request               -> next
    rule 2 route_no_retrieval      the router chose to retrieve             -> next
    rule 3 fits_full_context       scope_stats says 9,000 tokens, the gate is 4,000   -> next
    rule 4 wants_list_chunked      `list_strategy` is off in this profile   -> next
    ...
    rule 7 hybrid_available        vectors and keyword search both exist    -> hybrid_rrf
`plan()` returns this decision without searching, with a reason string naming the rule that fired.
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger("shared-rag")

OVERRIDE_STRATEGY = "$override"
NO_STRATEGY = "none"

DEFAULT_ADAPTIVE_RULES: tuple[tuple[str, str], ...] = (
    ("has_strategy_override", OVERRIDE_STRATEGY),
    ("route_no_retrieval", NO_STRATEGY),
    ("fits_full_context", "full_context"),
    ("wants_list_chunked", "find_then_read"),
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
    notes: list[str] = field(default_factory=list)


def _visible_hint_chars(rc: RuleContext) -> int | None:
    """The caller's doc_total_chars hint, limited to the visible range. Never trusted for a decision."""
    chars = rc.request.doc_total_chars
    if chars is None:
        return None
    if rc.filters.max_char_offset is not None:
        chars = min(chars, rc.filters.max_char_offset)
    if rc.filters.max_char_offset_by_doc:
        chars = min(chars, max(rc.filters.max_char_offset_by_doc.values()))
    return chars


def _hint_exceeds_gate(rc: RuleContext, gate: int) -> bool:
    """True when doc_total_chars proves the visible text cannot fit, avoiding a store call (§8.2)."""
    chars = _visible_hint_chars(rc)
    return chars is not None and chars * _MIN_TOKENS_PER_CHAR > gate


def fits_full_context(rc: RuleContext) -> bool:
    """Tier 0 test: does the whole visible scope fit the gate? It asks the store only when the hint cannot rule it out.

    The caller's `doc_total_chars` hint is used only to say "too big" without a store call; "it fits" is always
    decided from the store's own `scope_stats`, which stays the source of truth (§8.2).
    """
    gate = rc.profile.retrieval.full_context_max_tokens
    if gate <= 0 or not rc.caps.supports_full_scope_fetch or _hint_exceeds_gate(rc, gate):
        return False
    stats = rc.stats = rc.store.scope_stats(rc.filters)
    hint = _visible_hint_chars(rc)
    if stats.total_tokens == 0 and hint:
        # Diagnostic only: the store stays the source of truth (§4.2, INT-06)
        note = (
            f"scope_stats reports 0 tokens but doc_total_chars says {hint} visible chars; "
            "scope_stats must describe the text fetch_full_scope returns, not indexed rows"
        )
        logger.warning(f"[RAG] {note}")
        rc.notes.append(note)
    return 0 < stats.total_tokens <= gate


# The named conditions a rule can use. Each takes a RuleContext and returns True or False; they are looked up by
# name in the component registry, so an app can add its own predicate and name it in `adaptive_rules`.
BUILTIN_PREDICATES = {
    "has_strategy_override": lambda rc: bool(rc.request.strategy_override)
    and rc.request.strategy_override != AdaptiveRetrievalStrategy.name,
    "route_no_retrieval": lambda rc: rc.intent.route == "no_retrieval",
    "fits_full_context": fits_full_context,
    "wants_list_chunked": lambda rc: rc.intent.wants_list
    and rc.profile.retrieval.list_strategy == "find_then_read"
    and rc.caps.supports_full_scope_fetch,
    "wants_overview": lambda rc: rc.intent.scope == "document_overview"
    or rc.profile.retrieval.always_include_summaries,
    "exact_terms_only": lambda rc: bool(rc.intent.exact_terms) and rc.profile.retrieval.enable_keyword_first,
    "hybrid_available": lambda rc: rc.caps.supports_vector_search
    and (rc.caps.supports_keyword_fts or rc.caps.supports_native_hybrid_rpc),
    "vector_available": lambda rc: rc.caps.supports_vector_search,
    "keyword_available": lambda rc: rc.caps.supports_keyword_fts,
}


class AdaptiveRetrievalStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    """Runs the first rule that is true and whose strategy the store can serve (§8.9).

    It holds no state of its own: `decide_tier` is a pure function of the request, the filters, the store's
    declared capabilities and the profile, which is why `plan()` can call it without searching.
    """
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
            # The rule fired. A strategy the store cannot run is skipped, so the ladder falls through to the next rule
            is_override = configured == OVERRIDE_STRATEGY
            strategy = request.strategy_override if is_override else configured
            cls = components.get("strategy", strategy)
            if strategy != NO_STRATEGY and not is_override and not cls.is_runnable(caps):
                continue

            # The reason string is for humans reading a trace: which rule, and the numbers it was decided on
            measured = (
                f"scope {rc.stats.total_tokens} tokens, Tier 0 gate {profile.retrieval.full_context_max_tokens}"
                if rc.stats is not None
                else "Tier 0 not evaluated"
            )
            if rc.notes:
                measured += "; note: " + "; ".join(rc.notes)
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
        """Decide the tier, record the decision on the context for the trace, then delegate to the chosen strategy."""
        registry = ctx.registry or create_default_registry()
        plan = self.decide_tier(ctx.request, ctx.intent, ctx.filters, ctx.store, ctx.profile, registry)
        ctx.last_strategy_used = plan.strategy
        ctx.last_tier_reason = plan.tier_decision_reason
        ctx.last_rule = plan.rule
        ctx.scope_revision = plan.scope.revision if plan.scope is not None else None

        if plan.strategy == NO_STRATEGY:
            return []
        return registry.get("strategy", plan.strategy)().retrieve(ctx)
