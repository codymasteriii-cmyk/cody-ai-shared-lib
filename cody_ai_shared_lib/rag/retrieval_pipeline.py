"""Retrieval pipeline: filters, strategy, follow-up reuse, post-stages, relaxation, guards (§5.3, §7.6, §7.7, §8, §9)."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Generic, Literal

from .context.conversation import FollowupTermCheck, reusable_node_ids
from .errors import EmbeddingSignatureMismatch, ProfileConfigError
from .granularity import is_ranked, limit_ranked
from .guards import BUILTIN_GUARDS, build_turn_guards
from .models import MetaT, QueryIntent, RAGFilterSpec, RAGNode, RAGQueryRequest, RAGRetrievalResult, as_utc
from .ports import Guard, RAGStore, call_store, validate_scope
from .registry import ComponentDeps, split_component_name
from .runtime import RequestRun
from .strategies.adaptive import AdaptiveRetrievalStrategy
from .strategies.context import RetrievalContext, RetrievalStrategy
from .strategies.relaxation import RelaxationState, build_relaxation
from .transforms import QueryVariant, TransformContext, fair_interleave, fuse_variants

logger = logging.getLogger("shared-rag.retrieval")

# The stage the engine always runs itself; naming it in a profile changes nothing (§9).
ENGINE_OWNED_STAGES = frozenset({"guard_enforcement"})

CONVERSATION_REUSE = "conversation_reuse"


class RetrievalPipeline(Generic[MetaT]):
    """Turns a request and intent into guarded, ranked nodes."""
    filter_inferencer: Any = None   # set by RAGEngine (§5.5)


    def __init__(
        self,
        deps: ComponentDeps,
        store: RAGStore[MetaT],
        pool: Any,
        cache: Any,
        budget_guard: Any,
        engine_guards: list[Guard],
    ):
        self.deps = deps
        self.profile = deps.profile
        self.registry = deps.registry
        self.store = store
        self.pool = pool
        self.cache = cache
        self.budget_guard = budget_guard
        self.engine_guards = list(engine_guards)
        self.named_guards: list[Guard] = [
            self.registry.get("guard", name)(deps) for name in self.profile.guards if name not in BUILTIN_GUARDS
        ]
        self.stages = [self._build_stage(name) for name in self.profile.post_stages if name not in ENGINE_OWNED_STAGES]
        self.transforms = [self.registry.get("transform", name)(deps) for name in self.profile.transforms]
        self.relaxation_steps = [build_relaxation(step, self.registry) for step in self.profile.retrieval.relaxation]
        self.followup_terms = FollowupTermCheck(deps.analyzer)

    def _build_stage(self, configured: str) -> Any:
        name, argument = split_component_name(configured)
        factory = self.registry.get("stage", name)
        return factory(self.deps, argument) if argument is not None else factory(self.deps)

    # ── Filters and guards ────────────────────────────────────────────────

    def effective_filters(self, request: RAGQueryRequest, intent: QueryIntent | None) -> RAGFilterSpec:
        """Combine request filters, the cosine floor and the intent's soft date window (§5.5, §9.1)."""
        filters = request.filters.patch(
            {
                "tenant_or_user_id": request.tenant_id,
                "doc_ids": [request.document_id] if request.document_id else None,
            }
        )

        inferencer = getattr(self, "filter_inferencer", None)
        if inferencer is not None:
            proposed = inferencer.infer(request, intent)
            if proposed:
                filters = filters.patch(proposed)       # narrow-only: intersected, never unioned (§5.5)

        floor = self.profile.retrieval.min_similarity
        if floor is not None:
            filters.min_similarity = max(floor, filters.min_similarity or floor)

        window = self.soft_date_window(request, intent)
        if window is not None:
            filters.date_range_utc = window
        return filters

    @staticmethod
    def soft_date_window(
        request: RAGQueryRequest, intent: QueryIntent | None
    ) -> tuple[datetime | None, datetime | None] | None:
        """The intent's time window intersected with the caller's hard range (§5.5, D-04).

        None when the intent has no window, or when the window lies outside the hard range: a soft
        constraint never empties a hard scope, so the hard range alone applies then.
        """
        if intent is None or not intent.has_time_expression or not (intent.range_start_utc or intent.range_end_utc):
            return None
        start, end = _utc(intent.range_start_utc), _utc(intent.range_end_utc)
        hard = request.filters.date_range_utc
        if hard is not None:
            hard_start, hard_end = _utc(hard[0]), _utc(hard[1])
            start = max(start, hard_start) if (start and hard_start) else (start or hard_start)
            end = min(end, hard_end) if (end and hard_end) else (end or hard_end)
            if start and end and start > end:
                return None
        return start, end

    @classmethod
    def has_soft_date_window(cls, request: RAGQueryRequest, intent: QueryIntent | None) -> bool:
        return cls.soft_date_window(request, intent) is not None

    def turn_guards(self, request: RAGQueryRequest, filters: RAGFilterSpec, soft_date: bool = False) -> list[Guard]:
        """Build the guards active for this turn; a soft date window is not a guard (D-04)."""
        # The guard enforces only the caller's hard range; the soft window is the relaxable part (D-04)
        guard_filters = (
            filters.model_copy(update={"date_range_utc": request.filters.date_range_utc}) if soft_date else filters
        )
        return build_turn_guards(
            filters=guard_filters,
            analyzer=self.deps.analyzer,
            profile_guards=self.profile.guards,
            extra_guards=[*self.engine_guards, *self.named_guards, *request.guards],
            renderer=self.deps.renderer,
            store_date_fields=self.store.capabilities().supported_date_fields,
        )

    def scoped_filters(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent | None,
    ) -> tuple[RAGFilterSpec, list[Guard], bool]:
        """Return validated filters with every guard's patch applied, before any store call (§5.6)."""
        soft_date = self.has_soft_date_window(request, intent)
        filters = self.effective_filters(request, intent)
        guards = self.turn_guards(request, filters, soft_date)
        for guard in guards:
            filters = filters.patch(guard.to_filter())
        if not filters.empty_scope:
            # A hard conflict means no store call happens at all; a missing key is still an error
            validate_scope(filters, self.store.capabilities())
        return filters, guards, soft_date

    # ── Pipeline ──────────────────────────────────────────────────────────

    def run(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent,
        mode: Literal["query", "retrieve"],
        run: RequestRun,
    ) -> tuple[RAGRetrievalResult[MetaT], list[Guard]]:
        """Run retrieval and return the result with the guards that were enforced."""
        space = self.store.embedding_space()
        if space.model != self.profile.models.embedding:
            raise EmbeddingSignatureMismatch(
                f"Store embedding space '{space.model}' does not match profile '{self.profile.models.embedding}'",
                stage="retrieval",
            )
        run.deadline.check("retrieval")

        filters, guards, soft_date = self.scoped_filters(request, intent)
        if filters.is_empty_scope:
            return self._empty_scope_result(intent, filters, run), guards

        ctx = self._build_context(request, intent, filters, mode, run)
        ctx.variants = self._variants(request, intent, run)
        strategy = self._resolve_strategy(request)
        stages = self._stages_for(request)
        ctx.rerank_active = any(stage.name == "rerank" for stage in stages)

        nodes = self._execute(ctx, strategy, stages, guards, run)
        nodes, ctx, relaxations = self._relax(nodes, ctx, strategy, stages, guards, soft_date, request, run)

        limited = limit_ranked(nodes, ctx.target_limit())
        ctx.truncated = ctx.truncated or len(limited) < len(nodes)
        nodes = self._enforce_guards(limited, guards, run)

        run.trace.tier_decision_reason = ctx.last_tier_reason
        run.trace.scope_revision = ctx.scope_revision
        run.trace.rule = ctx.last_rule or None
        run.trace.components[strategy.name] = strategy.version

        result = RAGRetrievalResult[MetaT](
            nodes=nodes,
            intent=intent,
            strategy_used=ctx.last_strategy_used or strategy.name,
            tier_decision_reason=ctx.last_tier_reason,
            applied_filters=ctx.filters,
            relaxations=relaxations,
            reranked=ctx.reranked,
            truncated=ctx.truncated,
            degradations=run.degradations.degradations,
            usage=run.usage.stage_usages,
        )
        return result, guards

    def _empty_scope_result(
        self,
        intent: QueryIntent,
        filters: RAGFilterSpec,
        run: RequestRun,
    ) -> RAGRetrievalResult[MetaT]:
        reason = "empty_scope: hard filters match nothing"
        run.trace.tier_decision_reason = reason
        return RAGRetrievalResult[MetaT](
            intent=intent,
            strategy_used="none",
            tier_decision_reason=reason,
            applied_filters=filters,
            degradations=run.degradations.degradations,
            usage=run.usage.stage_usages,
        )

    def _build_context(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent,
        filters: RAGFilterSpec,
        mode: Literal["query", "retrieve"],
        run: RequestRun,
    ) -> RetrievalContext[MetaT]:
        return RetrievalContext[MetaT](
            request=request,
            intent=intent,
            variants=[],
            filters=filters,
            store=self.store,
            profile=self.profile,
            analyzer=self.deps.analyzer,
            renderer=self.deps.renderer,
            gateway=self.deps.gateway,
            deadline=run.deadline,
            usage_tracker=run.usage,
            degradation_tracker=run.degradations,
            mode=mode,
            budget_guard=self.budget_guard,
            pool=self.pool,
            cache=self.cache,
            registry=self.registry,
        )

    def _resolve_strategy(self, request: RAGQueryRequest) -> RetrievalStrategy[MetaT]:
        name = request.strategy_override or self.profile.retrieval.default_strategy
        cls = self.registry.get("strategy", name)
        if cls is None:
            raise ProfileConfigError(f"Unknown retrieval strategy '{name}'", unknown=[f"strategy:{name}"])
        return cls()

    def _variants(self, request: RAGQueryRequest, intent: QueryIntent, run: RequestRun) -> list[QueryVariant]:
        primary = QueryVariant(text=intent.search_query or request.question, weight=1.0, origin="primary")
        variants = [primary]
        seen = {primary.text}
        context = TransformContext(
            profile=self.profile,
            analyzer=self.deps.analyzer,
            gateway=self.deps.gateway,
            usage_tracker=run.usage,
            degradation_tracker=run.degradations,
            deadline=run.deadline,
            budget_guard=self.budget_guard,
        )
        for transform in self.transforms:
            for variant in transform.transform(request, intent, context):
                if variant.text not in seen:
                    seen.add(variant.text)
                    variants.append(variant)

        budget = self.profile.runtime.stage_budgets.get("secondary_queries", 0.0)
        if len(variants) > 1 and run.degradations.skip_for_deadline(run.deadline, "secondary_queries", budget):
            return [primary]
        return variants

    def _stages_for(self, request: RAGQueryRequest) -> list[Any]:
        """Apply the per-request reranker override to the profile's stage list."""
        override = request.enable_reranker_override
        if override is False:
            return [s for s in self.stages if s.name != "rerank"]
        if override is True and self.profile.models.reranker and not any(s.name == "rerank" for s in self.stages):
            stage = self.registry.get("stage", "rerank")(self.deps)
            insert_at = next((i for i, s in enumerate(self.stages) if s.name == "top_k"), len(self.stages))
            return [*self.stages[:insert_at], stage, *self.stages[insert_at:]]
        return self.stages

    def _execute(
        self,
        ctx: RetrievalContext[MetaT],
        strategy: RetrievalStrategy[MetaT],
        stages: list[Any],
        guards: list[Guard],
        run: RequestRun,
    ) -> list[RAGNode[MetaT]]:
        """Find candidates, then run every stage, with guards after each one (§5.3)."""
        prior, complete = self._prior_nodes(ctx, guards, run)
        if complete and self._reuse_alone_suffices(ctx, prior):
            ctx.last_strategy_used = CONVERSATION_REUSE
            ctx.last_tier_reason = f"{CONVERSATION_REUSE}: follow-up introduces no new terms"
            nodes = prior
        else:
            started = run.trace.elapsed_ms()
            ctx.last_strategy_used = ""
            nodes = self._fetch(ctx, strategy)
            ctx.last_strategy_used = ctx.last_strategy_used or strategy.name
            run.trace.record_stage(
                name=ctx.last_strategy_used,
                kind="strategy",
                started_ms=started,
                version=strategy.version,
                out_count=len(nodes),
                notes={"reason": ctx.last_tier_reason},
                nodes=nodes,
            )
            nodes = self._merge_prior(self._enforce_guards(nodes, guards, run), prior, ctx)

        for stage in stages:
            budget_key = getattr(stage, "deadline_budget_key", None)
            if budget_key is not None:
                budget = self.profile.runtime.stage_budgets.get(budget_key, 0.0)
                if run.degradations.skip_for_deadline(run.deadline, stage.name, budget):
                    continue
            started = run.trace.elapsed_ms()
            in_count = len(nodes)
            nodes = call_store(stage.name, lambda: stage.process(nodes, ctx))
            run.trace.record_stage(
                name=stage.name,
                kind="post_stage",
                started_ms=started,
                version=stage.version,
                in_count=in_count,
                out_count=len(nodes),
                nodes=nodes,
            )
            nodes = self._enforce_guards(nodes, guards, run)
        return nodes

    def _query_independent(self, ctx: RetrievalContext[MetaT], strategy: RetrievalStrategy[MetaT]) -> bool:
        """True when the tier this request runs gives the same result for every query variant (§8.2)."""
        if getattr(strategy, "query_independent", False):
            return True
        if not isinstance(strategy, AdaptiveRetrievalStrategy):
            return False
        plan = AdaptiveRetrievalStrategy.decide_tier(
            ctx.request, ctx.intent, ctx.filters, ctx.store, ctx.profile, ctx.registry
        )
        if plan.strategy == "none":
            return True
        chosen = (ctx.registry or self.deps.registry).get("strategy", plan.strategy)
        return bool(getattr(chosen, "query_independent", False))

    def _fetch(self, ctx: RetrievalContext[MetaT], strategy: RetrievalStrategy[MetaT]) -> list[RAGNode[MetaT]]:
        if len(ctx.variants) <= 1 or self._query_independent(ctx, strategy):
            # Tier 0 and "no retrieval" do not depend on the query: run them once, not per variant
            return call_store(strategy.name, lambda: strategy.retrieve(ctx))

        def run_variant(variant: QueryVariant) -> tuple[list[RAGNode[MetaT]], RetrievalContext[MetaT]]:
            intent = ctx.intent.model_copy(update={"search_query": variant.text})
            # Already on a pool worker: a strategy that waited on this pool could deadlock it
            variant_ctx = ctx.model_copy(update={"intent": intent, "pool": None})
            return call_store(strategy.name, lambda: strategy.retrieve(variant_ctx)), variant_ctx

        futures = [self.pool.submit(run_variant, variant) for variant in ctx.variants]
        results = [future.result() for future in futures]
        for _, variant_ctx in results:
            if variant_ctx.last_strategy_used:
                ctx.last_strategy_used = variant_ctx.last_strategy_used
                ctx.last_tier_reason = variant_ctx.last_tier_reason
                ctx.last_rule = variant_ctx.last_rule
                ctx.scope_revision = variant_ctx.scope_revision
                break
        # Only ranked nodes compete across variants; every variant sees the same ordered layers
        ordered: dict[str, RAGNode[MetaT]] = {}
        for nodes, _ in results:
            for node in nodes:
                if not is_ranked(node):
                    ordered.setdefault(node.node_id, node)
        ranked = fuse_variants(
            [[node for node in nodes if is_ranked(node)] for nodes, _ in results],
            [variant.weight for variant in ctx.variants],
            ctx.candidate_limit(),
            ctx.profile.retrieval.rrf_k,
        )
        return [*ordered.values(), *ranked]

    def _prior_nodes(
        self,
        ctx: RetrievalContext[MetaT],
        guards: list[Guard],
        run: RequestRun,
    ) -> tuple[list[RAGNode[MetaT]], bool]:
        """Re-fetch a follow-up's earlier nodes, newest first, under the current filters and guards (§7.7).

        Also returns whether every earlier node came back; one that is gone or now hidden needs a fresh search.
        """
        state = ctx.request.state
        if not (ctx.intent.is_followup and state and state.prior_nodes and self.store.capabilities().supports_fetch_nodes):
            return [], False
        limits = self.profile.context
        prior_ids = reusable_node_ids(state, limits.reused_turns, limits.max_reused_nodes)
        fetched = call_store("fetch_nodes", lambda: self.store.fetch_nodes(prior_ids, ctx.filters))
        position = {node_id: index for index, node_id in enumerate(prior_ids)}
        fetched.sort(key=lambda node: position.get(node.node_id, len(position)))
        guarded = self._enforce_guards(fetched, guards, run)
        return guarded, len(guarded) == len(prior_ids)

    def _reuse_alone_suffices(self, ctx: RetrievalContext[MetaT], prior: list[RAGNode[MetaT]]) -> bool:
        """True when a search would add nothing. Ordered layers and a reading position must stay current."""
        if ctx.request.cursor_offset is not None or not prior or not all(is_ranked(node) for node in prior):
            return False
        query = " ".join([ctx.intent.search_query or ctx.request.question, *ctx.intent.exact_terms, *ctx.intent.entity_terms])
        return not self.followup_terms.introduces_new_terms(query, (self.deps.renderer.prompt_text(n) for n in prior))

    def _merge_prior(
        self,
        fresh: list[RAGNode[MetaT]],
        prior: list[RAGNode[MetaT]],
        ctx: RetrievalContext[MetaT],
    ) -> list[RAGNode[MetaT]]:
        """Give earlier and new passages a fair share of the ranked slots (§7.7)."""
        ranked_prior = [node for node in prior if is_ranked(node)]
        if not ranked_prior:
            return fresh
        ordered = [node for node in fresh if not is_ranked(node)]
        ranked_fresh = [node for node in fresh if is_ranked(node)]
        limit = ctx.candidate_limit() if ctx.rerank_active else ctx.target_limit()
        merged = fair_interleave([ranked_prior, ranked_fresh], limit)
        distinct = {node.node_id for node in [*ranked_prior, *ranked_fresh]}
        ctx.truncated = ctx.truncated or len(distinct) > len(merged)
        return [*ordered, *merged]

    def _relax(
        self,
        nodes: list[RAGNode[MetaT]],
        ctx: RetrievalContext[MetaT],
        strategy: RetrievalStrategy[MetaT],
        stages: list[Any],
        guards: list[Guard],
        soft_date: bool,
        request: RAGQueryRequest,
        run: RequestRun,
    ) -> tuple[list[RAGNode[MetaT]], RetrievalContext[MetaT], list[str]]:
        """Walk the opt-in ladder while results are too few; hard limits stay out of reach (§7.6)."""
        applied: list[str] = []
        state = RelaxationState(
            ctx.filters, ctx.intent, soft_date, request.filters.min_similarity, request.filters.date_range_utc
        )
        budget = self.profile.runtime.stage_budgets.get("relaxation", 0.0)
        for step in self.relaxation_steps:
            if len(nodes) >= self.profile.retrieval.min_results_before_relax:
                break
            if run.degradations.skip_for_deadline(run.deadline, f"relaxation:{step.label}", budget):
                break
            relaxed = step.apply(state)
            if relaxed is None:
                continue
            relaxed_ctx = ctx.model_copy(update={"filters": relaxed.filters, "intent": relaxed.intent})
            relaxed_nodes = self._execute(relaxed_ctx, strategy, stages, guards, run)
            if len(relaxed_nodes) > len(nodes):
                nodes, ctx, state = relaxed_nodes, relaxed_ctx, relaxed
                applied.append(step.label)
        return nodes, ctx, applied

    def _enforce_guards(
        self,
        nodes: list[RAGNode[MetaT]],
        guards: list[Guard],
        run: RequestRun,
    ) -> list[RAGNode[MetaT]]:
        return apply_guards(nodes, guards, run.trace.record_guard_effect)


def apply_guards(
    nodes: list[RAGNode[MetaT]],
    guards: list[Guard],
    record_effect: Callable[[str, int, int], None] | None = None,
) -> list[RAGNode[MetaT]]:
    """Run every guard's verify over the nodes, reporting how many each dropped or truncated."""
    result = nodes
    for guard in guards:
        kept: list[RAGNode[MetaT]] = []
        truncated = 0
        for node in result:
            verified = guard.verify(node)
            if verified is None:
                continue
            if verified is not node and "truncated_by_guard" in verified.flags:
                truncated += 1
            kept.append(verified)
        if record_effect is not None and (truncated or len(kept) != len(result)):
            record_effect(guard.name, len(result) - len(kept), truncated)
        result = kept
    return result


def _utc(value: datetime | None) -> datetime | None:
    return as_utc(value) if value is not None else None
