"""Retrieval pipeline: filters, strategy, post-stages, relaxation, hydration, guards (§5.3, §7.6, §8, §9)."""
from __future__ import annotations

import logging
from typing import Any, Callable, Generic, Literal, TypeVar

from .errors import (
    EmbeddingSignatureMismatch,
    ProfileConfigError,
    RAGEngineError,
    RetrievalError,
)
from .guards import BUILTIN_GUARDS, build_turn_guards
from .models import MetaT, QueryIntent, RAGFilterSpec, RAGNode, RAGQueryRequest, RAGRetrievalResult
from .ports import Guard, RAGStore, validate_scope
from .registry import ComponentDeps, split_component_name
from .runtime import RequestRun
from .strategies.context import QueryVariant, RetrievalContext, RetrievalStrategy
from .strategies.relaxation import RelaxationState, build_relaxation
from .strategies.transforms import fuse_variants

logger = logging.getLogger("shared-rag.retrieval")

T = TypeVar("T")

# Stages the engine always runs itself; naming them in a profile changes nothing (§9).
ENGINE_OWNED_STAGES = frozenset({"hydrate", "guard_enforcement"})


class RetrievalPipeline(Generic[MetaT]):
    """Turns a request and intent into guarded, ranked, hydrated nodes."""

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

    def _build_stage(self, configured: str) -> Any:
        name, argument = split_component_name(configured)
        factory = self.registry.get("stage", name)
        return factory(self.deps, argument) if argument is not None else factory(self.deps)

    # ── Filters and guards ────────────────────────────────────────────────

    def effective_filters(self, request: RAGQueryRequest, intent: QueryIntent | None) -> RAGFilterSpec:
        """Combine request filters, the cosine floor and the intent's soft date window (§5.5, §9.1)."""
        filters = request.filters.model_copy(deep=True)
        if request.tenant_id and not filters.tenant_or_user_id:
            filters.tenant_or_user_id = request.tenant_id
        if request.document_id and not filters.doc_ids:
            filters.doc_ids = [request.document_id]

        floor = self.profile.retrieval.min_similarity
        if floor is not None:
            filters.min_similarity = max(floor, filters.min_similarity or floor)

        if self.has_soft_date_window(request, intent):
            filters.date_range_utc = (intent.range_start_utc, intent.range_end_utc)
        return filters

    @staticmethod
    def has_soft_date_window(request: RAGQueryRequest, intent: QueryIntent | None) -> bool:
        return (
            intent is not None
            and intent.has_time_expression
            and request.filters.date_range_utc is None
            and bool(intent.range_start_utc or intent.range_end_utc)
        )

    def turn_guards(self, request: RAGQueryRequest, filters: RAGFilterSpec, soft_date: bool = False) -> list[Guard]:
        """Build the guards active for this turn; a soft date window is not a guard (D-04)."""
        guard_filters = filters.model_copy(update={"date_range_utc": None}) if soft_date else filters
        return build_turn_guards(
            filters=guard_filters,
            analyzer=self.deps.analyzer,
            profile_guards=self.profile.guards,
            extra_guards=[*self.engine_guards, *self.named_guards, *request.guards],
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

        nodes = self._execute(ctx, strategy, stages, guards, run)
        nodes, ctx, relaxations = self._relax(nodes, ctx, strategy, stages, guards, soft_date, request, run)

        if ctx.last_strategy_used != "full_context":
            limit = ctx.target_limit()
            ctx.truncated = ctx.truncated or len(nodes) > limit
            nodes = nodes[:limit]

        nodes = self._hydrate(nodes, run)
        nodes = self._enforce_guards(nodes, guards, "guard_enforcement", run)

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
        for transform in self.transforms:
            for variant in transform.transform(request, intent):
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
        """Run the strategy and every stage, with guards after each one (§5.3)."""
        started = run.trace.elapsed_ms()
        nodes = self._fetch(ctx, strategy)
        run.trace.record_stage(
            name=ctx.last_strategy_used or strategy.name,
            kind="strategy",
            started_ms=started,
            version=strategy.version,
            out_count=len(nodes),
            notes={"reason": ctx.last_tier_reason},
            nodes=nodes,
        )
        nodes = self._enforce_guards(nodes, guards, "strategy", run)
        nodes = self._reuse_prior_nodes(nodes, ctx, guards, run)

        for stage in stages:
            budget_key = getattr(stage, "deadline_budget_key", None)
            if budget_key is not None:
                budget = self.profile.runtime.stage_budgets.get(budget_key, 0.0)
                if run.degradations.skip_for_deadline(run.deadline, stage.name, budget):
                    continue
            started = run.trace.elapsed_ms()
            in_count = len(nodes)
            nodes = self._store_call(stage.name, lambda: stage.process(nodes, ctx))
            run.trace.record_stage(
                name=stage.name,
                kind="post_stage",
                started_ms=started,
                version=stage.version,
                in_count=in_count,
                out_count=len(nodes),
                nodes=nodes,
            )
            nodes = self._enforce_guards(nodes, guards, stage.name, run)
        return nodes

    def _fetch(self, ctx: RetrievalContext[MetaT], strategy: RetrievalStrategy[MetaT]) -> list[RAGNode[MetaT]]:
        if len(ctx.variants) <= 1:
            return self._store_call(strategy.name, lambda: strategy.retrieve(ctx))

        def run_variant(variant: QueryVariant) -> tuple[list[RAGNode[MetaT]], RetrievalContext[MetaT]]:
            intent = ctx.intent.model_copy(update={"search_query": variant.text})
            # Already on a pool worker: a strategy that waited on this pool could deadlock it
            variant_ctx = ctx.model_copy(update={"intent": intent, "pool": None})
            return self._store_call(strategy.name, lambda: strategy.retrieve(variant_ctx)), variant_ctx

        futures = [self.pool.submit(run_variant, variant) for variant in ctx.variants]
        results = [future.result() for future in futures]
        for _, variant_ctx in results:
            if variant_ctx.last_strategy_used:
                ctx.last_strategy_used = variant_ctx.last_strategy_used
                ctx.last_tier_reason = variant_ctx.last_tier_reason
                ctx.last_rule = variant_ctx.last_rule
                ctx.scope_revision = variant_ctx.scope_revision
                break
        return fuse_variants(
            [nodes for nodes, _ in results],
            [variant.weight for variant in ctx.variants],
            ctx.target_limit(),
            ctx.profile.retrieval.rrf_k,
        )

    def _reuse_prior_nodes(
        self,
        nodes: list[RAGNode[MetaT]],
        ctx: RetrievalContext[MetaT],
        guards: list[Guard],
        run: RequestRun,
    ) -> list[RAGNode[MetaT]]:
        """Re-fetch a follow-up's prior nodes under the current filters and guards (§7.7)."""
        state = ctx.request.state
        if not (ctx.intent.is_followup and state and state.prior_nodes and self.store.capabilities().supports_fetch_nodes):
            return nodes
        prior_ids = [ref.node_id for ref in state.prior_nodes]
        fetched = self._store_call("fetch_nodes", lambda: self.store.fetch_nodes(prior_ids, ctx.filters))
        reused = self._enforce_guards(fetched, guards, "reuse", run)
        seen = {n.node_id for n in nodes}
        return [n for n in reused if n.node_id not in seen] + nodes

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
        state = RelaxationState(ctx.filters, ctx.intent, soft_date, request.filters.min_similarity)
        for step in self.relaxation_steps:
            if len(nodes) >= self.profile.retrieval.min_results_before_relax:
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

    def _hydrate(self, nodes: list[RAGNode[MetaT]], run: RequestRun) -> list[RAGNode[MetaT]]:
        if not nodes or not self.store.capabilities().supports_hydration:
            return nodes
        started = run.trace.elapsed_ms()
        projection = self.profile.hydration_projection
        metadata = self._store_call("hydrate", lambda: self.store.hydrate([n.node_id for n in nodes], projection=projection))
        for node in nodes:
            if node.node_id in metadata:
                node.metadata = metadata[node.node_id]
                node.hydrated = True
        run.trace.record_stage("hydrate", "post_stage", started, in_count=len(nodes), out_count=len(metadata))
        return nodes

    def _enforce_guards(
        self,
        nodes: list[RAGNode[MetaT]],
        guards: list[Guard],
        point: str,
        run: RequestRun,
    ) -> list[RAGNode[MetaT]]:
        return apply_guards(nodes, guards, run.trace.record_guard_effect)

    @staticmethod
    def _store_call(operation: str, call: Callable[[], T]) -> T:
        """Typed engine errors pass through; anything else from a store becomes RetrievalError (§13.3)."""
        try:
            return call()
        except RAGEngineError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Retrieval operation '{operation}' failed: {type(exc).__name__}",
                stage="retrieval",
                cause=exc,
            ) from exc


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
