"""IntentRouter implementations and HybridIntentRouter (§7.2, AC-RAG-12, AC-RAG-13, AC-RAG-14, AC-RAG-16)."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel

from ..degradation import DegradationTracker
from ..errors import ProfileConfigError
from ..models import ConversationState, QueryIntent, RAGQueryRequest
from ..profile import EngineProfile
from ..usage import BudgetGuard, UsageTracker
from .heuristic import HeuristicRouter
from .jev_gate import JevGateRouter
from .llm_rewriter import LLMIntentRouter
from .temporal import TemporalResolver


class RouterContext(BaseModel):
    """Context passed to router."""
    model_config = {"arbitrary_types_allowed": True}
    usage_tracker: UsageTracker
    degradation_tracker: DegradationTracker
    profile: EngineProfile
    budget_guard: BudgetGuard | None = None
    deadline: Any = None


@runtime_checkable
class IntentRouter(Protocol):
    name: str
    version: str

    def route(
        self,
        request: RAGQueryRequest,
        state: ConversationState | None,
        ctx: RouterContext,
    ) -> QueryIntent:
        ...


class PassthroughRouter:
    """Zero model calls; preserves query text unchanged (AC-RAG-12)."""
    name = "none"
    version = "1.0.0"

    def route(
        self,
        request: RAGQueryRequest,
        state: ConversationState | None,
        ctx: RouterContext,
    ) -> QueryIntent:
        return QueryIntent(
            route="rag",
            scope="local_lookup",
            search_query=request.question,
            router_source="passthrough",
        )


class HeuristicIntentRouter:
    """Rule-based router without LLM or Jev calls (AC-RAG-13)."""
    name = "heuristic"
    version = "1.0.0"

    def __init__(self, heuristic: HeuristicRouter | None = None):
        self.heuristic = heuristic or HeuristicRouter()

    def route(
        self,
        request: RAGQueryRequest,
        state: ConversationState | None,
        ctx: RouterContext,
    ) -> QueryIntent:
        intent = self.heuristic.classify(request.question)
        if intent is not None:
            return intent
        return QueryIntent(
            route="rag",
            scope="local_lookup",
            search_query=request.question,
            router_source="heuristic_passthrough",
        )


def _route_with_llm(llm_router: LLMIntentRouter, request: RAGQueryRequest, ctx: RouterContext) -> QueryIntent:
    return llm_router.route(
        query=request.question,
        usage_tracker=ctx.usage_tracker,
        degradation_tracker=ctx.degradation_tracker,
        history=[{"role": turn.role, "content": turn.content} for turn in request.history[-3:]],
        budget_guard=ctx.budget_guard,
    )


class LLMOnlyRouter:
    """Always uses the LLM rewriter (§7.5)."""
    name = "llm"
    version = "1.0.0"

    def __init__(self, llm_router: LLMIntentRouter):
        self.llm_router = llm_router

    def route(
        self,
        request: RAGQueryRequest,
        state: ConversationState | None,
        ctx: RouterContext,
    ) -> QueryIntent:
        return _route_with_llm(self.llm_router, request, ctx)


class HybridIntentRouter:
    """Hybrid layered router: Heuristic -> Jev gate (opt-in) -> LLM rewriter -> Passthrough."""
    name = "auto"
    version = "1.0.0"

    def __init__(
        self,
        heuristic: HeuristicRouter | None = None,
        jev_gate: JevGateRouter | None = None,
        llm_router: LLMIntentRouter | None = None,
        temporal_resolver: TemporalResolver | None = None,
    ):
        self.heuristic = heuristic or HeuristicRouter()
        self.jev_gate = jev_gate
        self.llm_router = llm_router
        self.temporal_resolver = temporal_resolver

    def _jev_gate_usable(self, ctx: RouterContext) -> bool:
        if not ctx.profile.router.enable_jev or self.jev_gate is None:
            return False
        if ctx.deadline is None:
            return True
        return not ctx.degradation_tracker.skip_for_deadline(ctx.deadline, "intent_jev", ctx.profile.router.jev_timeout)

    def route(
        self,
        request: RAGQueryRequest,
        state: ConversationState | None,
        ctx: RouterContext,
    ) -> QueryIntent:
        # Layer 1: Heuristic classification (0 ms, AC-RAG-13)
        h_intent = self.heuristic.classify(request.question)
        if h_intent is not None:
            if h_intent.route == "no_retrieval" or h_intent.scope == "document_overview":
                return h_intent

        # Layer 2: Jev System One fast path (if enabled in profile & constructed, AC-RAG-14)
        if self._jev_gate_usable(ctx):
            j_intent = self.jev_gate.route(
                query=request.question,
                usage_tracker=ctx.usage_tracker,
                degradation_tracker=ctx.degradation_tracker,
                budget_guard=ctx.budget_guard,
                fallbacks=ctx.profile.fallbacks,
            )
            if j_intent is not None:
                return j_intent

        # Layer 3: LLM Intent Router (if models configured)
        intent_models = ctx.profile.models.intent
        if self.llm_router is not None and intent_models:
            return _route_with_llm(self.llm_router, request, ctx)

        intent = h_intent if h_intent is not None else QueryIntent(
            route="rag",
            scope="local_lookup",
            search_query=request.question,
            router_source="passthrough",
        )

        if ctx.profile.router.enable_temporal_extraction and self.temporal_resolver and not intent.has_time_expression:
            temp_info = self.temporal_resolver.extract(request.question)
            if temp_info.get("has_time_expression"):
                intent.has_time_expression = True
                intent.temporal_scope = temp_info.get("temporal_scope", "recent")
                intent.date_window_days = temp_info.get("date_window_days")
                intent.range_start_utc = temp_info.get("range_start_utc")
                intent.range_end_utc = temp_info.get("range_end_utc")

        return intent


def _build_llm_rewriter(deps: Any) -> LLMIntentRouter | None:
    models = deps.profile.models.intent
    if not models:
        return None
    temporal = deps.temporal_resolver if deps.profile.router.enable_temporal_extraction else None
    return LLMIntentRouter(gateway=deps.gateway, models=models, temporal_resolver=temporal)


def build_passthrough_router(deps: Any) -> PassthroughRouter:
    return PassthroughRouter()


def build_heuristic_router(deps: Any) -> HeuristicIntentRouter:
    return HeuristicIntentRouter()


def build_llm_router(deps: Any) -> LLMOnlyRouter:
    rewriter = _build_llm_rewriter(deps)
    if rewriter is None:
        raise ProfileConfigError("Router 'llm' requires models.intent to be configured", missing=["models.intent"])
    return LLMOnlyRouter(rewriter)


def build_auto_router(deps: Any) -> HybridIntentRouter:
    jev_gate = None
    if deps.profile.router.enable_jev:
        jev_gate = JevGateRouter(gateway=deps.gateway, timeout=deps.profile.router.jev_timeout)
    return HybridIntentRouter(
        llm_router=_build_llm_rewriter(deps),
        jev_gate=jev_gate,
        temporal_resolver=deps.temporal_resolver,
    )
