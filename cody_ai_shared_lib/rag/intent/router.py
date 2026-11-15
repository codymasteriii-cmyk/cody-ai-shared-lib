"""IntentRouter implementations and HybridIntentRouter (§7.2, AC-RAG-12, AC-RAG-13, AC-RAG-14, AC-RAG-16).

A router reads the question (and a little history) and returns a `QueryIntent`: what kind of question it is, what
to search for, which names and exact terms matter, whether it is a follow-up. Four routers, chosen by the profile's
`router.name`:

    "none"       PassthroughRouter       no model call; searches with the question as written
    "heuristic"  HeuristicIntentRouter   regex rules only (intent/heuristic.py), no model call
    "llm"        LLMOnlyRouter           always asks the intent model (intent/llm_rewriter.py)
    "auto"       HybridIntentRouter      the layered default, below

HybridIntentRouter.route() tries cheap layers first and stops at the first that decides:

    route()
    ├─ heuristic.classify()      0 ms; decides at once for small talk and "summarize this" questions
    ├─ jev_gate.route()          (opt-in) a fast classifier; decides only when it is confident
    ├─ llm_router.route()        the intent model, when `models.intent` is set; the usual path for real questions
    └─ passthrough + dates       no model configured: use the heuristic result or the raw question, add date words

Dummy example: "thanks!" -> heuristic says no_retrieval (no search, no model). "summarize chapter 3" -> heuristic
says document_overview. "what did Li Wei do after the festival?" -> the heuristic has no opinion, so the intent
model runs and returns search_query "Li Wei actions after festival", entity_terms ["Li Wei"], is_followup False.
"""
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
    """What a router may use besides the question: where to record usage and degradations, the profile, the deadline."""
    model_config = {"arbitrary_types_allowed": True}
    usage_tracker: UsageTracker
    degradation_tracker: DegradationTracker
    profile: EngineProfile
    budget_guard: BudgetGuard | None = None
    deadline: Any = None


@runtime_checkable
class IntentRouter(Protocol):
    """The contract for a router. An app can supply its own (e.g. one that sets `rules` from app knowledge)
    through `RAGEngine(router=...)`; it must return a `QueryIntent` and may use `state` to resolve follow-ups."""
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
    """Ask the intent model, giving it only the last three turns: enough to resolve "he", cheap to send."""
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
        """The Jev gate is optional work: off unless the profile enables it, and skipped when the deadline is short (DG-10)."""
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
            # Only the two confident cases end here; other heuristic hits (exact terms, a list cue) are kept
            # as a fallback below and may still be refined by the model
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

        # No model is configured: the heuristic's reading, or the question as written, plus any date words in it
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


# The `build_*_router` functions are the registry factories: the engine calls them with its shared dependencies
# when the profile names a router, so a profile can say router.name = "auto" without building anything by hand.
def _build_llm_rewriter(deps: Any) -> LLMIntentRouter | None:
    """The intent-model router, or None when the profile configures no intent model."""
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
