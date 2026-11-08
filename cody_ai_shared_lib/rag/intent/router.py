"""IntentRouter implementations and AutoRouter (§7.2, AC-RAG-12, AC-RAG-13, AC-RAG-14, AC-RAG-16)."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel

from ..degradation import DegradationTracker
from ..gateway import ModelGateway
from ..models import ConversationState, QueryIntent, RAGQueryRequest
from ..profile import EngineProfile
from ..usage import UsageTracker
from .heuristic import HeuristicRouter
from .jev_gate import JevGateRouter
from .llm_rewriter import LLMIntentRouter, clamp_intent
from .temporal import TemporalExtractor


class RouterContext(BaseModel):
    """Context passed to router."""
    model_config = {"arbitrary_types_allowed": True}
    usage_tracker: UsageTracker
    degradation_tracker: DegradationTracker
    profile: EngineProfile


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
            search_query=request.query,
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
        intent = self.heuristic.classify(request.query)
        if intent is not None:
            return intent
        return QueryIntent(
            route="rag",
            scope="local_lookup",
            search_query=request.query,
            router_source="heuristic_passthrough",
        )


class AutoRouter:
    """Hybrid layered router: Heuristic -> Jev gate (opt-in) -> LLM rewriter -> Passthrough."""
    name = "auto"
    version = "1.0.0"

    def __init__(
        self,
        gateway: ModelGateway,
        heuristic: HeuristicRouter | None = None,
        jev_gate: JevGateRouter | None = None,
        llm_router: LLMIntentRouter | None = None,
        temporal_extractor: TemporalExtractor | None = None,
    ):
        self.gateway = gateway
        self.heuristic = heuristic or HeuristicRouter()
        self.jev_gate = jev_gate
        self.llm_router = llm_router
        self.temporal_extractor = temporal_extractor

    def route(
        self,
        request: RAGQueryRequest,
        state: ConversationState | None,
        ctx: RouterContext,
    ) -> QueryIntent:
        # Layer 1: Heuristic classification (0 ms, AC-RAG-13)
        h_intent = self.heuristic.classify(request.query)
        if h_intent is not None:
            # If no_retrieval or overview or pure greeting, return immediately
            if h_intent.route == "no_retrieval" or h_intent.scope == "document_overview":
                return h_intent

        # Layer 2: Jev System One fast path (if enabled in profile & constructed, AC-RAG-14)
        if ctx.profile.router.enable_jev and self.jev_gate is not None:
            j_intent = self.jev_gate.route(
                query=request.query,
                usage_tracker=ctx.usage_tracker,
                degradation_tracker=ctx.degradation_tracker,
            )
            if j_intent is not None:
                return j_intent

        # Layer 3: LLM Intent Router (if models configured)
        intent_models = ctx.profile.models.intent or ctx.profile.models.generation
        if self.llm_router is not None and intent_models:
            try:
                history_turns = []
                if request.conversation:
                    for turn in request.conversation.turns[-3:]:
                        history_turns.append({"role": turn.role, "text": turn.text})

                return self.llm_router.route(
                    query=request.query,
                    usage_tracker=ctx.usage_tracker,
                    degradation_tracker=ctx.degradation_tracker,
                    history=history_turns,
                )
            except Exception:
                # If LLM routing fails, fall back to heuristic or passthrough
                if h_intent is not None:
                    return h_intent

        if h_intent is not None:
            return h_intent

        return QueryIntent(
            route="rag",
            scope="local_lookup",
            search_query=request.query,
            router_source="passthrough",
        )
