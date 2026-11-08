"""RAGEngine public facade (§14, Phase P1).

Coordinates query understanding, retrieval tiers, guards, context assembly,
grounded generation, post-generation citation verification, sanitization,
spend accounting, and observable tracing.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Generic

from .context.assembler import ContextAssembler
from .context.conversation import ConversationManager
from .context.fence import ContextFencer
from .degradation import DegradationTracker
from .errors import GenerationError, MissingRequiredScopeError, ProfileConfigError, StoreCapabilityError
from .gateway import ModelGateway
from .generation.stuff import StuffStrategy
from .guards import ScopeGuard, SequentialBoundaryGuard, TemporalGuard, TenantGuard
from .intent.heuristic import HeuristicRouter
from .intent.jev_gate import JevGateRouter
from .intent.llm_rewriter import LLMIntentRouter, clamp_intent
from .intent.router import AutoRouter, HeuristicIntentRouter, IntentRouter, PassthroughRouter, RouterContext
from .intent.temporal import TemporalExtractor
from .models import (
    CitationCoverage,
    ConversationState,
    IngestionProgress,
    MetaT,
    NodeRef,
    QueryIntent,
    QueryPlan,
    RAGFilterSpec,
    RAGNode,
    RAGQueryRequest,
    RAGResponse,
    RAGRetrievalResult,
    SourceSpan,
    StageUsage,
)
from .ports import Guard, NodeRenderer, RAGStore
from .postgen.citations import CitationExtractor
from .postgen.quotes import QuoteVerifier
from .postgen.sanitizer import OutputSanitizer
from .postretrieval.dedup import IdentityDeduplicator, OffsetDeduplicator
from .postretrieval.rerank import JinaCrossEncoderReranker
from .postretrieval.threshold import ThresholdFilter
from .profile import EngineProfile
from .registry import ComponentRegistry, create_default_registry
from .render import DefaultRenderer
from .runtime import Clock, Deadline, Rng, SystemClock, SystemRng
from .strategies.adaptive import AdaptiveStrategy
from .strategies.context import QueryVariant, RetrievalContext, RetrievalStrategy
from .strategies.dense_vector import DenseVectorStrategy
from .strategies.full_context import FullContextStrategy
from .strategies.hybrid_rrf import HybridRRFStrategy
from .strategies.keyword import KeywordFirstStrategy, KeywordOnlyStrategy
from .strategies.transforms import SecondaryQueryFanOut, fair_interleave
from .trace import ObserverDispatcher, RAGObserver, RAGTrace, StageRecord
from .usage import BudgetGuard, UsageTracker


class RAGEngine(Generic[MetaT]):
    """Main facade for the shared RAG engine."""

    def __init__(
        self,
        profile: EngineProfile,
        store: RAGStore[MetaT],
        gateway: ModelGateway | None = None,
        clock: Clock | None = None,
        rng: Rng | None = None,
        registry: ComponentRegistry | None = None,
        observers: list[RAGObserver] | None = None,
        budget_guard: BudgetGuard | None = None,
        renderer: NodeRenderer[MetaT] | None = None,
        guards: list[Guard] | None = None,
    ):
        self.profile = profile
        self.store = store
        self.gateway = gateway or ModelGateway()
        self.clock = clock or SystemClock()
        self.rng = rng or SystemRng()
        self.registry = registry or create_default_registry()
        self.budget_guard = budget_guard
        self.renderer = renderer or DefaultRenderer()
        self.dispatcher = ObserverDispatcher(
            observers=observers,
            observer_errors=profile.observer_errors,
        )

        # 1. Startup validation (AC-RAG-49, AC-RAG-62)
        self.profile.validate(self.registry)

        # 2. Wire components
        self.temporal_extractor = TemporalExtractor(clock=self.clock)
        self.fencer = ContextFencer(rng=self.rng)
        self.assembler = ContextAssembler(
            profile=self.profile,
            renderer=self.renderer,
            fencer=self.fencer,
        )
        self.conversation_manager = ConversationManager()
        self.generator = StuffStrategy(gateway=self.gateway, profile=self.profile)
        self.citation_extractor = CitationExtractor()
        self.quote_verifier = QuoteVerifier()
        self.sanitizer = OutputSanitizer()

        # Guards
        self.guards = list(guards or [])
        if not self.guards:
            self._init_builtin_guards()

        # Router setup
        self._init_router()

    def _init_builtin_guards(self) -> None:
        """Instantiate guards configured in profile."""
        for g_name in self.profile.guards:
            if g_name == "sequential_boundary":
                self.guards.append(SequentialBoundaryGuard())
            elif g_name == "temporal":
                self.guards.append(TemporalGuard())
            elif g_name == "tenant":
                self.guards.append(TenantGuard())
            elif g_name == "scope":
                self.guards.append(ScopeGuard())

    def _init_router(self) -> None:
        """Instantiate router based on profile configuration."""
        r_name = self.profile.router.name
        heuristic = HeuristicRouter()

        if r_name == "none":
            self.router: IntentRouter = PassthroughRouter()
        elif r_name == "heuristic":
            self.router = HeuristicIntentRouter(heuristic=heuristic)
        else:
            # Auto / Hybrid
            jev_gate = None
            if self.profile.router.enable_jev:
                jev_gate = JevGateRouter(
                    gateway=self.gateway,
                    timeout=self.profile.router.jev_timeout,
                )

            intent_models = self.profile.models.intent or self.profile.models.generation
            llm_router = None
            if intent_models:
                llm_router = LLMIntentRouter(
                    gateway=self.gateway,
                    models=intent_models,
                    temporal_extractor=self.temporal_extractor if self.profile.router.enable_temporal_extraction else None,
                )

            self.router = AutoRouter(
                gateway=self.gateway,
                heuristic=heuristic,
                jev_gate=jev_gate,
                llm_router=llm_router,
                temporal_extractor=self.temporal_extractor,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def route(self, request: RAGQueryRequest) -> QueryIntent:
        """Public routing entry point (§7.2, AC-RAG-16)."""
        tracker = UsageTracker(request_id=request.query)
        degradations = DegradationTracker(fallbacks=self.profile.fallbacks)
        ctx = RouterContext(
            usage_tracker=tracker,
            degradation_tracker=degradations,
            profile=self.profile,
        )
        return self.router.route(request, request.conversation, ctx)

    def plan(self, request: RAGQueryRequest, intent: QueryIntent | None = None) -> QueryPlan:
        """Evaluates adaptive strategy rules without retrieval or generation (§8.12)."""
        resolved_intent = intent or QueryIntent(
            route="rag",
            scope="local_lookup",
            search_query=request.query,
            router_source="passthrough",
        )
        effective_filters = self._build_effective_filters(request)
        stats = self.store.scope_stats(effective_filters)
        max_t0 = self.profile.retrieval.full_context_max_tokens

        # Rule evaluation
        if max_t0 > 0 and stats.total_tokens > 0 and stats.total_tokens <= max_t0:
            return QueryPlan(
                strategy="full_context",
                tier_decision_reason=f"full_context: {stats.total_tokens} <= {max_t0} tokens",
                rule=1,
                requires_index=False,
                reads=frozenset(["supports_full_scope_fetch"]),
                applied_filters=effective_filters,
                scope=stats,
            )

        if resolved_intent.scope == "document_overview" and self.profile.retrieval.always_include_summaries:
            return QueryPlan(
                strategy="hierarchical",
                tier_decision_reason="summaries: document overview query",
                rule=2,
                requires_index=False,
                reads=frozenset(["fetch_summaries"]),
                applied_filters=effective_filters,
                scope=stats,
            )

        if resolved_intent.exact_terms and self.store.capabilities.supports_keyword_fts:
            return QueryPlan(
                strategy="keyword_first",
                tier_decision_reason="keyword_first: exact terms in query",
                rule=3,
                requires_index=True,
                reads=frozenset(["supports_keyword_fts"]),
                applied_filters=effective_filters,
                scope=stats,
            )

        return QueryPlan(
            strategy="hybrid_rrf" if self.store.capabilities.supports_native_hybrid else "dense_vector",
            tier_decision_reason=f"hybrid_rrf: scope {stats.total_tokens} tokens exceeded Tier 0 gate {max_t0}",
            rule=4,
            requires_index=True,
            reads=frozenset(["supports_vector_search"]),
            applied_filters=effective_filters,
            scope=stats,
        )

    def retrieve(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent | None = None,
    ) -> RAGRetrievalResult[MetaT]:
        """Ranked list retrieval mode (§8.10, AC-RAG-51)."""
        t0 = self.clock.monotonic()
        req_id = f"req-{uuid.uuid4().hex[:12]}"
        usage_tracker = UsageTracker(request_id=req_id, tenant_or_user_id=request.tenant_id)
        degradations = DegradationTracker(fallbacks=self.profile.fallbacks)
        deadline = Deadline(deadline_seconds=self.profile.runtime.deadline_seconds, clock=self.clock)

        # 1. Resolve Intent
        resolved_intent = intent or self.route(request)
        if intent is not None:
            resolved_intent = clamp_intent(intent.model_dump(), degradations)

        effective_filters = self._build_effective_filters(request)

        # 2. Run guards before_retrieval
        for guard in self.guards:
            patch = guard.to_filter()
            if patch is not None:
                effective_filters = effective_filters.patch(patch)

        # 3. Build Retrieval Context
        ret_ctx = RetrievalContext[MetaT](
            request=request,
            intent=resolved_intent,
            variants=[QueryVariant(text=resolved_intent.search_query or request.query, weight=1.0)],
            filters=effective_filters,
            store=self.store,
            profile=self.profile,
            analyzer=self.registry.get("analyzer", self.profile.analyzer)(),
            renderer=self.renderer,
            gateway=self.gateway,
            deadline=deadline,
            usage_tracker=usage_tracker,
            degradation_tracker=degradations,
        )

        # 4. Fan out query variants if comparisons present (AC-RAG-19)
        fanout = SecondaryQueryFanOut()
        ret_ctx.variants = fanout.transform(request, resolved_intent)

        # 5. Execute Strategy
        strat_name = self.profile.retrieval.default_strategy
        if strat_name == "adaptive":
            strategy: RetrievalStrategy[MetaT] = AdaptiveStrategy()
        elif strat_name == "full_context":
            strategy = FullContextStrategy()
        elif strat_name == "hybrid_rrf":
            strategy = HybridRRFStrategy()
        elif strat_name == "keyword_only":
            strategy = KeywordOnlyStrategy()
        elif strat_name == "keyword_first":
            strategy = KeywordFirstStrategy()
        else:
            strategy = DenseVectorStrategy()

        # Multi-variant execution and fair interleave
        if len(ret_ctx.variants) > 1:
            variant_results: list[list[RAGNode[MetaT]]] = []
            for v in ret_ctx.variants:
                v_intent = resolved_intent.model_copy(update={"search_query": v.text})
                v_ctx = ret_ctx.model_copy(update={"intent": v_intent})
                v_nodes = strategy.retrieve(v_ctx)
                variant_results.append(v_nodes)
            top_k = resolved_intent.suggested_top_k or self.profile.retrieval.top_k
            nodes = fair_interleave(variant_results, top_k=top_k)
        else:
            nodes = strategy.retrieve(ret_ctx)

        # 6. Apply post-retrieval guards
        for guard in self.guards:
            verified_nodes: list[RAGNode[MetaT]] = []
            for n in nodes:
                res = guard.verify(n)
                if res is not None:
                    verified_nodes.append(res)
            nodes = verified_nodes

        # 7. Post-retrieval pipeline stages
        # Threshold filter (AC-RAG-31)
        min_sim = self.profile.retrieval.min_similarity
        if min_sim is not None and min_sim > 0.0:
            nodes = ThresholdFilter[MetaT](min_similarity=min_sim).filter(nodes)

        # Deduplication (AC-RAG-32, AC-RAG-33)
        if "offset_dedup" in self.profile.post_stages:
            nodes = OffsetDeduplicator[MetaT]().deduplicate(nodes)
        if "identity_dedup" in self.profile.post_stages:
            nodes = IdentityDeduplicator[MetaT]().deduplicate(nodes)

        # Rerank if configured (AC-RAG-34)
        reranked = False
        if "rerank" in self.profile.post_stages and self.profile.models.reranker:
            reranker = JinaCrossEncoderReranker[MetaT](
                gateway=self.gateway,
                model=self.profile.models.reranker,
            )
            nodes = reranker.rerank(
                query=resolved_intent.search_query or request.query,
                nodes=nodes,
                usage_tracker=usage_tracker,
                degradation_tracker=degradations,
            )
            reranked = True

        tier_reason = request.metadata.get("tier_decision_reason", "") if request.metadata else ""
        strat_used = request.metadata.get("strategy_used", strategy.name) if request.metadata else strategy.name

        # Cap to top_k (Tier 0 full_context preserves full scope)
        if strat_used != "full_context":
            top_k = resolved_intent.suggested_top_k or self.profile.retrieval.top_k
            nodes = nodes[:top_k]

        trace = RAGTrace(
            request_id=req_id,
            profile_version=self.profile.version,
            total_latency_ms=(self.clock.monotonic() - t0) * 1000.0,
            level=self.profile.trace_level,
        )

        return RAGRetrievalResult[MetaT](
            nodes=nodes,
            intent=resolved_intent,
            strategy_used=strat_used,
            tier_decision_reason=tier_reason,
            applied_filters=effective_filters,
            reranked=reranked,
            degradations=degradations.degradations,
            usage=usage_tracker.stage_usages,
            trace=trace,
        )

    def query(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent | None = None,
    ) -> RAGResponse[MetaT]:
        """Main end-to-end RAG question answering pipeline (§14)."""
        t0 = self.clock.monotonic()
        req_id = f"req-{uuid.uuid4().hex[:12]}"
        usage_tracker = UsageTracker(request_id=req_id, tenant_or_user_id=request.tenant_id)
        degradations = DegradationTracker(fallbacks=self.profile.fallbacks)
        deadline = Deadline(deadline_seconds=self.profile.runtime.deadline_seconds, clock=self.clock)

        # 1. Routing
        resolved_intent = intent or self.route(request)
        if intent is not None:
            resolved_intent = clamp_intent(intent.model_dump(), degradations)

        # Check no-retrieval route (greetings / chitchat, AC-RAG-12)
        if resolved_intent.route == "no_retrieval":
            gen_res = self.gateway.generate(
                system_prompt="You are a helpful, polite assistant. Respond warmly to the user.",
                user_message=request.query,
                models=self.profile.models.generation,
                stage="generation_no_retrieval",
                usage_tracker=usage_tracker,
                degradation_tracker=degradations,
                budget_guard=self.budget_guard,
                request_id=req_id,
                tenant_or_user_id=request.tenant_id,
            )
            lat_ms = (self.clock.monotonic() - t0) * 1000.0
            total_u = usage_tracker.total()
            return RAGResponse[MetaT](
                answer_id=f"ans-{uuid.uuid4().hex[:12]}",
                answer=gen_res.text,
                answer_final=gen_res.text,
                grounded=True,
                answer_found="yes",
                retrieved=RAGRetrievalResult[MetaT](
                    intent=resolved_intent,
                    strategy_used="none",
                    tier_decision_reason="no_retrieval route selected",
                ),
                usage=usage_tracker.stage_usages,
                usage_total=total_u,
                degradations=degradations.degradations,
                latency_ms=lat_ms,
            )

        # 2. Candidate Retrieval
        ret_result = self.retrieve(request, intent=resolved_intent)
        for d in ret_result.degradations:
            degradations.record(d.id, d.stage, d.reason, d.impact, d.fallback_used)
        for u in ret_result.usage:
            usage_tracker.record(u)

        nodes = ret_result.nodes

        # 3. Context Assembly & Fencing (AC-RAG-37)
        state_fence_id = request.conversation.fence_id if request.conversation else None
        sys_prompt, user_msg, active_fence_id = self.assembler.assemble(
            request=request,
            nodes=nodes,
            fence_id=state_fence_id,
        )

        # 4. Generation
        gen_res = self.generator.generate(
            system_prompt=sys_prompt,
            user_message=user_msg,
            usage_tracker=usage_tracker,
            degradation_tracker=degradations,
            budget_guard=self.budget_guard,
            request_id=req_id,
            tenant_or_user_id=request.tenant_id,
        )

        # 5. Post-generation: citation extraction, quote verification, sanitization
        clean_text, citations = self.citation_extractor.extract(gen_res.text, nodes)
        if self.profile.citations.verify_quotes:
            citations, coverage = self.quote_verifier.verify(citations, nodes)
        else:
            coverage = CitationCoverage(
                total_citations=len(citations),
                verified_citations=len(citations),
                coverage_ratio=1.0,
            )

        final_answer = self.sanitizer.sanitize(clean_text)

        # Answer found classification
        if not nodes:
            answer_found = "no"
            citations = []
        else:
            answer_found = "yes"
            not_found_phrases = ["cannot be found", "not found in the provided", "not enough information"]
            if any(p in final_answer.lower() for p in not_found_phrases):
                answer_found = "no"

        # Update conversation state
        conv_state = request.conversation or ConversationState(fence_id=active_fence_id)
        conv_state.fence_id = active_fence_id
        for n in nodes:
            conv_state.prior_nodes.append(NodeRef(node_id=n.id))
        conv_state = self.conversation_manager.prune_state(conv_state)

        total_u = usage_tracker.total()
        lat_ms = (self.clock.monotonic() - t0) * 1000.0

        trace = RAGTrace(
            request_id=req_id,
            profile_version=self.profile.version,
            total_latency_ms=lat_ms,
            state_size_bytes=len(conv_state.model_dump_json().encode("utf-8")),
            level=self.profile.trace_level,
        )

        resp = RAGResponse[MetaT](
            answer_id=f"ans-{uuid.uuid4().hex[:12]}",
            answer=gen_res.text,
            answer_final=final_answer,
            citations=citations,
            sources=citations,
            retrieved=ret_result,
            grounded=True,
            answer_found=answer_found,
            citation_coverage=coverage,
            state=conv_state,
            usage=usage_tracker.stage_usages,
            usage_total=total_u,
            degradations=degradations.degradations,
            latency_ms=lat_ms,
            trace=trace,
        )

        self.dispatcher.notify_complete(trace, resp)
        return resp

    def ingest_document(
        self,
        document_id: str,
        text: str,
        metadata: MetaT | None = None,
        section_id: str = "main",
        chunk_size: int = 600,
        chunk_overlap: int = 50,
    ) -> IngestionProgress:
        """Splits document, embeds nodes, and atomically upserts into persistent store."""
        t0 = self.clock.monotonic()
        nodes = self.prepare_nodes(
            document_id=document_id,
            text=text,
            metadata=metadata,
            section_id=section_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Embed nodes
        texts_to_embed = [n.text for n in nodes]
        tracker = UsageTracker(request_id=f"ingest-{document_id}")
        emb_res = self.gateway.embed(
            texts=texts_to_embed,
            model=self.profile.models.embedding,
            stage="ingest_embedding",
            usage_tracker=tracker,
            task_type="RETRIEVAL_DOCUMENT",
        )

        for i, vec in enumerate(emb_res.vectors):
            nodes[i].embedding = vec

        # Upsert into store atomically
        self.store.upsert_section(
            document_id=document_id,
            section_id=section_id,
            nodes=nodes,
            metadata=metadata,
        )

        lat_ms = (self.clock.monotonic() - t0) * 1000.0
        return IngestionProgress(
            document_id=document_id,
            section_id=section_id,
            node_count=len(nodes),
            token_count=emb_res.usage.input_tokens,
            status="done",
            total_time_ms=lat_ms,
        )

    def prepare_nodes(
        self,
        document_id: str,
        text: str,
        metadata: MetaT | None = None,
        section_id: str = "main",
        chunk_size: int = 600,
        chunk_overlap: int = 50,
    ) -> list[RAGNode[MetaT]]:
        """Slice document into chunked RAGNode items with character offsets."""
        nodes: list[RAGNode[MetaT]] = []
        pos = 0
        order = 0
        length = len(text)

        while pos < length:
            end = min(pos + chunk_size, length)
            chunk_str = text[pos:end]
            node = RAGNode[MetaT](
                id=f"{document_id}_{section_id}_{order}",
                document_id=document_id,
                section_id=section_id,
                order_index=order,
                text=chunk_str,
                span=SourceSpan(
                    char_start=pos,
                    char_end=end,
                    position_ratio=pos / length if length > 0 else 0.0,
                ),
                metadata=metadata,
            )
            nodes.append(node)
            order += 1
            if end >= length:
                break
            pos = end - chunk_overlap

        return nodes

    def needs_reindex(self, document_id: str) -> bool:
        """Check whether stored embeddings match current embedder."""
        stored_fp = self.store.fingerprint()
        current_fp = self.profile.fingerprint()
        return stored_fp.profile_fingerprint != current_fp

    def _build_effective_filters(self, request: RAGQueryRequest) -> RAGFilterSpec:
        """Combine request-level filters, document_id, and tenant_id into RAGFilterSpec."""
        f = request.filters.model_copy() if request.filters else RAGFilterSpec()
        if request.tenant_id and not f.tenant_id:
            f.tenant_id = request.tenant_id
        if request.document_id and not f.document_ids:
            f.document_ids = [request.document_id]
        return f
