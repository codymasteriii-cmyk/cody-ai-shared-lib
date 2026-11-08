"""RAGEngine public facade (§14, Phase P1).

Coordinates query understanding, retrieval tiers, guards, context assembly,
grounded generation, post-generation citation verification, sanitization,
spend accounting, and observable tracing.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Callable, Generic

from .context.assembler import ContextAssembler
from .context.conversation import ConversationManager
from .context.fence import ContextFencer
from .degradation import DegradationTracker
from .embedder import BatchEmbedder
from .errors import (
    EmbeddingSignatureMismatch,
    GenerationError,
    MissingRequiredScopeError,
    ProfileConfigError,
    StoreCapabilityError,
)
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
    PreparedNode,
    QueryIntent,
    QueryPlan,
    RAGDocumentInput,
    RAGFilterSpec,
    RAGNode,
    RAGQueryRequest,
    RAGResponse,
    RAGRetrievalResult,
    RAGSectionInput,
    ReindexStatus,
    SourceSpan,
    StageUsage,
)
from .ports import Fingerprint, Guard, NodeRenderer, RAGStore
from .postgen.citations import CitationExtractor
from .postgen.quotes import QuoteVerifier
from .postgen.sanitizer import OutputSanitizer
from .summarizer import SectionSummarizer
from .text.analyzer import CjkAwareAnalyzer, estimate_tokens
from .text.splitter import PassThroughSplitter, SentenceSplitter
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
from .strategies.hierarchical import HierarchicalStrategy
from .strategies.hybrid_rrf import HybridRRFStrategy
from .strategies.keyword import KeywordFirstStrategy, KeywordOnlyStrategy
from .strategies.transforms import SecondaryQueryFanOut, fair_interleave
from .trace import ObserverDispatcher, RAGObserver, RAGTrace, StageRecord
from .usage import BudgetGuard, CallContext, UsageTracker


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

        # Text analyzer & splitter
        analyzer_cls = self.registry.get("analyzer", self.profile.analyzer) or CjkAwareAnalyzer
        self.analyzer = analyzer_cls() if callable(analyzer_cls) else analyzer_cls

        splitter_cls = self.registry.get("splitter", self.profile.ingestion.splitter) or SentenceSplitter
        if callable(splitter_cls):
            try:
                self.splitter = splitter_cls(
                    target_size=self.profile.ingestion.target_size,
                    overlap=self.profile.ingestion.overlap,
                    analyzer=self.analyzer,
                )
            except TypeError:
                self.splitter = splitter_cls()
        else:
            self.splitter = splitter_cls

        self.embedder = BatchEmbedder(
            gateway=self.gateway,
            model=self.profile.models.embedding,
            renderer=self.renderer,
        )

        self.summarizer = SectionSummarizer(
            gateway=self.gateway,
            summary_models=self.profile.models.summary,
            enabled=self.profile.ingestion.summaries.enabled,
            min_chars_for_summary=self.profile.ingestion.summaries.min_chars_for_summary,
            summary_unit_chars=self.profile.ingestion.summaries.summary_unit_chars,
            analyzer=self.analyzer,
            fallback_action=self.profile.fallbacks.get("summarizer", "raise"),
        )

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
        """Evaluates adaptive strategy rules without retrieval or generation (§8.12, AC-RAG-48, AC-RAG-58)."""
        resolved_intent = intent or QueryIntent(
            route="rag",
            scope="local_lookup",
            search_query=request.query,
            router_source="passthrough",
        )
        effective_filters = self._build_effective_filters(request)
        for g in self.guards + list(request.guards):
            patch = g.to_filter()
            if patch is not None:
                effective_filters = effective_filters.patch(patch)

        stats = self.store.scope_stats(effective_filters)
        max_t0 = self.profile.retrieval.full_context_max_tokens
        caps = self.store.capabilities() if callable(self.store.capabilities) else self.store.capabilities

        # Rule evaluation matching §8.9 Adaptive Rules
        if (
            max_t0 > 0
            and stats.total_tokens > 0
            and stats.total_tokens <= max_t0
            and caps.supports_full_scope_fetch
        ):
            return QueryPlan(
                strategy="full_context",
                tier_decision_reason=f"full_context: {stats.total_tokens} <= {max_t0} tokens",
                rule=1,
                requires_index=False,
                reads=frozenset(["supports_full_scope_fetch"]),
                applied_filters=effective_filters,
                scope=stats,
            )

        if (
            (resolved_intent.scope == "document_overview" or self.profile.retrieval.always_include_summaries)
            and caps.supports_parent_summaries
        ):
            return QueryPlan(
                strategy="hierarchical",
                tier_decision_reason="hierarchical: document overview query or always_include_summaries",
                rule=2,
                requires_index=True,
                reads=frozenset(["supports_parent_summaries", "supports_vector_search"]),
                applied_filters=effective_filters,
                scope=stats,
            )

        if resolved_intent.exact_terms and caps.supports_keyword_fts:
            return QueryPlan(
                strategy="keyword_first",
                tier_decision_reason="keyword_first: exact terms in query",
                rule=3,
                requires_index=True,
                reads=frozenset(["supports_keyword_fts"]),
                applied_filters=effective_filters,
                scope=stats,
            )

        strat_name = "hybrid_rrf" if caps.supports_native_hybrid else "dense_vector"
        return QueryPlan(
            strategy=strat_name,
            tier_decision_reason=f"{strat_name}: scope {stats.total_tokens} tokens exceeded Tier 0 gate {max_t0}",
            rule=4,
            requires_index=True,
            reads=frozenset(["supports_vector_search"]),
            applied_filters=effective_filters,
            scope=stats,
        )

    def validate(self) -> None:
        """Validate engine configuration and store compatibility (AC-RAG-07, AC-RAG-49)."""
        self.profile.validate(self.registry)
        store_space = self.store.embedding_space()
        if store_space.model != self.profile.models.embedding:
            raise ProfileConfigError(
                f"Store embedding space model '{store_space.model}' does not match "
                f"profile embedding model '{self.profile.models.embedding}'"
            )

    def retrieve(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent | None = None,
    ) -> RAGRetrievalResult[MetaT]:
        """Ranked list retrieval mode (§8.10, AC-RAG-51)."""
        # Check embedding space match (RAG-ING-7, AC-RAG-07)
        store_space = self.store.embedding_space()
        if store_space.model != self.profile.models.embedding:
            raise EmbeddingSignatureMismatch(
                f"Store embedding space '{store_space.model}' does not match profile '{self.profile.models.embedding}'"
            )

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
        all_guards = list(self.guards) + list(request.guards)

        # 2. Run guards before_retrieval
        for guard in all_guards:
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
        strat_name = request.strategy_override or self.profile.retrieval.default_strategy
        if strat_name == "adaptive":
            strategy: RetrievalStrategy[MetaT] = AdaptiveStrategy()
        elif strat_name == "hierarchical":
            strategy = HierarchicalStrategy()
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

        # Follow-up node reuse (§7.7, AC-RAG-29, AC-RAG-30)
        caps = self.store.capabilities() if callable(self.store.capabilities) else self.store.capabilities
        if request.conversation and request.conversation.prior_nodes and caps.supports_fetch_nodes:
            prior_ids = [ref.node_id for ref in request.conversation.prior_nodes]
            reused = self.store.fetch_nodes(prior_ids, effective_filters)
            guarded_reused: list[RAGNode[MetaT]] = []
            for rn in reused:
                passed = rn
                for g in all_guards:
                    passed = g.verify(passed) if passed is not None else None
                if passed is not None:
                    guarded_reused.append(passed)
            seen_ids = {n.id for n in nodes}
            nodes = [rn for rn in guarded_reused if rn.id not in seen_ids] + nodes

        # 6. Apply post-retrieval guards
        for guard in all_guards:
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
        doc: RAGDocumentInput[MetaT] | str,
        start_section_seq: int = 0,
        on_section_complete: Callable[[IngestionProgress], None] | None = None,
        text: str | None = None,
        metadata: MetaT | None = None,
        section_id: str = "main",
        chunk_size: int = 600,
        chunk_overlap: int = 50,
    ) -> IngestionProgress:
        """Splits document, embeds nodes, generates summaries, and atomically upserts into persistent store.

        Resumable via start_section_seq and append-only (RAG-ING-5, RAG-ING-6, AC-RAG-05).
        Enforces atomic sections (RAG-ING-10, AC-RAG-66) and budget guard (RAG-ING-12, AC-RAG-65).
        """
        t0 = self.clock.monotonic()

        # Handle legacy string/text positional arguments
        if isinstance(doc, str):
            doc_id = doc
            sec = RAGSectionInput[MetaT](
                section_id=section_id,
                section_seq=0,
                text=text or "",
                char_start=0,
                metadata=metadata or {},  # type: ignore[arg-type]
            )
            doc_input = RAGDocumentInput[MetaT](doc_id=doc_id, sections=[sec])
        else:
            doc_input = doc

        sections_to_process = [
            s for s in sorted(doc_input.sections, key=lambda x: x.section_seq)
            if s.section_seq >= start_section_seq
        ]

        total_passages_upserted = 0
        total_failed = 0
        last_progress = None
        completed_count = 0

        for sec in sections_to_process:
            tracker = UsageTracker(request_id=f"ingest-{doc_input.doc_id}-s{sec.section_seq}")
            deg_tracker = DegradationTracker(fallbacks=self.profile.fallbacks)

            # 1. Budget check before calls (RAG-ING-12, AC-RAG-65)
            if self.budget_guard:
                est_in_tokens = estimate_tokens(sec.text)
                self.budget_guard.before_call(
                    CallContext(
                        stage="ingest_embedding",
                        request_id=f"ingest-{doc_input.doc_id}",
                        est_input_tokens=est_in_tokens,
                        est_cost_usd=None,
                        request_spent=tracker.total(),
                    )
                )

            # 2. Summarize section (RAG-ING-3, RAG-ING-9, AC-RAG-08, AC-RAG-59)
            sum_res = self.summarizer.summarize_section(
                doc_id=doc_input.doc_id,
                section_seq=sec.section_seq,
                section_id=sec.section_id,
                text=sec.text,
                char_start=sec.char_start,
                metadata=sec.metadata,
                existing_summary=sec.existing_summary,
                usage_tracker=tracker,
                degradation_tracker=deg_tracker,
            )

            # 3. Split section text into passage nodes (RAG-ING-1, RAG-ING-2, AC-RAG-01)
            chunks = self.splitter.split(sec.text, base_offset=sec.char_start)
            nodes: list[RAGNode[MetaT]] = []
            for p_idx, chunk in enumerate(chunks):
                node_id = f"{doc_input.doc_id}#s{sec.section_seq}p{p_idx}"
                n = RAGNode[MetaT](
                    node_id=node_id,
                    doc_id=doc_input.doc_id,
                    section_id=sec.section_id,
                    section_seq=sec.section_seq,
                    granularity="passage",
                    text=chunk.text,
                    span=SourceSpan(
                        doc_id=doc_input.doc_id,
                        section_id=sec.section_id,
                        section_seq=sec.section_seq,
                        section_title=sec.title,
                        char_start=sec.char_start + chunk.local_start,
                        char_end=sec.char_start + chunk.local_end,
                        breadcrumb=list(sec.breadcrumb) + chunk.breadcrumb,
                    ),
                    metadata=sec.metadata,
                )
                n.token_estimate = estimate_tokens(n.text)
                if self.store.capabilities().uses_analyzer_tokens:
                    n.search_tokens = self.analyzer.tokenize(n.text)
                nodes.append(n)

            # 4. Batch embed nodes (RAG-ING-4, AC-RAG-04)
            valid_pairs = self.embedder.embed_nodes(nodes, tracker, deg_tracker)
            failed_in_sec = len(nodes) - len(valid_pairs)
            total_failed += failed_in_sec

            valid_nodes = [p[0] for p in valid_pairs]
            valid_embeddings = [p[1] for p in valid_pairs]

            # 5. Populate fingerprints on nodes
            dims = self.store.embedding_space().dims
            splitter_ver = self.store.embedding_space().splitter_version or getattr(self.splitter, "version", "1.0.0")
            emb_sig = f"{self.profile.models.embedding}|{dims}|RETRIEVAL_DOCUMENT|{splitter_ver}"
            fts_sig = f"{self.analyzer.name}|{self.analyzer.version}" if self.store.capabilities().uses_analyzer_tokens else None

            for n in valid_nodes:
                content_hash = hashlib.sha256(self.analyzer.normalize(self.renderer.embed_text(n)).encode("utf-8")).hexdigest()
                n.content_hash = content_hash

            # Also embed summary node if generated
            summary_node = sum_res.section_summary
            if summary_node:
                s_pairs = self.embedder.embed_nodes([summary_node], tracker, deg_tracker, stage="summary_embedding")
                if s_pairs:
                    summary_node.embedding = s_pairs[0][1]

            # Also add unit summaries to nodes if present
            for u_node in sum_res.unit_summaries:
                u_pairs = self.embedder.embed_nodes([u_node], tracker, deg_tracker, stage="summary_embedding")
                if u_pairs:
                    u_node.embedding = u_pairs[0][1]
                    valid_nodes.append(u_node)
                    valid_embeddings.append(u_pairs[0][1])

            # 6. Atomic upsert to store (RAG-ING-10, AC-RAG-66)
            written = self.store.upsert_section_nodes(
                doc_id=doc_input.doc_id,
                section_seq=sec.section_seq,
                nodes=valid_nodes,
                embeddings=valid_embeddings,
                summary=summary_node,
            )
            total_passages_upserted += written
            completed_count += 1

            last_progress = IngestionProgress(
                doc_id=doc_input.doc_id,
                section_seq=sec.section_seq,
                sections_completed=completed_count,
                sections_total=len(doc_input.sections),
                passages_upserted=written,
                passages_skipped_unchanged=0,
                passages_failed=failed_in_sec,
                summary_status=sum_res.summary_status,
                usage=tracker.stage_usages,
                degradations=deg_tracker.degradations,
            )

            if on_section_complete:
                on_section_complete(last_progress)

        return last_progress or IngestionProgress(
            doc_id=doc_input.doc_id,
            section_seq=0,
            sections_completed=0,
            sections_total=len(doc_input.sections),
            passages_upserted=0,
            summary_status="skipped",
        )

    def sync_document(self, doc: RAGDocumentInput[MetaT]) -> IngestionProgress:
        """Syncs document: embeds only changed nodes, deletes vanished ones, preserves node ids (§6.6, AC-RAG-06)."""
        stored_fps = self.store.get_fingerprints(doc.doc_id)
        all_section_nodes: list[RAGNode[MetaT]] = []
        for sec in sorted(doc.sections, key=lambda s: s.section_seq):
            chunks = self.splitter.split(sec.text, base_offset=sec.char_start)
            for p_idx, chunk in enumerate(chunks):
                n = RAGNode[MetaT](
                    node_id=f"{doc.doc_id}#s{sec.section_seq}p{p_idx}",
                    doc_id=doc.doc_id,
                    section_id=sec.section_id,
                    section_seq=sec.section_seq,
                    granularity="passage",
                    text=chunk.text,
                    span=SourceSpan(
                        doc_id=doc.doc_id,
                        section_id=sec.section_id,
                        section_seq=sec.section_seq,
                        section_title=sec.title,
                        char_start=sec.char_start + chunk.local_start,
                        char_end=sec.char_start + chunk.local_end,
                        breadcrumb=list(sec.breadcrumb) + chunk.breadcrumb,
                    ),
                    metadata=sec.metadata,
                )
                all_section_nodes.append(n)

        current_node_ids = {n.node_id for n in all_section_nodes}

        # Identify vanished nodes and delete them
        vanished_ids = [nid for nid in stored_fps.keys() if nid not in current_node_ids]
        if vanished_ids and self.store.capabilities().supports_delete:
            self.store.delete_nodes(vanished_ids)

        # Identify changed vs unchanged nodes
        nodes_to_embed: list[RAGNode[MetaT]] = []
        skipped_count = 0
        dims = self.store.embedding_space().dims
        splitter_ver = self.store.embedding_space().splitter_version or getattr(self.splitter, "version", "1.0.0")
        emb_sig = f"{self.profile.models.embedding}|{dims}|RETRIEVAL_DOCUMENT|{splitter_ver}"

        for n in all_section_nodes:
            embed_text = self.renderer.embed_text(n)
            chash = hashlib.sha256(self.analyzer.normalize(embed_text).encode("utf-8")).hexdigest()

            if n.node_id in stored_fps:
                st_fp = stored_fps[n.node_id]
                if st_fp.content_hash == chash and st_fp.embedding_signature == emb_sig:
                    skipped_count += 1
                    continue
            nodes_to_embed.append(n)

        # Embed changed/new nodes
        upserted_count = 0
        tracker = UsageTracker(request_id=f"sync-{doc.doc_id}")
        deg_tracker = DegradationTracker(fallbacks=self.profile.fallbacks)

        if nodes_to_embed:
            pairs = self.embedder.embed_nodes(nodes_to_embed, tracker, deg_tracker)
            by_sec: dict[int, list[tuple[RAGNode[MetaT], list[float]]]] = {}
            for node, vec in pairs:
                by_sec.setdefault(node.span.section_seq, []).append((node, vec))

            for s_seq, s_pairs in by_sec.items():
                s_nodes = [p[0] for p in s_pairs]
                s_vecs = [p[1] for p in s_pairs]
                w = self.store.upsert_section_nodes(
                    doc_id=doc.doc_id,
                    section_seq=s_seq,
                    nodes=s_nodes,
                    embeddings=s_vecs,
                )
                upserted_count += w

        return IngestionProgress(
            doc_id=doc.doc_id,
            section_seq=0,
            sections_completed=len(doc.sections),
            sections_total=len(doc.sections),
            passages_upserted=upserted_count,
            passages_skipped_unchanged=skipped_count,
            passages_failed=0,
            summary_status="skipped",
            usage=tracker.stage_usages,
        )

    def prepare_nodes(
        self,
        nodes_or_doc_id: list[RAGNode[MetaT]] | str,
        text: str | None = None,
        metadata: MetaT | None = None,
        section_id: str = "main",
        chunk_size: int = 600,
        chunk_overlap: int = 50,
    ) -> list[PreparedNode[MetaT]] | list[RAGNode[MetaT]]:
        """Compute embedding, fingerprint, token_estimate, search_tokens without store write (§6.4.1, RAG-ING-13, AC-RAG-71)."""
        if isinstance(nodes_or_doc_id, str):
            return self._legacy_prepare_nodes(nodes_or_doc_id, text or "", metadata, section_id, chunk_size, chunk_overlap)

        nodes = nodes_or_doc_id
        if not nodes:
            return []

        prepared: list[PreparedNode[MetaT]] = []
        tracker = UsageTracker(request_id="prepare-nodes")
        embed_texts = [self.renderer.embed_text(n) for n in nodes]

        emb_res = self.gateway.embed(
            texts=embed_texts,
            model=self.profile.models.embedding,
            stage="prepare_embedding",
            usage_tracker=tracker,
            task_type="RETRIEVAL_DOCUMENT",
        )

        dims = self.store.embedding_space().dims
        splitter_ver = self.store.embedding_space().splitter_version or getattr(self.splitter, "version", "1.0.0")

        for idx, node in enumerate(nodes):
            node_copy = node.model_copy()
            node_copy.token_estimate = estimate_tokens(node_copy.text)
            if self.store.capabilities().uses_analyzer_tokens:
                node_copy.search_tokens = self.analyzer.tokenize(node_copy.text)

            embed_text = self.renderer.embed_text(node_copy)
            content_hash = hashlib.sha256(self.analyzer.normalize(embed_text).encode("utf-8")).hexdigest()
            node_copy.content_hash = content_hash

            emb_sig = f"{self.profile.models.embedding}|{dims}|RETRIEVAL_DOCUMENT|{splitter_ver}"
            fts_sig = f"{self.analyzer.name}|{self.analyzer.version}" if self.store.capabilities().uses_analyzer_tokens else None
            fp = Fingerprint(
                content_hash=content_hash,
                embedding_signature=emb_sig,
                fts_signature=fts_sig,
            )

            raw_vec = emb_res.vectors[idx] if idx < len(emb_res.vectors) else []
            # Failed / zero vector rule (RAG-ING-4, AC-RAG-71): return [], never zero vector
            if not raw_vec or all(v == 0.0 for v in raw_vec):
                vec = []
            else:
                vec = raw_vec

            stage_u = tracker.stage_usages[0] if tracker.stage_usages else None
            prepared.append(
                PreparedNode[MetaT](
                    node=node_copy,
                    vector=vec,
                    fingerprint=fp,
                    usage=stage_u,
                )
            )

        return prepared

    def _legacy_prepare_nodes(
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

    def needs_reindex(self, doc_id: str) -> ReindexStatus:
        """Check whether stored embeddings or FTS tokens are stale (§6.6, AC-RAG-64)."""
        stored_fps = self.store.get_fingerprints(doc_id)
        if not stored_fps:
            return ReindexStatus(doc_id=doc_id, embeddings_stale=False, fts_stale=False)

        dims = self.store.embedding_space().dims
        splitter_ver = self.store.embedding_space().splitter_version or getattr(self.splitter, "version", "1.0.0")
        expected_emb_sig = f"{self.profile.models.embedding}|{dims}|RETRIEVAL_DOCUMENT|{splitter_ver}"
        expected_fts_sig = f"{self.analyzer.name}|{self.analyzer.version}" if self.store.capabilities().uses_analyzer_tokens else None

        emb_stale = False
        fts_stale = False

        for fp in stored_fps.values():
            if fp.embedding_signature != expected_emb_sig:
                emb_stale = True
            if self.store.capabilities().uses_analyzer_tokens and fp.fts_signature != expected_fts_sig:
                fts_stale = True

        return ReindexStatus(
            doc_id=doc_id,
            embeddings_stale=emb_stale,
            fts_stale=fts_stale,
        )

    def _build_effective_filters(self, request: RAGQueryRequest) -> RAGFilterSpec:
        """Combine request-level filters, document_id, and tenant_id into RAGFilterSpec."""
        f = request.filters.model_copy() if request.filters else RAGFilterSpec()
        if request.tenant_id and not f.tenant_id:
            f.tenant_id = request.tenant_id
        if request.document_id and not f.document_ids:
            f.document_ids = [request.document_id]
        return f
