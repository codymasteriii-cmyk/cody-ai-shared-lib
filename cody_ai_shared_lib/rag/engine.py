"""RAGEngine public facade (§2.2, §14).

Wires the components from the profile and registry, and exposes query, stream, retrieval,
ingestion and lifecycle calls. The work itself lives in the modules it delegates to.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Callable, Generic, Iterator

from ..embedding import EMBEDDING_MODEL
from ..pricing import get_model_pricing
from .answering import AnswerFinalizer
from .cache import Cache, InMemoryLRUCache
from .context.assembler import AssembledPrompt, ContextAssembler
from .context.conversation import ConversationManager
from .context.fence import ContextFencer
from .degradation import DegradationTracker
from .errors import (
    GenerationError,
    ProfileConfigError,
    RAGEngineError,
    StoreCapabilityError,
)
from .gateway import ModelGateway
from .generation.flag_buffer import FlagStreamBuffer
from .generation.stuff import GenerationContext, GenerationStrategy
from .ingestion import BatchEmbedder, DocumentSynchronizer, SectionSummarizer
from .ingestion.nodes import (
    build_section_nodes,
    embedding_signature,
    finalize_node,
    fts_signature,
    prepare_summaries,
)
from .intent.llm_rewriter import clamp_intent
from .intent.router import IntentRouter, RouterContext
from .intent.temporal import TemporalExtractor
from .models import (
    IngestionProgress,
    MetaT,
    PreparedNode,
    QueryIntent,
    QueryPlan,
    RAGCitation,
    RAGDocumentInput,
    RAGNode,
    RAGQueryRequest,
    RAGResponse,
    RAGRetrievalResult,
    RAGSectionInput,
    RAGStreamEvent,
    ReindexStatus,
    UsageTotal,
)
from .ports import Fingerprint, Guard, NodeRenderer, RAGStore
from .postgen.citations import CitationExtractor
from .postgen.quotes import QuoteVerifier
from .postgen.sanitizer import OutputSanitizer
from .profile import EngineProfile, ValidationReport
from .registry import ComponentDeps, ComponentRegistry, create_default_registry
from .retrieval_pipeline import RetrievalPipeline, apply_guards
from .runtime import Clock, Deadline, RequestRun, Rng, SystemClock, SystemRng, make_bounded_pool
from .strategies.adaptive import AdaptiveStrategy
from .trace import ObserverDispatcher, RAGObserver, TraceCollector
from .usage import BudgetGuard, UsageTracker
from .vectors import is_zero_vector

logger = logging.getLogger("shared-rag")


class RAGStream(Generic[MetaT]):
    """Iterator yielding RAGStreamEvent for stream_query (§3.11, §10.8, RAG-RES-5)."""

    def __init__(self, gen: Iterator[RAGStreamEvent], usage_tracker: UsageTracker):
        self._gen = gen
        self._usage_tracker = usage_tracker

    def __iter__(self) -> "RAGStream[MetaT]":
        return self

    def __next__(self) -> RAGStreamEvent:
        return next(self._gen)

    def close(self) -> None:
        """Close the stream, stopping generation and recording cancelled usage if needed."""
        self._gen.close()

    @property
    def usage_total(self) -> UsageTotal:
        """Total spend and tokens, readable at any time and after close (AC-RAG-70, RAG-OPS-2)."""
        return self._usage_tracker.total()


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
        cache: Cache | None = None,
        pool: Any = None,
        router: IntentRouter | None = None,
    ):
        self.profile = profile
        self.store = store
        self.gateway = gateway or ModelGateway()
        self.clock = clock or SystemClock()
        self.rng = rng or SystemRng()
        self.registry = registry or create_default_registry()
        self.budget_guard = budget_guard
        self._observers = list(observers or [])
        self._explicit_guards = list(guards or [])
        self._explicit_router = router
        self._variant_engines: dict[str, "RAGEngine[MetaT]"] = {}
        self._variant_lock = threading.Lock()
        self.dispatcher = ObserverDispatcher(observers=self._observers, observer_errors=profile.observer_errors)

        self._raise_on_startup_problems()

        self.cache: Cache | None = cache
        if self.cache is None and profile.cache.enabled:
            self.cache = InMemoryLRUCache(max_bytes=profile.cache.max_bytes)
        self._pool = pool or make_bounded_pool(profile.runtime.pool_size)

        self.renderer = renderer or self.registry.get("renderer", profile.renderer)()
        self.analyzer = self.registry.get("analyzer", profile.analyzer)()
        self.splitter = self.registry.get("splitter", profile.ingestion.splitter)(
            target_size=profile.ingestion.target_size,
            overlap=profile.ingestion.overlap,
            analyzer=self.analyzer,
        )
        self.temporal_extractor = TemporalExtractor(clock=self.clock)
        self._deps = ComponentDeps(
            profile=profile,
            gateway=self.gateway,
            renderer=self.renderer,
            analyzer=self.analyzer,
            clock=self.clock,
            temporal_extractor=self.temporal_extractor,
            registry=self.registry,
        )

        self.formatter = self.registry.get("formatter", profile.citations.formatter)(self.renderer)
        self.fencer = ContextFencer(rng=self.rng)
        self.assembler = ContextAssembler(profile, self.renderer, self.fencer, self.formatter)
        self.conversation_manager = ConversationManager(
            max_reused_turns=profile.context.reused_turns,
            max_reused_nodes=profile.context.max_reused_nodes,
            max_state_bytes=profile.context.state_max_bytes,
            rolling_summary_max_tokens=profile.context.rolling_summary_max_tokens,
        )
        self.generator: GenerationStrategy = self.registry.get("generator", profile.generation.strategy)(self._deps)
        self.citation_extractor = CitationExtractor(profile.citations.on_unverified, self.formatter)
        self.quote_verifier = QuoteVerifier(analyzer=self.analyzer)
        self.sanitizer = OutputSanitizer(allowed_link_hosts=profile.allowed_link_hosts)
        self.finalizer = AnswerFinalizer(
            profile,
            store,
            self.analyzer,
            self.citation_extractor,
            self.quote_verifier,
            self.sanitizer,
            self.conversation_manager,
        )
        self.embedder = BatchEmbedder(gateway=self.gateway, model=profile.models.embedding, renderer=self.renderer)
        self.summarizer = SectionSummarizer(
            gateway=self.gateway,
            summary_models=profile.models.summary,
            enabled=profile.ingestion.summaries.enabled,
            min_chars_for_summary=profile.ingestion.summaries.min_chars_for_summary,
            summary_unit_chars=profile.ingestion.summaries.summary_unit_chars,
            analyzer=self.analyzer,
            fallback_action=profile.fallbacks.get("summarizer", "raise"),
        )
        self.synchronizer = DocumentSynchronizer(
            store=store,
            splitter=self.splitter,
            analyzer=self.analyzer,
            renderer=self.renderer,
            embedder=self.embedder,
            summarizer=self.summarizer,
            embedding_model=profile.models.embedding,
            budget_guard=budget_guard,
        )
        self.pipeline: RetrievalPipeline[MetaT] = RetrievalPipeline(
            self._deps, store, self._pool, self.cache, budget_guard, self._explicit_guards
        )
        self.router: IntentRouter = router or self.registry.get("router", profile.router.name)(self._deps)

    # ── Validation ────────────────────────────────────────────────────────

    def _engine_problems(self) -> tuple[list[str], list[str], list[str]]:
        """Return (problems, missing, capability_gaps) knowable before the first request (§13.3)."""
        problems: list[str] = []
        missing: list[str] = []
        gaps: list[str] = []
        profile = self.profile

        space = self.store.embedding_space()
        if space.model != profile.models.embedding:
            problems.append(
                f"Store embedding space model '{space.model}' does not match "
                f"profile embedding model '{profile.models.embedding}'"
            )

        real_gateway = isinstance(self.gateway, ModelGateway)
        if real_gateway and self.gateway.embedding_client is None and profile.models.embedding != EMBEDDING_MODEL:
            problems.append(
                f"models.embedding '{profile.models.embedding}' is not the model the lib embeds with ('{EMBEDDING_MODEL}')"
            )

        caps = self.store.capabilities()
        strategy_cls = self.registry.get("strategy", profile.retrieval.default_strategy)
        if strategy_cls is not None and not strategy_cls.is_runnable(caps):
            gaps.append(f"Strategy '{strategy_cls.name}' needs store capabilities {sorted(strategy_cls.requires)}")

        if "rerank" in profile.post_stages and profile.models.reranker and real_gateway:
            if self.gateway.rerank_client is None and not os.environ.get("JINA_API_KEY"):
                missing.append("JINA_API_KEY")
                problems.append("Rerank stage is enabled but JINA_API_KEY is not set")
        if profile.router.enable_jev and not (os.getenv("TYPESAFE_API_KEY") or os.getenv("VERCEL_API_KEY")):
            missing.append("TYPESAFE_API_KEY")
            problems.append("Router configured with enable_jev=True without TYPESAFE_API_KEY")
        if profile.generation.stream_mode == "native" and real_gateway and not hasattr(self.gateway.llm_client, "stream"):
            problems.append("stream_mode='native' but the LLM client cannot stream")

        if self.budget_guard is not None and getattr(self.budget_guard, "enforces_cost", False):
            bindings = [
                *profile.models.generation,
                *(profile.models.intent or []),
                *(profile.models.summary or []),
                profile.models.embedding,
                *([profile.models.reranker] if profile.models.reranker else []),
            ]
            for model in bindings:
                if get_model_pricing(model) is None:
                    problems.append(f"BudgetGuard enforces_cost=True but model '{model}' has no price")
        return problems, missing, gaps

    def validate(self) -> ValidationReport:
        """Report every configuration problem without raising (§13.3)."""
        report = self.profile.check(self.registry)
        problems, missing, gaps = self._engine_problems()
        all_problems = [*report.problems, *problems, *gaps]
        return ValidationReport(
            valid=not all_problems,
            profile_fingerprint=self.profile.fingerprint(),
            problems=all_problems,
            missing=[*report.missing, *missing],
            unknown=report.unknown,
        )

    def _raise_on_startup_problems(self) -> None:
        report = self.profile.check(self.registry)
        problems, missing, gaps = self._engine_problems()
        if report.problems or problems:
            all_problems = [*report.problems, *problems, *gaps]
            raise ProfileConfigError(
                f"Engine validation failed: {'; '.join(all_problems)}",
                problems=all_problems,
                missing=[*report.missing, *missing],
                unknown=report.unknown,
            )
        if gaps:
            raise StoreCapabilityError(f"Engine validation failed: {'; '.join(gaps)}", stage="startup")

    def with_profile(self, profile: EngineProfile) -> "RAGEngine[MetaT]":
        """Return an engine for another profile that shares this engine's store, gateway and pool."""
        return RAGEngine[MetaT](
            profile=profile,
            store=self.store,
            gateway=self.gateway,
            clock=self.clock,
            rng=self.rng,
            registry=self.registry,
            observers=self._observers,
            budget_guard=self.budget_guard,
            renderer=self.renderer,
            guards=self._explicit_guards or None,
            cache=self.cache,
            pool=self._pool,
            router=self._explicit_router,
        )

    def _variant_engine(self, name: str) -> "RAGEngine[MetaT]":
        """Return the engine for a named profile variant (§12.1)."""
        if name not in self.profile.variants:
            raise ProfileConfigError(f"Unknown profile variant '{name}'", unknown=[f"variant:{name}"])
        with self._variant_lock:
            if name not in self._variant_engines:
                self._variant_engines[name] = self.with_profile(self.profile.variant(name))
            return self._variant_engines[name]

    # ── Per-request plumbing ──────────────────────────────────────────────

    def _begin(self, request: RAGQueryRequest) -> RequestRun:
        request_id = request.request_id or f"req-{uuid.uuid4().hex[:12]}"
        filters = request.filters
        degradations = DegradationTracker(fallbacks=self.profile.fallbacks)
        dispatcher = self.dispatcher.for_request(degradations)
        degradations.on_record = dispatcher.notify_degrade
        usage = UsageTracker(
            request_id=request_id,
            tenant_or_user_id=request.tenant_id or filters.tenant_or_user_id,
            doc_ids=filters.doc_ids,
            profile_name=self.profile.name,
            variant=request.variant,
            on_record=dispatcher.notify_usage,
        )
        return RequestRun(
            request_id=request_id,
            usage=usage,
            degradations=degradations,
            deadline=Deadline(self.profile.runtime.deadline_seconds, clock=self.clock),
            trace=TraceCollector(
                request_id,
                self.profile,
                self.clock,
                dispatcher,
                request.trace_level or self.profile.trace_level,
                request.variant,
            ),
        )

    @staticmethod
    def _attach_failure_context(
        exc: Exception,
        run: RequestRun,
        retrieval: RAGRetrievalResult[MetaT] | None = None,
    ) -> None:
        """Give a typed error what the request had spent and found before it failed (RAG-OPS-1, §10.7)."""
        if not isinstance(exc, RAGEngineError):
            return
        exc.request_id = exc.request_id or run.request_id
        exc.degradations = run.degradations.degradations
        exc.usage = run.usage.stage_usages
        exc.usage_total = run.usage.total()
        if isinstance(exc, GenerationError) and exc.retrieval is None:
            exc.retrieval = retrieval

    def _fail(self, exc: Exception, run: RequestRun, retrieval: RAGRetrievalResult[MetaT] | None = None) -> None:
        self._attach_failure_context(exc, run, retrieval)
        logger.error(f"[RAG] id={run.request_id} error={type(exc).__name__} stage={getattr(exc, 'stage', None)}")

    def _select_router(self, request: RAGQueryRequest) -> IntentRouter:
        """Selection order: request override, then the engine's router (§7.2)."""
        name = request.router_override
        if name is None:
            return self.router
        factory = self.registry.get("router", name)
        if factory is None:
            raise ProfileConfigError(f"Unknown router '{name}'", unknown=[f"router:{name}"])
        return factory(self._deps)

    def _resolve_intent(self, request: RAGQueryRequest, intent: QueryIntent | None, run: RequestRun) -> QueryIntent:
        if intent is not None:
            return clamp_intent(intent.model_dump(), run.degradations)
        started = run.trace.elapsed_ms()
        router = self._select_router(request)
        context = RouterContext(
            usage_tracker=run.usage,
            degradation_tracker=run.degradations,
            profile=self.profile,
            budget_guard=self.budget_guard,
            deadline=run.deadline,
        )
        resolved = router.route(request, request.state, context)
        run.trace.record_stage(router.name, "router", started, version=router.version)
        run.trace.router_decision = {
            "source": resolved.router_source,
            "route": resolved.route,
            "scope": resolved.scope,
        }
        return resolved

    def _prepare_generation(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent,
        run: RequestRun,
    ) -> tuple[AssembledPrompt[MetaT], RAGRetrievalResult[MetaT], list[Guard], str]:
        """Retrieve and assemble, or build the conversational prompt on the no-retrieval route."""
        state_fence = request.state.fence_id if request.state else None
        if intent.route == "no_retrieval":
            _, guards, _ = self.pipeline.scoped_filters(request, intent)
            retrieval = RAGRetrievalResult[MetaT](
                intent=intent,
                strategy_used="none",
                tier_decision_reason="no_retrieval route selected",
            )
            prompt = self.assembler.assemble_conversational(request, state_fence, guards)
            return prompt, retrieval, guards, "generation_no_retrieval"

        retrieval, guards = self.pipeline.run(request, intent, "query", run)
        return self.assembler.assemble_prompt(request, retrieval.nodes, state_fence, guards), retrieval, guards, "generation"

    def _generation_context(self, request: RAGQueryRequest, run: RequestRun, stage: str) -> GenerationContext:
        return GenerationContext(
            usage_tracker=run.usage,
            degradation_tracker=run.degradations,
            budget_guard=self.budget_guard,
            request_id=run.request_id,
            tenant_or_user_id=request.tenant_id,
            stage=stage,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def route(self, request: RAGQueryRequest) -> QueryIntent:
        """Public routing entry point (§7.2, AC-RAG-16)."""
        return self._resolve_intent(request, None, self._begin(request))

    def plan(self, request: RAGQueryRequest, intent: QueryIntent | None = None) -> QueryPlan:
        """Evaluate the adaptive rules without retrieval or generation (§8.12, AC-RAG-48, AC-RAG-58)."""
        resolved = intent or QueryIntent(search_query=request.question, router_source="passthrough")
        filters, _, _ = self.pipeline.scoped_filters(request, resolved)
        return AdaptiveStrategy.decide_tier(request, resolved, filters, self.store, self.profile, self.registry)

    def retrieve(self, request: RAGQueryRequest, intent: QueryIntent | None = None) -> RAGRetrievalResult[MetaT]:
        """Ranked list retrieval mode (§8.10, AC-RAG-51)."""
        if request.variant:
            return self._variant_engine(request.variant).retrieve(request.model_copy(update={"variant": None}), intent)
        run = self._begin(request)
        try:
            resolved = self._resolve_intent(request, intent, run)
            result, _ = self.pipeline.run(request, resolved, "retrieve", run)
        except Exception as exc:
            self._fail(exc, run)
            raise
        result.trace = run.trace.build(
            result.applied_filters, run.degradations.degradations, run.usage.stage_usages, run.usage.total()
        )
        run.trace.dispatcher.notify_complete(result.trace, result)
        return result

    def query(self, request: RAGQueryRequest, intent: QueryIntent | None = None) -> RAGResponse[MetaT]:
        """End-to-end question answering (§14)."""
        if request.variant:
            return self._variant_engine(request.variant).query(request.model_copy(update={"variant": None}), intent)
        run = self._begin(request)
        retrieval: RAGRetrievalResult[MetaT] | None = None
        try:
            resolved = self._resolve_intent(request, intent, run)
            prompt, retrieval, guards, stage = self._prepare_generation(request, resolved, run)
            return self._generate_and_finish(request, prompt, retrieval, guards, stage, run)
        except Exception as exc:
            self._fail(exc, run, retrieval)
            raise

    def generate_from_nodes(
        self,
        request: RAGQueryRequest,
        nodes: list[RAGNode[MetaT]],
        intent: QueryIntent | None = None,
    ) -> RAGResponse[MetaT]:
        """Run guards, assembly, generation and citation checks on caller-supplied nodes (§14.4)."""
        run = self._begin(request)
        resolved = intent or QueryIntent(search_query=request.question, router_source="passthrough")
        retrieval: RAGRetrievalResult[MetaT] | None = None
        try:
            filters, guards, _ = self.pipeline.scoped_filters(request, resolved)
            guarded = apply_guards(list(nodes), guards, run.trace.record_guard_effect)
            retrieval = RAGRetrievalResult[MetaT](
                nodes=guarded,
                intent=resolved,
                strategy_used="caller_supplied",
                tier_decision_reason="generate_from_nodes",
                applied_filters=filters,
            )
            state_fence = request.state.fence_id if request.state else None
            prompt = self.assembler.assemble_prompt(request, guarded, state_fence, guards)
            return self._generate_and_finish(request, prompt, retrieval, guards, "generation", run)
        except Exception as exc:
            self._fail(exc, run, retrieval)
            raise

    def _generate_and_finish(
        self,
        request: RAGQueryRequest,
        prompt: AssembledPrompt[MetaT],
        retrieval: RAGRetrievalResult[MetaT],
        guards: list[Guard],
        stage: str,
        run: RequestRun,
    ) -> RAGResponse[MetaT]:
        run.deadline.check("generation")
        started = run.trace.elapsed_ms()
        result = self.generator.generate(prompt, self._generation_context(request, run, stage))
        run.trace.record_stage(self.generator.name, "generation", started, version=self.generator.version)

        buffer = FlagStreamBuffer()
        visible = "".join(buffer.feed(result.text) + buffer.finalize())
        return self.finalizer.finish(request, result.text, visible, buffer.answer_found, prompt, retrieval, guards, run)

    def verify_answer(
        self,
        answer_text: str,
        response_or_nodes: RAGResponse[MetaT] | RAGRetrievalResult[MetaT] | list[RAGNode[MetaT]],
    ) -> list[RAGCitation]:
        """Run citation and quote checks on an edited answer, without model calls (§14.4)."""
        if isinstance(response_or_nodes, RAGResponse):
            nodes = response_or_nodes.retrieved.nodes
        elif isinstance(response_or_nodes, RAGRetrievalResult):
            nodes = response_or_nodes.nodes
        else:
            nodes = list(response_or_nodes)

        clean_text, citations = self.citation_extractor.extract(answer_text, nodes)
        if self.profile.citations.verify_quotes:
            citations, _ = self.quote_verifier.verify(citations, nodes, answer_text=clean_text)
        return citations

    def stream_query(self, request: RAGQueryRequest, intent: QueryIntent | None = None) -> RAGStream[MetaT]:
        """Event streaming query pipeline (§10.8, AC-RAG-45, RAG-OPS-1, RAG-OPS-2)."""
        if request.variant:
            return self._variant_engine(request.variant).stream_query(
                request.model_copy(update={"variant": None}), intent
            )
        run = self._begin(request)
        return RAGStream[MetaT](self._stream_events(request, intent, run), usage_tracker=run.usage)

    def _stream_events(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent | None,
        run: RequestRun,
    ) -> Iterator[RAGStreamEvent]:
        retrieval: RAGRetrievalResult[MetaT] | None = None
        chunks: Any = None
        emitted_degradations = 0

        def new_degradations() -> list[RAGStreamEvent]:
            nonlocal emitted_degradations
            pending = run.degradations.degradations[emitted_degradations:]
            emitted_degradations += len(pending)
            return [RAGStreamEvent(event="degradation", data=d.model_dump()) for d in pending]

        try:
            resolved = self._resolve_intent(request, intent, run)
            yield RAGStreamEvent(
                event="intent",
                data={
                    "route": resolved.route,
                    "scope": resolved.scope,
                    "search_query": resolved.search_query,
                    "router_source": resolved.router_source,
                },
            )
            yield from new_degradations()

            if resolved.route != "no_retrieval":
                yield RAGStreamEvent(event="status", data={"stage": "retrieving"})
            prompt, retrieval, guards, stage = self._prepare_generation(request, resolved, run)
            yield from new_degradations()
            if resolved.route != "no_retrieval":
                yield RAGStreamEvent(
                    event="retrieval",
                    data={
                        "nodes": [n.node_id for n in retrieval.nodes],
                        "node_count": len(retrieval.nodes),
                        "strategy_used": retrieval.strategy_used,
                        "applied_filters": retrieval.applied_filters.model_dump(),
                    },
                )

            yield RAGStreamEvent(event="status", data={"stage": "generating"})
            run.deadline.check("generation")
            started = run.trace.elapsed_ms()
            context = self._generation_context(request, run, stage)
            buffer = FlagStreamBuffer()
            raw_parts: list[str] = []
            visible_parts: list[str] = []

            def deltas_for(texts: list[str]) -> list[RAGStreamEvent]:
                visible_parts.extend(texts)
                return [RAGStreamEvent(event="delta", data={"text": text}) for text in texts]

            if self.profile.generation.stream_mode == "buffered":
                result = self.generator.generate(prompt, context)
                raw_parts.append(result.text)
                yield from new_degradations()
                yield from deltas_for(buffer.feed(result.text))
            else:
                chunks = self.generator.stream(prompt, context)
                for chunk in chunks:
                    yield from new_degradations()
                    if chunk.delta:
                        raw_parts.append(chunk.delta)
                        yield from deltas_for(buffer.feed(chunk.delta))
                chunks = None
            yield from deltas_for(buffer.finalize())
            yield from new_degradations()
            run.trace.record_stage(self.generator.name, "generation", started, version=self.generator.version)

            response = self.finalizer.finish(
                request,
                "".join(raw_parts),
                "".join(visible_parts),
                buffer.answer_found,
                prompt,
                retrieval,
                guards,
                run,
            )
            yield from new_degradations()

            citations = [c.model_dump() for c in response.citations]
            sources = [s.model_dump() for s in response.sources]
            coverage = response.citation_coverage
            yield RAGStreamEvent(
                event="citations",
                data={
                    "citations": citations,
                    "sources": sources,
                    "citation_coverage": coverage.ratio if coverage else 1.0,
                },
            )
            yield RAGStreamEvent(
                event="done",
                data={
                    "answer_id": response.answer_id,
                    "answer_final": response.answer_final,
                    "citations": citations,
                    "sources": sources,
                    "usage": [u.model_dump() for u in response.usage],
                    "usage_total": response.usage_total.model_dump(),
                    "degradations": [d.model_dump() for d in response.degradations],
                    "state": response.state.model_dump(),
                    "grounded": response.grounded,
                    "answer_found": response.answer_found,
                    "abstained": response.abstained,
                    "message_id": request.metadata.get("message_id"),
                },
            )
        except GeneratorExit:
            # Closing the model stream makes the gateway record the cancelled call (RAG-OPS-2)
            if chunks is not None:
                chunks.close()
            raise
        except Exception as exc:
            self._fail(exc, run, retrieval)
            yield RAGStreamEvent(
                event="error",
                data={
                    "type": type(exc).__name__,
                    "stage": getattr(exc, "stage", None) or "stream",
                    "message": str(exc)[:200],
                    "request_id": run.request_id,
                    "usage": [u.model_dump() for u in run.usage.stage_usages],
                    "usage_total": run.usage.total().model_dump(),
                },
            )
            raise

    # ── Ingestion ─────────────────────────────────────────────────────────

    def _ingestion_trackers(self, request_id: str, doc_id: str) -> tuple[UsageTracker, DegradationTracker]:
        usage = UsageTracker(request_id=request_id, doc_ids=[doc_id], profile_name=self.profile.name)
        return usage, DegradationTracker(fallbacks=self.profile.fallbacks)

    def ingest_document(
        self,
        doc: RAGDocumentInput[MetaT],
        start_section_seq: int = 0,
        on_section_complete: Callable[[IngestionProgress], None] | None = None,
    ) -> IngestionProgress:
        """Split, summarize, embed and atomically upsert a document, one section at a time.

        Resumable via start_section_seq and append-only (RAG-ING-5, RAG-ING-6, AC-RAG-05).
        """
        doc_input = doc
        sections = [
            s for s in sorted(doc_input.sections, key=lambda x: x.section_seq) if s.section_seq >= start_section_seq
        ]
        last_progress: IngestionProgress | None = None
        completed = 0

        for section in sections:
            tracker, deg_tracker = self._ingestion_trackers(
                f"ingest-{doc_input.doc_id}-s{section.section_seq}", doc_input.doc_id
            )
            try:
                last_progress = self._ingest_section(doc_input, section, completed + 1, tracker, deg_tracker)
            except RAGEngineError as exc:
                exc.request_id = exc.request_id or tracker.request_id
                exc.degradations = deg_tracker.degradations
                exc.usage = tracker.stage_usages
                exc.usage_total = tracker.total()
                raise
            completed += 1
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

    def _ingest_section(
        self,
        doc_input: RAGDocumentInput[MetaT],
        section: RAGSectionInput[MetaT],
        sections_completed: int,
        tracker: UsageTracker,
        deg_tracker: DegradationTracker,
    ) -> IngestionProgress:
        """Summarize, split, embed and atomically upsert one section (RAG-ING-3, RAG-ING-10, RAG-ING-12)."""
        uses_tokens = self.store.capabilities().uses_analyzer_tokens

        result = self.summarizer.summarize_section(
            doc_id=doc_input.doc_id,
            section_seq=section.section_seq,
            section_id=section.section_id,
            text=section.text,
            char_start=section.char_start,
            metadata=section.metadata,
            existing_summary=section.existing_summary,
            usage_tracker=tracker,
            degradation_tracker=deg_tracker,
            budget_guard=self.budget_guard,
        )

        nodes = build_section_nodes(doc_input.doc_id, section, self.splitter)
        for node in nodes:
            finalize_node(node, self.renderer, self.analyzer, uses_tokens)
        pairs = self.embedder.embed_nodes(nodes, tracker, deg_tracker, budget_guard=self.budget_guard)

        summaries = prepare_summaries(
            result, self.embedder, self.renderer, self.analyzer, uses_tokens, tracker, deg_tracker, self.budget_guard
        )
        written = self.store.upsert_section_nodes(
            doc_id=doc_input.doc_id,
            section_seq=section.section_seq,
            nodes=[p[0] for p in pairs] + summaries.unit_nodes,
            embeddings=[p[1] for p in pairs] + summaries.unit_vectors,
            summary=summaries.section_summary,
        )
        return IngestionProgress(
            doc_id=doc_input.doc_id,
            section_seq=section.section_seq,
            sections_completed=sections_completed,
            sections_total=len(doc_input.sections),
            passages_upserted=written,
            passages_failed=len(nodes) - len(pairs),
            summary_status=result.summary_status,
            usage=tracker.stage_usages,
            degradations=deg_tracker.degradations,
        )

    def sync_document(self, doc: RAGDocumentInput[MetaT]) -> IngestionProgress:
        """Embed only changed nodes, delete vanished ones, and preserve node ids (§6.6, AC-RAG-06)."""
        tracker, deg_tracker = self._ingestion_trackers(f"sync-{doc.doc_id}", doc.doc_id)
        try:
            return self.synchronizer.sync(doc, tracker, deg_tracker)
        except RAGEngineError as exc:
            exc.request_id = exc.request_id or tracker.request_id
            exc.degradations = deg_tracker.degradations
            exc.usage = tracker.stage_usages
            exc.usage_total = tracker.total()
            raise

    def prepare_nodes(self, nodes: list[RAGNode[MetaT]]) -> list[PreparedNode[MetaT]]:
        """Compute embedding, fingerprint, token estimate and search tokens without writing (RAG-ING-13)."""
        if not nodes:
            return []

        tracker, _ = self._ingestion_trackers("prepare-nodes", nodes[0].span.doc_id)
        vectors = self.gateway.embed(
            texts=[self.renderer.embed_text(n) for n in nodes],
            model=self.profile.models.embedding,
            stage="prepare_embedding",
            usage_tracker=tracker,
            task_type="RETRIEVAL_DOCUMENT",
            budget_guard=self.budget_guard,
        ).vectors

        uses_tokens = self.store.capabilities().uses_analyzer_tokens
        emb_sig = embedding_signature(self.profile.models.embedding, self.store.embedding_space(), self.splitter)
        fts_sig = fts_signature(self.analyzer, uses_tokens)
        stage_usage = tracker.stage_usages[0] if tracker.stage_usages else None

        prepared: list[PreparedNode[MetaT]] = []
        for idx, node in enumerate(nodes):
            node_copy = node.model_copy()
            finalize_node(node_copy, self.renderer, self.analyzer, uses_tokens)
            vector = vectors[idx] if idx < len(vectors) else []
            prepared.append(
                PreparedNode[MetaT](
                    node=node_copy,
                    vector=[] if is_zero_vector(vector) else vector,
                    fingerprint=Fingerprint(
                        content_hash=node_copy.content_hash,
                        embedding_signature=emb_sig,
                        fts_signature=fts_sig,
                    ),
                    usage=stage_usage,
                )
            )
        return prepared

    def needs_reindex(self, doc_id: str) -> ReindexStatus:
        """Check whether stored embeddings or FTS tokens are stale (§6.6, AC-RAG-64)."""
        stored = self.store.get_fingerprints(doc_id)
        uses_tokens = self.store.capabilities().uses_analyzer_tokens
        expected_embedding = embedding_signature(
            self.profile.models.embedding, self.store.embedding_space(), self.splitter
        )
        expected_fts = fts_signature(self.analyzer, uses_tokens)
        return ReindexStatus(
            doc_id=doc_id,
            embeddings_stale=any(fp.embedding_signature != expected_embedding for fp in stored.values()),
            fts_stale=uses_tokens and any(fp.fts_signature != expected_fts for fp in stored.values()),
        )
