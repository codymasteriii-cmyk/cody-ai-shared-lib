"""RAGEngine public facade (§2.2, §14).

Wires the components from the profile and registry, and exposes query, stream, retrieval,
ingestion and lifecycle calls. The work itself lives in the modules it delegates to.

How a question flows (read `query()` first; docs/CALL_TREES.md has the full trees):

    query()
      _begin()                  per-request trackers: usage, degradations, deadline, trace
      _resolve_intent()         router (or the caller's intent) -> clamp -> resolve dates
      _prepare_generation()     retrieval.pipeline.run() + clarification mark + prompt assembly
      _generate_and_finish()    one model call, then answering.AnswerFinalizer.finish()

What lives where:
    this file                 construction, validation, the public calls, streaming, ingestion entry points
    retrieval_pipeline.py     filters, guards, strategy, post stages, relaxation
    answering.py              everything after the model call: citations, state, the response
    batching.py               query_batched (list questions across many documents)
    clarification.py          deciding that a question is unclear, and the wording of the ask
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Callable, Generic, Iterator

from ..embedding import EMBEDDING_MODEL

# Dimensions of the embedding models the lib embeds with; others are not checked
EMBEDDING_DIMS = {EMBEDDING_MODEL: 768}
from ..pricing import get_model_pricing
from .answering import AnswerFinalizer
from .cache import Cache, InMemoryLRUCache
from .context.assembler import AssembledPrompt, ContextAssembler
from .batching import BatchedListAnswerer
from .clarification import ASKABLE_REASONS, CandidateKey, find_clarification, hedge_note, may_ask
from .context.conversation import ConversationManager, reusable_node_ids
from .context.fence import ContextFencer
from .degradation import DegradationTracker
from .errors import (
    NEVER_DEGRADABLE,
    EmbeddingProviderError,
    GenerationError,
    ProfileConfigError,
    RAGEngineError,
    RequestCancelled,
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
    embedding_is_stale,
    fts_signature,
    parse_node_id,
    prepare_passages,
    prepare_summaries,
    write_signatures,
)
from .intent.llm_rewriter import clamp_intent
from .intent.router import IntentRouter, RouterContext
from .intent.temporal import TemporalResolver
from .models import (
    CitationCoverage,
    Clarification,
    IngestionProgress,
    FailedNode,
    MetaT,
    PreparedBatch,
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
from .render import renderer_problems
from .ports import (
    Fingerprint,
    Guard,
    NodeRenderer,
    OwnerScope,
    RAGStore,
    SectionWrite,
    call_store,
    filter_value,
    store_overrides,
    validate_owner,
)
from .postgen.citations import CitationExtractor
from .postgen.quotes import QuoteValidator
from .postgen.sanitizer import OutputSanitizer
from .profile import EngineProfile, ValidationReport
from .registry import ComponentDeps, ComponentRegistry, create_default_registry
from .retrieval_pipeline import RetrievalPipeline, apply_guards
from .runtime import CancelToken, Clock, Deadline, RequestRun, Rng, SystemClock, SystemRng, make_bounded_pool
from .strategies.adaptive import AdaptiveRetrievalStrategy
from .trace import ObserverDispatcher, RAGObserver, TraceCollector
from .usage import BudgetGuard, UsageTracker

logger = logging.getLogger("shared-rag")


class RAGStream(Generic[MetaT]):
    """Iterator yielding RAGStreamEvent for stream_query (§3.11, §10.8, RAG-RES-5).

    Use it as a context manager so leaving early (a client disconnect, an exception, a `break`)
    always closes it, which stops model calls and records the cut-off call as cancelled:

        with engine.stream_query(request) as stream:
            for event in stream:
                ...
    """

    def __init__(self, gen: Iterator[RAGStreamEvent], usage_tracker: UsageTracker):
        self._gen = gen
        self._usage_tracker = usage_tracker
        self._closed = False

    def __iter__(self) -> "RAGStream[MetaT]":
        return self

    def __next__(self) -> RAGStreamEvent:
        return next(self._gen)

    def __enter__(self) -> "RAGStream[MetaT]":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the stream, stopping generation and recording cancelled usage if needed. Idempotent."""
        if not self._closed:
            self._closed = True
            self._gen.close()

    @property
    def usage_total(self) -> UsageTotal:
        """Total spend and tokens. Final only once the stream has ended or been closed: before that it
        does not yet include the call in progress (AC-RAG-70, RAG-OPS-2)."""
        return self._usage_tracker.total()


LIST_PARTIAL_NOTE = "This list may be incomplete: only part of the available text was read. Say so in the answer."


class RAGEngine(Generic[MetaT]):
    """Main facade for the shared RAG engine.

    One engine is built per profile and store, once, at app start; it is safe to share between requests
    because all per-request state lives in the `RequestRun` that `_begin()` creates. To vary the profile or
    the store per request, use `with_profile()` / `with_store()` (they share the pool, cache and gateway)
    or set `request.variant` (the named profile variants of §12.1).

    Public calls: `route`, `plan`, `retrieve`, `query`, `stream_query`, `query_batched`, `generate_from_nodes`,
    `verify_answer`; ingestion: `ingest_document`, `sync_document`, `prepare_node(s)`, `needs_reindex`;
    lifecycle: `validate`, `with_profile`, `with_store`.
    """

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
        variant_name: str | None = None,
        fencer: ContextFencer | None = None,
        filter_inferencer: Any = None,
        candidate_key: CandidateKey | None = None,
    ):
        """Build every component from the profile. The order matters:

        1. the renderer first, so a renderer that only stubs the protocol fails before anything else;
        2. startup validation, so a bad profile raises before any pool or client exists;
        3. text tools (analyzer, splitter), then the prompt side, the answer side, ingestion, retrieval, the router.

        The optional arguments let sibling engines (`with_profile`, `with_store`) share the heavy parts.
        """
        # 1. Settings and the pieces every other component reads
        self.profile = profile
        self._variant_name = variant_name
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
        # Bound before validation so a renderer that only stubs the protocol fails here (AC-RAG-111)
        self.renderer = renderer or self.registry.get("renderer", profile.renderer)()

        self._raise_on_startup_problems()

        # 2. Shared resources: a cache and a bounded worker pool (a sibling engine passes its own in)
        self.cache: Cache | None = cache
        if self.cache is None and profile.cache.enabled:
            self.cache = InMemoryLRUCache(max_bytes=profile.cache.max_bytes)
        self._pool = pool or make_bounded_pool(profile.runtime.pool_size)

        # 3. Text tools and the bundle of dependencies that registry factories receive
        self.analyzer = self.registry.get("analyzer", profile.analyzer)()
        self.splitter = self.registry.get("splitter", profile.ingestion.splitter)(
            target_size=profile.ingestion.target_size,
            overlap=profile.ingestion.overlap,
            analyzer=self.analyzer,
        )
        self.temporal_resolver = TemporalResolver(clock=self.clock)
        self._deps = ComponentDeps(
            profile=profile,
            gateway=self.gateway,
            renderer=self.renderer,
            analyzer=self.analyzer,
            clock=self.clock,
            temporal_resolver=self.temporal_resolver,
            registry=self.registry,
        )

        # 4. Prompt side: citation labels, fencing, assembly and conversation state
        self.formatter = self.registry.get("formatter", profile.citations.formatter)(self.renderer)
        # Siblings (with_profile, with_store) share the fencer, so stateless fence ids stay stable (D-06)
        self.fencer = fencer or ContextFencer(rng=self.rng)
        self.assembler = ContextAssembler(profile, self.renderer, self.fencer, self.formatter, self.analyzer)
        self.conversation_manager = ConversationManager(
            max_reused_turns=profile.context.reused_turns,
            max_reused_nodes=profile.context.max_reused_nodes,
            max_state_bytes=profile.context.state_max_bytes,
            rolling_summary_max_tokens=profile.context.rolling_summary_max_tokens,
        )
        # 5. Answer side: the generator, then the checks that run on its output, bundled into the finalizer
        self.generator: GenerationStrategy = self.registry.get("generator", profile.generation.strategy)(self._deps)
        self.citation_extractor = CitationExtractor(profile.citations.on_unverified, self.formatter)
        self.quote_validator = QuoteValidator(analyzer=self.analyzer)
        self.sanitizer = OutputSanitizer(allowed_link_hosts=profile.allowed_link_hosts)
        self.finalizer = AnswerFinalizer(
            profile,
            store,
            self.analyzer,
            self.citation_extractor,
            self.quote_validator,
            self.sanitizer,
            self.conversation_manager,
            self.renderer.prompt_text,
        )
        # 6. Ingestion: embed, summarize and synchronize documents (used by ingest_document / sync_document)
        self.embedder = BatchEmbedder(
            gateway=self.gateway,
            model=profile.models.embedding,
            renderer=self.renderer,
            task_type=store.embedding_space().task_type_documents,   # the task type the fingerprint names
        )
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
            embed_summaries=profile.ingestion.summaries.embed_summaries,
            granularity=profile.ingestion.granularity,
        )
        # 7. Retrieval and routing: the pipeline owns filters, guards, strategy and post stages
        self.pipeline: RetrievalPipeline[MetaT] = RetrievalPipeline(
            self._deps, store, self._pool, self.cache, budget_guard, self._explicit_guards
        )
        self.filter_inferencer = filter_inferencer          # §5.5: narrow-only, app-supplied
        # §7.10: node -> (key, label) | None. Only an app can tell referents apart (two characters both called
        # Li Ming); without it several_matches and too_broad never fire, because different documents are not different meanings.
        self.candidate_key = candidate_key
        self.pipeline.filter_inferencer = filter_inferencer
        self.router: IntentRouter = router or self.registry.get("router", profile.router.name)(self._deps)
        if router is None and profile.router.name == "auto" and not profile.models.intent:
            # Valid, but say so once: "auto" without an intent model is heuristic-only routing (§7)
            logger.warning(
                f"[RAG] profile '{profile.name}': router 'auto' has no models.intent; "
                "intents come from the heuristic router only"
            )

    # ── Validation ────────────────────────────────────────────────────────

    def _engine_problems(self) -> tuple[list[str], list[str], list[str]]:
        """Return (problems, missing, capability_gaps) knowable before the first request (§13.3).

        `problems` are settings that cannot work (a model or dimension mismatch, a stubbed renderer);
        `missing` names environment variables that are not set; `capability_gaps` are things the store
        cannot do that the chosen strategy needs. The first two raise `ProfileConfigError` at startup;
        a gap raises `StoreCapabilityError`.
        """
        problems: list[str] = []
        missing: list[str] = []
        gaps: list[str] = []
        profile = self.profile

        space = self.store.embedding_space()
        known_dims = EMBEDDING_DIMS.get(profile.models.embedding)
        if known_dims is not None and space.dims != known_dims:
            problems.append(
                f"Store embedding space has {space.dims} dims; '{profile.models.embedding}' produces {known_dims} (RAG-ING-7)"
            )
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

        problems.extend(renderer_problems(self.renderer))

        summaries = profile.ingestion.summaries
        if summaries.enabled and not summaries.embed_summaries and not store_overrides(self.store, "upsert_section"):
            problems.append(
                "ingestion.summaries.embed_summaries=False needs a store that overrides upsert_section; "
                "the legacy upsert_section_nodes takes one vector per node"
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
        """Fail fast in the constructor, so a bad profile never reaches a request (§13.3)."""
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

    def with_profile(self, profile: EngineProfile, variant_name: str | None = None) -> "RAGEngine[MetaT]":
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
            variant_name=variant_name,
            fencer=self.fencer,
            filter_inferencer=self.filter_inferencer,
            candidate_key=self.candidate_key,
        )

    def with_store(self, store: RAGStore[MetaT]) -> "RAGEngine[MetaT]":
        """Return an engine over another store that shares everything else: pool, cache, gateway,
        registry, observers and guards (§5.6).

        This is how an app uses a store scoped to the authenticated user per request without
        building a new pool and cache each time. The new store must have the same embedding space.
        """
        return RAGEngine[MetaT](
            profile=self.profile,
            store=store,
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
            variant_name=self._variant_name,
            fencer=self.fencer,
            filter_inferencer=self.filter_inferencer,
            candidate_key=self.candidate_key,
        )

    def _variant_engine(self, name: str) -> "RAGEngine[MetaT]":
        """Return the engine for a named profile variant (§12.1)."""
        if name not in self.profile.variants:
            raise ProfileConfigError(f"Unknown profile variant '{name}'", unknown=[f"variant:{name}"])
        with self._variant_lock:
            if name not in self._variant_engines:
                self._variant_engines[name] = self.with_profile(self.profile.variant(name), variant_name=name)
            return self._variant_engines[name]

    # ── Per-request plumbing ──────────────────────────────────────────────

    def _begin(self, request: RAGQueryRequest, cancel: CancelToken | None = None) -> RequestRun:
        """Create the per-request state: id, usage and degradation trackers, deadline and trace collector.

        Everything a request records goes into the returned `RequestRun`, never onto the engine, which is
        why one engine can serve many requests at once. Observers are notified from the trackers' callbacks.
        """
        request_id = request.request_id or f"req-{uuid.uuid4().hex[:12]}"
        filters = request.filters
        variant = request.variant or self._variant_name
        scope = {key: filter_value(filters, key) for key in self.store.capabilities().required_filter_keys}
        degradations = DegradationTracker(fallbacks=self.profile.fallbacks)
        dispatcher = self.dispatcher.for_request(degradations)
        degradations.on_record = dispatcher.notify_degrade
        usage = UsageTracker(
            request_id=request_id,
            tenant_or_user_id=request.tenant_id or filters.tenant_or_user_id,
            doc_ids=filters.doc_ids,
            profile_name=self.profile.name,
            variant=variant,
            scope=scope,
            on_record=dispatcher.notify_usage,
        )
        return RequestRun(
            request_id=request_id,
            usage=usage,
            degradations=degradations,
            deadline=Deadline(self.profile.runtime.deadline_seconds, clock=self.clock, cancel=cancel),
            trace=TraceCollector(
                request_id,
                self.profile,
                self.clock,
                dispatcher,
                request.trace_level or self.profile.trace_level,
                variant,
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
        """Common error path: attach what the request spent, log it, and tell observers. The caller re-raises."""
        self._attach_failure_context(exc, run, retrieval)
        level = logging.INFO if isinstance(exc, RequestCancelled) else logging.ERROR   # a cancel is the caller's choice
        logger.log(level, f"[RAG] id={run.request_id} error={type(exc).__name__} stage={getattr(exc, 'stage', None)}")
        try:
            trace = run.trace.build(None, run.degradations.degradations, run.usage.stage_usages, run.usage.total())
            run.trace.dispatcher.notify_complete(trace, None)
        except Exception:  # an observer must never replace the error the caller needs to see
            logger.error(f"[RAG] id={run.request_id} on_complete failed while reporting a failure", exc_info=True)

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
        """Decide what the question means: the caller's intent, or the router's, always clamped and date-resolved.

        An intent the caller passes in is trusted for its shape but still clamped to legal ranges (DG-05),
        so a hand-built intent cannot send an out-of-range value downstream.
        """
        if intent is not None:
            clamped = clamp_intent(intent.model_dump(exclude={"ext"}), run.degradations)
            return self.temporal_resolver.resolve(self._with_ext(clamped, intent))
        run.deadline.raise_if_cancelled("routing")
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
        run.deadline.raise_if_cancelled("routing")
        run.trace.router_decision = {
            "source": resolved.router_source,
            "route": resolved.route,
            "scope": resolved.scope,
        }
        return self.temporal_resolver.resolve(resolved)

    @staticmethod
    def _with_ext(clamped: QueryIntent, original: QueryIntent) -> QueryIntent:
        """Keep the caller's typed intent class and extension on the clamped values."""
        values = {name: getattr(clamped, name) for name in type(clamped).model_fields if name != "ext"}
        return original.model_copy(update=values)

    def _prepare_generation(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent,
        run: RequestRun,
    ) -> tuple[AssembledPrompt[MetaT], RAGRetrievalResult[MetaT], list[Guard], str]:
        """Retrieve and assemble, or build the conversational prompt on the no-retrieval route.

        Returns (prompt, retrieval, guards, stage name for usage accounting). Steps, in order:

        1. no_retrieval route: skip search and build a plain chat prompt;
        2. run the retrieval pipeline;
        3. mark an unclear question (`retrieval.clarification`), without a model call;
        4. collect "service notes" the model should see (degradations, a hedge, a partial list);
        5. assemble the prompt; and if that assembly shows the list became partial, assemble once more with the note.
        """
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
        retrieval.clarification = self._clarification(request, intent, retrieval, guards, run)
        # Service notes are short instructions appended to the prompt so the answer can be honest about its limits
        disclosure = request.degradation_disclosure or self.profile.generation.degradation_disclosure
        notes = (
            list(dict.fromkeys(d.impact for d in run.degradations.degradations if d.impact))
            if disclosure == "note" else None
        )
        if retrieval.clarification is not None and self._wants_hedge_note(request, retrieval):
            notes = [*(notes or []), hedge_note(retrieval.clarification)]
        if retrieval.completeness == "partial":
            notes = [*(notes or []), LIST_PARTIAL_NOTE]
        prompt = self.assembler.assemble_prompt(request, retrieval.nodes, state_fence, guards, service_notes=notes)
        # The note cannot always be known before assembly: the prompt budget may drop text, and only then is the list partial
        if retrieval.completeness != "partial" and self.finalizer.list_completeness(retrieval, prompt) == "partial":
            # The prompt budget left read text out, so the list became partial only now: tell the model too
            notes = [*(notes or []), LIST_PARTIAL_NOTE]
            prompt = self.assembler.assemble_prompt(request, retrieval.nodes, state_fence, guards, service_notes=notes)
        return prompt, retrieval, guards, "generation"

    @staticmethod
    def _done_data(
        response: RAGResponse[MetaT], run: RequestRun, request: RAGQueryRequest,
        citations: list[dict[str, Any]], sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """The stream's `done` payload; one place, so an ask and an answer cannot drift apart."""
        return {
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
            "completeness": response.completeness,
            "clarification": response.clarification.model_dump() if response.clarification else None,
            "request_id": run.request_id,
            "message_id": request.metadata.get("message_id"),
        }

    # ── Clarification (§7.10, OI-5) ───────────────────────────────────────

    def _clarification(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent,
        retrieval: RAGRetrievalResult[MetaT],
        guards: list[Guard],
        run: RequestRun,
    ) -> Clarification | None:
        """The engine's suggestion that this question is unclear, in every mode. No check without a reason."""

        def prior() -> list[RAGNode[MetaT]]:
            """The passages shown on earlier turns, re-fetched under today's filters and guards.

            Used to confirm a remembered name against text the user may still see: if the boundary moved
            back, a name from a hidden passage is not offered. Called lazily, only when a check needs it.
            """
            state = request.state
            if not (state and state.prior_nodes and self.store.capabilities().supports_fetch_nodes):
                return []
            limits = self.profile.context
            ids = reusable_node_ids(state, limits.reused_turns, limits.max_reused_nodes)
            filters, _, _ = self.pipeline.scoped_filters(request, intent)
            if filters.is_empty_scope:
                return []                                  # nothing is visible, so no store call (§5.5)
            fetched = call_store("fetch_nodes", lambda: self.store.fetch_nodes(ids, filters))
            return apply_guards(fetched, guards, run.trace.record_guard_effect)

        try:
            return find_clarification(
                state=request.state, intent=intent, nodes=retrieval.nodes, prior=prior,
                candidate_key=self.candidate_key, config=self.profile.clarification, prompt_text=self.renderer.prompt_text,
            )
        except NEVER_DEGRADABLE:
            raise
        except RAGEngineError as exc:
            # The mark is optional: a failed confirming lookup must not fail the answer (DG-16, 1.2.18). An app's own
            # candidate_key error is not an engine error and still propagates.
            run.degradations.record(
                dg_id="DG-16", stage="clarification", reason=f"Clarification lookup failed: {exc}",
                impact="No clarification mark this turn", error_type=type(exc).__name__,
            )
            return None

    def _asks(self, request: RAGQueryRequest, retrieval: RAGRetrievalResult[MetaT]) -> bool:
        """Ask instead of answering: opted in, an askable reason, never twice in a row, and only with a state sent back."""
        mode = request.clarify or self.profile.clarification.mode
        return (
            mode == "ask"
            and may_ask(retrieval.clarification)
            and request.state is not None
            and not request.state.clarification_asked
        )

    def _wants_hedge_note(self, request: RAGQueryRequest, retrieval: RAGRetrievalResult[MetaT]) -> bool:
        """Opt-in, or automatically the turn right after an ask, while the question is still unclear (§7.10).

        For either askable reason; `too_broad` is only a mark, so it never adds the note.
        """
        c = retrieval.clarification
        if c is None or c.reason not in ASKABLE_REASONS or self._asks(request, retrieval):
            return False
        after_ask = request.state is not None and request.state.clarification_asked
        return self.profile.clarification.hedge_note or after_ask

    def _generation_context(self, request: RAGQueryRequest, run: RequestRun, stage: str) -> GenerationContext:
        """What a generator needs besides the prompt: where to record usage and degradations, and the budget."""
        return GenerationContext(
            usage_tracker=run.usage,
            degradation_tracker=run.degradations,
            budget_guard=self.budget_guard,
            stage=stage,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def route(self, request: RAGQueryRequest, *, cancel: CancelToken | None = None) -> QueryIntent:
        """Public routing entry point (§7.2, AC-RAG-16)."""
        if request.variant:
            return self._variant_engine(request.variant).route(
                request.model_copy(update={"variant": None}), cancel=cancel
            )
        return self._resolve_intent(request, None, self._begin(request, cancel))

    def plan(self, request: RAGQueryRequest, intent: QueryIntent | None = None) -> QueryPlan:
        """Evaluate the adaptive rules without retrieval or generation (§8.12, AC-RAG-48, AC-RAG-58)."""
        if request.variant:
            return self._variant_engine(request.variant).plan(request.model_copy(update={"variant": None}), intent)
        resolved = intent or QueryIntent(search_query=request.question, router_source="passthrough")
        filters, _, _ = self.pipeline.scoped_filters(request, resolved)
        if filters.is_empty_scope:
            # Hard constraints conflict: nothing can match, so no store call and no index is needed (§5.5)
            return QueryPlan(
                strategy="none",
                tier_decision_reason="none: empty scope (hard constraints conflict)",
                rule=0,
                requires_index=False,
                reads=frozenset(),
                applied_filters=filters,
                scope=None,
            )
        return AdaptiveRetrievalStrategy.decide_tier(request, resolved, filters, self.store, self.profile, self.registry)

    def retrieve(
        self, request: RAGQueryRequest, intent: QueryIntent | None = None, *, cancel: CancelToken | None = None
    ) -> RAGRetrievalResult[MetaT]:
        """Ranked list retrieval mode (§8.10, AC-RAG-51)."""
        if request.variant:
            return self._variant_engine(request.variant).retrieve(
                request.model_copy(update={"variant": None}), intent, cancel=cancel
            )
        run = self._begin(request, cancel)
        try:
            resolved = self._resolve_intent(request, intent, run)
            result, guards = self.pipeline.run(request, resolved, "retrieve", run)
            result.clarification = self._clarification(request, resolved, result, guards, run)
        except Exception as exc:
            self._fail(exc, run)
            raise
        result.trace = run.trace.build(
            result.applied_filters, run.degradations.degradations, run.usage.stage_usages, run.usage.total()
        )
        run.trace.dispatcher.notify_complete(result.trace, result)
        return result

    def query(
        self, request: RAGQueryRequest, intent: QueryIntent | None = None, *, cancel: CancelToken | None = None
    ) -> RAGResponse[MetaT]:
        """End-to-end question answering (§14). `cancel` lets the caller stop it from another thread (§13.4).

        Call tree:

            query()
            ├─ _begin()                         per-request trackers, deadline, trace
            ├─ _resolve_intent()                router -> clamp -> resolve dates
            ├─ _prepare_generation()
            │  ├─ pipeline.run()                filters, guards, strategy, post stages, relaxation
            │  ├─ _clarification()              mark an unclear question (no model call)
            │  └─ assembler.assemble_prompt()   fence, budget and order the text the model sees
            ├─ _asks()  ->  finalizer.ask_response()     opt-in: return the question instead of an answer
            └─ _generate_and_finish()
               ├─ generator.generate()          the one model call
               └─ finalizer.finish()            citations, quotes, sanitize, state, trace -> RAGResponse
        """
        if request.variant:
            return self._variant_engine(request.variant).query(
                request.model_copy(update={"variant": None}), intent, cancel=cancel
            )
        run = self._begin(request, cancel)
        retrieval: RAGRetrievalResult[MetaT] | None = None     # kept so a failure can report what was found
        try:
            resolved = self._resolve_intent(request, intent, run)
            prompt, retrieval, guards, stage = self._prepare_generation(request, resolved, run)
            if self._asks(request, retrieval):
                return self.finalizer.ask_response(request, retrieval, prompt, run)
            return self._generate_and_finish(request, prompt, retrieval, guards, stage, run)
        except Exception as exc:
            self._fail(exc, run, retrieval)
            raise

    def generate_from_nodes(
        self,
        request: RAGQueryRequest,
        nodes: list[RAGNode[MetaT]],
        intent: QueryIntent | None = None,
        *,
        cancel: CancelToken | None = None,
    ) -> RAGResponse[MetaT]:
        """Run guards, assembly, generation and citation checks on caller-supplied nodes (§14.4)."""
        if request.variant:
            return self._variant_engine(request.variant).generate_from_nodes(
                request.model_copy(update={"variant": None}), nodes, intent, cancel=cancel
            )
        run = self._begin(request, cancel)
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
        """Make the model call and turn its raw text into the final response.

        The model's text carries a hidden answer-found flag; `FlagStreamBuffer` strips it from what the user
        sees and reports it separately. A cancelled request raises here, before any answer is saved.
        """
        run.deadline.check("generation")
        started = run.trace.elapsed_ms()
        result = self.generator.generate(prompt, self._generation_context(request, run, stage))
        run.trace.record_stage(self.generator.name, "generation", started, version=self.generator.version)
        run.deadline.raise_if_cancelled("generation")      # a cancelled request gets no answer to save

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
            citations, _ = self.quote_validator.verify(citations, nodes, answer_text=clean_text)
        return citations

    def query_batched(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent | None = None,
        *,
        doc_ids: list[str] | None = None,
        cancel: CancelToken | None = None,
    ) -> RAGResponse[MetaT]:
        """A list question across many documents: gather, read, answer in batches, merge (OI-4, §14).

        Returns the same `RAGResponse` as `query()`, with `batches` saying how much was read. Several model calls:
        at most `generation.list_max_batches` plus one merge. `doc_ids` names the documents when the app knows them.
        """
        if request.variant:
            return self._variant_engine(request.variant).query_batched(
                request.model_copy(update={"variant": None}), intent, doc_ids=doc_ids, cancel=cancel
            )
        run = self._begin(request, cancel)
        try:
            run.deadline.raise_if_cancelled("routing")
            resolved = self._resolve_intent(request, intent, run)
            return BatchedListAnswerer(self, self._pool).run(request, resolved, doc_ids, run)
        except Exception as exc:
            self._fail(exc, run)
            raise

    def stream_query(
        self, request: RAGQueryRequest, intent: QueryIntent | None = None, *, cancel: CancelToken | None = None
    ) -> RAGStream[MetaT]:
        """Event streaming query pipeline (§10.8, AC-RAG-45, RAG-OPS-1, RAG-OPS-2).

        Closing the stream stops generation; `cancel` also stops routing and retrieval from another thread (§13.4).
        """
        if request.variant:
            return self._variant_engine(request.variant).stream_query(
                request.model_copy(update={"variant": None}), intent, cancel=cancel
            )
        run = self._begin(request, cancel)
        return RAGStream[MetaT](self._stream_events(request, intent, run), usage_tracker=run.usage)

    def _stream_events(
        self,
        request: RAGQueryRequest,
        intent: QueryIntent | None,
        run: RequestRun,
    ) -> Iterator[RAGStreamEvent]:
        """The generator behind `stream_query`. It is the streaming twin of `query()`, with the same steps.

        Event order for an answer (§10.8):
            intent, [degradation], status(retrieving), [degradation], retrieval, status(generating),
            delta..., [degradation], citations, done
        Event order for an opt-in ask (§7.10): the same up to `retrieval`, then
            delta (the question), clarification, citations (empty), done
        On failure: `error` is the last event, then the exception is re-raised. When the consumer closes the
        stream early, `GeneratorExit` is raised at a `yield` and the `except GeneratorExit` branch cleans up.
        """
        retrieval: RAGRetrievalResult[MetaT] | None = None
        chunks: Any = None                  # the live model stream, so a cancel can close it (RAG-OPS-2)
        emitted_degradations = 0

        def new_degradations() -> list[RAGStreamEvent]:
            """Degradation events not yet sent; called between steps so the client sees them as they happen."""
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
                        "labels": [self.formatter.label(n) for n in retrieval.nodes],
                        "node_count": len(retrieval.nodes),
                        "strategy_used": retrieval.strategy_used,
                        "applied_filters": retrieval.applied_filters.model_dump(),
                    },
                )

            if self._asks(request, retrieval):
                # Opt-in ask: the question as a delta (a delta-only client shows it), then the structured event
                response = self.finalizer.ask_response(request, retrieval, prompt, run)
                yield RAGStreamEvent(event="delta", data={"text": response.answer_final})
                if response.clarification is not None:
                    yield RAGStreamEvent(event="clarification", data=response.clarification.model_dump())
                yield RAGStreamEvent(
                    event="citations",
                    data={
                        "citations": [], "sources": [], "citation_coverage": 1.0,
                        "citation_coverage_detail": CitationCoverage().model_dump(),
                    },
                )
                yield RAGStreamEvent(event="done", data=self._done_data(response, run, request, [], []))
                return

            yield RAGStreamEvent(event="status", data={"stage": "generating"})
            run.deadline.check("generation")
            started = run.trace.elapsed_ms()
            context = self._generation_context(request, run, stage)
            buffer = FlagStreamBuffer()
            raw_parts: list[str] = []
            visible_parts: list[str] = []

            def deltas_for(texts: list[str]) -> list[RAGStreamEvent]:
                """Turn the buffer's visible text into delta events, keeping a copy for the final answer."""
                visible_parts.extend(texts)
                return [RAGStreamEvent(event="delta", data={"text": text}) for text in texts]

            # "buffered" waits for the whole answer, then emits it; "native" forwards the model's chunks as they arrive
            if self.profile.generation.stream_mode == "buffered":
                result = self.generator.generate(prompt, context)
                run.deadline.raise_if_cancelled("generation")
                raw_parts.append(result.text)
                yield from new_degradations()
                yield from deltas_for(buffer.feed(result.text))
            else:
                chunks = self.generator.stream(prompt, context)
                for chunk in chunks:
                    run.deadline.raise_if_cancelled("generation")
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
                    "citation_coverage_detail": coverage.model_dump() if coverage else CitationCoverage().model_dump(),
                },
            )
            yield RAGStreamEvent(event="done", data=self._done_data(response, run, request, citations, sources))
        except GeneratorExit:
            # The consumer closed the stream early (a disconnect or a `break`)
            # Closing the model stream makes the gateway record the cancelled call (RAG-OPS-2)
            if chunks is not None:
                chunks.close()
            try:
                # Observers see every request, cancelled ones too (RAG-EVAL-8); outcome None = no answer
                trace = run.trace.build(
                    retrieval.applied_filters if retrieval is not None else None,
                    run.degradations.degradations, run.usage.stage_usages, run.usage.total(),
                )
                run.trace.dispatcher.notify_complete(trace, None)
            except Exception:
                logger.error(f"[RAG] id={run.request_id} on_complete failed for a cancelled stream", exc_info=True)
            raise
        except Exception as exc:
            if isinstance(exc, RequestCancelled) and chunks is not None:
                chunks.close()                      # records the cut-off model call as cancelled (RAG-OPS-2)
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

    def _ingestion_trackers(
        self, request_id: str, doc_ids: list[str], owner: OwnerScope | None
    ) -> tuple[UsageTracker, DegradationTracker]:
        """Trackers for one ingestion call: they carry the owner to the BudgetGuard and report to observers."""
        degradations = DegradationTracker(fallbacks=self.profile.fallbacks)
        dispatcher = self.dispatcher.for_request(degradations)
        degradations.on_record = dispatcher.notify_degrade
        usage = UsageTracker(
            request_id=request_id,
            tenant_or_user_id=owner.tenant_or_user_id if owner else None,
            doc_ids=doc_ids,
            profile_name=self.profile.name,
            variant=self._variant_name,
            scope=owner.as_scope() if owner else {},
            on_record=dispatcher.notify_usage,
        )
        return usage, degradations

    def ingest_document(
        self,
        doc: RAGDocumentInput[MetaT],
        start_section_seq: int = 0,
        on_section_complete: Callable[[IngestionProgress], None] | None = None,
        *,
        owner: OwnerScope | None = None,
    ) -> IngestionProgress:
        """Split, summarize, embed and atomically upsert a document, one section at a time.

        Resumable via start_section_seq and append-only (RAG-ING-5, RAG-ING-6, AC-RAG-05).
        `owner` must state every key in the store's `required_filter_keys` (§6.4.2); it is checked
        before any model or store call.
        """
        validate_owner(owner, self.store.capabilities())
        sections = [s for s in sorted(doc.sections, key=lambda x: x.section_seq) if s.section_seq >= start_section_seq]
        last_progress: IngestionProgress | None = None
        completed = len(doc.sections) - len(sections)  # a resumed job counts what earlier runs finished

        for section in sections:
            tracker, deg_tracker = self._ingestion_trackers(
                f"ingest-{doc.doc_id}-s{section.section_seq}", [doc.doc_id], owner
            )
            try:
                last_progress = self._ingest_section(doc, section, completed + 1, tracker, deg_tracker, owner)
            except RAGEngineError as exc:
                self._attach_ingestion_failure(exc, tracker, deg_tracker)
                raise
            completed += 1
            if on_section_complete:
                on_section_complete(last_progress)

        return last_progress or IngestionProgress(
            doc_id=doc.doc_id,
            section_seq=0,
            sections_completed=completed,
            sections_total=len(doc.sections),
            passages_upserted=0,
            summary_status="skipped",
        )

    @staticmethod
    def _attach_ingestion_failure(exc: RAGEngineError, tracker: UsageTracker, degradations: DegradationTracker) -> None:
        """Give an ingestion error what the call had spent before it failed (RAG-ING-12, RAG-OPS-1)."""
        exc.request_id = exc.request_id or tracker.request_id
        exc.degradations = degradations.degradations
        exc.usage = tracker.stage_usages
        exc.usage_total = tracker.total()

    def _ingest_section(
        self,
        doc_input: RAGDocumentInput[MetaT],
        section: RAGSectionInput[MetaT],
        sections_completed: int,
        tracker: UsageTracker,
        deg_tracker: DegradationTracker,
        owner: OwnerScope | None,
    ) -> IngestionProgress:
        """Summarize, split, embed and atomically upsert one section (RAG-ING-3, RAG-ING-10, RAG-ING-12).

        Steps: write signatures (what the stored rows will say they were built with) -> summarize ->
        split into passages -> embed passages -> embed summaries -> one atomic `upsert_section`.
        The write is atomic per section, so a failure leaves the store as it was and the job can resume.
        """
        uses_tokens = self.store.capabilities().uses_analyzer_tokens
        signatures = write_signatures(
            self.profile.models.embedding, self.store.embedding_space(), self.splitter, self.analyzer, uses_tokens
        )

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

        nodes = build_section_nodes(
            doc_input.doc_id, section, self.splitter, doc_input.revision, self.profile.ingestion.granularity
        )
        for summary_node in ([result.section_summary] if result.section_summary else []) + result.unit_summaries:
            summary_node.revision = doc_input.revision
        for node in nodes:
            finalize_node(node, self.renderer, self.analyzer, uses_tokens)
        pairs = self.embedder.embed_nodes(nodes, tracker, deg_tracker, budget_guard=self.budget_guard)

        summaries = prepare_summaries(
            result,
            self.embedder,
            self.renderer,
            self.analyzer,
            uses_tokens,
            tracker,
            deg_tracker,
            self.budget_guard,
            signatures=signatures,
            embed=self.profile.ingestion.summaries.embed_summaries,
        )
        write = SectionWrite[MetaT](
            doc_id=doc_input.doc_id,
            section_seq=section.section_seq,
            owner=owner or OwnerScope(),
            passages=prepare_passages(pairs, signatures),  # type: ignore[arg-type]
            unit_summaries=summaries.unit_summaries,
            section_summary=summaries.section_summary,
        )
        written = call_store("upsert_section", lambda: self.store.upsert_section(write))
        return IngestionProgress(
            doc_id=doc_input.doc_id,
            section_seq=section.section_seq,
            sections_completed=sections_completed,
            sections_total=len(doc_input.sections),
            passages_upserted=written,
            summary_status=result.summary_status,
            usage=tracker.stage_usages,
            degradations=deg_tracker.degradations,
        )

    def sync_document(self, doc: RAGDocumentInput[MetaT], *, owner: OwnerScope | None = None) -> IngestionProgress:
        """Embed only changed nodes, delete vanished ones, and preserve node ids (§6.6, AC-RAG-06).

        `owner` follows the same rule as `ingest_document` (§6.4.2).
        """
        validate_owner(owner, self.store.capabilities())
        tracker, deg_tracker = self._ingestion_trackers(f"sync-{doc.doc_id}", [doc.doc_id], owner)
        try:
            return self.synchronizer.sync(doc, tracker, deg_tracker, owner or OwnerScope())
        except RAGEngineError as exc:
            self._attach_ingestion_failure(exc, tracker, deg_tracker)
            raise

    def prepare_node(self, node: RAGNode[MetaT], *, owner: OwnerScope | None = None) -> PreparedNode[MetaT]:
        """Prepare one node for an app-side insert: it either succeeds completely or raises (§6.4.1)."""
        return self.prepare_nodes([node], owner=owner).prepared[0]

    def prepare_nodes(self, nodes: list[RAGNode[MetaT]], *, owner: OwnerScope | None = None) -> PreparedBatch[MetaT]:
        """Prepare a batch for app-side inserts, judging each node on its own (§6.4.1, RAG-ING-13).

        Nodes that fail are listed in `failed` and the good ones are returned, so the caller can
        keep those and retry the rest. Raises only when nothing succeeded or the budget ran out.
        Nothing is written, but the app inserts the result, so `owner` follows the same rule as
        `ingest_document` and the spend counts against that owner (§6.4.2).
        """
        validate_owner(owner, self.store.capabilities())
        if not nodes:
            return PreparedBatch[MetaT]()

        # Each node is embedded on its own, so one bad node cannot sink the batch
        doc_ids = list(dict.fromkeys(node.span.doc_id for node in nodes))
        tracker, degradations = self._ingestion_trackers(f"prepare-{uuid.uuid4().hex[:12]}", doc_ids, owner)
        try:
            embedded, failures = self.embedder.embed_each_with_reasons(
                nodes, tracker, degradations, "prepare_embedding", self.budget_guard
            )
            failed = [node for node, _ in failures]
            if not embedded:
                raise EmbeddingProviderError(
                    f"All {len(nodes)} nodes failed to embed with model '{self.profile.models.embedding}'",
                    failed_node_ids=[node.node_id for node in failed],
                    stage="prepare_embedding",
                )
        except RAGEngineError as exc:
            self._attach_ingestion_failure(exc, tracker, degradations)
            raise

        if failed:
            degradations.record(
                dg_id="DG-08",
                stage="prepare_embedding",
                reason=f"{len(failed)} of {len(nodes)} nodes failed to embed",
                impact="Failed nodes were returned in `failed`; the rest are ready to insert",
            )

        # The fingerprint travels with the node so the app's insert stores what `needs_reindex` later compares
        uses_tokens = self.store.capabilities().uses_analyzer_tokens
        emb_sig = embedding_signature(self.profile.models.embedding, self.store.embedding_space(), self.splitter)
        fts_sig = fts_signature(self.analyzer, uses_tokens)

        prepared: list[PreparedNode[MetaT]] = []
        for node, vector in embedded:
            node_copy = node.model_copy()
            finalize_node(node_copy, self.renderer, self.analyzer, uses_tokens)
            prepared.append(
                PreparedNode[MetaT](
                    node=node_copy,
                    vector=vector,
                    fingerprint=Fingerprint(
                        content_hash=node_copy.content_hash,
                        embedding_signature=emb_sig,
                        fts_signature=fts_sig,
                    ),
                )
            )
        return PreparedBatch[MetaT](
            prepared=prepared,
            failed=[FailedNode(node_id=node.node_id, reason=reason) for node, reason in failures],
            usage=tracker.stage_usages,
            degradations=degradations.degradations,
        )

    def needs_reindex(self, doc_id: str) -> ReindexStatus:
        """Check whether stored embeddings or FTS tokens are stale (§6.6, AC-RAG-64).

        A row stored without a vector by design is never stale for embeddings; a summary without
        a vector is stale when the profile now embeds summaries.
        """
        stored = self.store.get_fingerprints(doc_id)
        uses_tokens = self.store.capabilities().uses_analyzer_tokens
        expected_embedding = embedding_signature(
            self.profile.models.embedding, self.store.embedding_space(), self.splitter
        )
        expected_fts = fts_signature(self.analyzer, uses_tokens)
        embed_summaries = self.profile.ingestion.summaries.embed_summaries
        return ReindexStatus(
            doc_id=doc_id,
            embeddings_stale=any(
                embedding_is_stale(doc_id, node_id, fp, expected_embedding, embed_summaries)
                for node_id, fp in stored.items()
            ),
            # Only keyword-searched rows count: summaries never are (§4.2), so they carry no FTS signature
            fts_stale=uses_tokens and any(
                fp.fts_signature != expected_fts
                for node_id, fp in stored.items()
                if not ((parsed := parse_node_id(doc_id, node_id)) is not None and parsed.is_summary)
            ),
        )
