"""Turns raw model output into the final response; shared by query, stream and generate_from_nodes."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Generic, Literal, Sequence

from .context.assembler import AssembledPrompt
from .context.conversation import ConversationManager
from .models import (
    ConversationState,
    MetaT,
    NodeRef,
    QueryIntent,
    RAGCitation,
    RAGQueryRequest,
    RAGResponse,
    RAGRetrievalResult,
)
from .ports import Guard, RAGStore, call_store
from .postgen.citations import CitationExtractor
from .postgen.quotes import QuoteValidator, compute_coverage
from .postgen.sanitizer import OutputSanitizer
from .profile import EngineProfile
from .runtime import RequestRun
from .text.analyzer import TextAnalyzer

logger = logging.getLogger("shared-rag")


class AnswerFinalizer(Generic[MetaT]):
    """Verifies citations, sanitizes the answer, updates state, and reports the outcome."""

    def __init__(
        self,
        profile: EngineProfile,
        store: RAGStore[MetaT],
        analyzer: TextAnalyzer,
        citation_extractor: CitationExtractor[MetaT],
        quote_validator: QuoteValidator[MetaT],
        sanitizer: OutputSanitizer,
        conversation_manager: ConversationManager,
    ):
        self.profile = profile
        self.store = store
        self.analyzer = analyzer
        self.citation_extractor = citation_extractor
        self.quote_validator = quote_validator
        self.sanitizer = sanitizer
        self.conversation_manager = conversation_manager

    def finish(
        self,
        request: RAGQueryRequest,
        raw_text: str,
        visible_text: str,
        flag: Literal["yes", "no", "partial"] | None,
        prompt: AssembledPrompt[MetaT],
        retrieval: RAGRetrievalResult[MetaT],
        guards: Sequence[Guard],
        run: RequestRun,
    ) -> RAGResponse[MetaT]:
        """Build the response for a completed generation (§10.5, §11)."""
        nodes = prompt.included_nodes
        clean_text, citations = self.citation_extractor.extract(
            visible_text,
            nodes,
            guards=guards,
            current_revisions=self._current_revisions(nodes),
            on_unverified=self.profile.citations.on_unverified,
            turn_map=prompt.turn_map,
        )
        if self.profile.citations.verify_quotes:
            citations, coverage = self.quote_validator.verify(citations, nodes, answer_text=clean_text)
        else:
            coverage = compute_coverage(citations, clean_text, self.analyzer)
        final_answer = self.sanitizer.sanitize(clean_text, nodes=nodes, fence_id=prompt.fence_id)

        # RAG-RES-4: a search that found nothing is "no"; on the no-retrieval route nothing was searched.
        no_retrieval = retrieval.intent is not None and retrieval.intent.route == "no_retrieval"
        no_data = not retrieval.nodes
        answer_found = None if no_retrieval else ("no" if no_data else flag)
        if no_data:
            citations = []
        sources = self._deduplicated_sources(citations)

        state = self._updated_state(request, prompt, retrieval.intent)
        answer_id = f"ans-{uuid.uuid4().hex[:12]}"
        degradations = run.degradations.degradations
        usage = run.usage.stage_usages
        usage_total = run.usage.total()

        run.trace.prompt_version = prompt.prompt_version
        run.trace.zone_tokens = prompt.zone_tokens
        trace = run.trace.build(
            applied_filters=retrieval.applied_filters,
            degradations=degradations,
            usage=usage,
            usage_total=usage_total,
            answer_id=answer_id,
            state_size_bytes=len(state.model_dump_json().encode("utf-8")),
        )
        response = RAGResponse[MetaT](
            answer_id=answer_id,
            answer=raw_text,
            answer_final=final_answer,
            citations=citations,
            sources=sources,
            retrieved=retrieval,
            grounded=self._is_grounded(prompt, final_answer),
            answer_found=answer_found,
            abstained=answer_found == "no" and prompt.grounding_mode == "strict",
            degraded=bool(degradations),
            citation_coverage=coverage,
            state=state,
            usage=usage,
            usage_total=usage_total,
            degradations=degradations,
            latency_ms=trace.total_latency_ms,
            trace=trace,
        )
        run.trace.dispatcher.notify_complete(trace, response)
        self._log_summary(run, response)
        return response

    def _current_revisions(self, nodes: Sequence[Any]) -> dict[str, str] | None:
        """One batched staleness lookup for nodes that carry a revision (§11.2)."""
        if not self.store.capabilities().supports_revisions:
            return None
        ids = [n.node_id for n in nodes if n.revision is not None]
        return call_store("current_revisions", lambda: self.store.current_revisions(ids)) if ids else None

    def _is_grounded(self, prompt: AssembledPrompt[MetaT], answer: str) -> bool:
        """False only when a labelled-fallback answer used the general-knowledge label (§10.5)."""
        return not (
            prompt.grounding_mode == "labelled_fallback" and self.profile.generation.fallback_label in answer
        )

    @staticmethod
    def _deduplicated_sources(citations: list[RAGCitation]) -> list[RAGCitation]:
        seen: set[str] = set()
        sources: list[RAGCitation] = []
        for citation in citations:
            if citation.verified and citation.node_id not in seen:
                seen.add(citation.node_id)
                sources.append(citation)
        return sources

    def _updated_state(
        self,
        request: RAGQueryRequest,
        prompt: AssembledPrompt[MetaT],
        intent: QueryIntent | None,
    ) -> ConversationState:
        state = (
            request.state.model_copy(deep=True)
            if request.state is not None
            else ConversationState(fence_id=prompt.fence_id)
        )
        state.fence_id = prompt.fence_id
        state.turn_count += 1
        state.profile_fingerprint = self.profile.fingerprint()
        if intent is not None:
            state.last_intent = {"route": intent.route, "scope": intent.scope, "is_followup": intent.is_followup}
        state.prior_nodes.extend(
            NodeRef(node_id=node.node_id, revision=node.revision, turn=state.turn_count)
            for node in prompt.included_nodes
        )
        return self.conversation_manager.prune_state(state)

    def _log_summary(self, run: RequestRun, response: RAGResponse[MetaT]) -> None:
        """Emit the one INFO summary line per request (§13.8)."""
        total = response.usage_total
        cost = f"{total.cost_usd:.6f}" if total.cost_usd is not None else "na"
        intent = response.retrieved.intent
        degraded_ids = ",".join(d.id for d in response.degradations) or "none"
        verified = sum(1 for c in response.citations if c.verified)
        logger.info(
            f"[RAG] id={run.request_id} profile={self.profile.name}@{self.profile.version} "
            f"strategy={response.retrieved.strategy_used} rule={run.trace.rule} "
            f"router={intent.router_source if intent else 'none'} nodes={len(response.retrieved.nodes)} "
            f"reranked={response.retrieved.reranked} citations={verified}/{len(response.citations)} "
            f"degr={degraded_ids} tokens={total.input_tokens}/{total.output_tokens} cost={cost} "
            f"ms={response.latency_ms:.0f} outcome={'degraded' if response.degraded else 'ok'}"
        )
