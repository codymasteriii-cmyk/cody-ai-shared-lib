"""Turns raw model output into the final response; shared by query, stream, generate_from_nodes and query_batched.

It also owns the two other ways a request ends: the clarifying question (§7.10) and the conversation state every
answer hands back, so the state is updated in one place whichever entry point answered.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Generic, Literal, Sequence

from .clarification import question_text, remember_entities
from .context.assembler import AssembledPrompt
from .context.conversation import ConversationManager
from .models import (
    Completeness,
    ConversationState,
    MetaT,
    NodeRef,
    QueryIntent,
    RAGCitation,
    RAGNode,
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
        prompt_text: Callable[[Any], str],            # the renderer's view of a node: what the model reads
    ):
        self.profile = profile
        self.store = store
        self.analyzer = analyzer
        self.citation_extractor = citation_extractor
        self.quote_validator = quote_validator
        self.sanitizer = sanitizer
        self.conversation_manager = conversation_manager
        self.prompt_text = prompt_text

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
            current_revisions=self.current_revisions(nodes),
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
        completeness = self.list_completeness(retrieval, prompt)
        if completeness == "partial" and answer_found == "yes":
            answer_found = "partial"          # a list known to be missing items is not a plain "yes" (OI-4)
        if no_data:
            citations = []
        sources = self.deduplicated_sources(citations)

        state = self.next_state(request.state, prompt.fence_id, retrieval.intent, prompt.included_nodes)
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
            state_size_bytes=state_size(state),
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
            completeness=completeness,
            degraded=bool(degradations),
            citation_coverage=coverage,
            clarification=retrieval.clarification,
            state=state,
            usage=usage,
            usage_total=usage_total,
            degradations=degradations,
            latency_ms=trace.total_latency_ms,
            trace=trace,
        )
        self.complete(run, response)
        return response

    def ask_response(
        self,
        request: RAGQueryRequest,
        retrieval: RAGRetrievalResult[MetaT],
        prompt: AssembledPrompt[MetaT],
        run: RequestRun,
    ) -> RAGResponse[MetaT]:
        """The clarifying question as the answer, shaped so it cannot be mistaken for one; no generation call (§7.10)."""
        found, previous = retrieval.clarification, request.state
        assert found is not None and previous is not None      # the engine asks only with both (RAGEngine._asks)
        clarification = found.model_copy(update={"asked": True})
        retrieval.clarification = clarification
        question = question_text(clarification, request.question, self.profile.clarification.templates)
        # Nothing was shown, so no passage is remembered; the guard is set so the next turn never asks again
        state = self.next_state(previous, prompt.fence_id, retrieval.intent, [], asked=True)
        degradations = run.degradations.degradations
        usage, usage_total = run.usage.stage_usages, run.usage.total()
        answer_id = f"ans-{uuid.uuid4().hex[:12]}"
        trace = run.trace.build(
            applied_filters=retrieval.applied_filters, degradations=degradations, usage=usage, usage_total=usage_total,
            answer_id=answer_id, state_size_bytes=state_size(state),
        )
        response = RAGResponse[MetaT](
            answer_id=answer_id, answer=question, answer_final=question, citations=[], sources=[], retrieved=retrieval,
            grounded=False, answer_found=None, abstained=False, completeness=None, degraded=bool(degradations),
            citation_coverage=None, clarification=clarification, state=state, usage=usage, usage_total=usage_total,
            degradations=degradations, latency_ms=trace.total_latency_ms, trace=trace,
        )
        self.complete(run, response)
        return response

    def complete(self, run: RequestRun, response: RAGResponse[MetaT]) -> None:
        """Tell the observers and write the one summary line: the last step of every answered request."""
        if response.trace is not None:
            run.trace.dispatcher.notify_complete(response.trace, response)
        self.log_summary(run, response)

    @staticmethod
    def list_completeness(retrieval: RAGRetrievalResult[MetaT], prompt: AssembledPrompt[MetaT]) -> Completeness | None:
        """The retrieval's mark, downgraded to "partial" when the prompt budget left part of the read text out (OI-4)."""
        if retrieval.completeness not in ("whole_scope", "searched"):
            return retrieval.completeness
        return "partial" if prompt_cut(retrieval.nodes, prompt) else retrieval.completeness

    def current_revisions(self, nodes: Sequence[Any]) -> dict[str, str] | None:
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
    def deduplicated_sources(citations: list[RAGCitation]) -> list[RAGCitation]:
        seen: set[str] = set()
        sources: list[RAGCitation] = []
        for citation in citations:
            if citation.verified and citation.node_id not in seen:
                seen.add(citation.node_id)
                sources.append(citation)
        return sources

    def next_state(
        self,
        previous: ConversationState | None,
        fence_id: str,
        intent: QueryIntent | None,
        shown: Sequence[RAGNode[MetaT]],
        *,
        asked: bool = False,
    ) -> ConversationState:
        """The state to hand back: one more turn, the passages shown, and the never-ask-twice guard (§7.7, §7.10)."""
        state = previous.model_copy(deep=True) if previous is not None else ConversationState(fence_id=fence_id)
        state.fence_id = fence_id
        state.turn_count += 1
        state.profile_fingerprint = self.profile.fingerprint()
        state.clarification_asked = asked                 # set by an ask; the guard lasts one turn
        if intent is not None:
            state.recent_entities = remember_entities(state.recent_entities, intent, shown, self.prompt_text)
            state.last_intent = {"route": intent.route, "scope": intent.scope, "is_followup": intent.is_followup}
        state.prior_nodes.extend(
            NodeRef(node_id=node.node_id, revision=node.revision, turn=state.turn_count) for node in shown
        )
        return self.conversation_manager.prune_state(state)

    def log_summary(self, run: RequestRun, response: RAGResponse[MetaT]) -> None:
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


def prompt_cut(nodes: Sequence[RAGNode[Any]], prompt: AssembledPrompt[Any]) -> bool:
    """True when the prompt budget left one of `nodes` out, or shortened one (§10.1)."""
    included = {node.node_id for node in prompt.included_nodes}
    dropped = any(node.node_id not in included for node in nodes)
    return dropped or any("truncated_by_budget" in node.flags for node in prompt.included_nodes)


def state_size(state: ConversationState) -> int:
    return len(state.model_dump_json().encode("utf-8"))
