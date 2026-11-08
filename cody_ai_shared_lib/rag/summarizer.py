"""Section summarizer and summary unit generator for RAG ingestion.

Implements:
  - Parent section summaries (level=1, parent_kind="section") (RAG-ING-3)
  - Summary units for sections > summary_unit_chars (level=0, parent_kind="section_unit") (RAG-ING-9)
  - RAG-ING-8: Derived text must not look ahead; unit summaries summarize only their own span.
  - DG-14: Summarizer failure raises RAGStageError unless fallbacks["summarizer"] == "skip",
    which records DG-14, logs ERROR, and returns summary_status="failed".
"""
from __future__ import annotations

import logging
from typing import Generic, Literal
from pydantic import BaseModel

from .degradation import DegradationTracker
from .errors import RAGStageError
from .gateway import ModelGateway
from .models import MetaT, RAGNode, RAGNodeSpan
from .text.analyzer import CjkAwareAnalyzer, TextAnalyzer
from .usage import UsageTracker

logger = logging.getLogger("shared-rag.ingestion")


class SummarizerResult(BaseModel, Generic[MetaT]):
    """Result of section summarization."""
    section_summary: RAGNode[MetaT] | None = None
    unit_summaries: list[RAGNode[MetaT]] = []
    summary_status: Literal["generated", "provided", "skipped", "failed"] = "skipped"


class SectionSummarizer(Generic[MetaT]):
    """Generates hierarchical section summaries and unit summaries."""

    def __init__(
        self,
        gateway: ModelGateway,
        summary_models: list[str] | None,
        enabled: bool = True,
        min_chars_for_summary: int = 1500,
        summary_unit_chars: int = 8000,
        analyzer: TextAnalyzer | None = None,
        fallback_action: str = "raise",  # "raise" | "skip" (DG-14)
    ):
        self.gateway = gateway
        self.summary_models = summary_models or []
        self.enabled = enabled
        self.min_chars_for_summary = min_chars_for_summary
        self.summary_unit_chars = summary_unit_chars
        self.analyzer = analyzer or CjkAwareAnalyzer()
        self.fallback_action = fallback_action

    def summarize_section(
        self,
        doc_id: str,
        section_seq: int,
        section_id: str | None,
        text: str,
        char_start: int,
        metadata: MetaT,
        existing_summary: str | None,
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
        stage: str = "summarizer",
    ) -> SummarizerResult[MetaT]:
        """Summarize a section, generating section summary and summary units."""
        if not self.enabled:
            return SummarizerResult(summary_status="skipped")

        # 1. Check existing summary
        if existing_summary:
            sec_node = RAGNode[MetaT](
                node_id=f"{doc_id}#s{section_seq}_sum",
                doc_id=doc_id,
                section_id=section_id,
                section_seq=section_seq,
                granularity="parent_summary",
                parent_kind="section",
                parent_id=section_id or doc_id,
                level=1,
                text=existing_summary,
                span=RAGNodeSpan(char_start=char_start, char_end=char_start + len(text)),
                metadata=metadata,
            )
            return SummarizerResult(
                section_summary=sec_node,
                summary_status="provided",
            )

        if len(text) < self.min_chars_for_summary:
            return SummarizerResult(summary_status="skipped")

        if not self.summary_models:
            return SummarizerResult(summary_status="skipped")

        # 2. Generate section summary
        section_summary_node: RAGNode[MetaT] | None = None
        status: Literal["generated", "provided", "skipped", "failed"] = "skipped"

        system_prompt = (
            "You are a summarization assistant. Provide a concise 2-4 sentence summary of the "
            "following text in the same language as the text. Do not invent any facts."
        )

        try:
            res = self.gateway.generate(
                system_prompt=system_prompt,
                user_message=text,
                models=self.summary_models,
                stage=stage,
                usage_tracker=usage_tracker,
                degradation_tracker=degradation_tracker,
            )
            section_summary_node = RAGNode[MetaT](
                node_id=f"{doc_id}#s{section_seq}_sum",
                doc_id=doc_id,
                section_id=section_id,
                section_seq=section_seq,
                granularity="parent_summary",
                parent_kind="section",
                parent_id=section_id or doc_id,
                level=1,
                text=res.text.strip(),
                span=RAGNodeSpan(char_start=char_start, char_end=char_start + len(text)),
                metadata=metadata,
            )
            status = "generated"
        except Exception as exc:
            if self.fallback_action == "skip":
                logger.error(f"[Summarizer] Failed to summarize section {section_seq}: {exc} [DG-14]", exc_info=True)
                degradation_tracker.record(
                    dg_id="DG-14",
                    stage=stage,
                    reason=f"Summarizer failed: {exc}",
                    fallback_used="skip",
                    impact="Section ingested without parent summary",
                )
                return SummarizerResult(summary_status="failed")
            else:
                raise RAGStageError(
                    stage=stage,
                    message=f"Summarizer failed for section {section_seq}: {exc}",
                    cause=exc,
                )

        # 3. Generate summary units if section > summary_unit_chars (RAG-ING-9)
        unit_summaries: list[RAGNode[MetaT]] = []
        if len(text) > self.summary_unit_chars:
            units = self._slice_summary_units(text)
            for u_idx, (u_start, u_end) in enumerate(units):
                u_text = text[u_start:u_end]
                try:
                    u_res = self.gateway.generate(
                        system_prompt=system_prompt,
                        user_message=u_text,
                        models=self.summary_models,
                        stage=f"{stage}_unit",
                        usage_tracker=usage_tracker,
                        degradation_tracker=degradation_tracker,
                    )
                    unit_node = RAGNode[MetaT](
                        node_id=f"{doc_id}#s{section_seq}_u{u_idx}_sum",
                        doc_id=doc_id,
                        section_id=section_id,
                        section_seq=section_seq,
                        granularity="parent_summary",
                        parent_kind="section_unit",
                        parent_id=section_id or doc_id,
                        level=0,
                        text=u_res.text.strip(),
                        span=RAGNodeSpan(
                            char_start=char_start + u_start,
                            char_end=char_start + u_end,
                        ),
                        metadata=metadata,
                    )
                    unit_summaries.append(unit_node)
                except Exception as exc:
                    if self.fallback_action == "skip":
                        logger.error(f"[Summarizer] Unit {u_idx} failed: {exc}", exc_info=True)
                        degradation_tracker.record(
                            dg_id="DG-14",
                            stage=f"{stage}_unit",
                            reason=f"Summary unit {u_idx} failed: {exc}",
                            fallback_used="skip",
                            impact="Unit skipped",
                        )
                    else:
                        raise RAGStageError(
                            stage=stage,
                            message=f"Summary unit {u_idx} failed: {exc}",
                            cause=exc,
                        )

        return SummarizerResult(
            section_summary=section_summary_node,
            unit_summaries=unit_summaries,
            summary_status=status,
        )

    def _slice_summary_units(self, text: str) -> list[tuple[int, int]]:
        """Slice text into consecutive sentence-aligned units of ~summary_unit_chars."""
        sents = self.analyzer.sentences(text)
        if not sents:
            return [(0, len(text))]

        units: list[tuple[int, int]] = []
        curr_start = sents[0][0]
        curr_sent_idx = 0
        n_sents = len(sents)

        while curr_sent_idx < n_sents:
            start_char = sents[curr_sent_idx][0]
            next_sent = curr_sent_idx
            while next_sent < n_sents:
                end_char = sents[next_sent][1]
                if (end_char - start_char) >= self.summary_unit_chars and next_sent > curr_sent_idx:
                    break
                next_sent += 1

            unit_end_char = sents[next_sent - 1][1]
            units.append((start_char, unit_end_char))
            curr_sent_idx = next_sent

        return units
