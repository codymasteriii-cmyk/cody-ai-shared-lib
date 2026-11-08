"""Agreed degradations table and tracking (§13.3, AC-RAG-55).

Closed list of agreed graceful degradation rows DG-01 to DG-14.
Any failure outside this list or an agreed fallback that itself fails raises an error.
"""
from __future__ import annotations

import logging
from typing import Any

from .models import Degradation

logger = logging.getLogger("rag-degradation")


# Closed list table specification
AGREED_DEGRADATIONS: dict[str, dict[str, Any]] = {
    "DG-01": {
        "stage": "provider",
        "description": "Transient provider error (rate limit, 5xx, network)",
        "default": "on",
        "log_level": logging.WARNING,
        "fallback": "retry_with_backoff",
    },
    "DG-02": {
        "stage": "model_list",
        "description": "Model still fails after retries",
        "default": "on",
        "log_level": logging.ERROR,
        "fallback": "next_model",
    },
    "DG-03": {
        "stage": "intent",
        "description": "Jev error or timeout",
        "default": "on",
        "log_level": logging.ERROR,
        "fallback": "llm_path",
    },
    "DG-04": {
        "stage": "intent",
        "description": "Intent output cannot be parsed",
        "default": "on",
        "log_level": logging.WARNING,
        "fallback": "retry_then_next_model",
    },
    "DG-05": {
        "stage": "intent",
        "description": "Intent output parses but has out-of-range values",
        "default": "on",
        "log_level": logging.WARNING,
        "fallback": "clamp_to_range",
    },
    "DG-06": {
        "stage": "rerank",
        "description": "Reranker error or timeout",
        "default": "on",
        "log_level": logging.ERROR,
        "fallback": "keep_incoming_order",
    },
    "DG-07": {
        "stage": "retrieval",
        "description": "Query embedding is a zero vector",
        "default": "opt-in",
        "log_level": logging.ERROR,
        "fallback": "keyword_search",
    },
    "DG-08": {
        "stage": "ingestion",
        "description": "Some nodes in section embed to zero",
        "default": "on",
        "log_level": logging.ERROR,
        "fallback": "skip_failed_nodes",
    },
    "DG-09": {
        "stage": "generation",
        "description": "Output truncated (finish_reason='length')",
        "default": "on",
        "log_level": logging.WARNING,
        "fallback": "retry_or_next_model",
    },
    "DG-10": {
        "stage": "runtime",
        "description": "Deadline: optional stage will not fit in remaining time",
        "default": "on",
        "log_level": logging.ERROR,
        "fallback": "skip_optional_stage",
    },
    "DG-11": {
        "stage": "budget",
        "description": "BudgetGuard returns 'skip' for an optional stage",
        "default": "on",
        "log_level": logging.WARNING,
        "fallback": "skip_stage",
    },
    "DG-12": {
        "stage": "cache",
        "description": "Cache read or write error",
        "default": "on",
        "log_level": logging.ERROR,
        "fallback": "continue_without_cache",
    },
    "DG-13": {
        "stage": "observer",
        "description": "Observer raises",
        "default": "on",
        "log_level": logging.ERROR,
        "fallback": "continue",
    },
    "DG-14": {
        "stage": "summarizer",
        "description": "Summarizer fails at ingestion",
        "default": "opt-in",
        "log_level": logging.ERROR,
        "fallback": "continue_without_summary",
    },
}


class DegradationTracker:
    """Collects and logs degradations that occur during request execution."""

    def __init__(self, fallbacks: dict[str, str] | None = None):
        self._fallbacks = dict(fallbacks or {})
        self._degradations: list[Degradation] = []

    @property
    def degradations(self) -> list[Degradation]:
        return list(self._degradations)

    def record(
        self,
        dg_id: str,
        stage: str,
        reason: str,
        impact: str,
        fallback_used: str = "",
        error_type: str | None = None,
        message: str | None = None,
        attempts: int = 1,
    ) -> Degradation:
        spec = AGREED_DEGRADATIONS.get(dg_id, {})
        log_level = spec.get("log_level", logging.WARNING)

        deg = Degradation(
            id=dg_id,
            stage=stage,
            reason=reason,
            impact=impact,
            fallback_used=fallback_used or spec.get("fallback", ""),
            error_type=error_type,
            message=message,
            attempts=attempts,
        )
        self._degradations.append(deg)

        log_msg = f"[Degradation {dg_id}] ({stage}): {reason}. Fallback: {deg.fallback_used}. Impact: {impact}"
        logger.log(log_level, log_msg)
        return deg
