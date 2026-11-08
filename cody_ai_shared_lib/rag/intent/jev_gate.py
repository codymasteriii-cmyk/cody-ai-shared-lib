"""Jev System One fast gating router (§7.4, AC-RAG-14).

Optionally enabled to fast-path high-confidence local lookup and overview queries
without an LLM call. Emits DG-03 degradation on error or timeout and falls back to LLM.
"""
from __future__ import annotations

import logging
import os

from ...pricing import compute_cost
from ...tokens import estimate_tokens
from ..degradation import DegradationTracker
from ..errors import ProfileConfigError, RAGStageError
from ..gateway import ModelGateway
from ..models import QueryIntent, StageUsage
from ..usage import BudgetGuard, UsageTracker, invoke_budget_guard

logger = logging.getLogger("shared-rag.router")


class JevGateRouter:
    """Fast pre-router via TypeSafe AI Jev System One."""

    def __init__(
        self,
        gateway: ModelGateway,
        api_key: str | None = None,
        timeout: float = 2.0,
    ):
        key = api_key or os.getenv("TYPESAFE_API_KEY") or os.getenv("VERCEL_API_KEY")
        if not key:
            raise ProfileConfigError(
                "Router configured with enable_jev=True but TYPESAFE_API_KEY (or VERCEL_API_KEY) is not set.",
                missing=["TYPESAFE_API_KEY"],
                problems=["Router configured with enable_jev=True without TYPESAFE_API_KEY"],
            )
        self.gateway = gateway
        self.timeout = timeout

    def route(
        self,
        query: str,
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
        budget_guard: BudgetGuard | None = None,
        fallbacks: dict[str, str] | None = None,
    ) -> QueryIntent | None:
        """Attempt fast path via Jev. Returns QueryIntent if fast path taken, else None."""
        if not self.gateway.jev_client:
            return None

        if budget_guard is not None:
            est_in = estimate_tokens(query) or 20
            est_cost = compute_cost("jev", input_tokens=est_in)
            ctx = usage_tracker.call_context("intent_jev", "jev", True, est_in, est_cost)
            if invoke_budget_guard(budget_guard, ctx) == "skip":
                degradation_tracker.record(
                    dg_id="DG-11",
                    stage="intent_jev",
                    reason="BudgetGuard returned skip",
                    impact="Jev gate skipped by BudgetGuard",
                )
                return None

        try:
            from ...jev import Choice, Noul

            questions = {
                "scope": Choice(criteria={
                    "local_lookup": "Questions about specific details or facts",
                    "document_overview": "Broad summary or overview of the entire document",
                }),
                "needs_query_rewrite": Noul(),
                "has_time_expression": Noul(),
                "beyond_boundary_probe": Noul(),
            }

            jev_res = self.gateway.jev_client.system_one(
                state=query,
                questions=questions,
            )

            usage_tracker.record(
                StageUsage(
                    stage="intent_jev",
                    model=jev_res.model,
                    input_tokens=jev_res.tokens_in,
                    output_tokens=jev_res.tokens_out,
                    calls=1,
                    latency_ms=jev_res.ms,
                    cost_usd=jev_res.cost_usd,
                    price_table_version=jev_res.price_table_version,
                )
            )

            # Evaluate fast path criteria (AC-RAG-14)
            scope_ans = jev_res.answers.get("scope")
            rewrite_ans = jev_res.answers.get("needs_query_rewrite")
            time_ans = jev_res.answers.get("has_time_expression")
            probe_ans = jev_res.answers.get("beyond_boundary_probe")

            scope_conf = getattr(scope_ans, "confidence", 0.0) if scope_ans else 0.0
            scope_val = getattr(scope_ans, "value", "local_lookup") if scope_ans else "local_lookup"
            rewrite_prob = getattr(rewrite_ans, "probability", 1.0) if rewrite_ans else 1.0
            time_prob = getattr(time_ans, "probability", 1.0) if time_ans else 1.0
            probe_prob = getattr(probe_ans, "probability", 0.0) if probe_ans else 0.0

            if scope_conf >= 0.85 and rewrite_prob < 0.30 and time_prob < 0.30:
                return QueryIntent(
                    route="rag",
                    scope=scope_val,
                    search_query=query,
                    beyond_boundary_risk=probe_prob,
                    router_source="jev",
                )

            # Fast path condition not satisfied; proceed to LLM rewriter
            return None

        except Exception as exc:
            if (fallbacks or {}).get("jev") == "raise":
                raise RAGStageError(stage="intent_jev", message=f"Jev fast gate failed: {exc}", cause=exc) from exc

            # DG-03: Jev error or timeout continues on LLM path
            degradation_tracker.record(
                dg_id="DG-03",
                stage="intent",
                reason=f"Jev error or timeout: {exc}",
                fallback_used="llm_path",
                impact="Jev fast path skipped; continuing on LLM path",
                error_type=type(exc).__name__,
            )
            return None
