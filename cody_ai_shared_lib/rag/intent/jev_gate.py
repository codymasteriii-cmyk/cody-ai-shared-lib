"""Jev System One fast gating router (§7.4, AC-RAG-14).

Optionally enabled to fast-path high-confidence local lookup and overview queries
without an LLM call. Emits DG-03 degradation on error or timeout and falls back to LLM.

Jev answers a few yes/no and multiple-choice questions about the query, each with a confidence. The fast path
is taken only when Jev is sure the question is a plain lookup or overview (scope confidence >= 0.85) AND sure it
needs no rewriting (< 30%) AND has no time expression (< 30%). Anything less sure returns None, and the caller
goes on to the intent model: the gate may skip the model, but it must never guess in its place.
"""
from __future__ import annotations

import logging
import os
import time

from ...pricing import compute_cost
from ...tokens import estimate_tokens
from ..degradation import DegradationTracker
from ..errors import ProfileConfigError, RAGStageError
from ..gateway import ModelGateway
from ..models import QueryIntent, StageUsage
from ..usage import BudgetGuard, UsageTracker, invoke_budget_guard

logger = logging.getLogger("shared-rag.router")


class JevGateRouter:
    """Fast pre-router via TypeSafe AI Jev System One.

    Optional and off by default (`router.enable_jev`). It only ever returns an intent when it is confident, and
    a failure is a degradation (DG-03), never an error, unless the profile says `fallbacks["jev"] = "raise"`.
    """

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
        """Attempt fast path via Jev. Returns QueryIntent if fast path taken, else None.

        Order: check the budget guard -> ask Jev -> record the spend -> test the confidence thresholds.
        """
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

        # `recorded` tells the error path whether the spend was already booked, so it is never booked twice or lost
        recorded = False
        started = time.perf_counter()
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

            jev_res = self.gateway.jev_client.ask(
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
            recorded = True

            # Evaluate fast path criteria (AC-RAG-14). A missing answer counts as "not sure", so it fails the test.
            scope_ans = jev_res.answers.get("scope")
            rewrite_ans = jev_res.answers.get("needs_query_rewrite")
            time_ans = jev_res.answers.get("has_time_expression")
            probe_ans = jev_res.answers.get("beyond_boundary_probe")

            # Every answer is a JevAnswer (jev.py): a choice's `.probability` is Jev's confidence in `.value`; a noul's is P(yes)
            scope_conf = scope_ans.probability if scope_ans else 0.0
            scope_val = scope_ans.value if scope_ans else "local_lookup"
            rewrite_prob = rewrite_ans.probability if rewrite_ans else 1.0
            time_prob = time_ans.probability if time_ans else 1.0
            probe_prob = probe_ans.probability if probe_ans else 0.0

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
            if not recorded:
                # A failed attempt may still be billed; record it as unknown so the spend is not lost (RAG-OPS-1)
                usage_tracker.record(
                    StageUsage(
                        stage="intent_jev",
                        model="jev",
                        input_tokens=estimate_tokens(query),
                        calls=1,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        estimated=True,
                        billing="unknown",
                        outcome="error",
                    )
                )
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
