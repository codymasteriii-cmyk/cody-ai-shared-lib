"""ModelGateway coordinating model execution, fallbacks, budgets, and usage (§10.7, §13.1, §13.7).

Handles:
  - DG-01/DG-02 model fallback chains (AC-RAG-44)
  - DG-09 output truncation retry (AC-RAG-44)
  - Safety filter blocked output detection (AC-RAG-44)
  - DG-11 BudgetGuard pre-call limits and skip handling
  - Full usage and spend tracking on success and failure paths (RAG-OPS-1, AC-RAG-70)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from ..embedding import EmbeddingResult, generate_embedding_with_usage
from ..jev import JevClient, JevResult
from ..llm import LLMClient, LLMResult, LLMUsage, ModelCallError
from ..pricing import compute_cost
from ..rerank import RerankClient, RerankResult
from .degradation import DegradationTracker
from .errors import GenerationError
from .models import StageUsage
from .usage import BudgetGuard, CallContext, UsageTracker

logger = logging.getLogger("rag-gateway")


class ModelGateway:
    """Gateway orchestrating all LLM, embedding, Jev, and reranking calls."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        jev_client: JevClient | None = None,
        rerank_client: RerankClient | None = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.jev_client = jev_client
        self.rerank_client = rerank_client

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        models: list[str],
        stage: str,
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
        budget_guard: BudgetGuard | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_schema: type | None = None,
        response_mime_type: str | None = None,
        request_id: str = "",
        tenant_or_user_id: str | None = None,
    ) -> LLMResult:
        """Execute text generation through ordered model fallback list (DG-02, AC-RAG-44)."""
        last_exc: Exception | None = None

        for model_idx, model in enumerate(models):
            # Pre-call budget check (DG-11)
            if budget_guard is not None:
                spent = usage_tracker.total()
                est_cost = compute_cost(model, input_tokens=1000, output_tokens=max_tokens or 500)
                ctx = CallContext(
                    stage=stage,
                    request_id=request_id,
                    tenant_or_user_id=tenant_or_user_id,
                    est_input_tokens=1000,
                    est_cost_usd=est_cost,
                    request_spent=spent,
                )
                decision = budget_guard.before_call(ctx)
                if decision == "skip":
                    degradation_tracker.record(
                        dg_id="DG-11",
                        stage=stage,
                        reason="BudgetGuard returned skip",
                        impact="Generation stage skipped",
                    )
                    return LLMResult(
                        text="",
                        usage=LLMUsage(input_tokens=0, output_tokens=0),
                        finish_reason="stop",
                    )

            t0 = time.perf_counter()
            try:
                result = self.llm_client.generate_with_usage(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_schema=response_schema,
                    response_mime_type=response_mime_type,
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0

                stage_u = StageUsage(
                    stage=stage,
                    model=model,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    cached_tokens=result.usage.cached_tokens,
                    latency_ms=lat_ms,
                    cost_usd=result.usage.cost_usd,
                    price_table_version=result.usage.price_table_version,
                    estimated=result.usage.estimated,
                    billing="billed",
                    outcome="ok",
                )
                usage_tracker.record(stage_u)

                # Check safety block
                if result.finish_reason == "safety":
                    raise GenerationError(
                        f"Generation blocked by safety filters on model '{model}'",
                        reason="blocked",
                        stage=stage,
                    )

                # Check truncation (DG-09)
                if result.finish_reason == "length":
                    degradation_tracker.record(
                        dg_id="DG-09",
                        stage=stage,
                        reason="Output truncated (finish_reason='length')",
                        impact="Retrying with larger output allowance or next model",
                    )
                    # Retry once with larger token limit if available
                    retry_tokens = (max_tokens or 2048) * 2
                    try:
                        t0_retry = time.perf_counter()
                        retry_res = self.llm_client.generate_with_usage(
                            system_prompt=system_prompt,
                            user_message=user_message,
                            model=model,
                            max_tokens=retry_tokens,
                            temperature=temperature,
                            response_schema=response_schema,
                            response_mime_type=response_mime_type,
                        )
                        lat_retry = (time.perf_counter() - t0_retry) * 1000.0
                        usage_tracker.record(
                            StageUsage(
                                stage=f"{stage}_truncation_retry",
                                model=model,
                                input_tokens=retry_res.usage.input_tokens,
                                output_tokens=retry_res.usage.output_tokens,
                                cached_tokens=retry_res.usage.cached_tokens,
                                latency_ms=lat_retry,
                                cost_usd=retry_res.usage.cost_usd,
                                price_table_version=retry_res.usage.price_table_version,
                            )
                        )
                        if retry_res.finish_reason != "length":
                            return retry_res
                    except Exception:
                        pass
                    raise GenerationError(
                        f"Output from model '{model}' remained truncated",
                        reason="truncated",
                        stage=stage,
                    )

                return result

            except GenerationError:
                raise
            except ModelCallError as mce:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                if mce.usage:
                    usage_tracker.record(
                        StageUsage(
                            stage=stage,
                            model=model,
                            input_tokens=mce.usage.input_tokens,
                            output_tokens=mce.usage.output_tokens,
                            cached_tokens=mce.usage.cached_tokens,
                            latency_ms=lat_ms,
                            cost_usd=mce.usage.cost_usd,
                            price_table_version=mce.usage.price_table_version,
                            billing="billed",
                            outcome="error",
                        )
                    )
                last_exc = mce
            except Exception as exc:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                usage_tracker.record(
                    StageUsage(
                        stage=stage,
                        model=model,
                        latency_ms=lat_ms,
                        billing="unknown",
                        outcome="error",
                    )
                )
                last_exc = exc

            # If there's another model in the list, record DG-02 fallback and continue
            if model_idx < len(models) - 1:
                next_m = models[model_idx + 1]
                degradation_tracker.record(
                    dg_id="DG-02",
                    stage=stage,
                    reason=f"Model '{model}' failed: {last_exc}",
                    fallback_used=f"model:{next_m}",
                    impact=f"Switched to fallback model {next_m}",
                    error_type=type(last_exc).__name__ if last_exc else None,
                )

        raise GenerationError(
            f"All models failed for stage '{stage}': {last_exc}",
            reason="all_models_failed",
            stage=stage,
            cause=last_exc,
        )

    def embed(
        self,
        texts: list[str] | str,
        model: str,
        stage: str,
        usage_tracker: UsageTracker,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> EmbeddingResult:
        """Generate vector embeddings and track token usage/cost."""
        t0 = time.perf_counter()
        result = generate_embedding_with_usage(texts, task_type=task_type)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        usage_tracker.record(
            StageUsage(
                stage=stage,
                model=model,
                input_tokens=result.usage.input_tokens,
                output_tokens=0,
                latency_ms=lat_ms,
                cost_usd=result.usage.cost_usd,
                price_table_version=result.usage.price_table_version,
                estimated=result.usage.estimated,
            )
        )
        return result

    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        stage: str,
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
        top_n: int | None = None,
    ) -> RerankResult | None:
        """Call reranker with DG-06 degradation on error or timeout."""
        if not self.rerank_client:
            return None

        t0 = time.perf_counter()
        try:
            result = self.rerank_client.rerank(query, documents, model=model, top_n=top_n)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            usage_tracker.record(
                StageUsage(
                    stage=stage,
                    model=model,
                    input_tokens=result.usage.total_tokens,
                    output_tokens=0,
                    latency_ms=lat_ms,
                    cost_usd=result.usage.cost_usd,
                    price_table_version=result.usage.price_table_version,
                )
            )
            return result
        except Exception as exc:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            usage_tracker.record(
                StageUsage(
                    stage=stage,
                    model=model,
                    latency_ms=lat_ms,
                    billing="unknown",
                    outcome="error",
                )
            )
            degradation_tracker.record(
                dg_id="DG-06",
                stage=stage,
                reason=f"Reranker failed: {exc}",
                fallback_used="keep_incoming_order",
                impact="Results were not reranked; kept original order",
                error_type=type(exc).__name__,
            )
            return None
