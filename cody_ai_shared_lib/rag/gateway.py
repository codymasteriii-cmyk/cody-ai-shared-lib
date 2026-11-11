"""ModelGateway coordinating model execution, fallbacks, budgets, caching, and usage (§10.7, §13.1, §13.5, §13.7).

Handles:
  - DG-01/DG-02 model fallback chains (AC-RAG-44)
  - DG-09 output truncation retry (AC-RAG-44)
  - Safety filter blocked output detection (AC-RAG-44)
  - DG-11 BudgetGuard pre-call limits and skip handling
  - DG-12 Query embedding cache read/write fault tolerance (§13.5)
  - Full usage and spend tracking on success and failure paths (RAG-OPS-1, AC-RAG-70)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from ..embedding import EmbeddingResult, EmbeddingUsage, generate_embedding_with_usage
from ..jev import JevClient
from ..llm import LLMClient, LLMResult, LLMUsage, ModelCallError
from ..pricing import PRICE_TABLE_VERSION, compute_cost, estimate_cost
from ..tokens import estimate_tokens
from ..rerank import RerankClient, RerankResult
from .cache import Cache, deserialize_vector, make_embedding_cache_key, serialize_vector
from .degradation import DegradationTracker
from .errors import EmbeddingProviderError, GenerationError, RAGEngineError, RAGStageError, safe_message
from .models import StageUsage
from .text.analyzer import CjkAwareAnalyzer, TextAnalyzer
from .usage import BudgetGuard, UsageTracker, invoke_budget_guard

logger = logging.getLogger("shared-rag.generation")

_OPTIONAL_STAGES = frozenset({"summarizer", "summarizer_unit"})

# finish_reason of the empty result returned when a BudgetGuard skips an optional stage
FINISH_SKIPPED = "skipped"


class ModelGateway:
    """Gateway orchestrating all LLM, embedding, Jev, and reranking calls."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        jev_client: JevClient | None = None,
        rerank_client: RerankClient | None = None,
        cache: Cache | None = None,
        analyzer: TextAnalyzer | None = None,
        embedding_client: Any = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.jev_client = jev_client or JevClient()
        self.rerank_client = rerank_client or (RerankClient() if os.environ.get("JINA_API_KEY") else None)
        self.cache = cache
        self.analyzer = analyzer or CjkAwareAnalyzer()
        self.embedding_client = embedding_client

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
        optional: bool | None = None,
        cache_breakpoints: bool | list[int] | None = None,
    ) -> LLMResult:
        """Execute text generation through ordered model fallback list (DG-02, AC-RAG-44)."""
        last_exc: Exception | None = None
        is_opt = optional if optional is not None else stage in _OPTIONAL_STAGES
        est_in = estimate_tokens(system_prompt) + estimate_tokens(user_message) or 1000

        for model_idx, model in enumerate(models):
            # Pre-call budget check (DG-11)
            if budget_guard is not None:
                est_cost = estimate_cost(model, est_input_tokens=est_in, est_output_tokens=max_tokens or 500)
                ctx = usage_tracker.call_context(stage, model, is_opt, est_in, est_cost)
                decision = invoke_budget_guard(budget_guard, ctx)
                if decision == "skip":
                    degradation_tracker.record(
                        dg_id="DG-11",
                        stage=stage,
                        reason="BudgetGuard returned skip",
                        impact=f"Stage '{stage}' skipped by BudgetGuard",
                    )
                    return LLMResult(
                        text="",
                        usage=LLMUsage(input_tokens=0, output_tokens=0),
                        finish_reason=FINISH_SKIPPED,
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
                    cache_breakpoints=cache_breakpoints,
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0

                stage_u = StageUsage(
                    stage=stage,
                    model=model,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    cached_tokens=result.usage.cached_tokens,
                    calls=1,
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
                    retry_tokens = (max_tokens or result.usage.output_tokens or 2048) * 2
                    if budget_guard is not None:
                        est_cost = compute_cost(model, input_tokens=est_in, output_tokens=retry_tokens)
                        invoke_budget_guard(
                            budget_guard,
                            usage_tracker.call_context(f"{stage}_truncation_retry", model, False, est_in, est_cost),
                        )
                    t0_retry = time.perf_counter()
                    try:
                        retry_res = self.llm_client.generate_with_usage(
                            system_prompt=system_prompt,
                            user_message=user_message,
                            model=model,
                            max_tokens=retry_tokens,
                            temperature=temperature,
                            response_schema=response_schema,
                            response_mime_type=response_mime_type,
                            cache_breakpoints=cache_breakpoints,
                        )
                    except Exception as retry_exc:
                        usage_tracker.record(
                            StageUsage(
                                stage=f"{stage}_truncation_retry",
                                model=model,
                                input_tokens=est_in,
                                calls=1,
                                latency_ms=(time.perf_counter() - t0_retry) * 1000.0,
                                estimated=True,
                                billing="unknown",
                                outcome="error",
                            )
                        )
                        raise GenerationError(
                            f"Truncation retry on model '{model}' failed",
                            reason="truncated",
                            stage=stage,
                            cause=retry_exc,
                        ) from retry_exc
                    usage_tracker.record(
                        StageUsage(
                            stage=f"{stage}_truncation_retry",
                            model=model,
                            input_tokens=retry_res.usage.input_tokens,
                            output_tokens=retry_res.usage.output_tokens,
                            cached_tokens=retry_res.usage.cached_tokens,
                            calls=1,
                            latency_ms=(time.perf_counter() - t0_retry) * 1000.0,
                            cost_usd=retry_res.usage.cost_usd,
                            price_table_version=retry_res.usage.price_table_version,
                        )
                    )
                    if retry_res.finish_reason == "safety":
                        raise GenerationError(
                            f"Generation blocked by safety filters on model '{model}'",
                            reason="blocked",
                            stage=stage,
                        )
                    if retry_res.finish_reason != "length":
                        return retry_res
                    raise GenerationError(
                        f"Output from model '{model}' remained truncated",
                        reason="truncated",
                        stage=stage,
                    )

                return result

            except RAGEngineError:
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
                            calls=1,
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
                        input_tokens=est_in,
                        calls=1,
                        latency_ms=lat_ms,
                        estimated=True,
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
            f"All models failed for stage '{stage}': {safe_message(last_exc)}",
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
        budget_guard: BudgetGuard | None = None,
        degradation_tracker: DegradationTracker | None = None,
        use_cache: bool = True,
        cache: Cache | None = None,
        fallbacks: dict[str, str] | None = None,
        optional: bool = False,
        cache_ttl_seconds: int | None = None,
    ) -> EmbeddingResult:
        """Generate embeddings with optional query-embedding cache and budget check (§13.1, §13.5).

        `vectors` is always a list of vectors, one per input, including for a single string.
        """
        is_batch = isinstance(texts, list)
        text_list = texts if is_batch else [texts]
        est_in = sum(estimate_tokens(t) for t in text_list if t)
        active_cache = cache if cache is not None else self.cache

        # Query embedding cache check for single query (§13.5)
        cache_key: str | None = None
        if (
            use_cache
            and active_cache is not None
            and task_type == "RETRIEVAL_QUERY"
            and len(text_list) == 1
            and text_list[0]
        ):
            norm_q = self.analyzer.normalize(text_list[0])
            cache_key = make_embedding_cache_key(model, task_type, norm_q)
            try:
                t_c0 = time.perf_counter()
                cached_bytes = active_cache.get(cache_key)
                if cached_bytes is not None:
                    vec = deserialize_vector(cached_bytes)
                    lat_c = (time.perf_counter() - t_c0) * 1000.0
                    usage_tracker.record(
                        StageUsage(
                            stage=stage,
                            model=model,
                            input_tokens=0,
                            output_tokens=0,
                            cached_tokens=est_in,
                            calls=0,
                            latency_ms=lat_c,
                            cost_usd=0.0,
                            price_table_version=PRICE_TABLE_VERSION,
                            estimated=True,
                        )
                    )
                    return EmbeddingResult(
                        vectors=[vec],
                        usage=EmbeddingUsage(
                            input_tokens=0,
                            cost_usd=0.0,
                            price_table_version=PRICE_TABLE_VERSION,
                            estimated=True,
                        ),
                    )
            except Exception as exc:
                if (fallbacks or {}).get("cache") == "raise":
                    raise RAGStageError(stage="cache", message=f"Embedding cache read failed: {exc}", cause=exc) from exc
                logger.error(f"Embedding cache read failed: {exc} [DG-12]", exc_info=True)
                if degradation_tracker is not None:
                    degradation_tracker.record(
                        dg_id="DG-12",
                        stage="cache",
                        reason=f"Cache read error: {exc}",
                        fallback_used="live_embedding",
                        impact="Continued without embedding cache",
                        error_type=type(exc).__name__,
                    )

        if budget_guard is not None:
            est_cost = compute_cost(model, input_tokens=est_in)
            ctx = usage_tracker.call_context(stage, model, optional, est_in, est_cost)
            decision = invoke_budget_guard(budget_guard, ctx)
            if decision == "skip":
                if degradation_tracker is not None:
                    degradation_tracker.record(
                        dg_id="DG-11",
                        stage=stage,
                        reason="BudgetGuard returned skip",
                        impact=f"Embedding stage '{stage}' skipped",
                    )
                return EmbeddingResult(
                    vectors=[] if is_batch else [],
                    usage=EmbeddingUsage(input_tokens=0, cost_usd=0.0, price_table_version=PRICE_TABLE_VERSION),
                )

        t0 = time.perf_counter()
        try:
            if self.embedding_client is not None and hasattr(self.embedding_client, "embed_with_usage"):
                result = self.embedding_client.embed_with_usage(texts, task_type=task_type)
            else:
                result = generate_embedding_with_usage(texts, task_type=task_type)
        except Exception as exc:
            usage_tracker.record(
                StageUsage(
                    stage=stage,
                    model=model,
                    input_tokens=est_in,
                    calls=1,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    estimated=True,
                    billing="unknown",
                    outcome="error",
                )
            )
            raise EmbeddingProviderError(
                f"Embedding call failed: {type(exc).__name__}", stage=stage, cause=exc
            ) from exc
        lat_ms = (time.perf_counter() - t0) * 1000.0
        if not is_batch and result.vectors and isinstance(result.vectors[0], (int, float)):
            result = EmbeddingResult(vectors=[result.vectors], usage=result.usage)

        usage_tracker.record(
            StageUsage(
                stage=stage,
                model=model,
                input_tokens=result.usage.input_tokens,
                output_tokens=0,
                calls=1,
                latency_ms=lat_ms,
                cost_usd=result.usage.cost_usd,
                price_table_version=result.usage.price_table_version,
                estimated=result.usage.estimated,
            )
        )

        # Populate query embedding cache if valid non-zero vector
        if cache_key is not None and active_cache is not None:
            vec_to_cache = result.vectors[0] if result.vectors else None
            if vec_to_cache and any(x != 0.0 for x in vec_to_cache):
                try:
                    active_cache.set(cache_key, serialize_vector(vec_to_cache), ttl_seconds=cache_ttl_seconds)
                except Exception as exc:
                    if (fallbacks or {}).get("cache") == "raise":
                        raise RAGStageError(stage="cache", message=f"Embedding cache write failed: {exc}", cause=exc) from exc
                    logger.error(f"Embedding cache write failed: {exc} [DG-12]", exc_info=True)
                    if degradation_tracker is not None:
                        degradation_tracker.record(
                            dg_id="DG-12",
                            stage="cache",
                            reason=f"Cache write error: {exc}",
                            fallback_used="unbuffered",
                            impact="Continued without caching query embedding",
                            error_type=type(exc).__name__,
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
        budget_guard: BudgetGuard | None = None,
        fallbacks: dict[str, str] | None = None,
    ) -> RerankResult | None:
        """Call reranker with DG-06 degradation on error or timeout (unless fallbacks['rerank'] == 'raise')."""
        if not self.rerank_client:
            if (fallbacks or {}).get("rerank") == "raise":
                raise RAGStageError(stage=stage, message="Reranker is not configured")
            degradation_tracker.record(
                dg_id="DG-06",
                stage=stage,
                reason="Reranker is not configured",
                fallback_used="keep_incoming_order",
                impact="Results were not reranked; kept original order",
            )
            return None

        if budget_guard is not None:
            est_in = estimate_tokens(query) + sum(estimate_tokens(d) for d in documents)
            est_cost = compute_cost(model, input_tokens=est_in)
            ctx = usage_tracker.call_context(stage, model, True, est_in, est_cost)
            decision = invoke_budget_guard(budget_guard, ctx)
            if decision == "skip":
                degradation_tracker.record(
                    dg_id="DG-11",
                    stage=stage,
                    reason="BudgetGuard returned skip",
                    impact="Rerank stage skipped by BudgetGuard",
                )
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
                    calls=1,
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
                    calls=1,
                    latency_ms=lat_ms,
                    billing="unknown",
                    outcome="error",
                )
            )
            if (fallbacks or {}).get("rerank") == "raise":
                raise RAGStageError(stage=stage, message=f"Reranker failed: {exc}", cause=exc) from exc

            degradation_tracker.record(
                dg_id="DG-06",
                stage=stage,
                reason=f"Reranker failed: {exc}",
                fallback_used="keep_incoming_order",
                impact="Results were not reranked; kept original order",
                error_type=type(exc).__name__,
            )
            return None

    def stream(
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
        cache_breakpoints: list[int] | None = None,
        optional: bool | None = None,
    ) -> Any:
        """Stream generation through ordered model fallback list (§10.8, AC-RAG-44, AC-RAG-70)."""
        last_exc: Exception | None = None
        is_opt = optional if optional is not None else stage in _OPTIONAL_STAGES
        est_in = estimate_tokens(system_prompt) + estimate_tokens(user_message) or 1000

        for model_idx, model in enumerate(models):
            if budget_guard is not None:
                est_cost = estimate_cost(model, est_input_tokens=est_in, est_output_tokens=max_tokens or 500)
                ctx = usage_tracker.call_context(stage, model, is_opt, est_in, est_cost)
                decision = invoke_budget_guard(budget_guard, ctx)
                if decision == "skip":
                    degradation_tracker.record(
                        dg_id="DG-11",
                        stage=stage,
                        reason="BudgetGuard returned skip",
                        impact="Generation streaming stage skipped",
                    )
                    return

            t0 = time.perf_counter()
            received: list[str] = []
            usage_recorded = False

            def record_unfinished(outcome: str, billing: str) -> None:
                # Provider usage only arrives on the last chunk, so estimate from what was received
                output = estimate_tokens("".join(received))
                usage_tracker.record(
                    StageUsage(
                        stage=stage,
                        model=model,
                        input_tokens=est_in,
                        output_tokens=output,
                        calls=1,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        cost_usd=compute_cost(model, input_tokens=est_in, output_tokens=output)
                        if outcome == "cancelled" else None,
                        estimated=True,
                        billing=billing,
                        outcome=outcome,
                    )
                )

            try:
                chunk_iter = self.llm_client.stream(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    cache_breakpoints=cache_breakpoints,
                )
                for chunk in chunk_iter:
                    if chunk.delta:
                        received.append(chunk.delta)
                    if chunk.usage is not None:
                        usage_recorded = True
                        usage_tracker.record(
                            StageUsage(
                                stage=stage,
                                model=model,
                                input_tokens=chunk.usage.input_tokens,
                                output_tokens=chunk.usage.output_tokens,
                                cached_tokens=chunk.usage.cached_tokens,
                                calls=1,
                                latency_ms=(time.perf_counter() - t0) * 1000.0,
                                cost_usd=chunk.usage.cost_usd,
                                price_table_version=chunk.usage.price_table_version,
                                estimated=chunk.usage.estimated,
                                billing="billed",
                                outcome="ok",
                            )
                        )
                    if chunk.finish_reason == "safety":
                        raise GenerationError(
                            f"Generation blocked by safety filters on model '{model}'",
                            reason="blocked",
                            stage=stage,
                        )
                    yield chunk
                    if chunk.finish_reason == "length":
                        # Text was already sent to the caller, so a retry is no longer possible (§13.3)
                        raise GenerationError(
                            f"Output from model '{model}' was truncated while streaming",
                            reason="truncated",
                            stage=stage,
                        )
                return
            except GeneratorExit:
                if not usage_recorded:
                    record_unfinished("cancelled", "billed")
                raise
            except GenerationError:
                raise
            except Exception as exc:
                last_exc = exc
                if not usage_recorded:
                    record_unfinished("error", "unknown")
                if received:
                    raise GenerationError(
                        f"Stream from model '{model}' broke after it started",
                        reason="other",
                        stage=stage,
                        cause=exc,
                    ) from exc
                if model_idx < len(models) - 1:
                    degradation_tracker.record(
                        dg_id="DG-02",
                        stage=stage,
                        reason=f"Model '{model}' failed before streaming: {type(exc).__name__}",
                        fallback_used=f"model:{models[model_idx + 1]}",
                        impact=f"Switched from '{model}' to a backup model",
                        error_type=type(exc).__name__,
                    )

        raise GenerationError(
            f"All models failed for streaming stage '{stage}': {last_exc}",
            reason="all_models_failed",
            stage=stage,
            cause=last_exc,
        )
