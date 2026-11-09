"""Jina Reranker HTTP client with usage and cost tracking.

Calls the Jina Rerank API (https://api.jina.ai/v1/rerank).
Tracks token counts and computes USD cost using cody_ai_shared_lib.pricing ($0.05 / 1M tokens).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from .pricing import PRICE_TABLE_VERSION, compute_cost

__all__ = ["RerankClient", "RerankResult", "RerankUsage", "RerankItem"]

logger = logging.getLogger("shared-rerank")


@dataclass
class RerankUsage:
    """Usage accounting for a rerank call."""
    total_tokens: int
    cost_usd: float | None = None
    price_table_version: str | None = None
    estimated: bool = False


@dataclass
class RerankItem:
    """Individual ranked document item."""
    index: int
    relevance_score: float
    document: str | None = None


@dataclass
class RerankResult:
    """Result of rerank() containing ranked items and token usage/cost."""
    results: list[RerankItem]
    usage: RerankUsage


class RerankClient:
    """Client for Jina Cross-Encoder Rerank API with retries and cost tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = "https://api.jina.ai/v1/rerank",
        timeout_seconds: float = 4.0,
        retry_max: int = 2,
        retry_base_wait_secs: float = 1.0,
    ):
        self._api_key = api_key
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.retry_max = retry_max
        self.retry_base_wait_secs = retry_base_wait_secs

    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str = "jina-reranker-v2-base-multilingual",
        top_n: int | None = None,
    ) -> RerankResult:
        """Rerank a list of documents against a query.

        Args:
            query: The search query.
            documents: List of document strings to rerank.
            model: Reranker model ID (default: "jina-reranker-v2-base-multilingual").
            top_n: Optional number of top results to return.

        Returns:
            RerankResult with scored items and token/cost usage.

        Raises:
            RuntimeError: JINA_API_KEY missing or all retry attempts failed.
        """
        if not documents:
            return RerankResult(
                results=[],
                usage=RerankUsage(
                    total_tokens=0,
                    cost_usd=0.0,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                ),
            )

        api_key = self._api_key or os.getenv("JINA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "JINA_API_KEY is not set. Rerank calls cannot be made without it."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        last_exc = None
        for attempt in range(self.retry_max):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    resp.raise_for_status()
                resp.raise_for_status()
                data = resp.json()

                usage_data = data.get("usage", {})
                total_tokens = usage_data.get("total_tokens") or usage_data.get("prompt_tokens") or 0
                cost_usd = compute_cost("jina-reranker", input_tokens=total_tokens)
                usage = RerankUsage(
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                )

                raw_results = data.get("results", [])
                items: list[RerankItem] = []
                for r in raw_results:
                    doc = r.get("document", {})
                    doc_text = doc.get("text") if isinstance(doc, dict) else (doc if isinstance(doc, str) else None)
                    items.append(
                        RerankItem(
                            index=int(r.get("index", 0)),
                            relevance_score=float(r.get("relevance_score", 0.0)),
                            document=doc_text,
                        )
                    )
                return RerankResult(results=items, usage=usage)

            except Exception as exc:
                last_exc = exc
                if attempt < self.retry_max - 1:
                    wait = self.retry_base_wait_secs * (2 ** attempt)
                    logger.warning(
                        f"[Rerank] {model} retryable error (attempt {attempt + 1}/{self.retry_max}): {exc}. "
                        f"Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"[Rerank] {model} failed after {self.retry_max} attempts: {last_exc}"
                    ) from last_exc
