"""RAG Engine exception hierarchy.

Every error carries stage, request_id, degradations, and usage (including failed attempts)
per §13.3, §13.7, RAG-OPS-1.
"""
from __future__ import annotations

from typing import Any, Literal


class RAGEngineError(Exception):
    """Base error for all RAG engine failures."""

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        request_id: str | None = None,
        degradations: list[Any] | None = None,
        usage: list[Any] | None = None,
        usage_total: Any | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.request_id = request_id
        self.degradations = list(degradations or [])
        self.usage = list(usage or [])
        self.usage_total = usage_total
        self.cause = cause


class ProfileConfigError(RAGEngineError):
    """Raised at startup validation when profile configuration is invalid or missing bindings."""

    def __init__(
        self,
        message: str,
        problems: list[str] | None = None,
        missing: list[str] | None = None,
        unknown: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.problems = list(problems or [])
        self.missing = list(missing or [])
        self.unknown = list(unknown or [])


class StoreCapabilityError(RAGEngineError):
    """Raised when a store lacks capabilities declared or required by the profile."""
    pass


class MissingRequiredScopeError(RAGEngineError):
    """Raised when required filter keys (e.g. tenant_id) are missing from effective filters."""
    pass


class UnsupportedFilterError(RAGEngineError):
    """Raised when an unknown custom filter key or wrong value type is passed to a store."""
    pass


class EmbeddingProviderError(RAGEngineError):
    """Raised when embedding generation fails or produces all-zero vectors."""
    pass


class EmbeddingSignatureMismatch(RAGEngineError):
    """Raised when store embedding space does not match engine embedder model/dims."""
    pass


class IntentParseError(RAGEngineError):
    """Raised when model query intent output cannot be parsed after retries and fallback models."""
    pass


class RetrievalError(RAGEngineError):
    """Raised when persistent store retrieval operations fail."""
    pass


class GenerationError(RAGEngineError):
    """Raised when answer generation fails, is truncated, blocked by safety, or all models fail.

    Carries retrieval result so caller can still access retrieved sources (RAG-RES-6).
    """

    def __init__(
        self,
        message: str,
        reason: Literal["all_models_failed", "truncated", "blocked", "other"] = "other",
        retrieval: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.reason = reason
        self.retrieval = retrieval


class RAGStageError(RAGEngineError):
    """Raised when a specific pipeline stage (e.g. summarizer, reranker) fails."""

    def __init__(self, stage: str, message: str | None = None, **kwargs: Any):
        msg = message or f"Stage '{stage}' failed"
        super().__init__(msg, stage=stage, **kwargs)


class BudgetExceeded(RAGEngineError):
    """Raised when a BudgetGuard halts execution because token or cost limits are exceeded."""
    pass


class DeadlineExceeded(RAGEngineError):
    """Raised when a required pipeline stage cannot finish within the remaining deadline."""
    pass
