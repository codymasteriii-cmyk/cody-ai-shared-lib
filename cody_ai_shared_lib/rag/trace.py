"""RAG Trace and Observer architecture (§13.2, DG-13).

Light trace is always on and holds no raw question or document text.
Verbose trace captures per-stage snapshots and raw scores for evaluation.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from .models import Degradation, StageUsage

logger = logging.getLogger("rag-trace")


class StageRecord(BaseModel):
    """Snapshot record of one stage execution."""
    stage: str
    latency_ms: float = 0.0
    usage: StageUsage | None = None
    degradations: list[Degradation] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class RAGTrace(BaseModel):
    """Observable trace payload returned on every RAGResponse (§13.2)."""
    request_id: str
    profile_version: str
    stages: list[StageRecord] = Field(default_factory=list)
    level: Literal["light", "verbose"] = "light"
    total_latency_ms: float = 0.0
    state_size_bytes: int = 0
    trace_schema_version: str = "1.0.0"


@runtime_checkable
class RAGObserver(Protocol):
    """Observer receiving stage, usage, and degradation lifecycle events (§13.2)."""

    def on_stage(self, record: StageRecord) -> None:
        ...

    def on_usage(self, usage: StageUsage) -> None:
        ...

    def on_degrade(self, degradation: Degradation) -> None:
        ...

    def on_complete(self, trace: RAGTrace, outcome: Any) -> None:
        ...


class ObserverDispatcher:
    """Dispatches lifecycle events to registered observers with DG-13 fault tolerance."""

    def __init__(
        self,
        observers: list[RAGObserver] | None = None,
        observer_errors: Literal["log", "raise"] = "log",
    ):
        self._observers = list(observers or [])
        self.observer_errors = observer_errors

    def notify_stage(self, record: StageRecord) -> None:
        for obs in self._observers:
            try:
                obs.on_stage(record)
            except Exception as exc:
                self._handle_error("on_stage", exc)

    def notify_usage(self, usage: StageUsage) -> None:
        for obs in self._observers:
            try:
                obs.on_usage(usage)
            except Exception as exc:
                self._handle_error("on_usage", exc)

    def notify_degrade(self, degradation: Degradation) -> None:
        for obs in self._observers:
            try:
                obs.on_degrade(degradation)
            except Exception as exc:
                self._handle_error("on_degrade", exc)

    def notify_complete(self, trace: RAGTrace, outcome: Any) -> None:
        for obs in self._observers:
            try:
                obs.on_complete(trace, outcome)
            except Exception as exc:
                self._handle_error("on_complete", exc)

    def _handle_error(self, method: str, exc: Exception) -> None:
        if self.observer_errors == "raise":
            raise exc
        logger.error(
            f"[DG-13] Observer {method} failed: {exc}. Request continues.",
            exc_info=True,
        )
