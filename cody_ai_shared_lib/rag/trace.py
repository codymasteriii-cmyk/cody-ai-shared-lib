"""RAG trace and observers (§13.2, DG-13).

The light trace is always on and holds ids, counts, decisions and versions, never text.
The verbose trace adds per-stage snapshots, zone tokens and model call records.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .models import Degradation, RAGFilterSpec, RAGNode, StageUsage, UsageTotal

logger = logging.getLogger("shared-rag")


class StageRecord(BaseModel):
    """Record of one stage execution."""
    name: str
    version: str = "1.0.0"
    kind: str = "stage"
    started_ms: float = 0.0
    duration_ms: float = 0.0
    in_count: int | None = None
    out_count: int | None = None
    fallback: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)


class NodeSnapshot(BaseModel):
    node_id: str
    granularity: str
    score: float
    score_kind: str
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None


class StageSnapshot(BaseModel):
    stage: str
    nodes: list[NodeSnapshot]


class ModelCallRecord(BaseModel):
    stage: str
    model: str | None
    input_tokens: int
    output_tokens: int
    outcome: str


class RAGTrace(BaseModel):
    """Trace returned on every result (§13.2)."""
    trace_schema_version: int = 1
    request_id: str | None = None
    answer_id: str | None = None
    profile_name: str = ""
    profile_version: str = ""
    profile_fingerprint: str = ""
    variant: str | None = None
    prompt_version: str | None = None
    components: dict[str, str] = Field(default_factory=dict)
    router_decision: dict[str, Any] = Field(default_factory=dict)
    tier_decision_reason: str = ""
    applied_filters: RAGFilterSpec | None = None
    guard_effects: dict[str, dict[str, int]] = Field(default_factory=dict)
    stages: list[StageRecord] = Field(default_factory=list)
    degradations: list[Degradation] = Field(default_factory=list)
    usage_total: UsageTotal | None = None
    scope_revision: str | None = None
    clock_now: datetime | None = None
    level: Literal["light", "verbose"] = "light"
    total_latency_ms: float = 0.0
    state_size_bytes: int = 0
    snapshots: list[StageSnapshot] | None = None
    zone_tokens: dict[str, int] | None = None
    model_calls: list[ModelCallRecord] | None = None


@runtime_checkable
class RAGObserver(Protocol):
    """Receives stage, usage, degradation and completion events; must be fast (§13.2)."""

    def on_stage(self, record: StageRecord) -> None:
        ...

    def on_usage(self, usage: StageUsage) -> None:
        ...

    def on_degrade(self, degradation: Degradation) -> None:
        ...

    def on_complete(self, trace: RAGTrace, outcome: Any) -> None:
        ...


class ObserverDispatcher:
    """Sends lifecycle events to observers with DG-13 fault tolerance."""

    def __init__(
        self,
        observers: list[RAGObserver] | None = None,
        observer_errors: Literal["log", "raise"] = "log",
        degradation_tracker: Any = None,
    ):
        self._observers = list(observers or [])
        self.observer_errors = observer_errors
        self._degradation_tracker = degradation_tracker

    def for_request(self, degradation_tracker: Any) -> "ObserverDispatcher":
        """Return a dispatcher that records observer failures on this request."""
        return ObserverDispatcher(self._observers, self.observer_errors, degradation_tracker)

    def notify_stage(self, record: StageRecord) -> None:
        self._notify("on_stage", record)

    def notify_usage(self, usage: StageUsage) -> None:
        self._notify("on_usage", usage)

    def notify_degrade(self, degradation: Degradation) -> None:
        self._notify("on_degrade", degradation)

    def notify_complete(self, trace: RAGTrace, outcome: Any) -> None:
        self._notify("on_complete", trace, outcome)

    def _notify(self, method: str, *args: Any) -> None:
        for observer in self._observers:
            try:
                getattr(observer, method)(*args)
            except Exception as exc:
                self._handle_error(method, exc)

    def _handle_error(self, method: str, exc: Exception) -> None:
        if self.observer_errors == "raise":
            raise exc
        logger.error(f"[DG-13] Observer {method} failed: {exc}. Request continues.", exc_info=True)
        if self._degradation_tracker is not None:
            self._degradation_tracker.record(
                dg_id="DG-13",
                stage="observer",
                reason=f"Observer {method} failed: {type(exc).__name__}",
                impact="Observer missed an event; the request continued",
                notify=False,
            )


class TraceCollector:
    """Builds the trace for one request and reports each stage to observers."""

    def __init__(
        self,
        request_id: str,
        profile: Any,
        clock: Any,
        dispatcher: ObserverDispatcher,
        level: Literal["light", "verbose"],
        variant: str | None = None,
    ):
        self.request_id = request_id
        self.profile = profile
        self.clock = clock
        self.dispatcher = dispatcher
        self.level = level
        self.variant = variant
        self.started = clock.monotonic()
        self.stages: list[StageRecord] = []
        self.guard_effects: dict[str, dict[str, int]] = {}
        self.components: dict[str, str] = {}
        self.router_decision: dict[str, Any] = {}
        self.snapshots: list[StageSnapshot] = []
        self.zone_tokens: dict[str, int] | None = None
        self.prompt_version: str | None = None
        self.tier_decision_reason = ""
        self.scope_revision: str | None = None
        self.rule: int | None = None

    def elapsed_ms(self) -> float:
        return (self.clock.monotonic() - self.started) * 1000.0

    def record_stage(
        self,
        name: str,
        kind: str,
        started_ms: float,
        version: str = "1.0.0",
        in_count: int | None = None,
        out_count: int | None = None,
        notes: dict[str, Any] | None = None,
        nodes: list[RAGNode[Any]] | None = None,
    ) -> None:
        record = StageRecord(
            name=name,
            version=version,
            kind=kind,
            started_ms=started_ms,
            duration_ms=self.elapsed_ms() - started_ms,
            in_count=in_count,
            out_count=out_count,
            notes=notes or {},
        )
        self.stages.append(record)
        self.components[name] = version
        if self.level == "verbose" and nodes is not None:
            self.snapshots.append(StageSnapshot(stage=name, nodes=[self._snapshot(n) for n in nodes]))
        self.dispatcher.notify_stage(record)

    def record_guard_effect(self, guard: str, dropped: int, truncated: int) -> None:
        effect = self.guard_effects.setdefault(guard, {"dropped": 0, "truncated": 0})
        effect["dropped"] += dropped
        effect["truncated"] += truncated

    def build(
        self,
        applied_filters: RAGFilterSpec | None,
        degradations: list[Degradation],
        usage: list[StageUsage],
        usage_total: UsageTotal,
        answer_id: str | None = None,
        state_size_bytes: int = 0,
    ) -> RAGTrace:
        verbose = self.level == "verbose"
        return RAGTrace(
            request_id=self.request_id,
            answer_id=answer_id,
            profile_name=self.profile.name,
            profile_version=self.profile.version,
            profile_fingerprint=self.profile.fingerprint(),
            variant=self.variant,
            prompt_version=self.prompt_version,
            components=dict(self.components),
            router_decision=dict(self.router_decision),
            tier_decision_reason=self.tier_decision_reason,
            applied_filters=applied_filters,
            guard_effects={k: dict(v) for k, v in self.guard_effects.items()},
            stages=list(self.stages),
            degradations=degradations,
            usage_total=usage_total,
            scope_revision=self.scope_revision,
            clock_now=self.clock.now_utc(),
            level=self.level,
            total_latency_ms=self.elapsed_ms(),
            state_size_bytes=state_size_bytes,
            snapshots=list(self.snapshots) if verbose else None,
            zone_tokens=self.zone_tokens if verbose else None,
            model_calls=[
                ModelCallRecord(
                    stage=u.stage,
                    model=u.model,
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    outcome=u.outcome,
                )
                for u in usage
            ] if verbose else None,
        )

    @staticmethod
    def _snapshot(node: RAGNode[Any]) -> NodeSnapshot:
        return NodeSnapshot(
            node_id=node.node_id,
            granularity=node.granularity,
            score=node.score,
            score_kind=node.score_kind,
            vector_score=node.vector_score,
            keyword_score=node.keyword_score,
            rerank_score=node.rerank_score,
        )
