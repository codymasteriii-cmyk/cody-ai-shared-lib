"""Usage accounting, CallContext, BudgetGuard, and UsageTracker (§13.1, §15.4, RAG-OPS-1).

Every paid model call is accounted for, including calls that fail:

    before a call   gateway -> UsageTracker.call_context() -> BudgetGuard.before_call()   "allow", "skip" or raise
    after a call    gateway -> UsageTracker.record(StageUsage)                              tokens, cost, latency
    any time        UsageTracker.total()                                                   the request's running total

The app supplies the `BudgetGuard` (e.g. "no tenant may spend more than $5 a day"); the library only asks it before
each call and obeys. `skip` is allowed only for optional stages (a reranker may be skipped, the answer may not).

Dummy example: a guard that refuses optional work once a request has spent 2 cents. The reranker (optional) is skipped
with a DG-11 degradation and the answer is still generated; the generation call (required) is never skipped.
"""
from __future__ import annotations

import copy
import threading
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from .errors import ProfileConfigError
from .models import StageUsage, UsageTotal
from ..pricing import PRICE_TABLE_VERSION


class CallContext(BaseModel):
    """Context passed to BudgetGuard before each paid model call (§13.1).

    Everything a guard needs to decide without calling back into the engine: which stage and model, how big the call
    will be, what this request has already spent, and who is paying (tenant, scope, documents).
    """
    request_id: str | None = ""
    stage: str
    model: str = ""
    optional: bool = True
    est_input_tokens: int = 0
    est_cost_usd: float | None = None
    request_spent: UsageTotal = Field(default_factory=lambda: UsageTotal(stage="total"))
    tenant_or_user_id: str | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    doc_ids: list[str] | None = None
    profile_name: str = ""
    variant: str | None = None


@runtime_checkable
class BudgetGuard(Protocol):
    """App-supplied budget enforcement hook consulted before each paid call (§13.1).

    `enforces_cost=True` tells the engine at startup to check that every model in the profile has a known price,
    because a guard that limits cost cannot work on a model it cannot price.
    """
    enforces_cost: bool = False

    def before_call(
        self,
        stage: str,
        model: str,
        est_input_tokens: int,
        optional: bool,
        ctx: CallContext,
    ) -> Literal["allow", "skip"]:
        """Return 'allow', return 'skip' (only valid when optional=True), or raise BudgetExceeded."""
        ...


def invoke_budget_guard(guard: BudgetGuard, ctx: CallContext) -> Literal["allow", "skip"]:
    """Consult the guard before a paid call (§13.1).

    A guard may raise `BudgetExceeded` itself to stop the request. Returning "skip" for a required stage is a
    programming error in the guard, so it raises `ProfileConfigError` instead of quietly dropping a required step.

    >>> class StingyGuard:
    ...     enforces_cost = False
    ...     def before_call(self, stage, model, est_input_tokens, optional, ctx):
    ...         return "skip"
    >>> invoke_budget_guard(StingyGuard(), CallContext(stage="rerank", optional=True))
    'skip'
    """
    raw = guard.before_call(ctx.stage, ctx.model, ctx.est_input_tokens, ctx.optional, ctx)

    if raw == "skip":
        if not ctx.optional:
            raise ProfileConfigError(
                f"BudgetGuard returned 'skip' for required stage '{ctx.stage}'",
                stage=ctx.stage,
                request_id=ctx.request_id or "",
            )
        return "skip"
    return "allow"


class UsageTracker:
    """Thread-safe collector of per-stage usage records during a RAG request.

    Thread-safe because a request's model calls can run in parallel (the map step of a list answer, the two legs of a
    hybrid search). `total()` is computed from the records every time, so it is always consistent with them.

    >>> tracker = UsageTracker("req-1")
    >>> tracker.record(StageUsage(stage="router", model="m1", input_tokens=100, output_tokens=20, calls=1, cost_usd=0.001))
    >>> tracker.record(StageUsage(stage="generation", model="m2", input_tokens=900, output_tokens=150, calls=1, cost_usd=0.002))
    >>> total = tracker.total()
    >>> total.input_tokens, total.calls, total.cost_usd, total.cost_complete
    (1000, 2, 0.003, True)
    """

    def __init__(
        self,
        request_id: str,
        tenant_or_user_id: str | None = None,
        doc_ids: list[str] | None = None,
        profile_name: str = "",
        variant: str | None = None,
        scope: dict[str, Any] | None = None,
        on_record: Callable[[StageUsage], None] | None = None,
    ):
        self.request_id = request_id
        self.tenant_or_user_id = tenant_or_user_id
        self.doc_ids = doc_ids
        self.profile_name = profile_name
        self.variant = variant
        self.scope = dict(scope or {})
        self.on_record = on_record
        self._records: list[StageUsage] = []
        self._lock = threading.Lock()

    def record(self, usage: StageUsage) -> None:
        """Add a stage usage record.

        The record is copied, so the caller cannot change it afterwards; the `on_record` callback (the observers'
        `on_usage`) runs after the lock is released, so a slow observer cannot block other threads' accounting.
        """
        with self._lock:
            self._records.append(copy.deepcopy(usage))
        if self.on_record is not None:
            self.on_record(usage)

    def call_context(
        self,
        stage: str,
        model: str,
        optional: bool,
        est_input_tokens: int,
        est_cost_usd: float | None,
    ) -> CallContext:
        """Build what a BudgetGuard sees before a paid call in this request (§13.1)."""
        return CallContext(
            request_id=self.request_id,
            stage=stage,
            model=model,
            optional=optional,
            est_input_tokens=est_input_tokens,
            est_cost_usd=est_cost_usd,
            request_spent=self.total(),
            tenant_or_user_id=self.tenant_or_user_id,
            scope=dict(self.scope),
            doc_ids=self.doc_ids,
            profile_name=self.profile_name,
            variant=self.variant,
        )

    @property
    def stage_usages(self) -> list[StageUsage]:
        """A copy of the records so far, in the order they were recorded."""
        with self._lock:
            return list(self._records)

    def total(self) -> UsageTotal:
        """Aggregate total tokens and cost across all recorded stages.

        `cost_complete` is False when some call had no known price; `unpriced_models` names them. The total then
        counts only what is known, and says so, rather than presenting a partial sum as the whole. A failed call whose
        billing is unknown does not make the total incomplete: it is recorded, but its cost is not guessed.
        """
        records = self.stage_usages
        in_tok = sum(u.input_tokens for u in records)
        out_tok = sum(u.output_tokens for u in records)
        cached_tok = sum(u.cached_tokens for u in records)
        calls = sum(u.calls for u in records)
        lat = sum(u.latency_ms for u in records)

        cost_sum = 0.0
        cost_complete = True
        unpriced: list[str] = []

        for u in records:
            if u.cost_usd is not None:
                cost_sum += u.cost_usd
            elif u.billing != "unknown":  # a failed call has a price; whether it was billed is unknown
                cost_complete = False
                if u.model and u.model not in unpriced:
                    unpriced.append(u.model)

        total_cost = round(cost_sum, 6) if cost_complete or cost_sum > 0.0 else None

        return UsageTotal(
            stage="total",
            model=None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_tokens=cached_tok,
            calls=calls,
            latency_ms=lat,
            cost_usd=total_cost,
            price_table_version=PRICE_TABLE_VERSION,
            estimated=any(u.estimated for u in records),
            billing="billed",
            outcome="ok",
            cost_complete=cost_complete,
            unpriced_models=unpriced,
        )
