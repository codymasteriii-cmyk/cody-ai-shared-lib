"""Usage accounting, CallContext, BudgetGuard, and UsageTracker (§13.1, §15.4, RAG-OPS-1)."""
from __future__ import annotations

import copy
import threading
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from .errors import ProfileConfigError
from .models import StageUsage, UsageTotal
from ..pricing import PRICE_TABLE_VERSION


class CallContext(BaseModel):
    """Context passed to BudgetGuard before each paid model call (§13.1)."""
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
    """App-supplied budget enforcement hook consulted before each paid call (§13.1)."""
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
    """Consult the guard before a paid call (§13.1)."""
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
    """Thread-safe collector of per-stage usage records during a RAG request."""

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
        """Add a stage usage record."""
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
        with self._lock:
            return list(self._records)

    def total(self) -> UsageTotal:
        """Aggregate total tokens and cost across all recorded stages."""
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
