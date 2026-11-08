"""Usage accounting, CallContext, BudgetGuard, and UsageTracker (§13.1, §15.4, RAG-OPS-1)."""
from __future__ import annotations

import copy
from typing import Any, Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from .errors import BudgetExceeded
from .models import StageUsage, UsageTotal
from ..pricing import PRICE_TABLE_VERSION


class CallContext(BaseModel):
    """Context passed to BudgetGuard before each paid model call."""
    stage: str
    request_id: str
    tenant_or_user_id: str | None = None
    est_input_tokens: int = 0
    est_cost_usd: float | None = None
    request_spent: UsageTotal = Field(default_factory=lambda: UsageTotal(stage="total"))


@runtime_checkable
class BudgetGuard(Protocol):
    """App-supplied budget enforcement hook consulted before each paid call."""

    def before_call(self, ctx: CallContext) -> Literal["proceed", "skip"]:
        """Decide whether call proceeds, is skipped (DG-11), or raises BudgetExceeded."""
        ...


class UsageTracker:
    """Thread-safe collector of per-stage usage records during a RAG request."""

    def __init__(self, request_id: str, tenant_or_user_id: str | None = None):
        self.request_id = request_id
        self.tenant_or_user_id = tenant_or_user_id
        self._records: list[StageUsage] = []

    def record(self, usage: StageUsage) -> None:
        """Add a stage usage record."""
        self._records.append(copy.deepcopy(usage))

    @property
    def stage_usages(self) -> list[StageUsage]:
        return list(self._records)

    def total(self) -> UsageTotal:
        """Aggregate total tokens and cost across all recorded stages."""
        in_tok = sum(u.input_tokens for u in self._records)
        out_tok = sum(u.output_tokens for u in self._records)
        cached_tok = sum(u.cached_tokens for u in self._records)
        calls = sum(u.calls or 1 for u in self._records)
        lat = sum(u.latency_ms for u in self._records)

        cost_sum = 0.0
        cost_complete = True
        unpriced: list[str] = []

        for u in self._records:
            if u.cost_usd is not None:
                cost_sum += u.cost_usd
            else:
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
            estimated=any(u.estimated for u in self._records),
            billing="billed",
            outcome="ok",
            cost_complete=cost_complete,
            unpriced_models=unpriced,
        )
