"""Opt-in relaxation ladder over soft constraints only (§7.6, §5.5, D-04)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from datetime import datetime

from ..models import QueryIntent, RAGFilterSpec, as_utc
from ..registry import split_component_name


@dataclass(frozen=True)
class RelaxationState:
    """The soft constraints a step may change, and the hard limits it must respect."""
    filters: RAGFilterSpec
    intent: QueryIntent
    soft_date_window: bool
    hard_min_similarity: float | None
    hard_date_range: tuple[datetime | None, datetime | None] | None = None   # the caller's range: never exceeded


class RelaxationStep(Protocol):
    label: str

    def apply(self, state: RelaxationState) -> RelaxationState | None:
        """Return the relaxed state, or None when the step has nothing to relax."""
        ...


class DropRequiredEntities:
    label = "drop_required_entities"

    def apply(self, state: RelaxationState) -> RelaxationState | None:
        if state.intent.entity_mode != "require" or not state.intent.entity_terms:
            return None
        return replace(state, intent=state.intent.model_copy(update={"entity_mode": "ignore"}))


class WidenTimeWindow:
    def __init__(self, factor: float = 2.0):
        self.factor = factor
        self.label = f"widen_time_window(x{factor:g})"

    def apply(self, state: RelaxationState) -> RelaxationState | None:
        window = state.filters.date_range_utc
        if not state.soft_date_window or window is None or window[0] is None or window[1] is None:
            return None
        start, end = window
        new_start = end - (end - start) * self.factor
        hard = state.hard_date_range
        if hard is not None and hard[0] is not None:
            new_start = max(new_start, as_utc(hard[0]))
        if new_start >= start:
            return None                      # already at the caller's limit
        return replace(state, filters=state.filters.model_copy(update={"date_range_utc": (new_start, end)}))


class DropTimeWindow:
    label = "drop_time_window"

    def apply(self, state: RelaxationState) -> RelaxationState | None:
        if not state.soft_date_window or state.filters.date_range_utc is None:
            return None
        if state.filters.date_range_utc == state.hard_date_range:
            return None
        # Dropping the soft window leaves the caller's hard range, never no range at all
        return replace(state, filters=state.filters.model_copy(update={"date_range_utc": state.hard_date_range}))


class LowerThreshold:
    def __init__(self, delta: float = 0.05):
        self.delta = delta
        self.label = f"lower_threshold({delta:g})"

    def apply(self, state: RelaxationState) -> RelaxationState | None:
        current = state.filters.min_similarity
        if current is None:
            return None
        lowered = max(current - self.delta, state.hard_min_similarity or 0.0)
        if lowered >= current:
            return None
        return replace(state, filters=state.filters.model_copy(update={"min_similarity": lowered}))


def _factor(argument: str | None) -> float:
    return float(argument.lstrip("x×")) if argument else 2.0


BUILTIN_RELAXATIONS = {
    "drop_required_entities": lambda argument: DropRequiredEntities(),
    "widen_time_window": lambda argument: WidenTimeWindow(_factor(argument)),
    "drop_time_window": lambda argument: DropTimeWindow(),
    "lower_threshold": lambda argument: LowerThreshold(float(argument) if argument else 0.05),
}


def build_relaxation(step: str, registry: Any) -> RelaxationStep:
    name, argument = split_component_name(step)
    return registry.get("relaxation", name)(argument)
