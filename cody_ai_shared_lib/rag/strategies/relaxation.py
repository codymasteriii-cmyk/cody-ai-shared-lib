"""Opt-in relaxation ladder over soft constraints only (§7.6, §5.5, D-04).

When a search returns too few results the pipeline can loosen the question one step at a time, trying again after
each step. A step may change only what the system chose on the user's behalf (a date window guessed from "last
month", a similarity floor, names the router marked droppable). It never touches what the caller set (a hard date
range, a guard boundary, a tenant): relaxation can widen a search but never widen what a reader may see.

Steps are named in the profile, e.g. ["drop_required_entities", "widen_time_window(x2)", "lower_threshold(0.05)"],
and tried in that order; a step is kept only if it produces more ranked results, and the ladder stops
as soon as there are enough.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from datetime import datetime

from ..models import QueryIntent, RAGFilterSpec, TextRule, as_utc
from ..registry import split_component_name


@dataclass(frozen=True)
class RelaxationState:
    """The soft constraints a step may change, and the hard limits it must respect."""
    filters: RAGFilterSpec
    intent: QueryIntent
    soft_date_window: bool
    hard_min_similarity: float | None
    hard_date_range: tuple[datetime | None, datetime | None] | None = None   # the caller's range: never exceeded
    rules: tuple[TextRule, ...] = ()        # the request's text rules; only `droppable` ones may be dropped (§7.9)


class RelaxationStep(Protocol):
    """One rung of the ladder. `apply` returns the loosened state, or None when there is nothing left to loosen."""
    label: str

    def apply(self, state: RelaxationState) -> RelaxationState | None:
        """Return the relaxed state, or None when the step has nothing to relax."""
        ...


class DropRequiredEntities:
    """Stop requiring the names the router marked droppable, so the search can match without them."""
    label = "drop_required_entities"

    def apply(self, state: RelaxationState) -> RelaxationState | None:
        """Drop the rules marked droppable. Non-droppable rules and `exact_terms` are never relaxed (§7.9)."""
        dropped = [rule for rule in state.rules if rule.droppable]
        if not dropped:
            return None
        kept = tuple(rule for rule in state.rules if not rule.droppable)
        names_left = any(rule.group == "auto" for rule in kept)
        intent = state.intent
        if state.intent.entity_mode == "require" and not names_left:
            intent = intent.model_copy(update={"entity_mode": "ignore"})      # token groups follow the rules
        return replace(state, intent=intent, rules=kept)


class WidenTimeWindow:
    """Stretch a soft date window further into the past by `factor` (never past the caller's own range)."""
    def __init__(self, factor: float = 2.0):
        self.factor = factor
        self.label = f"widen_time_window(x{factor:g})"

    def apply(self, state: RelaxationState) -> RelaxationState | None:
        """Only a soft window (one the system guessed) can widen; a hard range from the caller is left alone."""
        window = state.filters.date_range_utc
        if not state.soft_date_window or window is None or window[0] is None or window[1] is None:
            return None
        start: datetime = window[0]
        end: datetime = window[1]
        new_start: datetime = end - (end - start) * self.factor
        hard = state.hard_date_range
        if hard is not None and hard[0] is not None:
            new_start = max(new_start, as_utc(hard[0]))
        if new_start >= start:
            return None                      # already at the caller's limit
        return replace(state, filters=state.filters.model_copy(update={"date_range_utc": (new_start, end)}))


class DropTimeWindow:
    """Remove the soft date window altogether (the caller's hard range, if any, stays)."""
    label = "drop_time_window"

    def apply(self, state: RelaxationState) -> RelaxationState | None:
        """Falls back to the caller's hard range, never to no range at all."""
        if not state.soft_date_window or state.filters.date_range_utc is None:
            return None
        if state.filters.date_range_utc == state.hard_date_range:
            return None
        # Dropping the soft window leaves the caller's hard range, never no range at all
        return replace(state, filters=state.filters.model_copy(update={"date_range_utc": state.hard_date_range}))


class LowerThreshold:
    """Lower the minimum similarity by `delta`, but not below the floor the caller set."""
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
    """The multiplier in "widen_time_window(x3)"; a leading x or × is allowed, and the default is 2.

    >>> _factor("x3"), _factor("×1.5"), _factor(None)
    (3.0, 1.5, 2.0)
    """
    return float(argument.lstrip("x×")) if argument else 2.0


# Name -> factory taking the optional argument from the profile entry; registered in the component registry
BUILTIN_RELAXATIONS = {
    "drop_required_entities": lambda argument: DropRequiredEntities(),
    "widen_time_window": lambda argument: WidenTimeWindow(_factor(argument)),
    "drop_time_window": lambda argument: DropTimeWindow(),
    "lower_threshold": lambda argument: LowerThreshold(float(argument) if argument else 0.05),
}


def build_relaxation(step: str, registry: Any) -> RelaxationStep:
    """Build a step from its profile entry: "name" or "name(argument)", looked up in the registry."""
    name, argument = split_component_name(step)
    return registry.get("relaxation", name)(argument)
