"""Granularity definitions and trait registry (§3.3).

Granularity is an open string. Engine stages MUST key on traits, never on
granularity names (D-15).
"""
from __future__ import annotations

# Built-in trait names
TRAIT_ATOMIC = "atomic"
TRAIT_ORDERED = "ordered"
TRAIT_OFFSET_ADDRESSABLE = "offset_addressable"
TRAIT_RERANKABLE = "rerankable"
TRAIT_SUMMARY = "summary"

# Two built-ins have identical traits, yet §9.2 and §10.1 treat them differently, so they are named.
GRANULARITY_FULL_DOC = "full_doc"
GRANULARITY_LOCAL_WINDOW = "local_window"

# Built-in granularities and their immutable trait sets
BUILTIN_GRANULARITIES: dict[str, set[str]] = {
    "passage": {TRAIT_OFFSET_ADDRESSABLE, TRAIT_RERANKABLE},
    "local_window": {TRAIT_OFFSET_ADDRESSABLE, TRAIT_ORDERED},
    "full_doc": {TRAIT_OFFSET_ADDRESSABLE, TRAIT_ORDERED},
    "parent_summary": {TRAIT_SUMMARY, TRAIT_ORDERED, TRAIT_ATOMIC},
    "article": {TRAIT_ATOMIC, TRAIT_RERANKABLE, TRAIT_OFFSET_ADDRESSABLE},
    "entry": {TRAIT_ATOMIC, TRAIT_RERANKABLE},
    "graph_entity": {TRAIT_SUMMARY, TRAIT_ATOMIC},
}


class GranularityRegistry:
    """Registry mapping granularity names to trait sets."""

    def __init__(self):
        self._registry: dict[str, set[str]] = {k: set(v) for k, v in BUILTIN_GRANULARITIES.items()}

    def register_granularity(self, name: str, traits: set[str] | list[str]) -> None:
        """Register or extend traits for a granularity name."""
        self._registry[name] = set(traits)

    def get_traits(self, granularity: str) -> set[str]:
        """Return registered traits for granularity (or empty set if unknown)."""
        return self._registry.get(granularity, set())

    def has_trait(self, granularity: str, trait: str) -> bool:
        """Check if a granularity has a specific trait."""
        return trait in self._registry.get(granularity, ())


# Module-level default registry
_DEFAULT_REGISTRY = GranularityRegistry()

register_granularity = _DEFAULT_REGISTRY.register_granularity
get_traits = _DEFAULT_REGISTRY.get_traits
has_trait = _DEFAULT_REGISTRY.has_trait
