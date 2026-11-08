"""Explicit ComponentRegistry for RAG stages, routers, and strategies (D-18, R-12)."""
from __future__ import annotations

from typing import Any


class ComponentRegistry:
    """Central registry mapping component names to implementations."""

    def __init__(self):
        self._categories: dict[str, dict[str, Any]] = {
            "router": {},
            "strategy": {},
            "reranker": {},
            "splitter": {},
            "guard": {},
            "analyzer": {},
            "observer": {},
            "renderer": {},
        }

    def register(self, category: str, name: str, component: Any) -> None:
        """Register a component under a category."""
        if category not in self._categories:
            self._categories[category] = {}
        self._categories[category][name] = component

    def get(self, category: str, name: str) -> Any:
        """Retrieve a registered component, or None if not found."""
        return self._categories.get(category, {}).get(name)

    def has(self, category: str, name: str) -> bool:
        """Check if component is registered."""
        return name in self._categories.get(category, {})

    def list_components(self, category: str) -> list[str]:
        """List all registered names under category."""
        return sorted(list(self._categories.get(category, {}).keys()))


# Pre-register built-in defaults
def create_default_registry() -> ComponentRegistry:
    reg = ComponentRegistry()
    from .guards import ScopeGuard, SequentialBoundaryGuard, TemporalGuard, TenantGuard
    from .render import DefaultRenderer
    from .text.analyzer import CjkAwareAnalyzer

    reg.register("analyzer", "cjk_aware", CjkAwareAnalyzer)
    reg.register("renderer", "default", DefaultRenderer)
    reg.register("guard", "sequential_boundary", SequentialBoundaryGuard)
    reg.register("guard", "temporal", TemporalGuard)
    reg.register("guard", "tenant", TenantGuard)
    reg.register("guard", "scope", ScopeGuard)

    # Standard strategy names
    for strat in ["dense_vector", "hybrid_rrf", "full_context", "keyword", "hierarchical", "adaptive"]:
        reg.register("strategy", strat, strat)

    # Standard router names
    for r in ["auto", "none", "heuristic", "llm"]:
        reg.register("router", r, r)

    # Standard splitters
    from .text.splitter import PassThroughSplitter, SentenceSplitter
    reg.register("splitter", "sentence", SentenceSplitter)
    reg.register("splitter", "pass_through", PassThroughSplitter)

    return reg
