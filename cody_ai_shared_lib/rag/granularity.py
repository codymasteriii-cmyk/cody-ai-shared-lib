"""Granularity definitions and trait registry (§3.3).

Granularity is an open string. Engine stages MUST key on traits, never on
granularity names (D-15).

A node's `granularity` says what kind of thing it is ("passage", "parent_summary", "article"...). Code that needs to treat
kinds differently never asks "is this a passage?"; it asks about a TRAIT, so a new kind an app invents behaves correctly
the moment it declares its traits:

    trait                what it means                                              examples
    ───────────────────  ─────────────────────────────────────────────────────────  ─────────────────────────────
    offset_addressable   its text sits at known character offsets, so it can be     passage, local_window, article
                         cut at a reader's position
    ordered              a context layer with its own budget, not a ranked candidate local_window, full_doc, parent_summary
    rerankable           a reranker may score it                                    passage, article, entry
    summary              it summarizes other text                                   parent_summary
    atomic               goes into a prompt whole or not at all, never cut          article, entry, parent_summary

The most-used question is `is_ranked(node)`: a node competes on relevance only if it is not an ordered layer and is not Tier 0
text. Stages that reorder, cut or rescore use `apply_to_ranked` and `limit_ranked`, which leave every other node where it is.

Dummy example: an app adds "chat_message" with traits {rerankable}. Reranking now includes chat messages and nothing else in the
library had to change. The table is one per process today (OI-1 records the wish for one per engine).
"""
from __future__ import annotations

from typing import Any

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
    """Registry mapping granularity names to trait sets.

    >>> sorted(get_traits("passage"))
    ['offset_addressable', 'rerankable']
    >>> has_trait("parent_summary", "summary"), has_trait("passage", "summary")
    (True, False)
    """

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


# Module-level default registry: the one process-wide table. `register_granularity`, `get_traits` and `has_trait` below are
# its methods, exported as plain functions for convenience.
_DEFAULT_REGISTRY = GranularityRegistry()

register_granularity = _DEFAULT_REGISTRY.register_granularity
get_traits = _DEFAULT_REGISTRY.get_traits
has_trait = _DEFAULT_REGISTRY.has_trait


FLAG_FULL_SCOPE = "full_scope"   # Tier 0 membership (§8.2): the node is part of the whole scope


def is_ranked(node: Any) -> bool:
    """True for nodes competing on relevance (§9.2, §10.1).

    Ordered layers have token budgets instead, and Tier 0 text is the whole scope, never candidates.

    >>> from cody_ai_shared_lib.rag.models import RAGNode
    >>> is_ranked(RAGNode(node_id="p", text="")), is_ranked(RAGNode(node_id="s", text="", granularity="parent_summary"))
    (True, False)
    """
    return not has_trait(node.granularity, TRAIT_ORDERED) and FLAG_FULL_SCOPE not in getattr(node, "flags", ())


def apply_to_ranked(nodes: list[Any], transform: Any) -> list[Any]:
    """Let `transform` reorder or rescore only the ranked nodes; every other node keeps its position.

    `transform` receives the ranked nodes and returns the same nodes in a new order.
    """
    ranked = [n for n in nodes if is_ranked(n)]
    if not ranked:
        return list(nodes)
    reordered = iter(transform(ranked))
    return [next(reordered) if is_ranked(n) else n for n in nodes]


def limit_ranked(nodes: list[Any], limit: int) -> list[Any]:
    """Keep at most `limit` ranked nodes, leaving every ordered node in place.

    >>> from cody_ai_shared_lib.rag.models import RAGNode
    >>> nodes = [RAGNode(node_id=f"p{i}", text="") for i in range(4)] + [RAGNode(node_id="s", text="", granularity="parent_summary")]
    >>> [n.node_id for n in limit_ranked(nodes, 2)]
    ['p0', 'p1', 's']
    """
    kept: list[Any] = []
    ranked = 0
    for node in nodes:
        if is_ranked(node):
            if ranked >= limit:
                continue
            ranked += 1
        kept.append(node)
    return kept
