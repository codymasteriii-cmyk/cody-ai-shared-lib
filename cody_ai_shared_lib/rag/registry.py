"""ComponentRegistry: named strategies, routers, stages and other pluggable parts (§8.8, D-18, R-12).

Why a registry: a profile is data ("router": "auto", "post_stages": ["threshold", "rerank", "top_k"]). Something has
to turn those names into objects. The registry is that lookup table, grouped by category:

    category         what a name resolves to              examples
    ───────────────  ───────────────────────────────────  ──────────────────────────────────────
    strategy         a RetrievalStrategy class            "hybrid_rrf", "find_then_read"
    predicate        a function of a RuleContext          "fits_full_context"
    router           a factory(deps) -> IntentRouter      "none", "heuristic", "llm", "auto"
    stage            a factory(deps) -> post stage        "threshold", "rerank", "top_k"
    transform        a query transform factory            "secondary_query_fanout"
    reranker, generator, guard, relaxation, formatter, splitter, analyzer, renderer

An app extends the library by registering its own under a new name (or replace=True to override a built-in), then
naming it in the profile. `create_default_registry()` builds the table of built-ins; each engine can take its own
copy, so tests and apps never share mutable global state (D-18).

Dummy example: an app writes `MyBoost`, a post stage. It calls `registry.register_stage("my_boost", lambda deps: MyBoost())`
and adds "my_boost" to `post_stages`. The pipeline finds it by name and builds it once, like a built-in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import ProfileConfigError


def split_component_name(configured: str) -> tuple[str, str | None]:
    """Split a configured name such as `identity_dedup(url)` into its name and argument.

    >>> split_component_name("identity_dedup(url)")
    ('identity_dedup', 'url')
    >>> split_component_name("top_k")
    ('top_k', None)
    """
    match = re.fullmatch(r"([A-Za-z_]+)(?:\(([^)]*)\))?", configured.strip())
    return (match.group(1), match.group(2)) if match else (configured, None)


@dataclass
class ComponentDeps:
    """What a registered factory may use to build its component.

    A factory is `deps -> component`. Passing one bundle keeps factories uniform, and keeps components from reaching
    into the engine: they get only what is listed here.
    """
    profile: Any
    gateway: Any
    renderer: Any
    analyzer: Any
    clock: Any
    temporal_resolver: Any
    registry: "ComponentRegistry"


class ComponentRegistry:
    """Maps component names to implementations. Instances keep tests and engines isolated (D-18).

    The typed `register_*` methods below are thin, readable wrappers over `register(category, name, ...)`.

    >>> registry = ComponentRegistry()
    >>> registry.register("stage", "mine", object)
    >>> registry.has("stage", "mine"), registry.list_components("stage")
    (True, ['mine'])
    >>> registry.get("stage", "absent") is None
    True
    """

    def __init__(self) -> None:
        self._categories: dict[str, dict[str, Any]] = {}

    def register(self, category: str, name: str, component: Any, replace: bool = False) -> None:
        """Register a component; a duplicate name raises unless replace=True (§8.8)."""
        bucket = self._categories.setdefault(category, {})
        if name in bucket and not replace:
            raise ProfileConfigError(f"Duplicate {category} name '{name}' (pass replace=True to override)")
        bucket[name] = component

    def register_strategy(self, name: str, cls: Any, replace: bool = False) -> None:
        self.register("strategy", name, cls, replace)

    def register_predicate(self, name: str, fn: Any, replace: bool = False) -> None:
        self.register("predicate", name, fn, replace)

    def register_router(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("router", name, factory, replace)

    def register_transform(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("transform", name, factory, replace)

    def register_stage(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("stage", name, factory, replace)

    def register_reranker(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("reranker", name, factory, replace)

    def register_generator(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("generator", name, factory, replace)

    def register_post_generation_stage(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("post_generation", name, factory, replace)

    def register_ingestion_stage(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("ingestion_stage", name, factory, replace)

    def register_guard(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("guard", name, factory, replace)

    def register_granularity(self, name: str, traits: frozenset[str]) -> None:
        """Register a node kind and its traits. Unlike the others, granularities are one table for the whole process (OI-1)."""
        from .granularity import register_granularity

        register_granularity(name, set(traits))

    def register_relaxation(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("relaxation", name, factory, replace)

    def register_formatter(self, name: str, factory: Any, replace: bool = False) -> None:
        self.register("formatter", name, factory, replace)

    def get(self, category: str, name: str) -> Any:
        """Return a registered component, or None if not found."""
        return self._categories.get(category, {}).get(name)

    def has(self, category: str, name: str) -> bool:
        """True when a component is registered under this name."""
        return name in self._categories.get(category, {})

    def list_components(self, category: str) -> list[str]:
        """The names registered in one category, sorted; used in error messages and by validation."""
        return sorted(self._categories.get(category, {}))


def create_default_registry() -> ComponentRegistry:
    """Return a new registry holding every built-in component.

    Each call builds a fresh registry, so an app may add or replace entries without affecting another engine.
    """
    # Imported here because these modules depend on the profile, which depends on this module.
    from .generation.stuff import StuffStrategy
    from .guards import BUILTIN_GUARDS
    from .ingestion.splitter import PassThroughSplitter, SentenceSplitter
    from .intent.router import (
        build_auto_router,
        build_heuristic_router,
        build_llm_router,
        build_passthrough_router,
    )
    from .postgen.citations import BUILTIN_FORMATTERS
    from .postretrieval.rerank import JinaCrossEncoderReranker
    from .postretrieval.stages import BUILTIN_STAGES
    from .render import DefaultRenderer
    from .strategies.adaptive import BUILTIN_PREDICATES, AdaptiveRetrievalStrategy
    from .strategies.dense_vector import DenseVectorStrategy
    from .strategies.find_then_read import FindThenReadStrategy
    from .strategies.full_context import FullContextStrategy
    from .strategies.hierarchical import HierarchicalSummaryStrategy
    from .strategies.hybrid_rrf import HybridRRFStrategy
    from .strategies.keyword import KeywordFirstStrategy, KeywordOnlyStrategy
    from .strategies.relaxation import BUILTIN_RELAXATIONS
    from .text.analyzer import CjkAwareAnalyzer
    from .transforms import SecondaryQueryFanOut

    reg = ComponentRegistry()
    reg.register("analyzer", "cjk_aware", CjkAwareAnalyzer)
    reg.register("renderer", "default", DefaultRenderer)
    reg.register("splitter", "sentence", SentenceSplitter)
    reg.register("splitter", "pass_through", PassThroughSplitter)

    for strategy in (
        AdaptiveRetrievalStrategy,
        DenseVectorStrategy,
        FullContextStrategy,
        FindThenReadStrategy,
        HierarchicalSummaryStrategy,
        HybridRRFStrategy,
        KeywordFirstStrategy,
        KeywordOnlyStrategy,
    ):
        reg.register_strategy(strategy.name, strategy)
    for name, predicate in BUILTIN_PREDICATES.items():
        reg.register_predicate(name, predicate)

    reg.register_router("none", build_passthrough_router)
    reg.register_router("heuristic", build_heuristic_router)
    reg.register_router("llm", build_llm_router)
    reg.register_router("auto", build_auto_router)

    reg.register_transform(SecondaryQueryFanOut.name, lambda deps: SecondaryQueryFanOut())
    reg.register_reranker(
        "jina",
        lambda deps: JinaCrossEncoderReranker(
            gateway=deps.gateway,
            model=deps.profile.models.reranker,
            renderer=deps.renderer,
            fallback_action=deps.profile.fallbacks.get("rerank", "keep_order"),
        ),
    )
    reg.register_generator(StuffStrategy.name, lambda deps: StuffStrategy(deps.gateway, deps.profile))

    for name, factory in BUILTIN_STAGES.items():
        reg.register_stage(name, factory)
    for name, guard in BUILTIN_GUARDS.items():
        reg.register_guard(name, guard)
    for name, factory in BUILTIN_RELAXATIONS.items():
        reg.register_relaxation(name, factory)
    for name, factory in BUILTIN_FORMATTERS.items():
        reg.register_formatter(name, factory)
    return reg
