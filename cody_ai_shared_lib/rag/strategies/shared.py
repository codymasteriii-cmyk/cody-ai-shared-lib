"""Query steps shared by the retrieval strategies (§8.3–§8.6)."""
from __future__ import annotations

from typing import Callable

from ..errors import EmbeddingProviderError
from ..granularity import is_ranked
from ..models import MetaT, RAGFilterSpec, RAGNode, TextRule
from ..text.analyzer import TokenGroup, tokenize_query_groups, widen_name_groups
from ..text.rules import resolve_group, satisfies
from .context import RetrievalContext


def query_text(ctx: RetrievalContext[MetaT]) -> str:
    return ctx.intent.search_query or ctx.request.question


def query_token_groups(ctx: RetrievalContext[MetaT]) -> list[TokenGroup] | None:
    """Analyzer tokens for stores that search them; None for stores that use raw text (§6.1)."""
    if not ctx.store.capabilities().uses_analyzer_tokens:
        return None
    return tokenize_query_groups(
        ctx.analyzer,
        query_text(ctx),
        ctx.intent.exact_terms,
        ctx.intent.entity_terms,
        ctx.intent.entity_mode,
    )


OVERFETCH = 4          # an engine-checked search asks the store for about this many times the candidates (§7.9)


def _rule_sets(rules: list[TextRule], applied_kinds: frozenset[str]) -> tuple[list[TextRule], list[TextRule]]:
    """Split rules into those the store applies and those the engine checks.

    An `any` set is kept whole on one side, because "at least one of" cannot be split between two checkers.
    """
    any_rules = [r for r in rules if r.group == "any"]
    any_in_store = all(r.kind in applied_kinds for r in any_rules)
    in_store = [r for r in rules if r.kind in applied_kinds and (r.group != "any" or any_in_store)]
    return in_store, [r for r in rules if r not in in_store]


def _checked_call(
    ctx: RetrievalContext[MetaT],
    top_k: int,
    rules: list[TextRule],
    groups: list[TokenGroup] | None,
    call: Callable[[RAGFilterSpec, list[TokenGroup] | None, int], list[RAGNode[MetaT]]],
) -> list[RAGNode[MetaT]]:
    """One store call under `rules`: the store applies what it declares, the engine checks the rest."""
    caps = ctx.store.capabilities()
    in_store, by_engine = _rule_sets(rules, caps.text_rule_kinds)
    filters = ctx.filters.model_copy(update={"text_rules": in_store})
    if not by_engine:
        return call(filters, groups, top_k)
    rows = call(filters, groups, min(max(top_k, top_k * OVERFETCH), caps.max_top_k))
    kept = [
        node for node in rows
        if not is_ranked(node) or satisfies(ctx.renderer.prompt_text(node), by_engine)
    ]
    for rule in by_engine:
        note = f"{rule.kind}:{rule.text}"
        if note not in ctx.rule_checks:
            ctx.rule_checks.append(note)
    return kept[:top_k]


def ranked_store_call(
    ctx: RetrievalContext[MetaT],
    top_k: int,
    call: Callable[[RAGFilterSpec, list[TokenGroup] | None, int], list[RAGNode[MetaT]]],
) -> list[RAGNode[MetaT]]:
    """Run one ranked search (vector, keyword or hybrid) with the request's text rules (§7.9).

    The search requires the names found in its own text; if the rows left after the store and the engine have
    applied the rules number fewer than the effective `top_k`, it widens to "any of those names". Narrow rows stay
    first and widened ones are appended without duplicates. Only `filter` rules reach the store; `prefer` rules
    never do. A token store also gets the names as token groups (§6.1).
    """
    groups = query_token_groups(ctx)
    narrow, wide = resolve_group([r for r in ctx.rules if r.strength == "filter"], query_text(ctx))
    wide_groups = widen_name_groups(groups) if groups else None
    rows = _checked_call(ctx, top_k, narrow, groups, call)
    if (wide is None and wide_groups is None) or len(rows) >= ctx.target_limit():
        return rows
    more = _checked_call(ctx, top_k, wide if wide is not None else narrow, wide_groups or groups, call)
    seen = {node.node_id for node in rows}
    return [*rows, *(node for node in more if node.node_id not in seen)]


def embed_query(ctx: RetrievalContext[MetaT]) -> list[float]:
    """Return the query vector, or an empty list when embedding failed."""
    ctx.deadline.raise_if_cancelled("retrieval_embedding")
    try:
        result = ctx.gateway.embed(
            texts=query_text(ctx),
            model=ctx.profile.models.embedding,
            stage="retrieval_embedding",
            usage_tracker=ctx.usage_tracker,
            degradation_tracker=ctx.degradation_tracker,
            budget_guard=ctx.budget_guard,
            task_type="RETRIEVAL_QUERY",
            use_cache=ctx.profile.cache.enable_embedding_cache,
            cache=ctx.cache,
            cache_ttl_seconds=ctx.profile.cache.query_embedding_ttl_seconds,
            fallbacks=ctx.profile.fallbacks,
        )
    except EmbeddingProviderError:
        return []
    return result.vectors[0] if result.vectors else []


def keyword_fallback(ctx: RetrievalContext[MetaT], top_k: int) -> list[RAGNode[MetaT]]:
    """Search by keyword after a failed query embedding when the app opted in; otherwise raise (DG-07)."""
    if ctx.profile.fallbacks.get("query_embedding") != "keyword" or not ctx.store.capabilities().supports_keyword_fts:
        raise EmbeddingProviderError(
            f"Query embedding failed or returned zero vector for model '{ctx.profile.models.embedding}'",
            stage="retrieval_embedding",
        )
    ctx.degradation_tracker.record(
        dg_id="DG-07",
        stage="retrieval_embedding",
        reason="Query embedding failed or returned zero vector",
        fallback_used="keyword_search",
        impact="Fell back to keyword-only retrieval",
    )
    return ranked_store_call(
        ctx,
        top_k,
        lambda filters, groups, k: ctx.store.keyword_search(
            query_text=query_text(ctx), top_k=k, filters=filters, analyzer_tokens=groups
        ),
    )


def fetch_window(ctx: RetrievalContext[MetaT], cursor: int | None) -> list[RAGNode[MetaT]]:
    """Fetch the local window around a reading position, when the profile and store support it."""
    radius = ctx.profile.retrieval.local_window_radius_chars
    if cursor is None or radius <= 0 or not ctx.store.capabilities().supports_local_window:
        return []
    doc_id = ctx.filters.doc_ids[0] if ctx.filters.doc_ids else ctx.request.document_id
    if not doc_id:
        return []
    return ctx.store.fetch_local_window(doc_id=doc_id, center_offset=cursor, radius_chars=radius, filters=ctx.filters)


def with_local_window(ctx: RetrievalContext[MetaT], results: list[RAGNode[MetaT]]) -> list[RAGNode[MetaT]]:
    """Put the window around the request's cursor ahead of the ranked results (§8.4)."""
    window = fetch_window(ctx, ctx.request.cursor_offset)
    seen = {n.node_id for n in window}
    return window + [n for n in results if n.node_id not in seen]
