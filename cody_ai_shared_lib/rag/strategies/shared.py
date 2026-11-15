"""Query steps shared by the retrieval strategies (§8.3–§8.6)."""
from __future__ import annotations

from typing import Callable

from ..errors import EmbeddingProviderError
from ..models import MetaT, RAGNode
from ..text.analyzer import TokenGroup, tokenize_query_groups, widen_name_groups
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


def search_widening_names(
    ctx: RetrievalContext[MetaT],
    call: Callable[[list[TokenGroup] | None], list[RAGNode[MetaT]]],
) -> list[RAGNode[MetaT]]:
    """Run one keyword-bearing store call; widen "all these names" to "any of them" when it returns too few (§7.9).

    "Too few" is fewer rows than the request's effective top_k, counted on this one call's rows. The narrow rows
    stay first and the widened ones are appended without duplicates. A search with fewer than two required names
    never widens.
    """
    groups = query_token_groups(ctx)
    rows = call(groups)
    wide = widen_name_groups(groups) if groups else None
    if wide is None or len(rows) >= ctx.target_limit():
        return rows
    seen = {node.node_id for node in rows}
    return [*rows, *(node for node in call(wide) if node.node_id not in seen)]


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
    return search_widening_names(
        ctx,
        lambda groups: ctx.store.keyword_search(
            query_text=query_text(ctx), top_k=top_k, filters=ctx.filters, analyzer_tokens=groups
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
