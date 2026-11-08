"""Tier 3 HierarchicalStrategy (§8.6, AC-RAG-22, §5.2).

Combines parent summaries, local window text, and detail passages.
Under a sequential guard, the section containing the boundary does not appear
as a section summary (its unit summaries and local window appear instead).
"""
from __future__ import annotations

from typing import ClassVar, Generic
from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy
from .dense_vector import DenseVectorStrategy
from .hybrid_rrf import HybridRRFStrategy


class HierarchicalStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "hierarchical"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_parent_summaries"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        caps = ctx.store.capabilities() if callable(ctx.store.capabilities) else ctx.store.capabilities
        if not caps.supports_parent_summaries:
            raise StoreCapabilityError(
                "HierarchicalStrategy requires store capability 'supports_parent_summaries'"
            )

        # 1. Fetch parent summaries
        all_summaries = ctx.store.fetch_parent_summaries(ctx.filters)
        # Apply boundary filtering: section containing boundary B never appears as whole section summary (§5.2)
        boundary = ctx.filters.max_char_offset
        summaries: list[RAGNode[MetaT]] = []
        for s in all_summaries:
            if boundary is not None and s.span.char_end > boundary:
                continue
            summaries.append(s)

        # 2. Fetch local window if cursor and capability are present
        local_window_nodes: list[RAGNode[MetaT]] = []
        radius = ctx.profile.retrieval.local_window_radius_chars
        if caps.supports_local_window and radius > 0:
            doc_id = ctx.filters.doc_ids[0] if ctx.filters.doc_ids else (ctx.request.document_id or "")
            cursor = None
            if boundary is not None:
                cursor = boundary
            elif ctx.request.metadata and "cursor" in ctx.request.metadata:
                cursor = ctx.request.metadata["cursor"]

            if doc_id and cursor is not None:
                local_window_nodes = ctx.store.fetch_local_window(
                    doc_id=doc_id,
                    center_offset=cursor,
                    radius_chars=radius,
                    filters=ctx.filters,
                )

        # 3. Retrieve detail passages via Tier 2 (or Tier 1)
        if caps.supports_vector_search and caps.supports_keyword_fts:
            passages = HybridRRFStrategy().retrieve(ctx)
        elif caps.supports_vector_search:
            passages = DenseVectorStrategy().retrieve(ctx)
        else:
            passages = []

        # 4. Return combined nodes (summaries, local window, detail passages)
        return summaries + local_window_nodes + passages
