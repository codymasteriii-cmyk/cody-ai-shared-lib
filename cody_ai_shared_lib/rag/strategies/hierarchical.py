"""Tier 3 HierarchicalSummaryStrategy (§8.6, AC-RAG-22, §5.2).

Combines parent summaries, local window text and detail passages. Under a sequential guard, the
store excludes the summary of the section containing the boundary; its unit summaries and local
window appear instead.
"""
from __future__ import annotations

from typing import ClassVar, Generic

from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy
from .dense_vector import DenseVectorStrategy
from .hybrid_rrf import HybridRRFStrategy
from .shared import fetch_window


def select_summary_layer(summaries: list[RAGNode[MetaT]]) -> list[RAGNode[MetaT]]:
    """Per section, the section summary when it was admitted, else that section's unit summaries (§6.3).

    The store admits a section summary only if it ends before the boundary, so a section with one is
    fully read and its units would repeat it. Document order is kept.
    """
    has_section_summary = {
        (n.span.doc_id, n.span.section_seq) for n in summaries if n.parent_kind != "section_unit"
    }
    return [
        n for n in summaries
        if n.parent_kind != "section_unit" or (n.span.doc_id, n.span.section_seq) not in has_section_summary
    ]


class HierarchicalSummaryStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "hierarchical"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_parent_summaries"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        caps = ctx.store.capabilities()
        if not self.is_runnable(caps):
            raise StoreCapabilityError("HierarchicalSummaryStrategy requires store capability 'supports_parent_summaries'")

        summaries = select_summary_layer(ctx.store.fetch_parent_summaries(ctx.filters))
        cursor =ctx.request.cursor_offset if ctx.request.cursor_offset is not None else ctx.filters.max_char_offset
        window = fetch_window(ctx, cursor)

        if HybridRRFStrategy.is_runnable(caps):
            passages = HybridRRFStrategy().retrieve(ctx)
        elif DenseVectorStrategy.is_runnable(caps):
            passages = DenseVectorStrategy().retrieve(ctx)
        else:
            passages = []

        combined: dict[str, RAGNode[MetaT]] = {}
        for node in [*summaries, *window, *passages]:
            combined.setdefault(node.node_id, node)
        return list(combined.values())
