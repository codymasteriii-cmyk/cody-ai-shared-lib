"""FindThenReadStrategy: find the documents a list question touches, then read the matching text whole (OI-4).

A list question ("list every model mentioned this year") needs the passages around each hit, not the few best
passages. The strategy runs the profile's ordinary search to find hits, groups them by document in rank order,
and reads each document as cheaply as the store allows (`reading.read_around_hits`):

  1. the whole document, when it fits the remaining Tier 0 budget;
  2. else the section(s) holding the hits, by character range (sections come from the section summaries; the
     range is applied together with any sequential-guard boundary, so a reader never sees ahead);
  3. else a window around the hit;
  4. else the hit passage as found.

The result is Tier 0 style text. A ranked finder can never prove it saw every match, so the best this strategy
reports is "searched" (the text around every hit was read, nothing capped or cut), never "whole_scope" (§8.13).
"""
from __future__ import annotations

from typing import Any, ClassVar, Generic

from ..errors import StoreCapabilityError
from ..granularity import FLAG_FULL_SCOPE, GRANULARITY_FULL_DOC, TRAIT_ATOMIC, has_trait, is_ranked
from ..models import MetaT, RAGNode
from ..text.analyzer import node_tokens
from .context import RetrievalContext, RetrievalStrategy
from .dense_vector import DenseVectorStrategy
from .hybrid_rrf import HybridRRFStrategy
from .keyword import KeywordOnlyStrategy
from .reading import read_around_hits


class FindThenReadStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "find_then_read"
    version: ClassVar[str] = "1.1.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_full_scope_fetch"])

    @classmethod
    def is_runnable(cls, capabilities: Any) -> bool:
        return bool(capabilities.supports_full_scope_fetch) and (
            HybridRRFStrategy.is_runnable(capabilities)
            or capabilities.supports_vector_search
            or capabilities.supports_keyword_fts
        )

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        caps = ctx.store.capabilities()
        if not self.is_runnable(caps):
            raise StoreCapabilityError("FindThenReadStrategy requires a searchable store with 'supports_full_scope_fetch'")
        cfg = ctx.profile.retrieval

        by_doc: dict[str, list[RAGNode[MetaT]]] = {}
        for hit in find_hits(ctx):
            if is_ranked(hit):
                by_doc.setdefault(hit.span.doc_id, []).append(hit)
        documents = list(by_doc)
        capped = len(documents) > cfg.list_max_documents
        documents = documents[: cfg.list_max_documents]

        read: dict[str, RAGNode[MetaT]] = {}
        remaining = cfg.full_context_max_tokens
        stopped = cut = False
        for doc_id in documents:
            part = read_around_hits(ctx, doc_id, list(by_doc[doc_id]), remaining)
            for node in part.nodes:
                if node.node_id not in read:
                    read[node.node_id] = node
            remaining -= node_tokens(part.nodes)
            cut = cut or part.cut
            if not part.complete:
                stopped = True
                break

        ctx.truncated = ctx.truncated or capped or stopped
        ctx.scope_searched = not (capped or stopped or cut)
        for node in read.values():
            # Tier 0 style: read text is the scope, never ranked candidates; atomic kinds keep their granularity
            if not has_trait(node.granularity, TRAIT_ATOMIC):
                node.granularity = GRANULARITY_FULL_DOC
            if FLAG_FULL_SCOPE not in node.flags:
                node.flags.append(FLAG_FULL_SCOPE)
        return list(read.values())


def find_hits(ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
    """The profile's ordinary search, sized like a feed search so more than a few documents come back."""
    caps = ctx.store.capabilities()
    search: RetrievalStrategy[MetaT]
    if HybridRRFStrategy.is_runnable(caps):
        search = HybridRRFStrategy()
    elif caps.supports_vector_search:
        search = DenseVectorStrategy()
    else:
        search = KeywordOnlyStrategy()
    wide = ctx.model_copy(
        update={"mode": "retrieve", "pool": None, "request": ctx.request.model_copy(update={"top_k_override": None})}
    )
    return search.retrieve(wide)
