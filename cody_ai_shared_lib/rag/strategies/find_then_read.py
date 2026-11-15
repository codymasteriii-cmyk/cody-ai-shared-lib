"""FindThenReadStrategy: find the documents a list question touches, then read the matching text whole (OI-4).

A list question ("list every model mentioned this year") needs the passages around each hit, not the few best
passages. The strategy runs the profile's ordinary search to find hits, groups them by document in rank order,
and reads each document as cheaply as the store allows:

  1. the whole document, when it fits the remaining Tier 0 budget;
  2. else the section(s) holding the hits, by character range (sections come from the section summaries; the
     range is applied together with any sequential-guard boundary, so a reader never sees ahead);
  3. else a window around the hit;
  4. else the hit passage as found.

The result is Tier 0 style text. A ranked finder can never prove it saw every match, so this strategy never
reports a complete scope (§7.9 boundaries).
"""
from __future__ import annotations

from typing import Any, ClassVar, Generic

from ..errors import StoreCapabilityError
from ..granularity import FLAG_FULL_SCOPE, GRANULARITY_FULL_DOC, TRAIT_ATOMIC, has_trait, is_ranked
from ..models import MetaT, RAGFilterSpec, RAGNode
from ..text.analyzer import estimate_tokens
from .context import RetrievalContext, RetrievalStrategy
from .dense_vector import DenseVectorStrategy
from .hybrid_rrf import HybridRRFStrategy
from .keyword import KeywordOnlyStrategy


def _tokens(nodes: list[RAGNode[Any]]) -> int:
    return sum(n.token_estimate if n.token_estimate else estimate_tokens(n.text) for n in nodes)


class FindThenReadStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "find_then_read"
    version: ClassVar[str] = "1.0.0"
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
        for hit in self._find(ctx):
            if is_ranked(hit):
                by_doc.setdefault(hit.span.doc_id, []).append(hit)
        documents = list(by_doc)
        capped = len(documents) > cfg.list_max_documents
        documents = documents[: cfg.list_max_documents]

        read: dict[str, RAGNode[MetaT]] = {}
        remaining = cfg.full_context_max_tokens
        stopped = False
        for doc_id in documents:
            nodes, complete = self._read_document(ctx, doc_id, by_doc[doc_id], remaining)
            for node in nodes:
                if node.node_id not in read:
                    read[node.node_id] = node
            remaining -= _tokens(nodes)
            if not complete:
                stopped = True
                break

        ctx.truncated = ctx.truncated or capped or stopped
        for node in read.values():
            # Tier 0 style: read text is the scope, never ranked candidates; atomic kinds keep their granularity
            if not has_trait(node.granularity, TRAIT_ATOMIC):
                node.granularity = GRANULARITY_FULL_DOC
            if FLAG_FULL_SCOPE not in node.flags:
                node.flags.append(FLAG_FULL_SCOPE)
        return list(read.values())

    # ── find ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find(ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
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

    # ── read ──────────────────────────────────────────────────────────────

    @staticmethod
    def _scoped(ctx: RetrievalContext[MetaT], doc_id: str, **extra: Any) -> RAGFilterSpec:
        """The request's filters narrowed to one document (and a range): the guard boundary only ever tightens."""
        return ctx.filters.patch({"doc_ids": [doc_id], **extra})

    def _read_document(
        self, ctx: RetrievalContext[MetaT], doc_id: str, hits: list[RAGNode[MetaT]], remaining: int
    ) -> tuple[list[RAGNode[MetaT]], bool]:
        """The text to read for one document, and whether all of it fit the budget."""
        caps = ctx.store.capabilities()
        cfg = ctx.profile.retrieval
        if remaining <= 0:
            return [], False

        whole = self._scoped(ctx, doc_id)
        if 0 < ctx.store.scope_stats(whole).total_tokens <= remaining:
            return ctx.store.fetch_full_scope(whole), True

        out: list[RAGNode[MetaT]] = []
        complete = True
        bounds = self._section_bounds(ctx, doc_id) if caps.supports_offset_range_scope else {}
        done: set[int] = set()
        for hit in hits:
            seq = hit.span.section_seq
            if seq in done:
                continue
            # Cheapest sufficient read that fits what is left: the section, then a window, then the hit as found
            attempts: list[Any] = []
            if seq in bounds:
                start, end = bounds[seq]
                attempts.append(
                    lambda start=start, end=end: ctx.store.fetch_full_scope(
                        self._scoped(ctx, doc_id, min_char_offset=start, max_char_offset=end)
                    )
                )
            if caps.supports_local_window:
                attempts.append(
                    lambda hit=hit: ctx.store.fetch_local_window(
                        doc_id=doc_id, center_offset=hit.span.char_start,
                        radius_chars=cfg.list_read_radius_chars, filters=ctx.filters,
                    )
                )
            attempts.append(lambda hit=hit: [hit])
            part = None
            for attempt in attempts:
                candidate = attempt()
                if candidate and _tokens(out) + _tokens(candidate) <= remaining:
                    part = candidate
                    break
            if part is None:
                complete = False
                break
            out.extend(part)
            done.add(seq)
        return out, complete

    def _section_bounds(self, ctx: RetrievalContext[MetaT], doc_id: str) -> dict[int, tuple[int, int]]:
        """Character range of each section the store can describe, from its section summaries."""
        if not ctx.store.capabilities().supports_parent_summaries:
            return {}
        bounds: dict[int, tuple[int, int]] = {}
        for summary in ctx.store.fetch_parent_summaries(self._scoped(ctx, doc_id)):
            if summary.parent_kind in (None, "section"):
                bounds[summary.span.section_seq] = (summary.span.char_start, summary.span.char_end)
        return bounds
