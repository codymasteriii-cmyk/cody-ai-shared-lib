"""Reading part or all of one document within a token budget (OI-4, §8.13).

Shared by `FindThenReadStrategy` (read around each hit) and `query_batched` (read the documents a list needs).
Every read narrows the request's filters to the document, so a guard boundary only ever tightens.

Which read is used, cheapest sufficient first (each step is only tried when the one before does not fit):

    read_around_hits()                          read_whole_document()
    ├─ whole document, if it fits               ├─ nothing visible? -> an empty, complete read
    ├─ per hit: its section (by char range)     ├─ whole document, if it fits
    ├─ else a window around the hit             └─ else section by section, until the budget ends
    └─ else the hit passage as found

Dummy example: an article of 9,000 tokens, a budget of 4,000 left, hits in sections 2 and 5. The whole article
does not fit, so section 2 (1,200 tokens) is read, then section 5 (900), 2,100 tokens in all; the read is
complete because every hit got its section. Had section 5 been 3,000 tokens, a 1,500-token window around that
hit would be read instead and the read would be marked `cut`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from ..models import MetaT, RAGFilterSpec, RAGNode
from ..text.analyzer import node_tokens
from .context import RetrievalContext


@dataclass
class DocumentRead:
    """What was read from one document."""
    nodes: list[RAGNode[Any]] = field(default_factory=list)
    complete: bool = True     # every read this document needed fit the budget
    cut: bool = False         # a hit got less text around it than the store offers, for lack of budget


def scoped(ctx: RetrievalContext[MetaT], doc_id: str, **extra: Any) -> RAGFilterSpec:
    """The request's filters narrowed to one document (and a range): the guard boundary only ever tightens."""
    return ctx.filters.patch({"doc_ids": [doc_id], **extra})


def section_bounds(ctx: RetrievalContext[MetaT], doc_id: str) -> dict[int, tuple[int, int]]:
    """Character range of each section the store can describe, from its section summaries."""
    if not ctx.store.capabilities().supports_parent_summaries:
        return {}
    bounds: dict[int, tuple[int, int]] = {}
    for summary in ctx.store.fetch_parent_summaries(scoped(ctx, doc_id)):
        if summary.parent_kind in (None, "section"):
            bounds[summary.span.section_seq] = (summary.span.char_start, summary.span.char_end)
    return bounds


def _read_range(ctx: RetrievalContext[MetaT], doc_id: str, start: int, end: int) -> list[RAGNode[Any]]:
    """Every passage of one document between two character offsets (a section), under the guards' filters."""
    return list(ctx.store.fetch_full_scope(scoped(ctx, doc_id, min_char_offset=start, max_char_offset=end)))


def _read_window(ctx: RetrievalContext[MetaT], doc_id: str, center: int, radius: int) -> list[RAGNode[Any]]:
    """The passages within `radius` characters of one position: the cheap fallback when a section does not fit."""
    return list(
        ctx.store.fetch_local_window(doc_id=doc_id, center_offset=center, radius_chars=radius, filters=ctx.filters)
    )


def _whole_if_it_fits(ctx: RetrievalContext[MetaT], doc_id: str, remaining: int) -> list[RAGNode[Any]] | None:
    """The whole document if its size (from the store's stats, no text fetched yet) fits `remaining` tokens."""
    whole = scoped(ctx, doc_id)
    if 0 < ctx.store.scope_stats(whole).total_tokens <= remaining:
        return list(ctx.store.fetch_full_scope(whole))
    return None


def read_around_hits(
    ctx: RetrievalContext[MetaT], doc_id: str, hits: list[RAGNode[Any]], remaining: int
) -> DocumentRead:
    """The whole document when it fits; else, per hit, its section, a window, or the hit as found (§8.13 step 2).

    A read that does not fit what is left is replaced by the next cheaper one, and the read is marked `cut`;
    when not even the hit fits, reading stops and the read is not `complete`.
    """
    if remaining <= 0:
        return DocumentRead(complete=False)       # the budget is already spent: this document cannot be read at all
    whole = _whole_if_it_fits(ctx, doc_id, remaining)
    if whole is not None:
        return DocumentRead(nodes=whole)

    caps = ctx.store.capabilities()
    radius = ctx.profile.retrieval.list_read_radius_chars
    bounds = section_bounds(ctx, doc_id) if caps.supports_offset_range_scope else {}
    read = DocumentRead()
    done: set[int] = set()
    for hit in hits:
        seq = hit.span.section_seq
        if seq in done:
            continue                               # an earlier hit in the same section already read it
        # Cheapest sufficient read first: the section, then a window, then the hit as found
        attempts: list[Callable[[], list[RAGNode[Any]]]] = []
        if seq in bounds:
            attempts.append(partial(_read_range, ctx, doc_id, *bounds[seq]))
        if caps.supports_local_window:
            attempts.append(partial(_read_window, ctx, doc_id, hit.span.char_start, radius))
        attempts.append(partial(list, [hit]))
        part: list[RAGNode[Any]] | None = None
        for attempt in attempts:
            candidate = attempt()
            if not candidate:
                continue
            if node_tokens(read.nodes) + node_tokens(candidate) <= remaining:
                part = candidate
                break
            read.cut = True                    # a larger read existed but did not fit
        if part is None:
            read.complete = False                  # not even the hit passage fits: stop, and say reading was incomplete
            break
        read.nodes.extend(part)
        done.add(seq)
    return read


def read_whole_document(ctx: RetrievalContext[MetaT], doc_id: str, remaining: int) -> DocumentRead:
    """A document the app named: whole if it fits, else section by section, else not read (and said so).

    A document with no visible text (unknown, or wholly past the guard boundary) has nothing to read: an empty,
    complete read, so it does not stop the documents after it.
    """
    if ctx.store.scope_stats(scoped(ctx, doc_id)).total_tokens == 0:
        return DocumentRead()
    whole = _whole_if_it_fits(ctx, doc_id, remaining)
    if whole is not None:
        return DocumentRead(nodes=whole)
    bounds = section_bounds(ctx, doc_id) if ctx.store.capabilities().supports_offset_range_scope else {}
    # Without section bounds there is nothing cheaper than the whole document, so the read is incomplete from the start
    read = DocumentRead(complete=bool(bounds))
    for _, (start, end) in sorted(bounds.items()):
        part = list(ctx.store.fetch_full_scope(scoped(ctx, doc_id, min_char_offset=start, max_char_offset=end)))
        if node_tokens(read.nodes) + node_tokens(part) > remaining:
            read.complete = False
            break
        read.nodes.extend(part)
    return read
