"""Tier 0 FullContextStrategy (§8.2, AC-RAG-10, AC-RAG-11, D-25).

Used when document or filtered corpus fits within full_context_max_tokens gate.
Fetches entire scope in document order with zero embedding calls.
"""
from __future__ import annotations

from typing import ClassVar, Generic
from ..errors import StoreCapabilityError
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy


class FullContextStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "full_context"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_full_scope_fetch"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        caps = ctx.store.capabilities() if callable(ctx.store.capabilities) else ctx.store.capabilities
        if not caps.supports_full_scope_fetch:
            raise StoreCapabilityError(
                "FullContextStrategy requires store capability 'supports_full_scope_fetch'"
            )

        nodes = ctx.store.fetch_full_scope(ctx.filters)
        # Full context nodes are ordered as fetched (document reading order)
        return nodes
