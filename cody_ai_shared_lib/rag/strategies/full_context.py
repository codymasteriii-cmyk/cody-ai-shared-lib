"""Tier 0 FullContextStrategy (§8.2, AC-RAG-10, AC-RAG-11, D-25).

Used when a document or filtered corpus fits within full_context_max_tokens.
Fetches the entire scope in document order with zero embedding calls.
"""
from __future__ import annotations

from typing import ClassVar, Generic

from ..errors import StoreCapabilityError
from ..granularity import GRANULARITY_FULL_DOC
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy


class FullContextStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "full_context"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_full_scope_fetch"])

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        if not self.is_runnable(ctx.store.capabilities()):
            raise StoreCapabilityError("FullContextStrategy requires store capability 'supports_full_scope_fetch'")
        nodes = ctx.store.fetch_full_scope(ctx.filters)
        for node in nodes:
            node.granularity = GRANULARITY_FULL_DOC  # §8.2: Tier 0 text is a whole, not ranked candidates
        return nodes
