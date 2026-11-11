"""Tier 0 FullContextStrategy (§8.2, AC-RAG-10, AC-RAG-11, D-25).

Used when a document or filtered corpus fits within full_context_max_tokens.
Fetches the entire scope in document order with zero embedding calls.
"""
from __future__ import annotations

from typing import ClassVar, Generic

from ..errors import StoreCapabilityError
from ..granularity import FLAG_FULL_SCOPE, GRANULARITY_FULL_DOC, TRAIT_ATOMIC, has_trait
from ..models import MetaT, RAGNode
from .context import RetrievalContext, RetrievalStrategy


class FullContextStrategy(RetrievalStrategy[MetaT], Generic[MetaT]):
    name: ClassVar[str] = "full_context"
    version: ClassVar[str] = "1.0.0"
    requires: ClassVar[frozenset[str]] = frozenset(["supports_full_scope_fetch"])
    query_independent: ClassVar[bool] = True       # the whole scope, whatever the question

    def retrieve(self, ctx: RetrievalContext[MetaT]) -> list[RAGNode[MetaT]]:
        if not self.is_runnable(ctx.store.capabilities()):
            raise StoreCapabilityError("FullContextStrategy requires store capability 'supports_full_scope_fetch'")
        nodes = ctx.store.fetch_full_scope(ctx.filters)
        for node in nodes:
            # §8.2: Tier 0 text is the whole scope, never ranked candidates. Text kinds become full_doc;
            # atomic kinds (an article, a KB entry) keep their granularity, so a budget drops them whole.
            if not has_trait(node.granularity, TRAIT_ATOMIC):
                node.granularity = GRANULARITY_FULL_DOC
            if FLAG_FULL_SCOPE not in node.flags:
                node.flags.append(FLAG_FULL_SCOPE)
        return nodes
