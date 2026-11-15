"""Deduplication post-retrieval stages (§9.2, §9.3, AC-RAG-32, AC-RAG-33).

Two different questions, two stages:

    identity_dedup   "is this the same item?"   same key (node id, or an app field such as a URL) -> keep the best score
    offset_dedup     "is this the same text?"   overlapping character ranges in one document -> keep the broader one

Dummy example for offset_dedup: a window covering characters 100-400 and a passage covering 150-300 of the same
document. The passage lies fully inside the window (100% >= the 80% threshold), so only the window is kept; the model
would otherwise read the same sentences twice.
"""
from __future__ import annotations

from typing import Any, Callable, ClassVar, Generic
from ..granularity import GRANULARITY_FULL_DOC, GRANULARITY_LOCAL_WINDOW, TRAIT_OFFSET_ADDRESSABLE, has_trait
from ..models import MetaT, RAGNode


class IdentityDeduplicator(Generic[MetaT]):
    """Deduplicates by key (default node id or url), preserving highest score (AC-RAG-33)."""
    name: ClassVar[str] = "identity_dedup"
    version: ClassVar[str] = "1.0.0"
    accepts_score_kinds: ClassVar[frozenset[str] | None] = None

    def process(self, nodes: list[RAGNode[MetaT]], ctx: Any) -> list[RAGNode[MetaT]]:
        """Stage entry point."""
        return self.deduplicate(nodes)

    def __init__(self, key: str | Callable[[RAGNode[MetaT]], Any] | None = None) -> None:
        self.key = key

    def _extract_key(self, node: RAGNode[MetaT]) -> Any:
        """The identity of a node: the app's function, or a named metadata field (e.g. "url"), else the node id."""
        if callable(self.key):
            return self.key(node)
        if isinstance(self.key, str):
            meta = node.metadata
            value = meta.get(self.key) if isinstance(meta, dict) else getattr(meta, self.key, None)
            if value is None:
                value = (getattr(meta, "extra", None) or {}).get(self.key)
            if value is not None:
                return value
        return node.node_id

    def deduplicate(self, nodes: list[RAGNode[MetaT]]) -> list[RAGNode[MetaT]]:
        """Keep one node per key: the highest-scored, in the position where the key first appeared."""
        seen: dict[Any, RAGNode[MetaT]] = {}
        ordered_keys: list[Any] = []

        for node in nodes:
            k = self._extract_key(node)
            if k not in seen:
                seen[k] = node
                ordered_keys.append(k)
            else:
                existing = seen[k]
                existing_score = existing.score if existing.score is not None else -1e9
                curr_score = node.score if node.score is not None else -1e9
                if curr_score > existing_score:
                    seen[k] = node

        return [seen[k] for k in ordered_keys]


class OffsetDeduplicator(Generic[MetaT]):
    """Deduplicates overlapping spans within same document (AC-RAG-32, RAG-RET-2).

    Removes any node at least 80% covered by an already-kept node with higher priority.
    Priority order: full_doc > local_window > higher-scored passage.
    """
    name: ClassVar[str] = "offset_dedup"
    version: ClassVar[str] = "1.0.0"
    accepts_score_kinds: ClassVar[frozenset[str] | None] = None

    def process(self, nodes: list[RAGNode[MetaT]], ctx: Any) -> list[RAGNode[MetaT]]:
        """Stage entry point."""
        return self.deduplicate(nodes)

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold

    def _priority(self, node: RAGNode[MetaT]) -> tuple[int, float]:
        """Which node survives an overlap: a whole document beats a window, a window beats a passage, then score."""
        gran = node.granularity
        if gran == GRANULARITY_FULL_DOC:
            p = 3
        elif gran == GRANULARITY_LOCAL_WINDOW:
            p = 2
        else:
            p = 1
        score = node.score if node.score is not None else 0.0
        return (p, score)

    def deduplicate(self, nodes: list[RAGNode[MetaT]]) -> list[RAGNode[MetaT]]:
        """Drop each node that is at least `threshold` covered by a higher-priority node of the same document.

        Nodes without character offsets (summaries, entity facts) cannot overlap by position and are always kept.
        """
        if not nodes:
            return []

        # Sort by (priority, score) descending to evaluate higher priority nodes first
        sorted_nodes = sorted(
            nodes,
            key=self._priority,
            reverse=True,
        )

        accepted: list[RAGNode[MetaT]] = []
        for cand in sorted_nodes:
            if not cand.span or not has_trait(cand.granularity, TRAIT_OFFSET_ADDRESSABLE):
                accepted.append(cand)
                continue

            c_start = cand.span.char_start
            c_end = cand.span.char_end
            if c_start is None or c_end is None or c_end <= c_start:
                accepted.append(cand)
                continue

            c_len = c_end - c_start
            covered = False

            for acc in accepted:
                if acc.span and acc.span.doc_id == cand.span.doc_id:
                    a_start = acc.span.char_start
                    a_end = acc.span.char_end
                    if a_start is not None and a_end is not None and a_end > a_start:
                        overlap_start = max(c_start, a_start)
                        overlap_end = min(c_end, a_end)
                        if overlap_end > overlap_start:
                            overlap_len = overlap_end - overlap_start
                            if (overlap_len / c_len) >= self.threshold:
                                covered = True
                                break

            if not covered:
                accepted.append(cand)

        # Restore original input order of accepted nodes
        accepted_ids = {n.node_id for n in accepted}
        return [n for n in nodes if n.node_id in accepted_ids]
