"""Deduplication post-retrieval stages (§9.2, §9.3, AC-RAG-32, AC-RAG-33)."""
from __future__ import annotations

from typing import Any, Callable, Generic
from ..granularity import TRAIT_OFFSET_ADDRESSABLE, has_trait
from ..models import MetaT, RAGNode


class IdentityDeduplicator(Generic[MetaT]):
    """Deduplicates by key (default node id or url), preserving highest score (AC-RAG-33)."""

    def __init__(self, key: str | Callable[[RAGNode[MetaT]], Any] | None = None) -> None:
        self.key = key

    def _extract_key(self, node: RAGNode[MetaT]) -> Any:
        if callable(self.key):
            return self.key(node)
        if isinstance(self.key, str):
            # Try metadata attribute or dict key, fallback to node attribute
            meta = node.metadata
            if meta is not None:
                if isinstance(meta, dict) and self.key in meta:
                    return meta[self.key]
                if hasattr(meta, self.key):
                    val = getattr(meta, self.key)
                    if val is not None:
                        return val
            if hasattr(node, self.key):
                return getattr(node, self.key)
            return node.id
        return node.id

    def deduplicate(self, nodes: list[RAGNode[MetaT]]) -> list[RAGNode[MetaT]]:
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

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold

    def _priority(self, node: RAGNode[MetaT]) -> tuple[int, float]:
        gran = node.granularity
        if gran == "full_doc":
            p = 3
        elif gran == "local_window":
            p = 2
        else:
            p = 1
        score = node.score if node.score is not None else 0.0
        return (p, score)

    def deduplicate(self, nodes: list[RAGNode[MetaT]]) -> list[RAGNode[MetaT]]:
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
                if acc.span and acc.document_id == cand.document_id:
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
        accepted_ids = {n.id for n in accepted}
        return [n for n in nodes if n.id in accepted_ids]
