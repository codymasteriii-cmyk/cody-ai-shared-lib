"""Quote verification and coverage (§11.4, AC-RAG-42)."""
from __future__ import annotations

import difflib
from typing import Generic
from ..models import CitationCoverage, MetaT, RAGCitation, RAGNode


class QuoteVerifier(Generic[MetaT]):
    """Verifies cited quotes match retrieved source node text."""

    def __init__(self, min_match_ratio: float = 0.85):
        self.min_match_ratio = min_match_ratio

    def verify(
        self,
        citations: list[RAGCitation],
        nodes: list[RAGNode[MetaT]],
    ) -> tuple[list[RAGCitation], CitationCoverage]:
        """Verify each citation quote against node text."""
        node_map = {n.id: n.text for n in nodes}
        verified_count = 0

        for cite in citations:
            if not cite.quote or cite.node_id not in node_map:
                cite.verified = False
                continue

            node_text = node_map[cite.node_id]
            # Exact check
            if cite.quote in node_text:
                cite.verified = True
                verified_count += 1
                continue

            # Fuzzy match check
            matcher = difflib.SequenceMatcher(None, cite.quote.lower(), node_text.lower())
            ratio = matcher.quick_ratio()
            if ratio >= self.min_match_ratio:
                cite.verified = True
                verified_count += 1
            else:
                cite.verified = False

        total = len(citations)
        ratio = verified_count / total if total > 0 else 1.0

        coverage = CitationCoverage(
            total_citations=total,
            verified_citations=verified_count,
            coverage_ratio=ratio,
        )
        return citations, coverage
