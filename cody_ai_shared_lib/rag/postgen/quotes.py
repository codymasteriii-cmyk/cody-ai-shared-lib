"""Quote verification and sentence-level citation coverage (§11.3, §11.4, AC-RAG-42)."""
from __future__ import annotations

import difflib
import re
from typing import Generic

from ..models import CitationCoverage, MetaT, RAGCitation, RAGNode
from ..text.analyzer import CjkAwareAnalyzer, TextAnalyzer

# A quoted span in English ("...", “...”) or CJK (「...」, 『...』) followed by its citation marker
_QUOTED_CITE_RE = re.compile(r'(?:"([^"]{4,})"|“([^”]{4,})”|「([^」]{4,})」|『([^』]{4,})』)\s*(\[[^\]]+\])')
_NORM_WS_RE = re.compile(r"[\s\W_]+", flags=re.UNICODE)
_MIN_COVERED_SENTENCE_CHARS = 20


def _normalize_for_quote(text: str) -> str:
    return _NORM_WS_RE.sub(" ", text.lower()).strip()


def quote_match_ratio(quote: str, source: str) -> float:
    """Best similarity of the quote against the source window aligned to their longest shared run."""
    if not quote:
        return 0.0
    norm_quote, norm_source = _normalize_for_quote(quote), _normalize_for_quote(source)
    if norm_quote and norm_quote in norm_source:
        return 1.0

    longest = difflib.SequenceMatcher(None, norm_quote, norm_source, autojunk=False).find_longest_match(
        0, len(norm_quote), 0, len(norm_source)
    )
    start = max(0, longest.b - longest.a)
    window = norm_source[start:start + len(norm_quote)]
    return difflib.SequenceMatcher(None, norm_quote, window, autojunk=False).ratio()


def compute_coverage(
    citations: list[RAGCitation],
    answer_text: str,
    analyzer: TextAnalyzer,
) -> CitationCoverage:
    """A sentence is cited when a verified marker is in it or in the next sentence (§11.4)."""
    sentences = [answer_text[start:end].strip() for start, end in analyzer.sentences(answer_text)]
    verified = [c for c in citations if c.verified]
    has_marker = [any(c.marker in sentence for c in verified) for sentence in sentences]

    total = cited = 0
    uncited: list[str] = []
    for i, sentence in enumerate(sentences):
        bare = re.sub(r"\[[^\]]*\]", "", sentence).strip()
        if len(bare) < _MIN_COVERED_SENTENCE_CHARS or bare.startswith("#") or bare.endswith((":", "：")):
            continue
        total += 1
        if has_marker[i] or (i + 1 < len(sentences) and has_marker[i + 1]):
            cited += 1
        else:
            uncited.append(sentence)

    return CitationCoverage(
        sentences_total=total,
        sentences_cited=cited,
        ratio=(cited / total) if total else 1.0,
        uncited_sentences=uncited,
        total_citations=len(citations),
        verified_citations=len(verified),
    )


class QuoteValidator(Generic[MetaT]):
    """Checks that every quoted string next to a citation is really in the cited node (§11.3)."""

    def __init__(
        self,
        min_match_ratio: float = 0.85,
        analyzer: TextAnalyzer | None = None,
        threshold: float | None = None,
    ):
        self.min_match_ratio = threshold if threshold is not None else min_match_ratio
        self.analyzer = analyzer or CjkAwareAnalyzer()

    def verify(
        self,
        citations: list[RAGCitation],
        nodes: list[RAGNode[MetaT]],
        answer_text: str | None = None,
    ) -> tuple[list[RAGCitation], CitationCoverage]:
        node_text = {n.node_id: n.text for n in nodes}

        quotes_by_marker: dict[str, list[str]] = {}
        for match in _QUOTED_CITE_RE.finditer(answer_text or ""):
            quote = next(group for group in match.groups()[:4] if group)
            quotes_by_marker.setdefault(match.group(5), []).append(quote)

        for citation in citations:
            if not citation.verified or citation.node_id not in node_text:
                citation.verified = False
                citation.verification_score = 0.0
                continue
            source = node_text[citation.node_id]
            quotes = quotes_by_marker.get(citation.marker) or ([citation.quote] if citation.quote else [])
            if not quotes:
                citation.verified = False
                citation.verification_score = 0.0
                continue

            score = min(quote_match_ratio(quote, source) for quote in quotes)
            if citation.marker in quotes_by_marker:
                citation.quote = quotes[0]
            citation.verification_score = round(score, 4)
            citation.verified = score >= self.min_match_ratio

        return citations, compute_coverage(citations, answer_text or "", self.analyzer)
