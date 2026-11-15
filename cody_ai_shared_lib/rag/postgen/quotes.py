"""Quote verification and sentence-level citation coverage (§11.3, §11.4, AC-RAG-42).

Two separate checks on an answer, both without a model call:

  * quotes: when the answer quotes words next to a citation ("he said "we leave at dawn" [3]"), those words must really occur
    in passage 3, allowing for small differences. A quote that is not in its source is a misquote; the citation is unverified.
  * coverage: what share of the answer's sentences carry a verified citation. It is reported, not enforced
    (`min_citation_coverage` is not enforced in this version).

Quote checking is opt-in (`citations.verify_quotes`), because it only helps when the model quotes; KnowledgeBase turns it on.
"""
from __future__ import annotations

import difflib
import re
from typing import Generic

from ..models import CitationCoverage, MetaT, RAGCitation, RAGNode
from ..text.analyzer import CjkAwareAnalyzer, TextAnalyzer

# A quoted span in English ("...", “...”) or CJK (「...」, 『...』) followed by its citation marker
# ASCII quotes need 4+ characters; CJK quotes may be 2+, since a name or term is often that short
_QUOTED_CITE_RE = re.compile(r'(?:"([^"]{4,})"|“([^”]{2,})”|「([^」]{2,})」|『([^』]{2,})』)\s*(\[[^\]]+\])')
_NUMBER_RE = re.compile(r"\d+")
_NORM_WS_RE = re.compile(r"[\s\W_]+", flags=re.UNICODE)
_MIN_COVERED_SENTENCE_CHARS = 20


def _normalize_for_quote(text: str) -> str:
    """Lower-case and collapse punctuation and spacing, so "Don't  go!" and "don t go" compare equal."""
    return _NORM_WS_RE.sub(" ", text.lower()).strip()


def quote_match_ratio(quote: str, source: str) -> float:
    """Best similarity of the quote against the source window aligned to their longest shared run.

    1.0 when the normalized quote occurs verbatim; otherwise the similarity (0 to 1) of the best-aligned stretch of the source.

    >>> quote_match_ratio("founded in 1999", "The company was founded in 1999.")
    1.0
    >>> quote_match_ratio("completely unrelated words", "The company was founded in 1999.") < 0.85
    True
    """
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
    """A sentence is cited when a verified marker is in it or in the next sentence (§11.4).

    "The next sentence" is allowed because models often put the marker after a sentence's full stop. Headings, very short
    sentences (under 20 characters) and sentences that end in a colon are not counted: they make no claim to cite.
    """
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
    """Checks that every quoted string next to a citation is really in the cited node (§11.3).

    A quote passes when its best match in the source scores at least `min_match_ratio` (default 0.85), which tolerates
    small differences such as punctuation or a changed word, but not an invented sentence.
    """

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
        """Verify each citation's quotes and return the citations with coverage computed on the answer.

        A citation with no quote next to it is left as it was: comparing a source with itself would always pass and
        prove nothing. A citation whose node was not shown is unverified.
        """
        node_text = {n.node_id: n.text for n in nodes}

        quotes_by_marker: dict[str, list[str]] = {}
        for match in _QUOTED_CITE_RE.finditer(answer_text or ""):
            quote = next(group for group in match.groups()[:4] if group)
            raw = match.group(5)
            # "[1, 2]" quotes for both [1] and [2]; other marker styles are keyed as written
            keys = [f"[{n}]" for n in _NUMBER_RE.findall(raw)] if re.fullmatch(r"\[[\d,\s]+\]", raw) else [raw]
            for key in keys:
                quotes_by_marker.setdefault(key, []).append(quote)

        for citation in citations:
            if not citation.verified or citation.node_id not in node_text:
                citation.verified = False
                citation.verification_score = 0.0
                continue
            source = node_text[citation.node_id]
            quotes = quotes_by_marker.get(citation.marker)
            if not quotes:
                # The answer quotes nothing next to this marker: the quote check does not apply, and the
                # citation keeps its marker verification. Comparing the source with its own excerpt would
                # always "pass" and prove nothing.
                continue

            score = min(quote_match_ratio(quote, source) for quote in quotes)
            if citation.marker in quotes_by_marker:
                citation.quote = quotes[0]
            citation.verification_score = round(score, 4)
            citation.verified = score >= self.min_match_ratio

        return citations, compute_coverage(citations, answer_text or "", self.analyzer)
