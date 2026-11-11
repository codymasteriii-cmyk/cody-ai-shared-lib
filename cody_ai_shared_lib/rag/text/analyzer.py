"""Text analysis, tokenization, sentence splitting, and token estimation.

Provides:
  - TextAnalyzer Protocol
  - CjkAwareAnalyzer default implementation (bigrams for CJK, lower-cased words for Latin)
  - estimate_tokens: 1.0 token/CJK char, 0.25 token/other char, +10% (from cody_ai_shared_lib.tokens)
  - TokenGroup & build_tsquery for PostgreSQL full-text search query construction
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal, Protocol, Sequence, runtime_checkable
from pydantic import BaseModel

from ...tokens import estimate_tokens, is_cjk_char  # estimate_tokens is public here (§2.5)


@runtime_checkable
class TextAnalyzer(Protocol):
    """Protocol for text tokenization, normalization and sentence boundary detection."""
    name: str
    version: str

    def normalize(self, text: str) -> str:
        """Normalize text: NFKC, case-fold, width-fold."""
        ...

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into search tokens."""
        ...

    def sentences(self, text: str) -> list[tuple[int, int]]:
        """Return half-open offsets (start, end) for sentences."""
        ...

    def last_sentence_end(self, text: str) -> int | None:
        """Return offset just past the last complete sentence in text, or None."""
        ...


# Closing punctuation that attaches to preceding sentence terminator
_CLOSING_PUNCT = set('"\'"\'”’」』）)》]')
# Chinese/CJK terminators that end a sentence wherever they appear (no trailing space required)
_CJK_TERMINATORS = ("。", "！", "？", "；", "……")
# Latin terminators that require space, closing quote/bracket, or end-of-text to avoid splitting 3.14 / urls
_LATIN_TERMINATORS = (".", "!", "?")


class CjkAwareAnalyzer:
    """Default text analyzer supporting Latin words, CJK bigrams, and sentence detection."""
    name: str = "cjk_aware"
    version: str = "1.0.0"

    def normalize(self, text: str) -> str:
        """Normalize string using NFKC normalization."""
        if not text:
            return ""
        return unicodedata.normalize("NFKC", text)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into search tokens.

        CJK runs become overlapping character bigrams (or single char if length 1).
        Latin runs become lower-cased words.
        Punctuation and whitespace are omitted.
        """
        if not text:
            return []

        tokens: list[str] = []
        # Group text into contiguous CJK vs non-CJK runs
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if is_cjk_char(ch):
                # Gather CJK run
                cjk_start = i
                while i < n and is_cjk_char(text[i]):
                    i += 1
                cjk_run = text[cjk_start:i]
                if len(cjk_run) == 1:
                    tokens.append(cjk_run)
                else:
                    for j in range(len(cjk_run) - 1):
                        tokens.append(cjk_run[j : j + 2])
            elif ch.isalnum():
                # Gather alphanumeric run
                latin_start = i
                while i < n and text[i].isalnum() and not is_cjk_char(text[i]):
                    i += 1
                latin_word = text[latin_start:i].lower()
                if latin_word:
                    tokens.append(latin_word)
            else:
                i += 1

        return tokens

    def sentences(self, text: str) -> list[tuple[int, int]]:
        """Return half-open char offset intervals (start, end) for sentences."""
        if not text:
            return []

        ranges: list[tuple[int, int]] = []
        n = len(text)
        sent_start = 0

        # Skip leading whitespace
        while sent_start < n and text[sent_start].isspace():
            sent_start += 1
        if sent_start >= n:
            return []

        i = sent_start
        while i < n:
            # Check double newline (paragraph break)
            if text[i : i + 2] == "\n\n" or text[i : i + 4] == "\r\n\r\n":
                # Sentence ends before \n\n
                sent_end = i
                if sent_end > sent_start:
                    ranges.append((sent_start, sent_end))
                # Skip whitespace to next sentence
                i += 2
                while i < n and text[i].isspace():
                    i += 1
                sent_start = i
                continue

            # Check CJK terminators
            is_cjk_term = False
            for cjk_term in _CJK_TERMINATORS:
                if text.startswith(cjk_term, i):
                    term_len = len(cjk_term)
                    # Absorb any trailing repeated terminators
                    while i + term_len < n and any(text.startswith(t, i + term_len) for t in _CJK_TERMINATORS):
                        matched = next(t for t in _CJK_TERMINATORS if text.startswith(t, i + term_len))
                        term_len += len(matched)
                    # Absorb trailing closing quotes / brackets
                    while i + term_len < n and text[i + term_len] in _CLOSING_PUNCT:
                        term_len += 1
                    sent_end = i + term_len
                    ranges.append((sent_start, sent_end))
                    i = sent_end
                    while i < n and text[i].isspace():
                        i += 1
                    sent_start = i
                    is_cjk_term = True
                    break

            if is_cjk_term:
                continue

            # Check Latin terminators (. ! ?)
            if text[i] in _LATIN_TERMINATORS:
                # Terminator rule: only split if followed by space, closing punct + space/EOF, or EOF
                term_len = 1
                # Absorb repeated terminators (e.g. ...)
                while i + term_len < n and text[i + term_len] in _LATIN_TERMINATORS:
                    term_len += 1
                # Absorb closing quotes
                while i + term_len < n and text[i + term_len] in _CLOSING_PUNCT:
                    term_len += 1

                # Check if this qualifies as a sentence end: followed by EOF or whitespace
                is_end = (i + term_len >= n) or text[i + term_len].isspace()
                if is_end:
                    sent_end = i + term_len
                    ranges.append((sent_start, sent_end))
                    i = sent_end
                    while i < n and text[i].isspace():
                        i += 1
                    sent_start = i
                    continue
                else:
                    # e.g., 3.14 or domain.com: do not split
                    i += term_len
                    continue

            i += 1

        # Any remaining tail
        if sent_start < n:
            tail_end = n
            # Strip trailing whitespace from sentence end
            while tail_end > sent_start and text[tail_end - 1].isspace():
                tail_end -= 1
            if tail_end > sent_start:
                ranges.append((sent_start, tail_end))

        return ranges

    def last_sentence_end(self, text: str) -> int | None:
        """Return offset just past the last complete sentence in text, or None if no complete sentence."""
        sents = self.sentences(text)
        if not sents:
            return None

        # Check if the last sentence actually terminated with sentence punctuation or \n\n,
        # or if it was just an unclosed tail running to the end of text.
        last_start, last_end = sents[-1]
        raw_slice = text[last_start:last_end].rstrip()
        if not raw_slice:
            return None
        # A paragraph that ends in a blank line is complete, even without terminal punctuation
        if text.startswith("\n\n", last_end) or text.startswith("\r\n\r\n", last_end):
            return last_end

        # Does raw_slice end with a terminator or closing punct?
        has_terminator = False
        # Strip closing punctuation to check preceding char
        stripped = raw_slice
        while stripped and stripped[-1] in _CLOSING_PUNCT:
            stripped = stripped[:-1]

        if any(stripped.endswith(t) for t in _CJK_TERMINATORS) or any(stripped.endswith(t) for t in _LATIN_TERMINATORS):
            has_terminator = True

        if has_terminator:
            return last_end
        elif len(sents) > 1:
            # The last sentence was an unclosed tail, so return the end of the previous sentence
            return sents[-2][1]
        else:
            return None


class TokenGroup(BaseModel):
    """Group of tokens for full-text search query construction."""
    tokens: list[str]
    mode: Literal["any", "phrase"]
    origin: Literal["free_text", "exact_term", "entity"] = "free_text"


def tokenize_query_groups(
    analyzer: TextAnalyzer,
    query: str,
    exact_terms: Sequence[str] | None = None,
    entity_terms: Sequence[str] | None = None,
    entity_mode: Literal["require", "ignore"] = "ignore",
) -> list[TokenGroup]:
    """Tokenize a query and its exact/entity terms into TokenGroups per §6.1.

    - Free-text query tokens become a 'free_text' group with mode='any'.
    - Each exact_term becomes an 'exact_term' group with mode='phrase'.
    - With entity_mode='require', each entity_term becomes an 'entity' group with mode='phrase'.
    """
    groups: list[TokenGroup] = []
    if query:
        free_tokens = analyzer.tokenize(analyzer.normalize(query))
        if free_tokens:
            groups.append(TokenGroup(tokens=free_tokens, mode="any", origin="free_text"))
    for term in exact_terms or []:
        toks = analyzer.tokenize(analyzer.normalize(term))
        if toks:
            groups.append(TokenGroup(tokens=toks, mode="phrase", origin="exact_term"))
    if entity_mode == "require":
        for ent in entity_terms or []:
            toks = analyzer.tokenize(analyzer.normalize(ent))
            if toks:
                groups.append(TokenGroup(tokens=toks, mode="phrase", origin="entity"))
    return groups


def build_tsquery(groups: list[TokenGroup]) -> str:
    """Build PostgreSQL websearch/tsquery-safe string from token groups.

    Guarantees no unescaped tsquery operators or quotes from token text.
    'phrase' groups use '<->' adjacency; 'any' groups use '|'; groups are joined with '&'.
    """
    clean_parts: list[str] = []
    # Strip any dangerous tsquery characters from tokens: only keep letters, digits, and CJK
    safe_re = re.compile(r"[^\w\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]", re.UNICODE)

    for g in groups:
        sanitized = [safe_re.sub("", t) for t in g.tokens]
        sanitized = [t for t in sanitized if t]
        if not sanitized:
            continue
        if g.mode == "phrase":
            if len(sanitized) == 1:
                clean_parts.append(sanitized[0])
            else:
                clean_parts.append(f"({' <-> '.join(sanitized)})")
        else:  # any
            if len(sanitized) == 1:
                clean_parts.append(sanitized[0])
            else:
                clean_parts.append(f"({' | '.join(sanitized)})")

    if not clean_parts:
        return ""
    return " & ".join(clean_parts)
