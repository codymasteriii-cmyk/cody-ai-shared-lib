"""Text splitters for RAG ingestion (§6.2).

Implements:
  - TextSplitter protocol
  - SentenceSplitter: sentence-aligned splitter preserving exact character offsets (RAG-ING-1/2, AC-RAG-01)
  - PassThroughSplitter: one node per unit (AC-RAG-03)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from pydantic import BaseModel, Field

from ..text.analyzer import CjkAwareAnalyzer, TextAnalyzer


class SplitChunk(BaseModel):
    """A sliced text chunk with exact local offsets in the parent text."""
    text: str
    local_start: int
    local_end: int
    breadcrumb: list[str] = Field(default_factory=list)


@runtime_checkable
class TextSplitter(Protocol):
    """Protocol for text splitting strategies."""
    name: str
    version: str

    def split(self, text: str, base_offset: int = 0) -> list[SplitChunk]:
        """Split text into chunks with local character offsets."""
        ...


class SentenceSplitter:
    """Sentence-aligned text splitter with target size and overlap.

    Respects CJK and Latin sentence terminators. Preserves exact character offsets
    such that text[chunk.local_start:chunk.local_end] == chunk.text (RAG-ING-1).
    """
    name: str = "sentence"
    version: str = "1.0.0"

    def __init__(
        self,
        target_size: int = 1200,
        overlap: int = 150,
        analyzer: TextAnalyzer | None = None,
    ):
        self.target_size = target_size
        self.overlap = overlap
        self.analyzer = analyzer or CjkAwareAnalyzer()

    def split(self, text: str, base_offset: int = 0) -> list[SplitChunk]:
        """Split text into sentence-aligned chunks."""
        if not text:
            return []

        sentences = self.analyzer.sentences(text)
        if not sentences:
            stripped = text.strip()
            if not stripped:
                return []
            start = text.index(stripped)
            end = start + len(stripped)
            return [SplitChunk(text=text[start:end], local_start=start, local_end=end)]

        effective_sentences: list[tuple[int, int]] = []
        max_sentence_len = int(self.target_size * 1.5)

        for s_start, s_end in sentences:
            s_len = s_end - s_start
            if s_len <= max_sentence_len:
                effective_sentences.append((s_start, s_end))
            else:
                clause_ranges = self._split_clauses(text, s_start, s_end, max_len=self.target_size)
                effective_sentences.extend(clause_ranges)

        chunks: list[SplitChunk] = []
        n_sents = len(effective_sentences)
        sent_idx = 0

        while sent_idx < n_sents:
            chunk_start_sent = sent_idx
            chunk_start_char = effective_sentences[chunk_start_sent][0]

            curr_sent = chunk_start_sent
            while curr_sent < n_sents:
                end_char = effective_sentences[curr_sent][1]
                curr_len = end_char - chunk_start_char
                if curr_len > self.target_size and curr_sent > chunk_start_sent:
                    break
                curr_sent += 1

            chunk_end_char = effective_sentences[curr_sent - 1][1]
            chunk_text = text[chunk_start_char:chunk_end_char]
            chunks.append(
                SplitChunk(
                    text=chunk_text,
                    local_start=chunk_start_char,
                    local_end=chunk_end_char,
                )
            )

            if curr_sent >= n_sents:
                break

            next_start_sent = curr_sent
            while next_start_sent > chunk_start_sent + 1:
                overlap_len = chunk_end_char - effective_sentences[next_start_sent - 1][0]
                if overlap_len <= self.overlap:
                    next_start_sent -= 1
                else:
                    break

            if next_start_sent <= chunk_start_sent:
                next_start_sent = chunk_start_sent + 1

            sent_idx = next_start_sent

        return chunks

    def _split_clauses(self, text: str, start: int, end: int, max_len: int) -> list[tuple[int, int]]:
        """Subdivide an oversized sentence at clause punctuation, preserving exact offsets."""
        clause_puncts = (",", ";", "，", "、", "；")
        sub_ranges: list[tuple[int, int]] = []
        curr_start = start

        i = start
        while i < end:
            if text[i] in clause_puncts:
                clause_end = i + 1
                if clause_end - curr_start >= max_len:
                    sub_ranges.append((curr_start, clause_end))
                    curr_start = clause_end
            elif (i - curr_start) >= max_len * 2:
                sub_ranges.append((curr_start, i))
                curr_start = i
            i += 1

        if curr_start < end:
            sub_ranges.append((curr_start, end))

        return sub_ranges


class PassThroughSplitter:
    """Pass-through splitter: one input text unit yields exactly one node (AC-RAG-03)."""
    name: str = "pass_through"
    version: str = "1.0.0"

    def __init__(
        self,
        target_size: int | None = None,
        overlap: int | None = None,
        analyzer: TextAnalyzer | None = None,
    ):
        """Takes the arguments every splitter is built with; none apply to one chunk per unit."""

    def split(self, text: str, base_offset: int = 0) -> list[SplitChunk]:
        """Wrap entire text as a single chunk."""
        if not text:
            return []
        return [
            SplitChunk(
                text=text,
                local_start=0,
                local_end=len(text),
            )
        ]
