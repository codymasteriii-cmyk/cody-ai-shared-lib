"""Ingestion subpackage (§2.5, §6): splitters, batch embedder, and section summarizer."""
from .embedder import BatchEmbedder
from .splitter import PassThroughSplitter, SentenceSplitter, SplitChunk, TextSplitter
from .summarizer import SectionSummarizer, SummarizerResult
from .sync import DocumentSynchronizer

__all__ = [
    "BatchEmbedder",
    "DocumentSynchronizer",
    "PassThroughSplitter",
    "SectionSummarizer",
    "SentenceSplitter",
    "SplitChunk",
    "SummarizerResult",
    "TextSplitter",
]
