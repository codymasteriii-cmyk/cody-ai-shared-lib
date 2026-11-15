"""Ingestion subpackage (§2.5, §6): splitters, batch embedder, and section summarizer.

Ingestion turns a document into rows a store can search. The pieces, in the order a document meets them:

    document -> sections                       the app's own structure (chapters, articles)           models.RAGDocumentInput
    section  -> summaries (optional)           summarizer.SectionSummarizer: one model call per section or unit
    section  -> passages                       splitter: sentence-aligned chunks with exact character offsets
    passages -> vectors                        embedder.BatchEmbedder: embedding calls, zero-vector guard
    all      -> one atomic store write         RAGStore.upsert_section, once per section
    later    -> only what changed              sync.DocumentSynchronizer: compare fingerprints, redo the difference

`nodes.py` holds the shared steps (build nodes, hash, sign) and `ids.py` the node id format. The engine's
`ingest_document`, `sync_document` and `prepare_nodes` drive these; see docs/CALL_TREES.md section 8.

Dummy example: a 3-chapter novel. Each chapter is a section. Chapter 1 (9,000 characters) is split into about eight passages
of ~1,200 characters with 150 overlap, embedded in one batch, summarized, and written in one atomic call. If the author later
edits only chapter 2, `sync_document` finds that chapter 2's content hash changed, re-embeds only its passages and summaries,
and leaves chapters 1 and 3 untouched.
"""
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
