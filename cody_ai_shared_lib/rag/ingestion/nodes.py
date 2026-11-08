"""Node construction and fingerprints shared by ingestion and sync (§6.6, RAG-ING-1)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Generic

from ..degradation import DegradationTracker
from ..models import MetaT, RAGNode, RAGSectionInput, SourceSpan
from ..ports import EmbeddingSpace
from ..render import NodeRenderer
from ..text.analyzer import TextAnalyzer, estimate_tokens
from ..usage import BudgetGuard, UsageTracker
from .embedder import BatchEmbedder
from .splitter import TextSplitter
from .summarizer import SummarizerResult


def build_section_nodes(
    doc_id: str,
    section: RAGSectionInput[MetaT],
    splitter: TextSplitter,
) -> list[RAGNode[MetaT]]:
    """Split a section into passage nodes carrying global character offsets."""
    nodes: list[RAGNode[MetaT]] = []
    for idx, chunk in enumerate(splitter.split(section.text, base_offset=section.char_start)):
        nodes.append(
            RAGNode[MetaT](
                node_id=f"{doc_id}#s{section.section_seq}p{idx}",
                granularity="passage",
                text=chunk.text,
                span=SourceSpan(
                    doc_id=doc_id,
                    section_id=section.section_id,
                    section_seq=section.section_seq,
                    section_title=section.title,
                    char_start=section.char_start + chunk.local_start,
                    char_end=section.char_start + chunk.local_end,
                    breadcrumb=list(section.breadcrumb) + chunk.breadcrumb,
                ),
                metadata=section.metadata,
            )
        )
    return nodes


def parse_node_id(doc_id: str, node_id: str) -> tuple[int, bool] | None:
    """Return (section_seq, is_summary) for an engine-built node id, or None for any other id."""
    match = re.fullmatch(re.escape(doc_id) + r"#s(\d+)(p\d+|_sum|_u\d+_sum)", node_id)
    if match is None:
        return None
    return int(match.group(1)), match.group(2).endswith("_sum")


def finalize_node(
    node: RAGNode[MetaT],
    renderer: NodeRenderer[MetaT],
    analyzer: TextAnalyzer,
    uses_analyzer_tokens: bool,
) -> None:
    """Fill the fields the store persists with each node (§4.4, §6.6)."""
    node.content_hash = hashlib.sha256(
        analyzer.normalize(renderer.embed_text(node)).encode("utf-8")
    ).hexdigest()
    node.token_estimate = estimate_tokens(node.text)
    if uses_analyzer_tokens:
        node.search_tokens = analyzer.tokenize(node.text)


def embedding_signature(model: str, space: EmbeddingSpace, splitter: TextSplitter) -> str:
    """The one formula for `model|dims|task_type|splitter_version` (§4.4, §6.6)."""
    splitter_version = space.splitter_version or getattr(splitter, "version", "1.0.0")
    return f"{model}|{space.dims}|{space.task_type_documents}|{splitter_version}"


def fts_signature(analyzer: TextAnalyzer, uses_analyzer_tokens: bool) -> str | None:
    """The one formula for `analyzer_name|analyzer_version` (§4.4)."""
    return f"{analyzer.name}|{analyzer.version}" if uses_analyzer_tokens else None


@dataclass
class PreparedSummaries(Generic[MetaT]):
    """Summary nodes ready to upsert: the section summary and the embedded unit summaries."""
    section_summary: RAGNode[MetaT] | None = None
    unit_nodes: list[RAGNode[MetaT]] = field(default_factory=list)
    unit_vectors: list[list[float]] = field(default_factory=list)


def prepare_summaries(
    result: SummarizerResult[MetaT],
    embedder: BatchEmbedder[MetaT],
    renderer: NodeRenderer[MetaT],
    analyzer: TextAnalyzer,
    uses_analyzer_tokens: bool,
    usage_tracker: UsageTracker,
    degradation_tracker: DegradationTracker,
    budget_guard: BudgetGuard | None = None,
) -> PreparedSummaries[MetaT]:
    """Fingerprint and embed the summaries of one section."""
    prepared: PreparedSummaries[MetaT] = PreparedSummaries(section_summary=result.section_summary)
    if result.section_summary is not None:
        finalize_node(result.section_summary, renderer, analyzer, uses_analyzer_tokens)
        pairs = embedder.embed_nodes(
            [result.section_summary], usage_tracker, degradation_tracker, "summary_embedding", budget_guard
        )
        if pairs:
            result.section_summary.embedding = pairs[0][1]

    for unit in result.unit_summaries:
        finalize_node(unit, renderer, analyzer, uses_analyzer_tokens)
        pairs = embedder.embed_nodes([unit], usage_tracker, degradation_tracker, "summary_embedding", budget_guard)
        if pairs:
            unit.embedding = pairs[0][1]
            prepared.unit_nodes.append(unit)
            prepared.unit_vectors.append(pairs[0][1])
    return prepared
