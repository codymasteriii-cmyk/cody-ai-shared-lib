"""Node construction and fingerprints shared by ingestion and sync (§6.6, RAG-ING-1)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Generic

from ..degradation import DegradationTracker
from ..models import MetaT, PreparedNode, RAGNode, RAGSectionInput, SourceSpan, node_class_for
from ..ports import EmbeddingSpace, Fingerprint, SummaryWrite
from ..render import NodeRenderer
from ..text.analyzer import TextAnalyzer, estimate_tokens
from ..usage import BudgetGuard, UsageTracker
from .embedder import BatchEmbedder
from .ids import format_node_id, parse_node_id  # noqa: F401  (parse_node_id is used by sync)
from .splitter import TextSplitter
from .summarizer import SummarizerResult


def build_section_nodes(
    doc_id: str,
    section: RAGSectionInput[MetaT],
    splitter: TextSplitter,
    revision: str | None = None,
    granularity: str = "passage",
) -> list[RAGNode[MetaT]]:
    """Split a section into passage nodes carrying global character offsets and the document revision."""
    nodes: list[RAGNode[MetaT]] = []
    for idx, chunk in enumerate(splitter.split(section.text, base_offset=section.char_start)):
        nodes.append(
            node_class_for(section.metadata)(
                node_id=format_node_id(doc_id, section.section_seq, "passage", idx),
                granularity=granularity,
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
                revision=revision,
            )
        )
    return nodes


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


def embedding_is_stale(
    doc_id: str, node_id: str, fingerprint: Fingerprint, expected: str, embed_summaries: bool
) -> bool:
    """True when this row's vector must be (re)made under the current signature (§6.6).

    A row with a signature is stale when the signature differs. A row stored without a vector is
    stale only when it is a summary and the profile now embeds summaries.
    """
    if fingerprint.embedding_signature is not None:
        return fingerprint.embedding_signature != expected
    parsed = parse_node_id(doc_id, node_id)
    return embed_summaries and parsed is not None and parsed.is_summary


def write_signatures(
    model: str, space: EmbeddingSpace, splitter: TextSplitter, analyzer: TextAnalyzer, uses_analyzer_tokens: bool
) -> "WriteSignatures":
    """Both signatures for one write, from the one formula each."""
    return WriteSignatures(
        embedding=embedding_signature(model, space, splitter),
        fts=fts_signature(analyzer, uses_analyzer_tokens),
    )


@dataclass(frozen=True)
class WriteSignatures:
    """The signatures the engine stamps on every node it writes (§6.6). Computed once per call."""
    embedding: str
    fts: str | None

    def fingerprint(self, node: RAGNode[MetaT], has_vector: bool = True) -> Fingerprint:
        """A row stored without a vector carries no embedding signature (§6.6)."""
        return Fingerprint(
            content_hash=node.content_hash or "",
            embedding_signature=self.embedding if has_vector else None,
            fts_signature=self.fts,
        )


@dataclass
class PreparedSummaries(Generic[MetaT]):
    """Summaries ready to write: the section summary and the unit summaries, each with a fingerprint."""
    section_summary: SummaryWrite[MetaT] | None = None
    unit_summaries: list[SummaryWrite[MetaT]] = field(default_factory=list)

    @property
    def node_ids(self) -> list[str]:
        items = ([self.section_summary] if self.section_summary else []) + self.unit_summaries
        return [item.node.node_id for item in items]


def _unit_seq(doc_id: str, node: RAGNode[MetaT], position: int) -> int:
    parsed = parse_node_id(doc_id, node.node_id)
    return parsed.number if parsed is not None and parsed.kind == "unit_summary" and parsed.number is not None else position


def prepare_summaries(
    result: SummarizerResult[MetaT],
    embedder: BatchEmbedder[MetaT],
    renderer: NodeRenderer[MetaT],
    analyzer: TextAnalyzer,
    uses_analyzer_tokens: bool,
    usage_tracker: UsageTracker,
    degradation_tracker: DegradationTracker,
    budget_guard: BudgetGuard | None = None,
    *,
    signatures: WriteSignatures,
    embed: bool = True,
) -> PreparedSummaries[MetaT]:
    """Fingerprint the summaries of one section and, when `embed` is set, embed them in one batch.

    The unit's `unit_seq` is the index in its engine-built id, so a skipped unit leaves a gap
    instead of shifting the units after it (INT-01).
    """
    summaries = ([result.section_summary] if result.section_summary is not None else []) + list(result.unit_summaries)
    for node in summaries:
        finalize_node(node, renderer, analyzer, uses_analyzer_tokens)

    vectors: dict[str, list[float]] = {}
    if embed and summaries:
        pairs = embedder.embed_nodes(summaries, usage_tracker, degradation_tracker, "summary_embedding", budget_guard)
        for node, vector in pairs:
            node.embedding = vector
            vectors[node.node_id] = vector

    prepared: PreparedSummaries[MetaT] = PreparedSummaries()
    if result.section_summary is not None:
        node = result.section_summary
        vector = vectors.get(node.node_id)
        prepared.section_summary = SummaryWrite[MetaT](
            node=node, vector=vector, fingerprint=signatures.fingerprint(node, has_vector=vector is not None)
        )
    for position, unit in enumerate(result.unit_summaries):
        vector = vectors.get(unit.node_id)
        prepared.unit_summaries.append(
            SummaryWrite[MetaT](
                node=unit,
                vector=vector,
                fingerprint=signatures.fingerprint(unit, has_vector=vector is not None),
                unit_seq=_unit_seq(unit.span.doc_id, unit, position),
            )
        )
    return prepared


def prepare_passages(
    pairs: list[tuple[RAGNode[MetaT], list[float]]],
    signatures: WriteSignatures,
) -> list[PreparedNode[MetaT]]:
    """Pair each embedded passage with its fingerprint, in document order."""
    return [
        PreparedNode[MetaT](node=node, vector=vector, fingerprint=signatures.fingerprint(node)) for node, vector in pairs
    ]
