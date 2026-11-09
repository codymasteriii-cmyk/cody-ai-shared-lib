"""Fingerprint-driven document sync (§6.6, AC-RAG-06)."""
from __future__ import annotations

from typing import Any, Generic

from ..degradation import DegradationTracker
from ..errors import StoreCapabilityError
from ..models import IngestionProgress, MetaT, RAGDocumentInput, RAGNode
from ..ports import Fingerprint, RAGStore, call_store
from ..render import NodeRenderer
from ..text.analyzer import TextAnalyzer
from ..usage import BudgetGuard, UsageTracker
from .embedder import BatchEmbedder
from .nodes import (
    build_section_nodes,
    embedding_signature,
    finalize_node,
    parse_node_id,
    prepare_summaries,
)
from .splitter import TextSplitter
from .summarizer import SectionSummarizer

_STATUS_PRIORITY = {"skipped": 0, "provided": 1, "generated": 2, "failed": 3}


class DocumentSynchronizer(Generic[MetaT]):
    """Brings a stored document in line with its current text, re-embedding only what changed."""

    def __init__(
        self,
        store: RAGStore[MetaT],
        splitter: TextSplitter,
        analyzer: TextAnalyzer,
        renderer: NodeRenderer[MetaT],
        embedder: BatchEmbedder[MetaT],
        summarizer: SectionSummarizer[MetaT],
        embedding_model: str,
        budget_guard: BudgetGuard | None = None,
    ):
        self.store = store
        self.splitter = splitter
        self.analyzer = analyzer
        self.renderer = renderer
        self.embedder = embedder
        self.summarizer = summarizer
        self.embedding_model = embedding_model
        self.budget_guard = budget_guard

    def sync(
        self,
        doc: RAGDocumentInput[MetaT],
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
    ) -> IngestionProgress:
        """Embed new or changed nodes, delete vanished ones, and skip unchanged ones.

        A section whose text changed gets fresh summaries, because its old ones describe text
        that no longer exists. A section whose text is unchanged keeps them, so a model upgrade
        (signature change only) re-embeds passages without paying to summarize again.
        """
        caps = self.store.capabilities()
        if not (caps.supports_fingerprints and caps.supports_delete):
            raise StoreCapabilityError(
                "sync_document requires store capabilities 'supports_fingerprints' and 'supports_delete'",
                stage="sync",
            )

        stored = call_store("get_fingerprints", lambda: self.store.get_fingerprints(doc.doc_id))
        signature = embedding_signature(self.embedding_model, self.store.embedding_space(), self.splitter)
        sections = sorted(doc.sections, key=lambda s: s.section_seq)

        current_ids: set[str] = set()
        section_nodes: dict[int, list[RAGNode[MetaT]]] = {}
        for section in sections:
            nodes = build_section_nodes(doc.doc_id, section, self.splitter)
            for node in nodes:
                finalize_node(node, self.renderer, self.analyzer, caps.uses_analyzer_tokens)
                current_ids.add(node.node_id)
            section_nodes[section.section_seq] = nodes

        changed_sections = self._sections_with_changed_text(doc.doc_id, stored, section_nodes, current_ids)

        upserted = skipped = 0
        summary_status = "skipped"
        regenerated_ids: set[str] = set()

        for section in sections:
            seq = section.section_seq
            to_embed = [n for n in section_nodes[seq] if self._is_stale(n, stored.get(n.node_id), signature)]
            skipped += len(section_nodes[seq]) - len(to_embed)
            if not to_embed and seq not in changed_sections:
                continue

            pairs = (
                self.embedder.embed_nodes(to_embed, usage_tracker, degradation_tracker, budget_guard=self.budget_guard)
                if to_embed
                else []
            )
            nodes = [p[0] for p in pairs]
            vectors = [p[1] for p in pairs]

            section_summary = None
            if seq in changed_sections:
                result = self.summarizer.summarize_section(
                    doc_id=doc.doc_id,
                    section_seq=seq,
                    section_id=section.section_id,
                    text=section.text,
                    char_start=section.char_start,
                    metadata=section.metadata,
                    existing_summary=section.existing_summary,
                    usage_tracker=usage_tracker,
                    degradation_tracker=degradation_tracker,
                    budget_guard=self.budget_guard,
                )
                if _STATUS_PRIORITY[result.summary_status] > _STATUS_PRIORITY[summary_status]:
                    summary_status = result.summary_status
                summaries = prepare_summaries(
                    result,
                    self.embedder,
                    self.renderer,
                    self.analyzer,
                    caps.uses_analyzer_tokens,
                    usage_tracker,
                    degradation_tracker,
                    self.budget_guard,
                )
                section_summary = summaries.section_summary
                nodes += summaries.unit_nodes
                vectors += summaries.unit_vectors
                regenerated_ids.update(n.node_id for n in summaries.unit_nodes)
                if section_summary is not None:
                    regenerated_ids.add(section_summary.node_id)

            if nodes or section_summary is not None:
                upserted += call_store(
                    "upsert_section_nodes",
                    lambda: self.store.upsert_section_nodes(
                        doc_id=doc.doc_id,
                        section_seq=seq,
                        nodes=nodes,
                        embeddings=vectors,
                        summary=section_summary,
                    ),
                )

        # Delete last: a failed sync leaves the old rows readable, and a re-run converges.
        section_seqs = {s.section_seq for s in sections}
        to_delete = [
            node_id
            for node_id in stored
            if node_id not in current_ids
            and node_id not in regenerated_ids
            and not self._is_kept_summary(doc.doc_id, node_id, section_seqs, changed_sections)
        ]
        if to_delete:
            call_store("delete_nodes", lambda: self.store.delete_nodes(to_delete))

        return IngestionProgress(
            doc_id=doc.doc_id,
            section_seq=0,
            sections_completed=len(doc.sections),
            sections_total=len(doc.sections),
            passages_upserted=upserted,
            passages_skipped_unchanged=skipped,
            summary_status=summary_status,  # type: ignore[arg-type]
            usage=usage_tracker.stage_usages,
            degradations=degradation_tracker.degradations,
        )

    @staticmethod
    def _is_stale(node: RAGNode[Any], fingerprint: Fingerprint | None, signature: str) -> bool:
        return (
            fingerprint is None
            or fingerprint.content_hash != node.content_hash
            or fingerprint.embedding_signature != signature
        )

    @staticmethod
    def _sections_with_changed_text(
        doc_id: str,
        stored: dict[str, Fingerprint],
        section_nodes: dict[int, list[RAGNode[Any]]],
        current_ids: set[str],
    ) -> set[int]:
        changed = {
            seq
            for seq, nodes in section_nodes.items()
            if any(n.node_id not in stored or stored[n.node_id].content_hash != n.content_hash for n in nodes)
        }
        for node_id in stored:
            parsed = parse_node_id(doc_id, node_id)
            if parsed is not None and not parsed[1] and node_id not in current_ids:
                changed.add(parsed[0])
        return changed

    @staticmethod
    def _is_kept_summary(
        doc_id: str,
        node_id: str,
        section_seqs: set[int],
        changed_sections: set[int],
    ) -> bool:
        parsed = parse_node_id(doc_id, node_id)
        if parsed is None or not parsed[1]:
            return False
        seq = parsed[0]
        return seq in section_seqs and seq not in changed_sections
