"""Fingerprint-driven document sync (§6.6, AC-RAG-06)."""
from __future__ import annotations

from typing import Any, Generic

from ..degradation import DegradationTracker
from ..errors import StoreCapabilityError
from ..models import IngestionProgress, MetaT, RAGDocumentInput, RAGNode
from ..models import RAGFilterSpec
from ..ports import Fingerprint, OwnerScope, RAGStore, SectionWrite, StoreCapabilities, SummaryWrite, call_store
from ..render import NodeRenderer
from ..text.analyzer import TextAnalyzer
from ..usage import BudgetGuard, UsageTracker
from .embedder import BatchEmbedder
from .nodes import (
    WriteSignatures,
    build_section_nodes,
    embedding_is_stale,
    finalize_node,
    parse_node_id,
    prepare_passages,
    prepare_summaries,
    write_signatures,
)
from .splitter import TextSplitter
from .summarizer import SectionSummarizer

_STATUS_PRIORITY = {"skipped": 0, "provided": 1, "generated": 2, "partial": 3, "failed": 4}


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
        embed_summaries: bool = True,
        granularity: str = "passage",
    ):
        self.store = store
        self.splitter = splitter
        self.analyzer = analyzer
        self.renderer = renderer
        self.embedder = embedder
        self.summarizer = summarizer
        self.embedding_model = embedding_model
        self.budget_guard = budget_guard
        self.embed_summaries = embed_summaries
        self.granularity = granularity

    def sync(
        self,
        doc: RAGDocumentInput[MetaT],
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
        owner: OwnerScope | None = None,
    ) -> IngestionProgress:
        """Bring every stored row of the document to the current text and signatures (§6.6).

        - Passages: new, changed or stale ones (text, embedding signature, or FTS signature) are
          re-embedded; unchanged ones are skipped; vanished ones are deleted.
        - A section whose text changed gets fresh summaries, because its old ones describe text
          that no longer exists.
        - A section whose text is unchanged keeps its summaries. If they are stale (model or
          analyzer upgrade, or embed_summaries switched), they are refreshed from their stored text
          via fetch_nodes, without summarizing again; a store without fetch_nodes regenerates them.
        """
        caps = self.store.capabilities()
        if not (caps.supports_fingerprints and caps.supports_delete):
            raise StoreCapabilityError(
                "sync_document requires store capabilities 'supports_fingerprints' and 'supports_delete'",
                stage="sync",
            )

        stored = call_store("get_fingerprints", lambda: self.store.get_fingerprints(doc.doc_id))
        signatures = write_signatures(
            self.embedding_model, self.store.embedding_space(), self.splitter, self.analyzer, caps.uses_analyzer_tokens
        )
        sections = sorted(doc.sections, key=lambda s: s.section_seq)

        current_ids: set[str] = set()
        section_nodes: dict[int, list[RAGNode[MetaT]]] = {}
        for section in sections:
            nodes = build_section_nodes(doc.doc_id, section, self.splitter, doc.revision, self.granularity)
            for node in nodes:
                finalize_node(node, self.renderer, self.analyzer, caps.uses_analyzer_tokens)
                current_ids.add(node.node_id)
            section_nodes[section.section_seq] = nodes

        changed_sections = self._sections_with_changed_text(doc.doc_id, stored, section_nodes, current_ids)

        upserted = skipped = 0
        summary_status = "skipped"
        regenerated_ids: set[str] = set()
        resummarized: set[int] = set()

        for section in sections:
            seq = section.section_seq
            to_embed = [n for n in section_nodes[seq] if self._is_stale(n, stored.get(n.node_id), signatures)]
            skipped += len(section_nodes[seq]) - len(to_embed)
            stale_summaries = (
                [] if seq in changed_sections else self._stale_summary_ids(doc.doc_id, seq, stored, signatures, caps)
            )
            refreshed: list[SummaryWrite[MetaT]] | None = None
            if stale_summaries:
                refreshed = self._refresh_summaries(
                    doc.doc_id, stale_summaries, signatures, caps, owner, usage_tracker, degradation_tracker
                )
            resummarize = seq in changed_sections or (bool(stale_summaries) and refreshed is None)
            if not to_embed and not resummarize and not refreshed:
                continue

            pairs = (
                self.embedder.embed_nodes(to_embed, usage_tracker, degradation_tracker, budget_guard=self.budget_guard)
                if to_embed
                else []
            )
            write = SectionWrite[MetaT](
                doc_id=doc.doc_id,
                section_seq=seq,
                owner=owner or OwnerScope(),
                passages=prepare_passages(pairs, signatures),
            )
            if refreshed:
                write.unit_summaries = [w for w in refreshed if w.unit_seq is not None]
                write.section_summary = next((w for w in refreshed if w.unit_seq is None), None)
                regenerated_ids.update(w.node.node_id for w in refreshed)
            if resummarize:
                resummarized.add(seq)
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
                for summary_node in ([result.section_summary] if result.section_summary else []) + result.unit_summaries:
                    summary_node.revision = doc.revision
                summaries = prepare_summaries(
                    result,
                    self.embedder,
                    self.renderer,
                    self.analyzer,
                    caps.uses_analyzer_tokens,
                    usage_tracker,
                    degradation_tracker,
                    self.budget_guard,
                    signatures=signatures,
                    embed=self.embed_summaries,
                )
                write.unit_summaries = summaries.unit_summaries
                write.section_summary = summaries.section_summary
                regenerated_ids.update(summaries.node_ids)

            if write.passages or write.unit_summaries or write.section_summary is not None:
                section_write = write
                upserted += call_store("upsert_section", lambda: self.store.upsert_section(section_write))

        # Delete last: a failed sync leaves the old rows readable, and a re-run converges.
        section_seqs = {s.section_seq for s in sections}
        to_delete = [
            node_id
            for node_id in stored
            if node_id not in current_ids
            and node_id not in regenerated_ids
            and not self._is_kept_summary(doc.doc_id, node_id, section_seqs, changed_sections | resummarized)
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
    def _is_stale(node: RAGNode[Any], fingerprint: Fingerprint | None, signatures: WriteSignatures) -> bool:
        """A passage is stale when its text, its vector signature or its search-token signature differ."""
        return (
            fingerprint is None
            or fingerprint.content_hash != node.content_hash
            or fingerprint.embedding_signature != signatures.embedding
            or (signatures.fts is not None and fingerprint.fts_signature != signatures.fts)
        )

    def _stale_summary_ids(
        self,
        doc_id: str,
        seq: int,
        stored: dict[str, Fingerprint],
        signatures: WriteSignatures,
        caps: StoreCapabilities,
    ) -> list[str]:
        """Stored summaries of an unchanged section that no longer match the current signatures."""
        stale = []
        for node_id, fp in stored.items():
            parsed = parse_node_id(doc_id, node_id)
            if parsed is None or not parsed.is_summary or parsed.section_seq != seq:
                continue
            vector_stale = embedding_is_stale(doc_id, node_id, fp, signatures.embedding, self.embed_summaries)
            unwanted_vector = not self.embed_summaries and fp.embedding_signature is not None
            tokens_stale = caps.uses_analyzer_tokens and fp.fts_signature != signatures.fts
            if vector_stale or unwanted_vector or tokens_stale:
                stale.append(node_id)
        return stale

    def _refresh_summaries(
        self,
        doc_id: str,
        node_ids: list[str],
        signatures: WriteSignatures,
        caps: StoreCapabilities,
        owner: OwnerScope | None,
        usage_tracker: UsageTracker,
        degradation_tracker: DegradationTracker,
    ) -> list[SummaryWrite[MetaT]] | None:
        """Re-sign stored summaries from their stored text, or None when they must be regenerated."""
        if not caps.supports_fetch_nodes:
            return None
        filters = _owner_filters(doc_id, owner, caps)
        fetched = call_store("fetch_nodes", lambda: self.store.fetch_nodes(node_ids, filters))
        if {n.node_id for n in fetched} != set(node_ids):
            return None
        for node in fetched:
            finalize_node(node, self.renderer, self.analyzer, caps.uses_analyzer_tokens)
        vectors: dict[str, list[float]] = {}
        if self.embed_summaries:
            pairs = self.embedder.embed_nodes(
                fetched, usage_tracker, degradation_tracker, "summary_embedding", self.budget_guard
            )
            vectors = {node.node_id: vector for node, vector in pairs}
        writes: list[SummaryWrite[MetaT]] = []
        for node in fetched:
            parsed = parse_node_id(doc_id, node.node_id)
            vector = vectors.get(node.node_id)
            node.embedding = vector
            writes.append(
                SummaryWrite[MetaT](
                    node=node,
                    vector=vector,
                    fingerprint=signatures.fingerprint(node, has_vector=vector is not None),
                    unit_seq=parsed.number if parsed is not None and parsed.kind == "unit_summary" else None,
                )
            )
        return writes

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
            if parsed is not None and not parsed.is_summary and node_id not in current_ids:
                changed.add(parsed.section_seq)
        return changed

    @staticmethod
    def _is_kept_summary(
        doc_id: str,
        node_id: str,
        section_seqs: set[int],
        changed_sections: set[int],
    ) -> bool:
        parsed = parse_node_id(doc_id, node_id)
        if parsed is None or not parsed.is_summary:
            return False
        return parsed.section_seq in section_seqs and parsed.section_seq not in changed_sections


def _owner_filters(doc_id: str, owner: OwnerScope | None, caps: StoreCapabilities) -> RAGFilterSpec:
    """Read filters that state the owner's scope, so a sync read passes the store's required keys."""
    fields: dict[str, Any] = {"doc_ids": [doc_id]}
    if owner is not None:
        if owner.tenant_or_user_id is not None:
            fields["tenant_or_user_id"] = owner.tenant_or_user_id
        if owner.domain_id is not None:
            fields["domain_id"] = owner.domain_id
        custom = {k: v for k, v in owner.custom.items() if k in caps.supported_custom_filters}
        if custom:
            fields["custom"] = custom
    return RAGFilterSpec(**fields)
