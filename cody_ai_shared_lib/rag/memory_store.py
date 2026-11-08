"""In-memory RAGStore reference implementation (§4.5).

Implements every capability in pure Python with thread safety:
  - Exact cosine search;
  - BM25 over TextAnalyzer tokens;
  - Native hybrid search with Reciprocal Rank Fusion (RRF);
  - Summary units and levels;
  - Offset addressable windows and sequential boundaries;
  - Fingerprints, revisions, atomic section commits, and JSONL fixture round-tripping.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Generic

from .errors import EmbeddingProviderError
from .granularity import TRAIT_OFFSET_ADDRESSABLE, has_trait
from .models import (
    DefaultNodeMeta,
    MetaT,
    RAGFilterSpec,
    RAGNode,
    SourceSpan,
)
from .ports import (
    EmbeddingSpace,
    Fingerprint,
    RAGStore,
    ScopeStats,
    StoreCapabilities,
    validate_scope,
)
from .text.analyzer import CjkAwareAnalyzer, TokenGroup, estimate_tokens
from .vectors import is_zero_vector


class InMemoryRAGStore(RAGStore[MetaT], Generic[MetaT]):
    """Pure-Python in-memory implementation of RAGStore for tests and eval."""

    def __init__(
        self,
        capabilities: StoreCapabilities | None = None,
        embedding_space: EmbeddingSpace | None = None,
        analyzer: CjkAwareAnalyzer | None = None,
    ):
        self._capabilities = capabilities or StoreCapabilities(
            supports_vector_search=True,
            supports_keyword_fts=True,
            supports_native_hybrid_rpc=True,
            returns_raw_scores=True,
            uses_analyzer_tokens=True,
            returns_snippets=True,
            supports_parent_summaries=True,
            supports_summary_levels=True,
            supports_local_window=True,
            supports_full_scope_fetch=True,
            supports_sequential_boundary=True,
            supports_hydration=True,
            supports_fetch_nodes=True,
            supports_fingerprints=True,
            supports_delete=True,
            supports_revisions=True,
            supports_embedding_return=True,
            required_filter_keys=frozenset(),
            supported_custom_filters={"favorites_only": bool, "active": bool, "language": str},
            supported_date_fields=frozenset(["created_at", "updated_at"]),
            time_basis="created_at = ingestion time",
        )
        self._embedding_space = embedding_space or EmbeddingSpace(
            model="gemini-embedding-2",
            dims=768,
            task_type_documents="RETRIEVAL_DOCUMENT",
            splitter_version="1.0.0",
        )
        self.analyzer = analyzer or CjkAwareAnalyzer()
        self._lock = threading.RLock()

        # Storage collections:
        # node_id -> RAGNode[MetaT]
        self._nodes: dict[str, RAGNode[MetaT]] = {}
        # (doc_id, section_seq) -> list[node_id]
        self._section_nodes: dict[tuple[str, int], list[str]] = {}
        # (doc_id, section_seq) -> RAGNode[MetaT] (summary)
        self._section_summaries: dict[tuple[str, int], RAGNode[MetaT]] = {}
        # doc_id -> {node_id: Fingerprint}
        self._fingerprints: dict[str, dict[str, Fingerprint]] = {}
        # Store-level revision counter
        self._revision_counter = 1

    def capabilities(self) -> StoreCapabilities:
        return self._capabilities

    def embedding_space(self) -> EmbeddingSpace:
        return self._embedding_space

    # ── Filter Validation and Enforcement ──────────────────────────────────────

    def _validate_filters(self, filters: RAGFilterSpec) -> None:
        """Enforce required filter keys and supported custom filter types."""
        validate_scope(filters, self._capabilities)

    def _passes_filters(self, node: RAGNode[MetaT], filters: RAGFilterSpec) -> bool:
        """Return True if node matches all criteria in filters."""
        # doc_ids filter
        if filters.doc_ids is not None and node.span.doc_id not in filters.doc_ids:
            return False

        # tenant_or_user_id
        if filters.tenant_or_user_id is not None:
            extra = getattr(node.metadata, "extra", {}) if hasattr(node.metadata, "extra") else {}
            if isinstance(node.metadata, dict):
                node_tenant = node.metadata.get("tenant_id")
            else:
                node_tenant = getattr(node.metadata, "tenant_id", None) or (
                    extra.get("tenant_id") if isinstance(extra, dict) else None
                )
            if node_tenant is not None and node_tenant != filters.tenant_or_user_id:
                return False

        # domain_id
        if filters.domain_id is not None:
            extra = getattr(node.metadata, "extra", {}) if hasattr(node.metadata, "extra") else {}
            if isinstance(node.metadata, dict):
                node_domain = node.metadata.get("domain_id")
            else:
                node_domain = getattr(node.metadata, "domain_id", None) or (
                    extra.get("domain_id") if isinstance(extra, dict) else None
                )
            if node_domain is not None and node_domain != filters.domain_id:
                return False

        # categories
        if filters.categories is not None:
            extra = getattr(node.metadata, "extra", {}) if hasattr(node.metadata, "extra") else {}
            if isinstance(node.metadata, dict):
                node_cat = node.metadata.get("category")
            else:
                node_cat = getattr(node.metadata, "category", None) or (
                    extra.get("category") if isinstance(extra, dict) else None
                )
            if node_cat is not None and node_cat not in filters.categories:
                return False

        # Sequential boundary: single-doc shorthand or per-doc boundary
        boundary = filters.max_char_offset
        if filters.max_char_offset_by_doc and node.span.doc_id in filters.max_char_offset_by_doc:
            doc_bound = filters.max_char_offset_by_doc[node.span.doc_id]
            boundary = min(boundary, doc_bound) if boundary is not None else doc_bound

        if boundary is not None:
            is_summary_node = getattr(node, "is_summary", False) or has_trait(node.granularity, "summary")
            if is_summary_node or not has_trait(node.granularity, TRAIT_OFFSET_ADDRESSABLE):
                # Summary or non-offset node must not exceed boundary
                if node.span.char_end > boundary:
                    return False
            else:
                # Verbatim offset node must start before boundary
                if node.span.char_start >= boundary:
                    return False

        # min_char_offset
        if filters.min_char_offset is not None:
            if node.span.char_end <= filters.min_char_offset:
                return False

        # date_range_utc
        if filters.date_range_utc is not None:
            dt_start, dt_end = filters.date_range_utc
            extra = getattr(node.metadata, "extra", {}) if hasattr(node.metadata, "extra") else {}
            date_field = filters.date_field or "created_at"
            if isinstance(node.metadata, dict):
                node_dt = node.metadata.get(date_field)
            else:
                node_dt = getattr(node.metadata, date_field, None) or (
                    extra.get(date_field) if isinstance(extra, dict) else None
                )
            if node_dt is not None and isinstance(node_dt, datetime):
                if dt_start is not None and node_dt < dt_start:
                    return False
                if dt_end is not None and node_dt > dt_end:
                    return False

        # custom filters
        if filters.custom:
            extra = getattr(node.metadata, "extra", {}) if hasattr(node.metadata, "extra") else {}
            for k, expected_v in filters.custom.items():
                if isinstance(node.metadata, dict):
                    node_v = node.metadata.get(k)
                else:
                    node_v = getattr(node.metadata, k, None)
                    if node_v is None and isinstance(extra, dict):
                        node_v = extra.get(k)
                if node_v is not None and node_v != expected_v:
                    return False

        return True

    # ── Cosine & BM25 Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm1 * norm2)))

    def _node_tokens(self, node: RAGNode[MetaT]) -> list[str]:
        return node.search_tokens if node.search_tokens is not None else self.analyzer.tokenize(node.text)

    @staticmethod
    def _matches_group(tokens: list[str], group: TokenGroup) -> bool:
        """Apply one token group: any, all, or adjacent in order (§6.1)."""
        if group.mode == "any":
            return bool(set(group.tokens) & set(tokens))
        if group.mode == "all":
            return set(group.tokens) <= set(tokens)
        size = len(group.tokens)
        return any(tokens[i:i + size] == group.tokens for i in range(len(tokens) - size + 1))

    @staticmethod
    def _bm25_scores(query_terms: list[str], documents: list[list[str]]) -> list[float]:
        """Okapi BM25 over the candidate documents (k1=1.2, b=0.75)."""
        if not documents:
            return []
        k1, b = 1.2, 0.75
        average_length = sum(len(tokens) for tokens in documents) / len(documents) or 1.0
        frequencies = [Counter(tokens) for tokens in documents]
        terms = list(dict.fromkeys(query_terms))
        doc_frequency = {term: sum(1 for counts in frequencies if term in counts) for term in terms}

        scores: list[float] = []
        for tokens, counts in zip(documents, frequencies):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (len(documents) - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens) / average_length))
            scores.append(score)
        return scores

    # ── Retrieval Methods (§4.2) ───────────────────────────────────────────────

    def vector_search(
        self,
        query_vector: list[float],
        filters: RAGFilterSpec,
        top_k: int,
    ) -> list[RAGNode[MetaT]]:
        # Conformance kit check 6: zero vector query must be rejected
        if is_zero_vector(query_vector):
            raise EmbeddingProviderError("Zero-vector query rejected by vector_search")

        self._validate_filters(filters)

        with self._lock:
            candidates: list[tuple[float, RAGNode[MetaT]]] = []
            for node in self._nodes.values():
                if not self._passes_filters(node, filters):
                    continue
                if is_zero_vector(node.embedding):
                    continue
                sim = self._cosine_similarity(query_vector, node.embedding)
                if filters.min_similarity is not None and sim < filters.min_similarity:
                    continue
                candidates.append((sim, copy.deepcopy(node)))

            # Sort descending by cosine similarity, break ties stably by node_id
            candidates.sort(key=lambda item: (-item[0], item[1].node_id))
            results: list[RAGNode[MetaT]] = []
            for rank, (score, node) in enumerate(candidates[:top_k], start=1):
                node.score = score
                node.score_kind = "cosine"
                node.vector_score = score
                node.vector_rank = rank
                results.append(node)
            return results

    def keyword_search(
        self,
        query_text: str,
        filters: RAGFilterSpec,
        top_k: int,
        analyzer_tokens: list[TokenGroup] | None = None,
    ) -> list[RAGNode[MetaT]]:
        self._validate_filters(filters)

        groups = analyzer_tokens or [
            TokenGroup(tokens=self.analyzer.tokenize(self.analyzer.normalize(query_text)), mode="any")
        ]
        query_terms = [token for group in groups for token in group.tokens]

        with self._lock:
            matched = []
            for node in self._nodes.values():
                if not self._passes_filters(node, filters):
                    continue
                tokens = self._node_tokens(node)
                if all(self._matches_group(tokens, group) for group in groups if group.tokens):
                    matched.append((node, tokens))

            candidates: list[tuple[float, RAGNode[MetaT]]] = []
            for (node, _), score in zip(matched, self._bm25_scores(query_terms, [t for _, t in matched])):
                if score > 0.0:
                    node_copy = copy.deepcopy(node)
                    if self._capabilities.returns_snippets:
                        node_copy.snippet = f"Excerpt: {node.text[:100]}"
                        node_copy.snippet_kind = "match_excerpt"
                    candidates.append((score, node_copy))

            candidates.sort(key=lambda item: (-item[0], item[1].node_id))
            results: list[RAGNode[MetaT]] = []
            for rank, (score, node) in enumerate(candidates[:top_k], start=1):
                node.score = score
                node.score_kind = "bm25"
                node.keyword_score = score
                node.keyword_rank = rank
                results.append(node)
            return results

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        filters: RAGFilterSpec,
        top_k: int,
        rrf_k: int = 60,
        w_vec: float = 1.0,
        w_fts: float = 1.0,
        analyzer_tokens: list[TokenGroup] | None = None,
    ) -> list[RAGNode[MetaT]]:
        self._validate_filters(filters)

        # Run vector search and keyword search internally
        vec_hits = self.vector_search(query_vector, filters, top_k=top_k * 2) if query_vector else []
        kw_hits = self.keyword_search(query_text, filters, top_k=top_k * 2, analyzer_tokens=analyzer_tokens)

        # Merge with Reciprocal Rank Fusion (RRF)
        vec_by_id = {n.node_id: n for n in vec_hits}
        kw_by_id = {n.node_id: n for n in kw_hits}
        all_ids = set(vec_by_id.keys()) | set(kw_by_id.keys())

        merged: list[tuple[float, RAGNode[MetaT]]] = []
        for nid in all_ids:
            v_node = vec_by_id.get(nid)
            k_node = kw_by_id.get(nid)

            v_rank = v_node.vector_rank if v_node else None
            k_rank = k_node.keyword_rank if k_node else None

            rrf_score = 0.0
            if v_rank is not None:
                rrf_score += w_vec * (1.0 / (rrf_k + v_rank))
            if k_rank is not None:
                rrf_score += w_fts * (1.0 / (rrf_k + k_rank))

            chosen_node = copy.deepcopy(v_node or k_node)  # type: ignore[arg-type]
            chosen_node.score = rrf_score
            chosen_node.score_kind = "rrf"
            chosen_node.vector_score = v_node.vector_score if v_node else None
            chosen_node.vector_rank = v_rank
            chosen_node.keyword_score = k_node.keyword_score if k_node else None
            chosen_node.keyword_rank = k_rank
            merged.append((rrf_score, chosen_node))

        merged.sort(key=lambda item: (-item[0], item[1].node_id))
        return [node for _, node in merged[:top_k]]

    def fetch_parent_summaries(
        self,
        filters: RAGFilterSpec,
        level: int | None = None,
    ) -> list[RAGNode[MetaT]]:
        self._validate_filters(filters)
        with self._lock:
            seen_ids: set[str] = set()
            summaries: list[RAGNode[MetaT]] = []
            for summary in self._section_summaries.values():
                if level is not None and summary.level != level:
                    continue
                if not self._passes_filters(summary, filters):
                    continue
                summaries.append(copy.deepcopy(summary))
                seen_ids.add(summary.node_id)

            for node in self._nodes.values():
                if node.node_id in seen_ids:
                    continue
                if not (node.is_summary or node.granularity == "parent_summary"):
                    continue
                if level is not None and node.level != level:
                    continue
                if not self._passes_filters(node, filters):
                    continue
                summaries.append(copy.deepcopy(node))
                seen_ids.add(node.node_id)

            summaries.sort(key=lambda s: (s.span.doc_id, s.span.section_seq, s.span.char_start))
            return summaries

    def fetch_local_window(
        self,
        doc_id: str,
        center_offset: int,
        radius_chars: int,
        filters: RAGFilterSpec,
    ) -> list[RAGNode[MetaT]]:
        self._validate_filters(filters)
        with self._lock:
            doc_nodes = [
                n for n in self._nodes.values()
                if n.span.doc_id == doc_id and not n.is_summary and self._passes_filters(n, filters)
            ]
            if not doc_nodes:
                return []

            start_offset = max(0, center_offset - radius_chars)
            end_offset = center_offset + radius_chars

            boundary = filters.max_char_offset
            if filters.max_char_offset_by_doc and doc_id in filters.max_char_offset_by_doc:
                doc_bound = filters.max_char_offset_by_doc[doc_id]
                boundary = min(boundary, doc_bound) if boundary is not None else doc_bound
            if boundary is not None:
                end_offset = min(end_offset, boundary)
            if start_offset >= end_offset:
                return []

            # Passages overlap, so each one contributes only the text after the previous one ended
            window_parts: list[str] = []
            window_start: int | None = None
            position = start_offset
            for n in sorted(doc_nodes, key=lambda x: x.span.char_start):
                local_start = max(position, n.span.char_start)
                local_end = min(end_offset, n.span.char_end)
                if local_end <= local_start:
                    continue
                if window_start is None:
                    window_start = local_start
                elif local_start > position:
                    break  # a gap in the indexed text ends the contiguous window
                window_parts.append(n.text[local_start - n.span.char_start:local_end - n.span.char_start])
                position = local_end

            if window_start is None:
                return []
            window_node = RAGNode[MetaT](
                node_id=f"window_{doc_id}_{window_start}_{position}",
                text="".join(window_parts),
                span=SourceSpan(doc_id=doc_id, char_start=window_start, char_end=position),
                granularity="local_window",
                metadata=DefaultNodeMeta(),  # type: ignore[arg-type]
            )
            return [window_node]

    def scope_stats(self, filters: RAGFilterSpec) -> ScopeStats:
        self._validate_filters(filters)
        with self._lock:
            matched_nodes = [n for n in self._nodes.values() if self._passes_filters(n, filters)]
            docs = {n.span.doc_id for n in matched_nodes}
            char_count = sum(len(n.text) for n in matched_nodes)
            est_tok = sum(
                n.token_estimate if (n.token_estimate is not None and n.token_estimate > 0) else estimate_tokens(n.text)
                for n in matched_nodes
            )
            has_sums = any(
                self._passes_filters(s, filters) for s in self._section_summaries.values()
            ) or any(n.is_summary or n.granularity == "parent_summary" for n in matched_nodes)
            revision = f"rev_{self._revision_counter}_{len(matched_nodes)}"
            return ScopeStats(
                doc_count=len(docs),
                node_count=len(matched_nodes),
                char_count=char_count,
                est_tokens=est_tok,
                revision=revision,
                has_summaries=has_sums,
            )

    def fetch_full_scope(self, filters: RAGFilterSpec) -> list[RAGNode[MetaT]]:
        self._validate_filters(filters)
        with self._lock:
            matched = [
                copy.deepcopy(n) for n in self._nodes.values()
                if self._passes_filters(n, filters) and not n.is_summary
            ]
            matched.sort(key=lambda n: (n.span.doc_id, n.span.section_seq, n.span.char_start))
            return matched

    def fetch_nodes(self, node_ids: list[str], filters: RAGFilterSpec) -> list[RAGNode[MetaT]]:
        self._validate_filters(filters)
        with self._lock:
            results: list[RAGNode[MetaT]] = []
            for nid in node_ids:
                node = self._nodes.get(nid)
                if node and self._passes_filters(node, filters):
                    results.append(copy.deepcopy(node))
            return results

    def hydrate(self, node_ids: list[str], projection: str) -> dict[str, MetaT]:
        with self._lock:
            return {
                nid: copy.deepcopy(self._nodes[nid].metadata)
                for nid in node_ids
                if nid in self._nodes
            }

    # ── Lifecycle Methods (§4.4) ───────────────────────────────────────────────

    def upsert_section_nodes(
        self,
        doc_id: str,
        section_seq: int,
        nodes: list[RAGNode[MetaT]],
        embeddings: list[list[float]] | None = None,
        summary: RAGNode[MetaT] | None = None,
    ) -> int:
        """Atomic upsert of section passages and summary (RAG-ING-10)."""
        with self._lock:
            written_ids: list[str] = []
            doc_fprints = self._fingerprints.setdefault(doc_id, {})

            if embeddings is None:
                embeddings = [n.embedding or [0.0] * self._embedding_space.dims for n in nodes]

            # Prepare all passage nodes with embeddings and signatures
            for node, emb in zip(nodes, embeddings):
                node_copy = copy.deepcopy(node)
                if not node_copy.span.doc_id and doc_id:
                    node_copy.span.doc_id = doc_id
                node_copy.embedding = emb
                node_copy.content_hash = node_copy.content_hash or self._hash_text(node_copy.text)
                node_copy.token_estimate = node_copy.token_estimate or estimate_tokens(node_copy.text)
                if self._capabilities.uses_analyzer_tokens and node_copy.search_tokens is None:
                    node_copy.search_tokens = self.analyzer.tokenize(node_copy.text)

                self._nodes[node_copy.node_id] = node_copy
                written_ids.append(node_copy.node_id)

                doc_fprints[node_copy.node_id] = self._fingerprint(node_copy)

            # A partial upsert (sync) adds to the section; it does not replace it
            existing_ids = self._section_nodes.get((doc_id, section_seq), [])
            self._section_nodes[(doc_id, section_seq)] = list(dict.fromkeys(existing_ids + written_ids))

            if summary is not None:
                sum_copy = copy.deepcopy(summary)
                if not sum_copy.span.doc_id and doc_id:
                    sum_copy.span.doc_id = doc_id
                sum_copy.is_summary = True
                sum_copy.content_hash = sum_copy.content_hash or self._hash_text(sum_copy.text)
                self._section_summaries[(doc_id, section_seq)] = sum_copy
                doc_fprints[sum_copy.node_id] = self._fingerprint(sum_copy)

            self._revision_counter += 1
            return len(written_ids)

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(self.analyzer.normalize(text).encode("utf-8")).hexdigest()

    def _fingerprint(self, node: RAGNode[MetaT]) -> Fingerprint:
        space = self._embedding_space
        uses_tokens = self._capabilities.uses_analyzer_tokens
        return Fingerprint(
            content_hash=node.content_hash or self._hash_text(node.text),
            embedding_signature=f"{space.model}|{space.dims}|{space.task_type_documents}|{space.splitter_version}",
            fts_signature=f"{self.analyzer.name}|{self.analyzer.version}" if uses_tokens else None,
        )

    def get_fingerprints(self, doc_id: str) -> dict[str, Fingerprint]:
        with self._lock:
            return dict(self._fingerprints.get(doc_id, {}))

    def delete_nodes(self, node_ids: list[str]) -> int:
        with self._lock:
            deleted = 0
            for nid in node_ids:
                if nid in self._nodes:
                    node = self._nodes.pop(nid)
                    self._fingerprints.get(node.span.doc_id, {}).pop(nid, None)
                    deleted += 1

            targets = set(node_ids)
            for key, summary in list(self._section_summaries.items()):
                if summary.node_id in targets:
                    del self._section_summaries[key]
                    self._fingerprints.get(key[0], {}).pop(summary.node_id, None)
                    deleted += 1
            self._revision_counter += 1
            return deleted

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            to_delete = [nid for nid, n in self._nodes.items() if n.span.doc_id == doc_id]
            for nid in to_delete:
                del self._nodes[nid]
            self._fingerprints.pop(doc_id, None)
            sec_keys = [k for k in self._section_summaries if k[0] == doc_id]
            for k in sec_keys:
                del self._section_summaries[k]
            self._revision_counter += 1
            return len(to_delete)

    def delete_sections(self, doc_id: str, from_seq: int) -> int:
        with self._lock:
            to_delete_nodes: list[str] = []
            for (d_id, seq), nids in list(self._section_nodes.items()):
                if d_id == doc_id and seq >= from_seq:
                    to_delete_nodes.extend(nids)
                    del self._section_nodes[(d_id, seq)]

            to_delete_nodes.extend(
                summary.node_id
                for (d_id, seq), summary in self._section_summaries.items()
                if d_id == doc_id and seq >= from_seq
            )
            return self.delete_nodes(to_delete_nodes)

    def current_revisions(self, node_ids: list[str]) -> dict[str, str]:
        with self._lock:
            return {
                nid: self._nodes[nid].revision or "v1"
                for nid in node_ids
                if nid in self._nodes
            }

    # ── Fixture Serialization (§14.1 RAG-EVAL-10) ─────────────────────────────

    def to_jsonl(self, path: str | Path) -> None:
        """Serialize nodes, summaries, fingerprints, and store header to JSONL (§14.1 RAG-EVAL-10)."""
        file_path = Path(path)
        with self._lock:
            lines: list[str] = []
            header = {
                "_kind": "header",
                "embedding_space": self._embedding_space.model_dump(mode="json"),
                "revision_counter": self._revision_counter,
            }
            lines.append(json.dumps(header, ensure_ascii=False))
            for nid, node in self._nodes.items():
                doc_id = node.span.doc_id
                fp = self._fingerprints.get(doc_id, {}).get(nid)
                row = {
                    "_kind": "node",
                    "node": node.model_dump(mode="json"),
                    "fingerprint": fp.model_dump(mode="json") if fp else None,
                }
                lines.append(json.dumps(row, ensure_ascii=False))
            for (doc_id, sec_seq), sum_node in self._section_summaries.items():
                row = {
                    "_kind": "summary",
                    "doc_id": doc_id,
                    "section_seq": sec_seq,
                    "node": sum_node.model_dump(mode="json"),
                    "fingerprint": self._fingerprint(sum_node).model_dump(mode="json"),
                }
                lines.append(json.dumps(row, ensure_ascii=False))
            file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        capabilities: StoreCapabilities | None = None,
        analyzer: CjkAwareAnalyzer | None = None,
    ) -> InMemoryRAGStore[Any]:
        """Load an InMemoryRAGStore snapshot from a JSONL file (§14.1 RAG-EVAL-10)."""
        file_path = Path(path)
        raw_lines = [ln.strip() for ln in file_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        emb_space: EmbeddingSpace | None = None
        rev_counter = 1
        nodes_to_load: list[tuple[RAGNode[Any], Fingerprint | None]] = []
        summaries_to_load: list[tuple[str, int, RAGNode[Any]]] = []
        summary_fingerprints: list[tuple[str, str, Fingerprint]] = []

        for ln in raw_lines:
            record = json.loads(ln)
            kind = record.get("_kind")
            if kind == "header":
                if "embedding_space" in record and record["embedding_space"]:
                    emb_space = EmbeddingSpace.model_validate(record["embedding_space"])
                rev_counter = int(record.get("revision_counter", 1))
            elif kind == "node":
                node = RAGNode.model_validate(record["node"])
                fp = Fingerprint.model_validate(record["fingerprint"]) if record.get("fingerprint") else None
                nodes_to_load.append((node, fp))
            elif kind == "summary":
                s_node = RAGNode.model_validate(record["node"])
                summaries_to_load.append((record["doc_id"], int(record["section_seq"]), s_node))
                if record.get("fingerprint"):
                    summary_fingerprints.append(
                        (record["doc_id"], s_node.node_id, Fingerprint.model_validate(record["fingerprint"]))
                    )

        store: InMemoryRAGStore[Any] = cls(
            capabilities=capabilities,
            embedding_space=emb_space,
            analyzer=analyzer,
        )
        store._revision_counter = rev_counter
        for node, fp in nodes_to_load:
            store._nodes[node.node_id] = node
            doc_id = node.span.doc_id
            sec_seq = node.span.section_seq
            store._section_nodes.setdefault((doc_id, sec_seq), []).append(node.node_id)
            if fp is not None:
                store._fingerprints.setdefault(doc_id, {})[node.node_id] = fp
        for doc_id, sec_seq, s_node in summaries_to_load:
            store._section_summaries[(doc_id, sec_seq)] = s_node
        for doc_id, node_id, fp in summary_fingerprints:
            store._fingerprints.setdefault(doc_id, {})[node_id] = fp
        return store
