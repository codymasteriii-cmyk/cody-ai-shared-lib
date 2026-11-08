"""Query transforms and multi-variant result fusion (§7.8, AC-RAG-19)."""
from __future__ import annotations

import math
from typing import Protocol, Sequence
from ..models import MetaT, QueryIntent, RAGNode, RAGQueryRequest
from .context import QueryVariant


class QueryTransform(Protocol):
    name: str
    version: str

    def transform(self, request: RAGQueryRequest, intent: QueryIntent) -> list[QueryVariant]:
        ...


class SecondaryQueryFanOut:
    """Fans out secondary queries from query intent (e.g. comparison queries, AC-RAG-19)."""
    name = "secondary_fanout"
    version = "1.0.0"

    def transform(self, request: RAGQueryRequest, intent: QueryIntent) -> list[QueryVariant]:
        variants = [
            QueryVariant(
                text=intent.search_query or request.question,
                weight=1.0,
                origin="primary",
            )
        ]
        for sec in intent.secondary_queries:
            if sec.strip():
                variants.append(
                    QueryVariant(
                        text=sec.strip(),
                        weight=0.8,
                        origin="secondary",
                    )
                )
        return variants


def fair_interleave(variant_results: list[list[RAGNode[MetaT]]], top_k: int) -> list[RAGNode[MetaT]]:
    """Merge variant results so each one keeps at least ceil(top_k / n) slots when it has candidates."""
    if not variant_results:
        return []
    per_variant = math.ceil(top_k / len(variant_results))
    selected: dict[str, RAGNode[MetaT]] = {}

    for results in variant_results:
        taken = 0
        for node in results:
            if taken >= per_variant or len(selected) >= top_k:
                break
            if node.node_id not in selected:
                selected[node.node_id] = node
                taken += 1

    for rank in range(max(map(len, variant_results), default=0)):
        for results in variant_results:
            if len(selected) >= top_k:
                return list(selected.values())
            if rank < len(results):
                selected.setdefault(results[rank].node_id, results[rank])
    return list(selected.values())


def fuse_variants(
    variant_results: list[list[RAGNode[MetaT]]],
    weights: Sequence[float],
    top_k: int,
    rrf_k: int = 60,
) -> list[RAGNode[MetaT]]:
    """Weighted RRF across variants, then fair_interleave, ranked by the fused score (§7.8)."""
    fused: dict[str, float] = {}
    for results, weight in zip(variant_results, weights):
        for rank, node in enumerate(results, start=1):
            fused[node.node_id] = fused.get(node.node_id, 0.0) + weight / (rrf_k + rank)

    by_fused_rank = [sorted(results, key=lambda n: (-fused[n.node_id], n.node_id)) for results in variant_results]
    selected = fair_interleave(by_fused_rank, top_k)
    return sorted(selected, key=lambda n: (-fused[n.node_id], n.node_id))
