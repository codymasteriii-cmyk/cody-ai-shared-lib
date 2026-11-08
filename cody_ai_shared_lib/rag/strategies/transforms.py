"""Query transforms and multi-variant result fusion (§7.8, AC-RAG-19)."""
from __future__ import annotations

import math
from typing import Generic, Sequence
from ..models import MetaT, QueryIntent, RAGNode, RAGQueryRequest
from .context import QueryVariant


class SecondaryQueryFanOut:
    """Fans out secondary queries from query intent (e.g. comparison queries, AC-RAG-19)."""

    def transform(self, request: RAGQueryRequest, intent: QueryIntent) -> list[QueryVariant]:
        variants = [
            QueryVariant(
                text=intent.search_query or request.query,
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


def fair_interleave(
    variant_results: list[list[RAGNode[MetaT]]],
    top_k: int,
) -> list[RAGNode[MetaT]]:
    """Merge candidate lists from multiple query variants with fair representation (AC-RAG-19).

    Each variant gets at least floor(top_k / n_variants) slots if available.
    """
    if not variant_results:
        return []
    if len(variant_results) == 1:
        return variant_results[0][:top_k]

    n_variants = len(variant_results)
    min_per_variant = max(1, math.floor(top_k / n_variants))
    seen_ids: set[str] = set()
    selected: list[RAGNode[MetaT]] = []

    # Round 1: Take up to min_per_variant from each variant
    for res_list in variant_results:
        taken = 0
        for node in res_list:
            if node.id not in seen_ids and taken < min_per_variant and len(selected) < top_k:
                seen_ids.add(node.id)
                selected.append(node)
                taken += 1

    # Round 2: Fill remaining slots round-robin by score/rank
    max_len = max(len(rl) for rl in variant_results)
    for idx in range(max_len):
        for res_list in variant_results:
            if len(selected) >= top_k:
                break
            if idx < len(res_list):
                node = res_list[idx]
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    selected.append(node)

    return selected[:top_k]
