"""Merging several ranked lists of nodes (§7.7, §7.8, AC-RAG-19)."""
from __future__ import annotations

import math
from typing import Sequence

from ..models import MetaT, RAGNode


def fair_interleave(variant_results: list[list[RAGNode[MetaT]]], top_k: int) -> list[RAGNode[MetaT]]:
    """Merge lists in turn, so each keeps at least ceil(top_k / n) slots when it has candidates."""
    if not variant_results:
        return []
    quota = math.ceil(top_k / len(variant_results))
    selected: dict[str, RAGNode[MetaT]] = {}
    positions = [0] * len(variant_results)

    def take_next(index: int) -> bool:
        results = variant_results[index]
        while positions[index] < len(results):
            node = results[positions[index]]
            positions[index] += 1
            if node.node_id not in selected:
                selected[node.node_id] = node
                return True
        return False

    # First every list gets its fair share, then any free slots go to whichever still has nodes.
    taken = [0] * len(variant_results)
    for fair_share in (True, False):
        progressed = True
        while progressed and len(selected) < top_k:
            progressed = False
            for index in range(len(variant_results)):
                if len(selected) >= top_k or (fair_share and taken[index] >= quota):
                    continue
                if take_next(index):
                    taken[index] += 1
                    progressed = True
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
    selected = sorted(fair_interleave(by_fused_rank, top_k), key=lambda n: (-fused[n.node_id], n.node_id))
    for node in selected:
        node.score = fused[node.node_id]
        node.score_kind = "rrf"
    return selected
