"""Retrieval diff helper for comparing two retrieval runs (§14.2)."""
from __future__ import annotations

import math
from typing import Any
from ..models import RAGRetrievalResult, RetrievalDiff


def diff_retrievals(
    a: RAGRetrievalResult[Any],
    b: RAGRetrievalResult[Any],
    k: int = 10,
) -> RetrievalDiff:
    """Compare two RAGRetrievalResult objects at top-k (§14.2).

    Computes Jaccard overlap@k, Spearman rank correlation on shared node IDs,
    exclusively retrieved IDs, and rank deltas.
    """
    ids_a = [n.node_id for n in a.nodes[:k]]
    ids_b = [n.node_id for n in b.nodes[:k]]

    set_a = set(ids_a)
    set_b = set(ids_b)
    union = set_a | set_b
    shared = [nid for nid in ids_a if nid in set_b]

    overlap_at_k = len(shared) / len(union) if union else 1.0

    rank_a = {nid: idx + 1 for idx, nid in enumerate(ids_a)}
    rank_b = {nid: idx + 1 for idx, nid in enumerate(ids_b)}

    rank_deltas: dict[str, tuple[int, int]] = {
        nid: (rank_a[nid], rank_b[nid]) for nid in shared
    }

    # Spearman rank correlation on shared items
    n_shared = len(shared)
    if n_shared >= 2:
        # Remap shared items to relative ranks 1..n_shared
        rel_a_order = sorted(shared, key=lambda x: rank_a[x])
        rel_b_order = sorted(shared, key=lambda x: rank_b[x])
        rel_rank_a = {nid: i + 1 for i, nid in enumerate(rel_a_order)}
        rel_rank_b = {nid: i + 1 for i, nid in enumerate(rel_b_order)}
        d_sq_sum = sum((rel_rank_a[nid] - rel_rank_b[nid]) ** 2 for nid in shared)
        denom = n_shared * (n_shared ** 2 - 1)
        rank_corr = 1.0 - (6.0 * d_sq_sum) / denom if denom > 0 else 1.0
    elif n_shared == 1:
        rank_corr = 1.0
    else:
        rank_corr = 0.0

    only_in_a = [nid for nid in ids_a if nid not in set_b]
    only_in_b = [nid for nid in ids_b if nid not in set_a]

    return RetrievalDiff(
        k=k,
        overlap_at_k=overlap_at_k,
        rank_correlation=rank_corr if not math.isnan(rank_corr) else 0.0,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        rank_deltas=rank_deltas,
    )
