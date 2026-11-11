"""Built-in post-retrieval stages and their registry factories (§9)."""
from __future__ import annotations

from typing import Any, ClassVar

from ..granularity import is_ranked
from ..transforms.fusion import fair_interleave
from ..granularity import limit_ranked
from ..models import RAGNode, ScoreKind
from ..registry import ComponentDeps
from .boosters import RecencyBooster, ScoreFusionStage
from .dedup import IdentityDeduplicator, OffsetDeduplicator
from .threshold import ThresholdFilter


class TopKStage:
    """Truncates ranked results to the request's target size; ordered layers are not counted (§8.10)."""
    name: ClassVar[str] = "top_k"
    version: ClassVar[str] = "1.0.0"
    accepts_score_kinds: ClassVar[frozenset[ScoreKind] | None] = None

    def process(self, nodes: list[RAGNode[Any]], ctx: Any) -> list[RAGNode[Any]]:
        kept = limit_ranked(nodes, ctx.target_limit())
        ctx.truncated = ctx.truncated or len(kept) < len(nodes)
        return kept


class RerankStage:
    """Applies the configured reranker, unless the deadline leaves no room for it (§9.3, DG-10)."""
    name: ClassVar[str] = "rerank"
    version: ClassVar[str] = "1.0.0"
    accepts_score_kinds: ClassVar[frozenset[ScoreKind] | None] = None
    deadline_budget_key: ClassVar[str] = "rerank"

    def __init__(self, reranker: Any):
        self.reranker = reranker

    def process(self, nodes: list[RAGNode[Any]], ctx: Any) -> list[RAGNode[Any]]:
        limit = ctx.target_limit()
        variants = list(getattr(ctx, "variants", None) or [])
        if len(variants) <= 1:
            reranked = self.reranker.rerank(ctx.intent.search_query or ctx.request.question, nodes, limit, ctx)
        else:
            # §9.3: rerank against each variant and merge fairly, so the second subject of a
            # comparison is not scored against the first subject's query and crowded out
            per_variant = [
                [n for n in self.reranker.rerank(v.text, [n.model_copy() for n in nodes], limit, ctx) if is_ranked(n)]
                for v in variants
            ]
            merged = fair_interleave(per_variant, limit)
            reranked = [n for n in nodes if not is_ranked(n)] + merged
        ctx.reranked = any(n.rerank_score is not None for n in reranked)
        return reranked


def _build_rerank_stage(deps: ComponentDeps, argument: str | None = None) -> RerankStage:
    return RerankStage(deps.registry.get("reranker", "jina")(deps))


BUILTIN_STAGES = {
    "threshold": lambda deps, argument=None: ThresholdFilter(
        min_similarity=deps.profile.retrieval.min_similarity or 0.0, kind=argument or "cosine"
    ),
    "identity_dedup": lambda deps, argument=None: IdentityDeduplicator(key=argument),
    "offset_dedup": lambda deps, argument=None: OffsetDeduplicator(),
    "score_fusion": lambda deps, argument=None: ScoreFusionStage(feature=argument),
    "rerank": _build_rerank_stage,
    "recency": lambda deps, argument=None: RecencyBooster(clock=deps.clock),
    "top_k": lambda deps, argument=None: TopKStage(),
}
