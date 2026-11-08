"""Post-retrieval boosters and score fusion (§9.4, AC-RAG-35).

Implements:
  - RecencyBooster: time-decayed multiplier based on age relative to an injected clock.
  - ScoreFusionStage: min-max normalized weighted fusion of semantic, keyword, recency, and app features.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar, Generic, Sequence
from ..models import MetaT, RAGNode
from ..ports import NodeRenderer
from ..render import DefaultRenderer
from ..runtime import Clock, SystemClock


class RecencyBooster(Generic[MetaT]):
    """Multiplies node score by 1 + weight * 2^(-age / half_life) (§9.4, AC-RAG-35)."""
    name: ClassVar[str] = "recency"
    version: ClassVar[str] = "1.0.0"
    accepts_score_kinds: ClassVar[frozenset[str] | None] = None

    def process(self, nodes: list[RAGNode[MetaT]], ctx: Any) -> list[RAGNode[MetaT]]:
        return self.boost(nodes, renderer=ctx.renderer)

    def __init__(
        self,
        half_life_days: float = 7.0,
        weight: float = 0.3,
        clock: Clock | None = None,
    ):
        self.half_life_days = max(half_life_days, 0.001)
        self.weight = weight
        self.clock = clock or SystemClock()

    def boost(
        self,
        nodes: Sequence[RAGNode[MetaT]],
        renderer: NodeRenderer[MetaT] | None = None,
    ) -> list[RAGNode[MetaT]]:
        """Apply recency boost deterministically under the injected clock."""
        now = self.clock.now_utc()
        view = renderer or DefaultRenderer()
        result: list[RAGNode[MetaT]] = []

        for node in nodes:
            dt = view.timestamp(node.metadata)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
                factor = 1.0 + self.weight * (2.0 ** (-age_days / self.half_life_days))
                node.score = node.score * factor
            result.append(node)

        # Re-sort descending by score, tie-break by node_id
        result.sort(key=lambda n: (-n.score, n.node_id))
        return result


class ScoreFusionStage(Generic[MetaT]):
    """Combines min-max normalized signals and app features into score_kind='fused' (§9.4, AC-RAG-35)."""
    name: ClassVar[str] = "score_fusion"
    version: ClassVar[str] = "1.0.0"
    accepts_score_kinds: ClassVar[frozenset[str] | None] = None

    def process(self, nodes: list[RAGNode[MetaT]], ctx: Any) -> list[RAGNode[MetaT]]:
        return self.fuse(nodes, renderer=ctx.renderer, clock=ctx.deadline.clock)

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights or {
            "semantic": 0.5,
            "keyword": 0.2,
            "recency": 0.1,
            "features": 0.2,
        })

    def _min_max_normalize(self, values: list[float]) -> list[float]:
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v < 1e-9:
            return [1.0 for _ in values]
        return [(v - min_v) / (max_v - min_v) for v in values]

    def fuse(
        self,
        nodes: Sequence[RAGNode[MetaT]],
        renderer: NodeRenderer[MetaT] | None = None,
        clock: Clock | None = None,
    ) -> list[RAGNode[MetaT]]:
        """Combine signals into a fused score."""
        if not nodes:
            return []

        node_list = list(nodes)

        # 1. Semantic (vector_score or rerank_score)
        v_scores = [
            n.rerank_score if n.rerank_score is not None else (n.vector_score if n.vector_score is not None else n.score)
            for n in node_list
        ]
        norm_v = self._min_max_normalize(v_scores)

        # 2. Keyword (keyword_score)
        k_scores = [n.keyword_score if n.keyword_score is not None else 0.0 for n in node_list]
        norm_k = self._min_max_normalize(k_scores)

        # 3. Recency factor
        clk = clock or SystemClock()
        now = clk.now_utc()
        view = renderer or DefaultRenderer()
        r_scores: list[float] = []
        for node in node_list:
            dt = view.timestamp(node.metadata)
            if isinstance(dt, datetime):
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
                r_scores.append(2.0 ** (-age_days / 7.0))
            else:
                r_scores.append(0.5)
        norm_r = self._min_max_normalize(r_scores)

        # 4. App feature signal (e.g. confidence or priority_score)
        feat_scores: list[float] = []
        for node in node_list:
            features = view.features(node.metadata)
            feat_scores.append(sum(features.values()) / len(features) if features else 0.5)
        norm_f = self._min_max_normalize(feat_scores)

        w_v = self.weights.get("semantic", 0.5)
        w_k = self.weights.get("keyword", 0.2)
        w_r = self.weights.get("recency", 0.1)
        w_f = self.weights.get("features", 0.2)
        total_w = w_v + w_k + w_r + w_f or 1.0

        for i, node in enumerate(node_list):
            fused = (
                w_v * norm_v[i] +
                w_k * norm_k[i] +
                w_r * norm_r[i] +
                w_f * norm_f[i]
            ) / total_w
            node.score = fused
            node.score_kind = "fused"

        node_list.sort(key=lambda item: (-item.score, item.node_id))
        return node_list
