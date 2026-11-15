"""Engine-owned stage: passages that satisfy a `prefer` text rule rank higher (§7.9).

It runs whenever `prefer` rules exist, after rerank and before the final cut, whatever the profile lists in
`post_stages`. Nothing is removed. It only reorders what the search fetched; with a boost active the context asks
for the same wider pool a reranker gets (`RetrievalContext.candidate_limit`). The boost is rank-based (a matching passage moves up as if its rank were
divided by the factor), so it works for every score kind and keeps a rerank's fair interleaving.
"""
from __future__ import annotations

from typing import Any, ClassVar

from ..granularity import apply_to_ranked
from ..models import RAGNode, ScoreKind, TextRule
from ..text.rules import holds, mentions


class TextRuleBoost:
    """Reorders ranked passages so those satisfying a `prefer` rule move up, and those a "not about X" rule dislikes move down.

    Dummy example (factor 2): a passage at rank 4 that mentions the preferred name acts as if it were at rank 2
    (4 / 2); one at rank 3 that mentions an excluded name acts as if it were at rank 6 (3 * 2). Nothing is removed.
    """
    name: ClassVar[str] = "text_rule_boost"
    version: ClassVar[str] = "1.0.0"
    accepts_score_kinds: ClassVar[frozenset[ScoreKind] | None] = None

    def process(self, nodes: list[RAGNode[Any]], ctx: Any) -> list[RAGNode[Any]]:
        """Stage entry point. Does nothing unless there are `prefer` rules and the profile's factor is above 1."""
        prefer = [rule for rule in ctx.rules if rule.strength == "prefer"]
        factor = ctx.profile.text_rules.prefer_boost
        if not prefer or factor <= 1.0:                    # the profile rejects values below 1.0; 1.0 is off
            return nodes
        return apply_to_ranked(nodes, lambda ranked: self._reorder(ranked, prefer, factor, ctx.renderer))

    @staticmethod
    def _reorder(
        ranked: list[RAGNode[Any]], prefer: list[TextRule], factor: float, renderer: Any
    ) -> list[RAGNode[Any]]:
        wanted = [rule for rule in prefer if rule.kind != "must_not"]
        unwanted = [rule for rule in prefer if rule.kind == "must_not"]

        def key(item: tuple[int, RAGNode[Any]]) -> tuple[float, int]:
            """Sort key: the node's rank (1-based) scaled up or down by the boost; ties keep their original order."""
            position, node = item
            text = renderer.prompt_text(node)
            weight = 1.0
            if wanted and any(holds(text, rule) for rule in wanted):
                weight /= factor
            if unwanted and any(mentions(text, rule) for rule in unwanted):
                weight *= factor
            return ((position + 1) * weight, position)

        return [node for _, node in sorted(enumerate(ranked), key=key)]
