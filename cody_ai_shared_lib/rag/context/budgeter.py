"""Token budgets per prompt zone and history selection (§10.1, §10.2, AC-RAG-38)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..models import ChatTurn
from ..profile import ContextConfig
from ..text.analyzer import CjkAwareAnalyzer, TextAnalyzer, estimate_tokens


@dataclass
class BudgetAllocation:
    """Computed token budgets per prompt zone (§10.1)."""
    usable_budget: int
    history_budget: int
    summary_budget: int
    local_window_budget: int
    retrieved_budget: int


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut text proportionally so its estimated size fits, marking the cut."""
    tokens = estimate_tokens(text)
    if tokens <= max_tokens:
        return text
    return text[: max(1, int(len(text) * max_tokens / tokens * 0.95))] + "…"


def _turn_pairs(history: Sequence[ChatTurn]) -> list[list[ChatTurn]]:
    """Group turns into exchanges; each user turn starts a new one."""
    pairs: list[list[ChatTurn]] = []
    for turn in history:
        if turn.role == "user" or not pairs:
            pairs.append([])
        pairs[-1].append(turn)
    return pairs


def _render_pair(pair: Sequence[ChatTurn]) -> str:
    return "\n".join(f"{turn.role.capitalize()}: {turn.content or ''}" for turn in pair)


class TokenBudgeter:
    """Computes token budgets per zone and selects the history that fits (§10.1, §10.2, AC-RAG-38)."""

    def __init__(self, config: ContextConfig, analyzer: TextAnalyzer | None = None):
        self.config = config
        self.analyzer = analyzer or CjkAwareAnalyzer()

    def allocate(
        self,
        has_history: bool = True,
        has_summaries: bool = True,
        has_local_window: bool = True,
    ) -> BudgetAllocation:
        usable = max(
            256,
            self.config.max_context_tokens
            - self.config.reserved_instruction_tokens
            - self.config.question_reserve_tokens
            - self.config.answer_reserve_tokens,
        )
        hist_b = min(int(usable * self.config.history_ratio), self.config.reserved_history_tokens) if has_history else 0
        sum_b = int(usable * self.config.summary_ratio) if has_summaries else 0
        win_b = int(usable * self.config.local_window_ratio) if has_local_window else 0
        ret_b = usable - hist_b - sum_b - win_b
        return BudgetAllocation(
            usable_budget=usable,
            history_budget=hist_b,
            summary_budget=sum_b,
            local_window_budget=win_b,
            retrieved_budget=max(128, ret_b),
        )

    def _tokens(self, text: str) -> set[str]:
        return set(self.analyzer.tokenize(self.analyzer.normalize(text)))

    def _retention(self, pair_text: str, question_tokens: set[str], age: int) -> float:
        """Retention(i) = e^(-λ·Δt) + α·VocabOverlap(i, question) (§10.2)."""
        overlap = len(self._tokens(pair_text) & question_tokens) / len(question_tokens) if question_tokens else 0.0
        return math.exp(-self.config.history_decay_lambda * age) + self.config.history_overlap_alpha * overlap

    def select_history(
        self,
        history: Sequence[ChatTurn],
        rolling_summary: str | None,
        budget_tokens: int,
        question: str = "",
    ) -> tuple[str, int]:
        """Return (history text, tokens used): recent exchanges always, older ones by score (§10.2)."""
        if budget_tokens <= 0 or (not history and not rolling_summary):
            return "", 0

        pairs = [_render_pair(pair) for pair in _turn_pairs(history)]
        keep = max(1, self.config.min_recent_turns_kept)
        recent, older = pairs[-keep:], pairs[:-keep]

        if sum(estimate_tokens(text) for text in recent) > budget_tokens:
            each = max(1, budget_tokens // len(recent))
            recent = [truncate_to_tokens(text, each) for text in recent]
        used = sum(estimate_tokens(text) for text in recent)

        summary_line = f"[Prior Conversation Summary]: {rolling_summary}" if rolling_summary else ""
        if summary_line:
            summary_tokens = estimate_tokens(summary_line)
            if used + summary_tokens <= budget_tokens:
                used += summary_tokens
            else:
                summary_line = ""

        question_tokens = self._tokens(question)
        best_first = sorted(
            range(len(older)),
            key=lambda i: (-self._retention(older[i], question_tokens, len(older) - i + keep - 1), -i),
        )
        kept: set[int] = set()
        for index in best_first:
            tokens = estimate_tokens(older[index])
            if used + tokens <= budget_tokens:
                kept.add(index)
                used += tokens

        lines = ([summary_line] if summary_line else []) + [older[i] for i in sorted(kept)] + recent
        return "\n".join(lines), used
