"""Token budgets per prompt zone and history selection (§10.1, §10.2, AC-RAG-38).

The model's window is finite, so before any text is added the window is divided into zones:

    max_context_tokens
    ├─ instructions         the system prompt (never less than `reserved_instruction_tokens`)
    ├─ question reserve     room for the user's question
    ├─ answer reserve       room for the answer to be written
    └─ usable               what is left, split into: history | summaries | local window | retrieved passages

Dummy example: a 6,000-token window, 800 for instructions, 200 for the question and 1,000 for the answer leaves
4,000 usable. With illustrative ratios of 15% for history and 15% for summaries that is 600 each
(when there is any history or any summary); the passages get what remains, never less than 128.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..errors import ProfileConfigError
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
    """Cut text proportionally so its estimated size fits, marking the cut.

    The cut is an estimate (tokens are not counted exactly), so it keeps 95% of the proportional length to stay
    safely under the budget.

    >>> truncate_to_tokens("short", 100)
    'short'
    >>> truncate_to_tokens("word " * 500, 10).endswith("…")
    True
    """
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
    """One exchange as text: "User: ..." then "Assistant: ..."."""
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
        instruction_tokens: int | None = None,
    ) -> BudgetAllocation:
        """Split `max_context_tokens` into zones (§10.1).

        `instruction_tokens` is the estimated size of the finished system prompt: the persona, rules and
        citation instruction the app supplies are part of it. The instructions zone is that size, never
        less than `reserved_instruction_tokens`, so a long preamble takes its room from the passages instead
        of silently overrunning the window. Without it (no prompt yet) the reserve alone is used.
        """
        instructions = self.config.reserved_instruction_tokens
        if instruction_tokens is not None:
            if instruction_tokens > self.config.max_context_tokens // 2:
                raise ProfileConfigError(
                    f"The instructions are about {instruction_tokens} tokens, more than half of "
                    f"max_context_tokens ({self.config.max_context_tokens}); shorten the persona or custom "
                    "instruction, or raise max_context_tokens (§10.1)",
                    stage="context",
                )
            instructions = max(instructions, instruction_tokens)
        usable = max(
            256,
            self.config.max_context_tokens
            - instructions
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
        """The distinct words of a text, as the analyzer sees them, for measuring overlap with the question."""
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
        """Return (history text, tokens used): recent exchanges always, older ones by score (§10.2).

        Order of decisions: the most recent `min_recent_turns_kept` exchanges are always kept (shortened evenly if
        they alone are too big); then the rolling summary, if it fits; then older exchanges, best score first, while
        room remains. Output keeps chronological order, summary first.
        """
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

        # An older exchange scores high when it is recent or shares words with the question ("he" -> the earlier talk
        # about that person); `age` counts exchanges back from now
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
