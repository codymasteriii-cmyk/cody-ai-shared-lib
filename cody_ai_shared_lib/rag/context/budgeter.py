"""Token budgets per prompt zone and history trimming (§10.1, §10.2, AC-RAG-38)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models import ChatTurn
from ..profile import ContextConfig
from ..text.analyzer import estimate_tokens


@dataclass
class BudgetAllocation:
    """Computed token budgets per prompt zone (§10.1)."""
    usable_budget: int
    history_budget: int
    summary_budget: int
    local_window_budget: int
    retrieved_budget: int


class TokenBudgeter:
    """Computes token budgets per zone and trims content to fit (§10.1, AC-RAG-38)."""

    def __init__(self, config: ContextConfig):
        self.config = config

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
        hist_b = int(usable * self.config.history_ratio) if has_history else 0
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

    def format_history_with_decay(
        self,
        history: Sequence[ChatTurn],
        rolling_summary: str | None,
        budget_tokens: int,
    ) -> tuple[str, int]:
        """Format conversation history applying exponential turn decay (§10.2).

        Returns:
            (formatted_history_text, tokens_used)
        """
        if budget_tokens <= 0 or (not history and not rolling_summary):
            return "", 0

        lines: list[str] = []
        used_tokens = 0

        if rolling_summary:
            sum_line = f"[Prior Conversation Summary]: {rolling_summary}"
            sum_tok = estimate_tokens(sum_line)
            if sum_tok <= budget_tokens:
                lines.append(sum_line)
                used_tokens += sum_tok

        if not history:
            return "\n".join(lines), used_tokens

        recent_full = max(1, self.config.recent_turns_full * 2)  # pairs -> individual messages
        decay = self.config.turn_decay
        cutoff = self.config.turn_summary_cutoff_tokens

        # Walk from newest to oldest so newest turns get priority
        selected_Newest_first: list[str] = []
        for rev_idx, turn in enumerate(reversed(history)):
            age_pairs = rev_idx // 2
            content = turn.content or ""
            toks = estimate_tokens(content)

            if rev_idx >= recent_full and toks > cutoff:
                allowed_toks = max(32, int(toks * (decay ** age_pairs)))
                if allowed_toks < toks:
                    # Trim proportionally by character length
                    ratio = allowed_toks / max(1, toks)
                    max_chars = max(40, int(len(content) * ratio))
                    content = content[:max_chars] + "…"
                    toks = estimate_tokens(content)

            formatted = f"{turn.role.capitalize()}: {content}"
            line_toks = estimate_tokens(formatted)
            if used_tokens + line_toks > budget_tokens:
                break
            selected_Newest_first.append(formatted)
            used_tokens += line_toks

        selected_Newest_first.reverse()
        lines.extend(selected_Newest_first)
        return "\n".join(lines), used_tokens
