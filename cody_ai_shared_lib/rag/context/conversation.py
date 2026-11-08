"""Conversation state bounds (§7.7, RAG-CTX-3, AC-RAG-67)."""
from __future__ import annotations

from ..models import ConversationState
from ..text.analyzer import estimate_tokens


class ConversationManager:
    """Keeps the client-held conversation state small."""

    def __init__(
        self,
        max_reused_turns: int = 3,
        max_reused_nodes: int = 24,
        max_state_bytes: int = 16_384,
        rolling_summary_max_tokens: int = 400,
    ):
        self.max_reused_turns = max_reused_turns
        self.max_reused_nodes = max_reused_nodes
        self.max_state_bytes = max_state_bytes
        self.rolling_summary_max_tokens = rolling_summary_max_tokens

    def _size(self, state: ConversationState) -> int:
        return len(state.model_dump_json().encode("utf-8"))

    def prune_state(self, state: ConversationState) -> ConversationState:
        """Apply the turn and node caps, then prune oldest node refs first, then the summary."""
        oldest_kept_turn = state.turn_count - self.max_reused_turns
        state.prior_nodes = [ref for ref in state.prior_nodes if ref.turn > oldest_kept_turn][-self.max_reused_nodes:]

        while state.rolling_summary and estimate_tokens(state.rolling_summary) > self.rolling_summary_max_tokens:
            state.rolling_summary = state.rolling_summary[: int(len(state.rolling_summary) * 0.9)]

        while state.prior_nodes and self._size(state) > self.max_state_bytes:
            state.prior_nodes.pop(0)
        while state.rolling_summary and self._size(state) > self.max_state_bytes:
            state.rolling_summary = state.rolling_summary[: len(state.rolling_summary) // 2]
        return state
