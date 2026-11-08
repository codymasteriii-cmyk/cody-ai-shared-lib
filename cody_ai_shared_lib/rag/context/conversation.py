"""Conversation state management and bounded history (§7.7, AC-RAG-67)."""
from __future__ import annotations

import json
from typing import Any
from ..models import ConversationState, NodeRef


class ConversationManager:
    """Manages untrusted conversation history and bounds serialized state size."""

    def __init__(
        self,
        max_reused_turns: int = 3,
        max_reused_nodes: int = 24,
        max_state_bytes: int = 16_384,
    ):
        self.max_reused_turns = max_reused_turns
        self.max_reused_nodes = max_reused_nodes
        self.max_state_bytes = max_state_bytes

    def prune_state(self, state: ConversationState) -> ConversationState:
        """Prune node refs and summary if serialized size exceeds bounds (AC-RAG-67)."""
        # 1. Cap node refs to max_reused_nodes
        if len(state.prior_nodes) > self.max_reused_nodes:
            state.prior_nodes = state.prior_nodes[-self.max_reused_nodes :]

        # 2. Check serialized size in bytes
        dumped = state.model_dump_json()
        if len(dumped.encode("utf-8")) > self.max_state_bytes:
            # Drop rolling summary
            state.rolling_summary = None
            # Prune older node refs until within limit
            while len(state.prior_nodes) > 4 and len(state.model_dump_json().encode("utf-8")) > self.max_state_bytes:
                state.prior_nodes.pop(0)

        return state
