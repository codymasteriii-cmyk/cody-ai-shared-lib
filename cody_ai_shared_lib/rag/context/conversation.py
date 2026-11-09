"""Conversation state bounds and follow-up reuse (§7.7, RAG-CTX-3, AC-RAG-67)."""
from __future__ import annotations

from typing import Iterable

from ..models import ConversationState
from ..text.analyzer import TextAnalyzer, estimate_tokens

# Words that say how to continue rather than what about, so they never count as new terms (§7.7).
_FOLLOWUP_WORDS = (
    "a about again also an and are as at be been but by can could did do does else elaborate expand explain for "
    "from further give go had has have he her here him his how i in into is it its keep me more my of on or our "
    "please say she should show so some tell than that the their them then there these they this those to us "
    "was we were what when where which who whose why will with would you your continue describe detail details "
    "next said"
)
_FOLLOWUP_PHRASES = (
    "那他", "那她", "然后呢", "继续", "详细说说", "接下来呢", "还有呢", "为什么这样", "为什么", "怎么样", "怎么",
    "告诉我", "多说一点", "请问", "这个", "那个", "他们", "她们", "什么",
)


class FollowupTermCheck:
    """Decides, by comparing words locally, whether a follow-up asks about something new (§7.7)."""

    def __init__(self, analyzer: TextAnalyzer):
        self.analyzer = analyzer
        ignored = _FOLLOWUP_WORDS.split() + list(_FOLLOWUP_PHRASES)
        self.ignored = {token for phrase in ignored for token in self._tokens(phrase)}

    def _tokens(self, text: str) -> list[str]:
        return self.analyzer.tokenize(self.analyzer.normalize(text))

    def introduces_new_terms(self, query: str, known_texts: Iterable[str]) -> bool:
        """True when the query has a meaningful word that none of the known texts contain."""
        query_tokens = set(self._tokens(query)) - self.ignored
        if not query_tokens:
            return False
        known: set[str] = set()
        for text in known_texts:
            known.update(self._tokens(text))
        return bool(query_tokens - known)


def reusable_node_ids(state: ConversationState, max_turns: int, max_nodes: int) -> list[str]:
    """Node ids a follow-up may reuse, newest first; state comes from the client, so it is capped here too."""
    oldest_kept_turn = state.turn_count - max_turns
    ids: list[str] = []
    seen: set[str] = set()
    for ref in reversed(state.prior_nodes):
        if ref.turn > oldest_kept_turn and ref.node_id not in seen:
            seen.add(ref.node_id)
            ids.append(ref.node_id)
            if len(ids) >= max_nodes:
                break
    return ids


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
        seen: set[str] = set()
        kept = []
        for ref in reversed(state.prior_nodes):  # newest first, so a repeated node keeps its latest turn
            if ref.turn > oldest_kept_turn and ref.node_id not in seen:
                seen.add(ref.node_id)
                kept.append(ref)
        state.prior_nodes = list(reversed(kept[: self.max_reused_nodes]))

        while state.rolling_summary and estimate_tokens(state.rolling_summary) > self.rolling_summary_max_tokens:
            state.rolling_summary = state.rolling_summary[: int(len(state.rolling_summary) * 0.9)]

        while state.prior_nodes and self._size(state) > self.max_state_bytes:
            state.prior_nodes.pop(0)
        while state.rolling_summary and self._size(state) > self.max_state_bytes:
            state.rolling_summary = state.rolling_summary[: len(state.rolling_summary) // 2]
        return state
