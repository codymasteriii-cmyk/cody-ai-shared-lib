"""Streaming buffer for [[answer_found=yes|no|partial]] flag detection (§10.5, AC-RAG-72).

Checks output start character-by-character and releases text at the first moment it cannot
be a flag. Never delays the first token when the text has ruled a flag out.
"""
from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger("shared-rag.generation")

FLAG_PREFIX = "[[answer_found="
FLAG_SUFFIX = "]]"
MAX_HELD_CHARS = 32
VALID_VALUES = frozenset(["yes", "no", "partial"])


class FlagStreamBuffer:
    """State machine buffering stream start for [[answer_found=...]] flag."""

    def __init__(self) -> None:
        self.state: Literal["leading_ws", "matching_prefix", "reading_value", "discard_ws_after", "done"] = "leading_ws"
        self.held: list[str] = []
        self.held_non_ws_count: int = 0
        self.answer_found: Literal["yes", "no", "partial"] | None = None
        self.prefix_index: int = 0

    def feed(self, chunk: str) -> list[str]:
        """Feed a new chunk of streamed text. Return deltas to emit."""
        if not chunk:
            return []
        if self.state == "done":
            return [chunk]

        deltas: list[str] = []

        for idx, char in enumerate(chunk):
            if self.state == "leading_ws":
                if char.isspace():
                    # Rule 1: Leading whitespace is discarded
                    continue
                else:
                    self.state = "matching_prefix"
                    # Fall through to matching_prefix for this character

            if self.state == "matching_prefix":
                expected = FLAG_PREFIX[self.prefix_index]
                if char.lower() == expected.lower():
                    self.held.append(char)
                    self.held_non_ws_count += 1
                    self.prefix_index += 1
                    if self.prefix_index == len(FLAG_PREFIX):
                        self.state = "reading_value"
                else:
                    # Mismatch! Rule 2: Release everything held + remainder of chunk
                    self.held.append(char)
                    released = "".join(self.held) + chunk[idx + 1:]
                    deltas.append(released)
                    self.held = []
                    self.state = "done"
                    self.answer_found = None
                    break
                continue

            if self.state == "reading_value":
                self.held.append(char)
                self.held_non_ws_count += 1
                held_str = "".join(self.held)

                if held_str.endswith(FLAG_SUFFIX):
                    # Found closing ]]
                    raw_val = held_str[len(FLAG_PREFIX):-len(FLAG_SUFFIX)].strip().lower()
                    if raw_val in VALID_VALUES:
                        self.answer_found = raw_val  # type: ignore[assignment]
                    else:
                        logger.warning(f"Invalid answer_found flag: {raw_val!r} (removed from output)")
                        self.answer_found = None

                    # Strip flag from output, discard following whitespace
                    self.held = []
                    self.state = "discard_ws_after"
                elif self.held_non_ws_count >= MAX_HELD_CHARS:
                    # Rule 4: Cap exceeded (32 non-whitespace chars without closing ]])
                    released = held_str + chunk[idx + 1:]
                    deltas.append(released)
                    self.held = []
                    self.state = "done"
                    self.answer_found = None
                    break
                continue

            if self.state == "discard_ws_after":
                if char.isspace():
                    continue
                else:
                    self.state = "done"
                    deltas.append(chunk[idx:])
                    break

        return deltas

    def finalize(self) -> list[str]:
        """Called at end of stream. Releases any held text as a last delta."""
        if self.state == "done":
            return []

        deltas: list[str] = []
        if self.held:
            # End of stream while text is held: released as a last delta (§10.5 rule 5)
            deltas.append("".join(self.held))
            self.held = []
            self.answer_found = None

        self.state = "done"
        return deltas
