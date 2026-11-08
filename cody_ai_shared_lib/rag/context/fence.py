"""ID-tagged context fence isolation and sanitization (§10.4, AC-RAG-37).

Encloses untrusted retrieved context and history in cryptographically unpredictable fences:
  <data-ID> ... </data-ID>
Sanitizes raw text to neutralize forged closing or opening fences.
Rotates fence ID on collision.
"""
from __future__ import annotations

import re
from typing import Any
from ..runtime import Rng, SystemRng


class ContextFencer:
    """Isolates untrusted data inside collision-free ID-tagged fences."""

    def __init__(self, rng: Rng | None = None):
        self.rng = rng or SystemRng()

    def generate_fence_id(self) -> str:
        """Generate a random 16-character hex tag."""
        return self.rng.hex(8)

    def sanitize_content(self, text: str) -> str:
        """Neutralize forged opening and closing fences inside data text (§10.4)."""
        # Replace <data- or </data- with sanitized equivalents
        sanitized = re.sub(r"<\s*/?\s*data-", "<sanitized-data-", text, flags=re.IGNORECASE)
        return sanitized

    def fence_block(self, content: str, fence_id: str | None = None) -> tuple[str, str]:
        """Sanitize content and enclose in <data-ID> ... </data-ID>.

        Returns:
            tuple of (fenced_text, active_fence_id)
        """
        sanitized = self.sanitize_content(content)
        tag_id = fence_id or self.generate_fence_id()

        # Check collision: if tag_id appears in sanitized content, rotate tag
        attempts = 0
        while tag_id in sanitized and attempts < 10:
            tag_id = self.generate_fence_id()
            attempts += 1

        fenced = f"<data-{tag_id}>\n{sanitized}\n</data-{tag_id}>"
        return fenced, tag_id
