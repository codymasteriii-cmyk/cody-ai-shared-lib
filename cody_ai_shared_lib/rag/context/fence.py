"""ID-tagged context fences and scrubbing of untrusted text (§10.4, AC-RAG-37).

Untrusted retrieved context, history and questions are enclosed in fences with an unpredictable id:
  <data-ID> ... </data-ID>, <history-ID> ... </history-ID>, <question-ID> ... </question-ID>
"""
from __future__ import annotations

import hashlib
import html
import re

from ..runtime import Rng, SystemRng

_ZERO_WIDTH_RE = re.compile(r"[​-‍⁠﻿]")

# The `<` of anything that could be read as one of our structural tags (§10.4)
_FENCE_TAG_RE = re.compile(r"<(?=\s*/?\s*(?:data|history|question|node)\b)", flags=re.IGNORECASE)

_FENCE_ID_RE = re.compile(r"[0-9a-zA-Z_\-]{4,64}")

FENCE_ID_HEX_CHARS = 16


def escape_xml_attr(value: str) -> str:
    """XML-escape attribute values (`\"`, `'`, `<`, `>`, `&`) per §10.4."""
    return html.escape(value or "", quote=True).replace("'", "&#x27;")


class ContextFencer:
    """Isolates untrusted data inside collision-free ID-tagged fences (§10.4)."""

    def __init__(self, rng: Rng | None = None):
        self.rng = rng or SystemRng()
        self._secret = self.rng.hex(16)

    def generate_fence_id(self) -> str:
        """Generate a random 16-character hex id (64 bits from rng)."""
        return self.rng.hex(FENCE_ID_HEX_CHARS // 2)

    def stable_id(self, scope: str) -> str:
        """A fence id for stateless calls: unpredictable, but the same for the same scope (D-06)."""
        return hashlib.sha256(f"{self._secret}|{scope}".encode("utf-8")).hexdigest()[:FENCE_ID_HEX_CHARS]

    def sanitize_content(self, text: str) -> str:
        """Remove zero-width characters, then neutralize forged structural tags (§10.4)."""
        if not text:
            return ""
        return _FENCE_TAG_RE.sub("‹", _ZERO_WIDTH_RE.sub("", text))

    def ensure_collision_free_id(self, texts: list[str], preferred_id: str | None = None) -> str:
        """Return a fence id that appears in none of the given texts, rotating it on a collision."""
        tag_id = preferred_id if preferred_id and _FENCE_ID_RE.fullmatch(preferred_id) else self.generate_fence_id()
        combined = _ZERO_WIDTH_RE.sub("", "\n".join(texts)).casefold()
        attempts = 0
        while tag_id.casefold() in combined and attempts < 10:
            tag_id = self.generate_fence_id()
            attempts += 1
        return tag_id

    def fence_block(self, content: str, fence_id: str | None = None, tag_prefix: str = "data") -> tuple[str, str]:
        """Sanitize content and enclose it in <{tag_prefix}-ID> tags; returns (fenced text, id used)."""
        sanitized = self.sanitize_content(content)
        tag_id = self.ensure_collision_free_id([sanitized], preferred_id=fence_id)
        return f"<{tag_prefix}-{tag_id}>\n{sanitized}\n</{tag_prefix}-{tag_id}>", tag_id
