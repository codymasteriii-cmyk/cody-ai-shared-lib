"""ID-tagged context fence isolation and sanitization (§10.4, AC-RAG-37).

Encloses untrusted retrieved context, conversation history, and user questions in
cryptographically unpredictable fences:
  <data-ID> ... </data-ID>
  <history-ID> ... </history-ID>
  <question-ID> ... </question-ID>
Sanitizes raw text by removing zero-width characters and neutralizing forged opening/closing
fence or node tags. Rotates fence ID on collision.
"""
from __future__ import annotations

import html
import re
from ..runtime import Rng, SystemRng

# Zero-width characters removed before regex scrubbing (§10.4)
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")

# Matches <data-, </data-, etc.
_DATA_HYPHEN_RE = re.compile(r"[<‹]\s*/?\s*data-", flags=re.IGNORECASE)

# Matches any < or ‹ followed by data, history, question, or node (§10.4)
_FENCE_TAG_RE = re.compile(r"[<‹]\s*/?\s*(data|history|question|node)\b", flags=re.IGNORECASE)


def escape_xml_attr(value: str) -> str:
    """XML-escape attribute values (`\"`, `'`, `<`, `>`, `&`) per §10.4."""
    return html.escape(value or "", quote=True).replace("'", "&#x27;")


class ContextFencer:
    """Isolates untrusted data inside collision-free ID-tagged fences (§10.4)."""

    def __init__(self, rng: Rng | None = None):
        self.rng = rng or SystemRng()

    def generate_fence_id(self) -> str:
        """Generate a random 16-character hex tag (64 bits from rng)."""
        return self.rng.hex(8)

    def sanitize_content(self, text: str) -> str:
        """Neutralize zero-width bypasses and forged opening/closing fences inside untrusted text (§10.4)."""
        if not text:
            return ""
        # Step 1: Strip zero-width characters so `</d\u200bata>` cannot evade detection
        cleaned = _ZERO_WIDTH_RE.sub("", text)
        # Step 2: Replace `<data-` / `</data-` with `<sanitized-data-`
        cleaned = _DATA_HYPHEN_RE.sub("<sanitized-data-", cleaned)
        # Step 3: Neutralize any remaining < or ‹ followed by data, history, question, or node tags
        cleaned = _FENCE_TAG_RE.sub(r"<sanitized-\1", cleaned)
        return cleaned

    def ensure_collision_free_id(self, texts: list[str], preferred_id: str | None = None) -> str:
        """Return a fence_id guaranteed not to appear inside any of the given texts."""
        tag_id = preferred_id or self.generate_fence_id()
        # Scrub any malicious fence_id injection
        if not re.fullmatch(r"[0-9a-fA-Fa-zA-Z_\-]{4,64}", tag_id):
            tag_id = self.generate_fence_id()

        combined = "\n".join(texts)
        attempts = 0
        while tag_id in combined and attempts < 10:
            tag_id = self.generate_fence_id()
            attempts += 1
        return tag_id

    def fence_block(self, content: str, fence_id: str | None = None, tag_prefix: str = "data") -> tuple[str, str]:
        """Sanitize content and enclose in <{tag_prefix}-ID> ... </{tag_prefix}-ID>.

        Returns:
            tuple of (fenced_text, active_fence_id)
        """
        sanitized = self.sanitize_content(content)
        tag_id = self.ensure_collision_free_id([sanitized], preferred_id=fence_id)
        fenced = f"<{tag_prefix}-{tag_id}>\n{sanitized}\n</{tag_prefix}-{tag_id}>"
        return fenced, tag_id
