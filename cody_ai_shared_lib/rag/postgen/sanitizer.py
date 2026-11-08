"""Output sanitizer stripping fence leaks and exfiltration markdown (§11.5, AC-RAG-43)."""
from __future__ import annotations

import re


class OutputSanitizer:
    """Sanitizes generation output to prevent exfiltration and tag leaks."""

    def sanitize(self, text: str) -> str:
        # 1. Strip markdown images: ![alt](url) to prevent prompt injection data exfiltration
        cleaned = re.sub(r"!\[(.*?)\]\((.*?)\)", r"[image omitted: \1]", text)

        # 2. Strip fence tags: <data-...> or </data-...> or <sanitized-data-...>
        cleaned = re.sub(r"<\s*/?\s*(?:sanitized-)?data-[a-zA-Z0-9_\-]+\s*>", "", cleaned)

        return cleaned.strip()
