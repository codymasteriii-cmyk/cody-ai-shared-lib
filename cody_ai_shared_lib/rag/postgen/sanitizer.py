"""Output sanitizer stripping fence leaks, control flags, and unauthorized external links/images (§11.6, AC-RAG-43)."""
from __future__ import annotations

import re
from typing import Any, Sequence
from urllib.parse import urlparse


class OutputSanitizer:
    """Sanitizes generation output to prevent exfiltration, unauthorized links, and tag leaks (§11.6)."""

    def __init__(self, allowed_link_hosts: Sequence[str] | None = None):
        self.allowed_link_hosts = {h.lower() for h in (allowed_link_hosts or []) if h}

    def _collect_allowed_hosts(self, nodes: Sequence[Any] | None) -> set[str]:
        hosts = set(self.allowed_link_hosts)
        for node in nodes or []:
            meta = getattr(node, "metadata", None)
            url = None
            if isinstance(meta, dict):
                url = meta.get("url")
            elif meta is not None:
                extra = getattr(meta, "extra", {}) if hasattr(meta, "extra") else {}
                url = getattr(meta, "url", None) or (extra.get("url") if isinstance(extra, dict) else None)
            if url and isinstance(url, str):
                parsed = urlparse(url)
                if parsed.hostname:
                    hosts.add(parsed.hostname.lower())
        return hosts

    def sanitize(
        self,
        text: str,
        nodes: Sequence[Any] | None = None,
        allowed_link_hosts: Sequence[str] | None = None,
        fence_id: str | None = None,
    ) -> str:
        if not text:
            return ""

        # 1. Keep images and links only on trusted hosts; otherwise keep the text and drop the URL (§11.6)
        permitted_hosts = self._collect_allowed_hosts(nodes)
        if allowed_link_hosts:
            permitted_hosts.update(h.lower() for h in allowed_link_hosts if h)

        def _is_trusted(url: str) -> bool:
            host = (urlparse(url.strip()).hostname or "").lower()
            return bool(host) and any(host == h or host.endswith("." + h) for h in permitted_hosts)

        cleaned = re.sub(
            r"!\[(.*?)\]\(\s*<?([^)]*?)>?\s*\)",
            lambda m: m.group(0) if _is_trusted(m.group(2)) else m.group(1),
            text,
        )
        cleaned = re.sub(
            r"(?<!!)\[([^\]]+)\]\(\s*<?((?:https?:)?//[^)]*?)>?\s*\)",
            lambda m: m.group(0) if _is_trusted(m.group(2)) else m.group(1),
            cleaned,
            flags=re.IGNORECASE,
        )

        # 3. Strip fence tags: <data-...>, <history-...>, <question-...>, <node...>, </node>
        cleaned = re.sub(
            r"<\s*/?\s*(?:sanitized-)?(?:data|history|question)-[a-zA-Z0-9_\-]+\s*>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"<\s*/?\s*node\b[^>]*>", "", cleaned, flags=re.IGNORECASE)

        # 4. Strip raw fence_id if leaked
        if fence_id:
            cleaned = cleaned.replace(fence_id, "")

        # 5. Strip any residual [[answer_found=...]] control flags
        cleaned = re.sub(r"\[\[\s*answer_found\s*=\s*[^\]]*\]\]", "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()
