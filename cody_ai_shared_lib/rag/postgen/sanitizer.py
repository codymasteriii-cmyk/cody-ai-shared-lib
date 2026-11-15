"""Output sanitizer stripping fence leaks, control flags, and unauthorized external links/images (§11.6, AC-RAG-43).

The last step before the answer leaves the engine. A retrieved document or the user's own text could try to make the model
write a link or an image whose URL carries data out (an exfiltration trick), or the model could echo internal markup.
The sanitizer removes, in order: images and links on untrusted hosts (the visible text is kept, the URL dropped); reference-style
link definitions and autolinks on untrusted hosts; our own fence tags and the fence id; and any leftover [[answer_found=...]] flag.

A host is trusted when the profile lists it (`allowed_link_hosts`) or when a source passage's own metadata URL uses it.
"""
from __future__ import annotations

import re
from typing import Any, Sequence
from urllib.parse import urlparse


class OutputSanitizer:
    """Sanitizes generation output to prevent exfiltration, unauthorized links, and tag leaks (§11.6).

    >>> OutputSanitizer().sanitize("See [the site](https://evil.example/x) now")
    'See the site now'
    >>> OutputSanitizer(allowed_link_hosts=["example.com"]).sanitize("[ok](https://docs.example.com/a)")
    '[ok](https://docs.example.com/a)'
    """

    def __init__(self, allowed_link_hosts: Sequence[str] | None = None):
        self.allowed_link_hosts = {h.lower() for h in (allowed_link_hosts or []) if h}

    def _collect_allowed_hosts(self, nodes: Sequence[Any] | None) -> set[str]:
        """The configured hosts plus the host of each source passage's own `url` metadata (a news article may link its source)."""
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
        """Return the answer text made safe to show. `nodes` and `fence_id` come from the prompt the answer was written for."""
        if not text:
            return ""

        # Keep images and links only on trusted hosts; otherwise keep the text and drop the URL (§11.6)
        permitted_hosts = self._collect_allowed_hosts(nodes)
        if allowed_link_hosts:
            permitted_hosts.update(h.lower() for h in allowed_link_hosts if h)

        def _is_trusted(url: str) -> bool:
            host = (urlparse(url.strip()).hostname or "").lower()
            return bool(host) and any(host == h or host.endswith("." + h) for h in permitted_hosts)

        cleaned = re.sub(
            r"!\[(.*?)\]\(\s*<?((?:[^()]|\([^()]*\))*?)>?\s*\)",
            lambda m: m.group(0) if _is_trusted(m.group(2)) else m.group(1),
            text,
        )
        cleaned = re.sub(
            r"(?<!!)\[([^\]]+)\]\(\s*<?((?:[^()]|\([^()]*\))*?)>?\s*\)",
            lambda m: m.group(0) if _is_trusted(m.group(2)) else m.group(1),
            cleaned,
        )
        # Reference-style definitions `[x]: https://...` would turn `![img][x]` into a remote image
        cleaned = re.sub(
            r"^[ \t]*\[[^\]]+\]:[ \t]*<?(\S+?)>?(?:[ \t]+\S.*)?$",
            lambda m: m.group(0) if _is_trusted(m.group(1)) else "",
            cleaned,
            flags=re.MULTILINE,
        )
        # Autolinks `<scheme://host/...>`: keep the text form only for trusted hosts
        cleaned = re.sub(
            r"<([a-zA-Z][a-zA-Z0-9+.\-]*:[^\s<>]+)>",
            lambda m: m.group(0) if _is_trusted(m.group(1)) else "",
            cleaned,
        )

        # Strip fence tags, including a scrubbed copy: <data-...>, <history-...>, <question-...>, <node...>, </node>
        cleaned = re.sub(
            r"[<‹]\s*/?\s*(?:data|history|question)-[a-zA-Z0-9_\-]+\s*>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[<‹]\s*/?\s*node\b[^>]*>", "", cleaned, flags=re.IGNORECASE)

        # Strip the raw fence id if it leaked
        if fence_id:
            cleaned = cleaned.replace(fence_id, "")

        # Strip any residual [[answer_found=...]] control flags
        cleaned = re.sub(r"\[\[\s*answer_found\s*=\s*[^\]]*\]\]", "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()
