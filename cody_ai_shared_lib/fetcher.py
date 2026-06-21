"""Article content fetcher via Jina Reader API.

Shared across all Cody AI projects. Strips common boilerplate (cookie walls,
navigation, consent dialogs, footers) at HTML level using Jina's content-
targeting headers before the page is converted to markdown — preventing
cookie-consent content from polluting stored article text.

Service layer: projects should not call requests.get(r.jina.ai/...) directly.
Import fetch_article from cody_ai_shared_lib.fetcher (or from a project-level
thin wrapper that re-exports it) so all Jina behaviour stays in one place.
"""
import logging
import os

import requests

logger = logging.getLogger("shared-fetcher")

_JINA_BASE = "https://r.jina.ai/"
_DEFAULT_TIMEOUT = 30

# Boilerplate removed at HTML level before Jina converts to markdown.
# Covers cookie consent walls, navigation bars, footers, and GDPR dialogs
# across the majority of modern news and blog sites.
_DEFAULT_REMOVE_SELECTOR = (
    "nav, header, footer, aside, "
    "[class*='cookie'], [id*='cookie'], "
    "[class*='consent'], [id*='consent'], "
    "[class*='banner'], [id*='banner'], "
    "[class*='gdpr'], [id*='gdpr'], "
    "script, style"
)

# Bias toward article content containers when present.
# Falls back to full page gracefully if no selector matches.
_DEFAULT_TARGET_SELECTOR = (
    "article, main, [role='main'], "
    ".post-content, .entry-content, .article-body"
)


def fetch_article(
    url: str,
    timeout: int = _DEFAULT_TIMEOUT,
    remove_selector: str | None = _DEFAULT_REMOVE_SELECTOR,
    target_selector: str | None = _DEFAULT_TARGET_SELECTOR,
) -> str:
    """Fetch article text via Jina Reader.

    Args:
        url:             Full article URL to fetch.
        timeout:         Request timeout in seconds (default 30).
        remove_selector: CSS selectors stripped before markdown conversion.
                         Pass None to skip removal (Jina default behaviour).
        target_selector: CSS selectors for content extraction.
                         Pass None to use the full page.

    Returns:
        Article text as markdown string.

    Raises:
        requests.HTTPError: On non-2xx response.
        requests.Timeout:   If the request exceeds timeout seconds.
    """
    jina_url = f"{_JINA_BASE}{url}"
    headers = {"Accept": "text/plain"}

    if api_key := os.getenv("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    if remove_selector:
        headers["X-Remove-Selector"] = remove_selector
    if target_selector:
        headers["X-Target-Selector"] = target_selector

    logger.info(f"[Fetcher] Fetching: {url}")
    response = requests.get(jina_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text
