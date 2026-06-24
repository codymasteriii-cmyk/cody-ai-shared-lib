"""Article content fetcher — Jina Reader for HTML, direct extraction for PDFs.

Shared across all Cody AI projects. Strips common boilerplate (cookie walls,
navigation, consent dialogs, footers) at HTML level using Jina's content-
targeting headers before the page is converted to markdown.

PDF handling: Jina is an HTML→Markdown converter and cannot render PDF binary.
Direct .pdf URLs bypass Jina entirely and are extracted with pypdf. If Jina
returns an empty Markdown Content section for any other URL, a ValueError is
raised so the caller knows the fetch failed — no silent empty-text storage.

Security: validate_public_url() is called once at the fetch_article() entry
point, blocking SSRF attempts via private IP literals, localhost, and non-HTTP
schemes. Both the Jina path and the direct PDF download path use the validated
URL, so the guard is applied regardless of which path is taken.

Service layer: projects should not call requests.get() or r.jina.ai directly.
Import fetch_article from cody_ai_shared_lib.fetcher so all fetch behaviour
stays in one place.
"""
import io
import logging
import os
from urllib.parse import urlparse

import requests

from .url_validator import validate_public_url

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

# Sent with direct PDF downloads. Many academic servers reject the default
# Python requests User-Agent; a browser-like string avoids most 403s.
_PDF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_pdf_url(url: str) -> bool:
    """Return True if the URL path ends with .pdf (case-insensitive)."""
    return urlparse(url).path.lower().endswith(".pdf")


def _has_empty_jina_content(text: str) -> bool:
    """Return True when Jina's Markdown Content section is effectively blank.

    Jina's text response always includes a 'Markdown Content:' section header.
    When it cannot render the page body (bot detection, paywall, auth gate,
    or a PDF binary served at an HTML URL), the header is present but the body
    is empty. Threshold of 50 chars filters out noise like a lone newline.
    """
    marker = "Markdown Content:"
    idx = text.find(marker)
    if idx == -1:
        return False
    return len(text[idx + len(marker):].strip()) < 50


def _fetch_pdf(url: str, timeout: int) -> str:
    """Download a PDF and extract page text using pypdf.

    Output is formatted to match the Jina Reader response structure so callers
    handle both fetch paths identically.

    Raises:
        requests.HTTPError: On non-2xx response (e.g. 403 for auth-gated PDFs
                            such as SSRN — no fix possible without credentials).
        pypdf.errors.PdfReadError: If the downloaded content is not a valid PDF.
    """
    import pypdf  # lazy import — not needed for HTML-only callers

    logger.info(f"[Fetcher] Downloading PDF directly: {url}")
    response = requests.get(url, headers=_PDF_HEADERS, timeout=timeout)
    response.raise_for_status()

    reader = pypdf.PdfReader(io.BytesIO(response.content))
    page_texts = [
        page.extract_text() or ""
        for page in reader.pages
    ]
    # Drop blank pages; join remaining with paragraph spacing
    full_text = "\n\n".join(t for t in page_texts if t.strip())
    n_pages = len(reader.pages)

    return (
        f"URL Source: {url}\n\n"
        f"Number of Pages: {n_pages}\n\n"
        f"Markdown Content:\n{full_text}"
    )


def _fetch_via_jina(
    url: str,
    timeout: int,
    remove_selector: str | None,
    target_selector: str | None,
) -> str:
    jina_url = f"{_JINA_BASE}{url}"
    headers = {"Accept": "text/plain"}
    if api_key := os.getenv("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    if remove_selector:
        headers["X-Remove-Selector"] = remove_selector
    if target_selector:
        headers["X-Target-Selector"] = target_selector

    response = requests.get(jina_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_article(
    url: str,
    timeout: int = _DEFAULT_TIMEOUT,
    remove_selector: str | None = _DEFAULT_REMOVE_SELECTOR,
    target_selector: str | None = _DEFAULT_TARGET_SELECTOR,
) -> str:
    """Fetch article text, routing to the appropriate extractor.

    - Direct .pdf URLs bypass Jina and extract text with pypdf.
    - All other URLs use Jina Reader. If Jina returns a 200 OK but with empty
      Markdown Content (paywall, bot detection, auth gate), a ValueError is
      raised — callers must not silently store empty article text.

    Args:
        url:             Full article URL to fetch.
        timeout:         Request timeout in seconds (default 30).
        remove_selector: CSS selectors stripped before Jina markdown conversion.
                         Pass None to skip (Jina default behaviour).
        target_selector: CSS selectors for Jina content targeting.
                         Pass None to use the full page.

    Returns:
        Article text. Format matches Jina Reader output in all cases so callers
        need no special handling for the PDF path.

    Raises:
        ValueError:         If url fails the SSRF safety check.
        requests.HTTPError: On non-2xx response (e.g. 403 for auth-gated PDFs).
        requests.Timeout:   If the request exceeds timeout seconds.
    """
    # SSRF guard applied once here — covers both Jina and direct download paths.
    validate_public_url(url)

    if _is_pdf_url(url):
        return _fetch_pdf(url, timeout)

    logger.info(f"[Fetcher] Fetching via Jina: {url}")
    result = _fetch_via_jina(url, timeout, remove_selector, target_selector)

    # A 200 OK with empty Markdown Content is a content failure, not an HTTP
    # failure — raise_for_status() won't catch it. Surface it explicitly so
    # callers don't silently store empty article text. Possible causes: bot
    # detection, paywall, auth gate, or a PDF served without a .pdf extension
    # (add .pdf handling at the call site or rename the URL if that's the case).
    if _has_empty_jina_content(result):
        raise ValueError(
            f"Jina returned empty content for {url}. "
            "Possible causes: bot detection, paywall, authentication required, "
            "or a PDF binary served without a .pdf URL extension."
        )

    return result
