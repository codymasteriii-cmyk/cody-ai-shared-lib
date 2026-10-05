"""Article content fetcher — Jina Reader for HTML, direct extraction for PDFs.

Shared across all Cody AI projects. Strips common boilerplate (cookie walls,
navigation, consent dialogs, footers) at HTML level using Jina's content-
targeting headers before the page is converted to markdown.

PDF handling: Jina is an HTML→Markdown converter and cannot render PDF binary.
Direct .pdf URLs bypass Jina entirely and are extracted with pymupdf4llm. This
preserves tables, multi-column layouts, and outputs native Markdown matching the
Jina HTML response. If an empty Markdown Content section is returned for any URL,
a ValueError is raised so the caller knows the fetch failed — no silent empty-text storage.

Security: validate_public_url() is called once at the fetch_article() entry
point, blocking non-HTTP schemes and any host that resolves to a non-public
address. The direct PDF download additionally goes through safe_get(), which
re-validates every redirect hop, so a public URL cannot bounce the server to an
internal address.

Service layer: projects should not call requests.get() or r.jina.ai directly.
Import fetch_article from cody_ai_shared_lib.fetcher so all fetch behaviour
stays in one place.
"""
import logging
import os
from urllib.parse import urlparse

import requests

from .url_validator import safe_get, validate_public_url

logger = logging.getLogger("shared-fetcher")

_JINA_BASE = "https://r.jina.ai/"
_DEFAULT_TIMEOUT = 30

# Direct PDF downloads are held in memory (and parsed from that buffer), so both the
# download size and the number of converted pages are capped. Defaults suit a 512 MB
# host; callers with more headroom can override them via fetch_article().
#   - An oversized PDF is SKIPPED (ValueError): a truncated download cannot be parsed.
#   - A PDF with too many pages keeps its first _MAX_PDF_PAGES pages.
_MAX_PDF_BYTES = 30 * 1024 * 1024
_MAX_PDF_PAGES = 150

# Boilerplate removed at HTML level before Jina converts to markdown.
# Covers cookie consent walls, navigation bars, footers, and GDPR dialogs
# across the majority of modern news and blog sites.
#
# CMP-specific patterns (consent management platforms that don't use 'cookie'
# or 'consent' in their class/id names):
#   CookieYes  — cky-* prefix (cky-modal, cky-btn-revisit, cky-notice, ...)
#   OneTrust   — onetrust-* (most IDs contain 'consent', but some use onetrust-
#                only, and the ot-sdk-* class prefix isn't caught otherwise)
#   Cookiebot  — CybotCookiebot* (CamelCase, not caught by lowercase matchers)
_DEFAULT_REMOVE_SELECTOR = (
    "nav, header, footer, aside, "
    "[class*='cookie'], [id*='cookie'], "
    "[class*='consent'], [id*='consent'], "
    "[class*='banner'], [id*='banner'], "
    "[class*='gdpr'], [id*='gdpr'], "
    "[class*='cky'], [id*='cky'], "
    "[class*='onetrust'], [id*='onetrust'], "
    "[id*='CybotCookiebot'], [class*='CybotCookiebot'], "
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

# Substrings that Jina embeds in the response body when the target URL errors.
# Checked case-insensitively. Callers use this to detect content failures
# before attempting to classify or store the returned text.
JINA_ERROR_SIGNALS: tuple[str, ...] = (
    "target url returned error",
    "403: forbidden",
    "404: not found",
    "403 forbidden",
    "404 not found",
    "502 bad gateway",
    "503 service unavailable",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_pdf_url(url: str) -> bool:
    """Return True if the URL refers to a PDF document.

    Handles two cases:
    - Path ends with .pdf (query-string-safe: 'paper.pdf?X-Amz-Signature=...' → True)
    - arXiv PDF URLs: arxiv.org/pdf/<id> serves PDFs without a .pdf extension
    """
    parsed = urlparse(url)
    if parsed.path.lower().endswith(".pdf"):
        return True
    # arXiv /pdf/<id> URLs serve PDF binaries without a .pdf extension
    if "arxiv.org" in parsed.netloc and parsed.path.lower().startswith("/pdf/"):
        return True
    return False


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


def _fetch_pdf(
    url: str,
    timeout: int,
    max_bytes: int = _MAX_PDF_BYTES,
    max_pages: int = _MAX_PDF_PAGES,
) -> str:
    """Download a PDF and extract structural Markdown using pymupdf4llm.

    Output is formatted to match the Jina Reader response structure so callers
    handle both fetch paths identically. pymupdf4llm natively produces Markdown
    that perfectly preserves tables and multi-column layouts, matching Jina's
    style.

    Memory strategy: pages are converted one at a time via the pages=[i] argument
    rather than converting the whole document in a single call. pymupdf4llm renders
    each page through the C-level MuPDF engine, which allocates a large working buffer
    per page. Processing the full document at once holds all page buffers simultaneously;
    page-by-page processing keeps peak RAM bounded to roughly one page at a time,
    preventing OOM crashes on low-memory hosts (e.g. Render hobby plan, 512 MB RAM)
    when processing large academic or quantitative research PDFs (50+ pages).

    Download strategy: the body is streamed and counted as it arrives. A declared
    Content-Length over max_bytes is rejected before any body is read; the running
    count also stops responses with no (or a false) Content-Length. Counting happens
    after decompression, so a compressed response cannot slip past the cap.

    Raises:
        ValueError:          If the PDF is larger than max_bytes.
        requests.HTTPError:  On non-2xx response (e.g. 403 for auth-gated PDFs
                             such as SSRN — no fix possible without credentials).
        fitz.FileDataError:  If the downloaded content is not a valid PDF.
    """
    import fitz          # lazy import — provided by pymupdf
    import pymupdf4llm   # lazy import — not needed for HTML-only callers

    logger.info(f"[Fetcher] Downloading PDF directly: {url}")
    # This is the one path where the SERVER itself fetches an article URL (the Jina
    # path is fetched by Jina). safe_get() validates the URL and every redirect hop.
    response = safe_get(url, headers=_PDF_HEADERS, timeout=timeout, stream=True)
    try:
        response.raise_for_status()

        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_bytes:
            raise ValueError(
                f"PDF is {int(declared) / 1_048_576:.1f} MB, over the "
                f"{max_bytes / 1_048_576:g} MB limit: {url}"
            )
        pdf_bytes = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            pdf_bytes.extend(chunk)
            if len(pdf_bytes) > max_bytes:
                raise ValueError(
                    f"PDF exceeds the {max_bytes / 1_048_576:g} MB limit while downloading: {url}"
                )
    finally:
        response.close()

    # Open PDF from memory stream. Explicit close() — fitz.Document wraps
    # native MuPDF memory that Python's GC doesn't account for, and this path
    # runs inside batch loops (backfill/regenerate scripts processing many PDFs).
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        n_pages = len(doc)
        converted = min(n_pages, max_pages)
        if converted < n_pages:
            logger.warning(
                f"[Fetcher] PDF has {n_pages} pages; converting the first {converted} only: {url}"
            )
        # Convert one page at a time to bound peak RAM usage. Each page's MuPDF
        # render buffer is freed before the next page begins. Results are joined
        # with double-newlines to preserve paragraph separation across page breaks.
        page_markdowns = [
            pymupdf4llm.to_markdown(doc, pages=[i])
            for i in range(converted)
        ]
        full_text = "\n\n".join(page_markdowns)
    finally:
        doc.close()

    truncation_note = (
        f"Pages Converted: {converted} of {n_pages} (page limit)\n\n" if converted < n_pages else ""
    )
    return (
        f"URL Source: {url}\n\n"
        f"Number of Pages: {n_pages}\n\n"
        f"{truncation_note}"
        f"Markdown Content:\n{full_text}"
    )


def _fetch_via_jina(
    url: str,
    timeout: int,
    remove_selector: str | None,
    target_selector: str | None,
    retain_images: bool = False,
) -> str:
    jina_url = f"{_JINA_BASE}{url}"
    headers = {"Accept": "text/plain"}
    if api_key := os.getenv("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    if remove_selector:
        headers["X-Remove-Selector"] = remove_selector
    if target_selector:
        headers["X-Target-Selector"] = target_selector
    if not retain_images:
        # Tells Jina not to emit ![alt](url) markdown for <img> tags.
        # Image URLs are never useful for text-based LLM classification.
        headers["X-Retain-Images"] = "none"

    response = requests.get(jina_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_article(
    url: str,
    timeout: int = _DEFAULT_TIMEOUT,
    remove_selector: str | None = _DEFAULT_REMOVE_SELECTOR,
    target_selector: str | None = _DEFAULT_TARGET_SELECTOR,
    retain_images: bool = False,
    max_pdf_bytes: int = _MAX_PDF_BYTES,
    max_pdf_pages: int = _MAX_PDF_PAGES,
) -> str:
    """Fetch article text, routing to the appropriate extractor.

    - Direct .pdf URLs bypass Jina and extract structural Markdown with pymupdf4llm.
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
        retain_images:   If False (default), Jina strips all ![alt](url) image
                         markdown — image URLs are never useful for LLM text
                         classification. Pass True to keep them (e.g. for visual
                         content audits).
        max_pdf_bytes:   Largest direct-download PDF accepted (default 30 MB).
                         Larger files raise ValueError.
        max_pdf_pages:   Pages converted per PDF (default 150); extra pages dropped.

    Returns:
        Article text. Format matches Jina Reader output in all cases so callers
        need no special handling for the PDF path.

    Raises:
        ValueError:         If url fails the SSRF safety check, or a PDF exceeds
                            max_pdf_bytes.
        requests.HTTPError: On non-2xx response (e.g. 403 for auth-gated PDFs).
        requests.Timeout:   If the request exceeds timeout seconds.
    """
    # SSRF guard applied once here — covers both Jina and direct download paths.
    validate_public_url(url)

    if is_pdf_url(url):
        return _fetch_pdf(url, timeout, max_pdf_bytes, max_pdf_pages)

    logger.info(f"[Fetcher] Fetching via Jina: {url}")
    result = _fetch_via_jina(url, timeout, remove_selector, target_selector, retain_images)

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
