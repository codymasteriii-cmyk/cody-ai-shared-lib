"""Article content fetcher — Jina Reader for HTML, direct extraction for PDFs.

Shared across all Cody AI projects. Strips common boilerplate (cookie walls,
navigation, consent dialogs, footers) at HTML level using Jina's content-
targeting headers before the page is converted to markdown.

PDF handling: Jina is an HTML→Markdown converter and cannot render PDF binary.
Direct .pdf URLs bypass Jina entirely and are extracted as plain text with pdfminer.six. This
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
import re
import time
from urllib.parse import urlparse, urlsplit, urlunsplit

import requests

from .pdf_text import PdfTextError
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
# Pages sampled to learn repeated headers/footers. Kept small because fetch_article callers usually
# stop reading after a few thousand words (see max_pdf_chars); these pages are read anyway.
_PDF_PROFILE_PAGES = 10
# Pages are read until this many characters have been collected, then the cleaned text is returned
# whole. Cleaning (header/footer removal) can drop a little, so the stop point carries a margin.
_PDF_CHARS_MARGIN = 1.15

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

def normalize_url(url: str) -> str:
    """Return canonical form of a URL for deduplication and Jina fetching.

    - Strips whitespace.
    - Lowercases scheme and host (RFC 3986).
    - Collapses accidental consecutive slashes in the path (e.g. '//writing/x' -> '/writing/x').
    - Strips trailing slash on non-root paths while leaving root '/' and query/fragment intact.
    """
    url = (url or "").strip()
    if not url:
        return url
    parts = urlsplit(url)
    path = re.sub(r"/+", "/", parts.path)
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, parts.fragment))


def extract_title_and_markdown(raw_fetched: str, fallback_url: str = "") -> tuple[str, str]:
    """Extract human-readable (title, markdown_body) from Jina Reader or _fetch_pdf output.

    1. Looks for 'Title: <title>' in the Jina header block.
    2. Splits body after 'Markdown Content:'.
    3. Falls back to the first '# ' H1 heading in the Markdown body, or the URL path slug.
    """
    title = ""
    for line in raw_fetched.splitlines()[:10]:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            break

    marker = "Markdown Content:"
    idx = raw_fetched.find(marker)
    if idx != -1:
        body = raw_fetched[idx + len(marker):].strip()
    else:
        body = raw_fetched.strip()

    if not title:
        for line in body.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

    if not title and fallback_url:
        parsed = urlparse(fallback_url)
        clean_path = parsed.path.strip("/").split("/")[-1]
        if clean_path.lower().endswith(".pdf"):
            clean_path = clean_path[:-4]
        title = (
            re.sub(r"[-_]+", " ", clean_path).strip().title()
            if clean_path
            else parsed.netloc
        )

    return (title or "Untitled Article"), body


def extract_pdf_text(
    pdf_bytes: bytes | bytearray,
    max_pages: int = _MAX_PDF_PAGES,
    max_chars: int | None = None,
    max_seconds: float | None = None,
    source_label: str = "document.pdf",
) -> tuple[str, int, int]:
    """Extract plain text (paragraphs, no Markdown) from an in-memory PDF via pdfminer.six.

    Reads one page at a time through pdf_text.PdfTextReader: pure Python, roughly 50-70 MB for an
    ordinary PDF where the pymupdf4llm path needed 200-450 MB. Repeated headers/footers and page
    numbers are removed. Tables come out as ordinary text lines, not Markdown pipes.

    Args:
        max_pages: Pages read at most; later pages are dropped.
        max_chars: When given, stop reading once about this many characters have been collected
                   (a caller that only uses the first N characters need not read the whole file).
                   The cleaned text is returned in full, so it can exceed max_chars by up to a page.
        max_seconds: When given, stop starting new pages after this many seconds of reading and
                   return what was read (pdfminer needs about 0.3-0.5 s per page, so a long deck of
                   mostly images could otherwise run for minutes). At least the profile pages are
                   always read first, whatever the budget.

    Returns:
        Tuple of (text, total_pages, converted_pages).

    Raises:
        PdfEncryptedError: The PDF is password-protected.
        PdfTextError:      The PDF is corrupt/unreadable or has no pages.
        ValueError:        The PDF has no extractable text (for example a scan without a text layer).
    """
    from .pdf_text import PdfTextReader

    started = time.monotonic()
    reader = PdfTextReader(pdf_bytes, profile_pages=_PDF_PROFILE_PAGES)
    try:
        reader.prepare()    # reads and caches the profile pages up front, so the loop below reuses them
        n_pages = len(reader)
        limit = min(n_pages, max_pages)
        stop_at = int(max_chars * _PDF_CHARS_MARGIN) if max_chars else None
        pages = []
        collected = converted = 0
        for i in range(limit):
            converted = i + 1
            try:
                page = reader.page_boxes(i)
            except MemoryError:
                raise
            except Exception as exc:    # a damaged page costs only that page
                logger.warning(f"[Fetcher] PDF page {i + 1} skipped ({exc}): {source_label}")
                continue
            pages.append(page)
            collected += sum(len(line) for box in page.boxes for line in box.lines)
            if stop_at and collected >= stop_at:
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                logger.warning(f"[Fetcher] PDF read stopped at {max_seconds:g}s, page {i + 1} of {n_pages}: {source_label}")
                break
        if converted < n_pages:
            logger.info(f"[Fetcher] PDF has {n_pages} pages; read the first {converted}: {source_label}")
        text = reader.clean_pages(pages).strip()
    finally:
        reader.close()

    if not text:
        raise ValueError(f"PDF has no extractable text (scanned images without a text layer?): {source_label}")
    return text, n_pages, converted


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
    max_chars: int | None = None,
    max_seconds: float | None = None,
) -> str:
    """Download a PDF and extract plain text with pdfminer.six (see extract_pdf_text).

    Output is framed like the Jina Reader response ("URL Source", "Number of Pages",
    "Markdown Content:") so callers handle both fetch paths identically. The body is
    plain paragraphs, not Markdown.

    Memory strategy: pdfminer is pure Python and the reader works one page at a time, so
    peak RAM stays near 50-70 MB on ordinary PDFs. The pymupdf4llm path this replaced
    reached 380-450 MB on a 512 MB host (Render hobby plan).

    Download strategy: the body is streamed and counted as it arrives. A declared
    Content-Length over max_bytes is rejected before any body is read; the running
    count also stops responses with no (or a false) Content-Length. Counting happens
    after decompression, so a compressed response cannot slip past the cap.

    Raises:
        ValueError:          If the PDF is larger than max_bytes, is password-protected,
                             is not a readable PDF, or has no extractable text.
        requests.HTTPError:  On non-2xx response (e.g. 403 for auth-gated PDFs
                             such as SSRN — no fix possible without credentials).
    """
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

    try:
        full_text, n_pages, converted = extract_pdf_text(
            pdf_bytes, max_pages=max_pages, max_chars=max_chars, max_seconds=max_seconds, source_label=url
        )
    except PdfTextError as exc:
        # Same class as the other content failures (oversized, empty), so callers treat a bad PDF
        # as a permanent content error rather than a transient fetch error.
        raise ValueError(f"{exc.message}: {url}") from exc

    truncation_note = (
        f"Pages Converted: {converted} of {n_pages}\n\n" if converted < n_pages else ""
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
    if response.status_code == 422 and target_selector:
        # Jina returns 422 AssertionFailureError if target selector doesn't match the page.
        # Fall back to full page extraction without X-Target-Selector.
        logger.warning(
            f"[Fetcher] Target selector '{target_selector}' failed on {url} (422). Retrying without target selector."
        )
        headers.pop("X-Target-Selector", None)
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
    max_pdf_chars: int | None = None,
    max_pdf_seconds: float | None = None,
) -> str:
    """Fetch article text, routing to the appropriate extractor.

    - Direct .pdf URLs bypass Jina and are read as plain text with pdfminer.six.
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
        max_pdf_pages:   Pages read per PDF (default 150); extra pages dropped.
        max_pdf_chars:   Stop reading PDF pages once about this many characters are collected
                         (default None: read up to max_pdf_pages). A caller that only uses the
                         first N characters of the text passes N here to save time.
        max_pdf_seconds: Stop reading PDF pages after this many seconds (default None: no limit)
                         and return the text read so far. Does not cover the download, which
                         `timeout` bounds.

    Returns:
        Article text. Format matches Jina Reader output in all cases so callers
        need no special handling for the PDF path.

    Raises:
        ValueError:         If url fails the SSRF safety check, or a PDF exceeds
                            max_pdf_bytes.
        requests.HTTPError: On non-2xx response (e.g. 403 for auth-gated PDFs).
        requests.Timeout:   If the request exceeds timeout seconds.
    """
    url = normalize_url(url)
    # SSRF guard applied once here — covers both Jina and direct download paths.
    validate_public_url(url)

    if is_pdf_url(url):
        return _fetch_pdf(url, timeout, max_pdf_bytes, max_pdf_pages, max_pdf_chars, max_pdf_seconds)

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
