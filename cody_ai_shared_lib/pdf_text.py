"""Plain-text extraction from PDFs for reading aloud, built on pdfminer.six layout boxes.

Opt-in module: nothing here is imported by `fetch_article`, so News Intelligence output is unchanged.
Why not a Markdown converter: pymupdf4llm needs 200-425 MB for one PDF and emits `~~`/`<u>` markers
and table pipes that a voice would read aloud. pdfminer.six is pure Python, needs about 50-70 MB for
ordinary PDFs, and its layout boxes carry page positions, which makes repeated header/footer removal
and paragraph grouping reliable.

Usage (one page at a time, so callers keep their own time budgets, progress and checkpoints):

    reader = PdfTextReader(pdf_bytes_or_path)
    try:
        reader.prepare()                                   # optional: samples pages to learn headers/footers
        pages = [reader.page_boxes(i) for i in range(start, end)]
        text = reader.clean_pages(pages)                   # paragraphs separated by blank lines
    finally:
        reader.close()

Cleanup applied by `clean_pages`:
 1. repeated headers/footers (top/bottom zone, same text on many pages) and bare page numbers dropped
 2. one paragraph per layout box; line-end hyphens rejoined using the document's own vocabulary
    (a compound seen elsewhere in the document, like "red-teaming", keeps its hyphen)
 3. paragraphs that continue across a box or page break are joined
 4. unicode hyphens, `(cid:N)` tokens, dot leaders, bullet glyphs, ligatures, odd quote glyphs and
    spaces before punctuation are normalised; number-only paragraphs (chart labels) are dropped
"""

from __future__ import annotations

import io
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

__all__ = [
    "PROFILE_PAGES",
    "PdfTextReader",
    "PdfPageText",
    "TextBox",
    "PdfTextError",
    "PdfEncryptedError",
    "PdfEmptyError",
    "placeholder_page",
]

logger = logging.getLogger("cody-ai-pdf-text")

PROFILE_PAGES = 30   # pages sampled to learn repeated headers/footers and the document vocabulary
ZONE = 0.08          # top/bottom 8% of a page is the header/footer zone
_TERMINAL = tuple('.!?:;"”’)]')
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
_PAGE_NUMBER_RE = re.compile(r"^(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
_ODD_QUOTES = "˝˛ˇ"   # curly quotes mis-decoded by some fonts
_BULLET_RE = re.compile(r"^[■□▪▫●○•◦‣▶►◆◇✓✔➢➤]\s*")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_COMPOUND_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)+")


class PdfTextError(Exception):
    """The PDF could not be opened. `code` is a stable machine-readable reason."""

    code = "invalid_pdf"

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class PdfEncryptedError(PdfTextError):
    code = "encrypted_pdf"


class PdfEmptyError(PdfTextError):
    code = "invalid_pdf"


@dataclass
class TextBox:
    lines: List[str]
    y0: float
    y1: float


@dataclass
class PdfPageText:
    height: float
    boxes: List[TextBox] = field(default_factory=list)


def placeholder_page(message: str) -> PdfPageText:
    """A page that stands in for one that could not be read, so offsets stay canonical and the
    listener hears that content is missing. Positioned mid-page so it is never mistaken for a header."""
    return PdfPageText(height=100.0, boxes=[TextBox(lines=[message], y0=45.0, y1=55.0)])


@dataclass
class _Profile:
    repeated: Set[str]
    hyphenated: Set[str]
    odd_quotes: bool


def _norm(s: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", s.lower())).strip()


def _in_zone(box: TextBox, height: float) -> bool:
    return box.y1 >= (1 - ZONE) * height or box.y0 <= ZONE * height


def _line_fixes(raw: str) -> str:
    """Fixes applied before lines are joined, so hyphen rejoining sees plain ASCII hyphens."""
    raw = raw.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")   # unicode hyphens
    raw = re.sub(r"\(cid:\d+\)", " ", raw)                                           # unmapped symbol glyphs
    raw = re.sub(r"\s*\.{4,}\s*", " ", raw)                                          # dot leaders in contents lines
    return raw


def _compounds(pages: Sequence[PdfPageText]) -> Set[str]:
    found: Set[str] = set()
    for pg in pages:
        for b in pg.boxes:
            for ln in b.lines:
                found.update(w.lower() for w in _COMPOUND_RE.findall(ln))
    return found


def _join_lines(out: str, nxt: str, hyphenated: Set[str]) -> str:
    """Join two consecutive lines, rejoining a word that was hyphenated at the line end."""
    m = re.search(r"([A-Za-z]{2,})-$", out)
    n = re.match(r"([a-z]+)", nxt)
    if m and n:
        if f"{m.group(1)}-{n.group(1)}".lower() in hyphenated:   # a real compound seen elsewhere
            return out + nxt
        return out[:-1] + nxt                                     # soft line-break hyphen
    return out + " " + nxt


def _box_to_paragraph(lines: List[str], hyphenated: Set[str]) -> str:
    out = lines[0]
    for nxt in lines[1:]:
        out = _join_lines(out, nxt, hyphenated)
    return out


def _hygiene(text: str, odd_quotes: bool) -> str:
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = text.replace("\xa0", " ").replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    text = _CONTROL_RE.sub("", text)
    if odd_quotes:
        text = text.replace("˝", "“").replace("˛", "”").replace("ˇ", "’")
    text = _BULLET_RE.sub("", text)
    text = re.sub(r"\s+([,;:!?])", r"\1", text)               # 'scams , which' -> 'scams, which'
    text = re.sub(r"(?<=[a-z\)])\s+\.(?=\s|$)", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


class PdfTextReader:
    """Page-at-a-time PDF text reader. Not thread-safe; close() it (it holds the file open)."""

    def __init__(self, source: Union[bytes, bytearray, str, Path], profile_pages: int = PROFILE_PAGES):
        from pdfminer.converter import PDFPageAggregator
        from pdfminer.layout import LAParams
        from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError, PDFPasswordIncorrect
        from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfparser import PDFParser

        self._profile_pages = profile_pages
        self._cache: Dict[int, PdfPageText] = {}
        self._profile: Optional[_Profile] = None
        self._fp = None
        try:
            if isinstance(source, (bytes, bytearray)):
                self._fp = io.BytesIO(source)
            else:
                self._fp = open(Path(source), "rb")
            self._doc = PDFDocument(PDFParser(self._fp))
            self._pages = list(PDFPage.create_pages(self._doc))
        except (PDFPasswordIncorrect, PDFEncryptionError) as exc:
            self.close()
            raise PdfEncryptedError("PDF is password-protected or encrypted; decryption is not supported") from exc
        except Exception as exc:
            self.close()
            raise PdfTextError("The PDF is corrupt or unreadable") from exc
        if not self._pages:
            self.close()
            raise PdfEmptyError("PDF document contains no pages")

        self._rsrc = PDFResourceManager()
        self._device = PDFPageAggregator(self._rsrc, laparams=LAParams())
        self._interp = PDFPageInterpreter(self._rsrc, self._device)

    def __len__(self) -> int:
        return len(self._pages)

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def close(self) -> None:
        fp, self._fp = self._fp, None
        if fp is not None:
            fp.close()

    # ── table of contents ────────────────────────────────────────────────

    def toc(self) -> List[Tuple[int, str, int]]:
        """Outline entries as (level, title, page_index_0_based); empty when absent or unresolvable."""
        from pdfminer.pdftypes import resolve1
        from pdfminer.psparser import PSLiteral

        page_of: Dict[int, int] = {p.pageid: i for i, p in enumerate(self._pages)}

        def page_index(dest, action) -> Optional[int]:
            if dest is None and action is not None:
                action = resolve1(action)
                if isinstance(action, dict):
                    dest = action.get("D")
            dest = resolve1(dest)
            if isinstance(dest, (bytes, str, PSLiteral)):    # a named destination
                name = dest.name if isinstance(dest, PSLiteral) else dest
                dest = resolve1(self._doc.get_dest(name))
            if isinstance(dest, dict):
                dest = resolve1(dest.get("D"))
            if isinstance(dest, list) and dest:
                head = dest[0]
                objid = getattr(head, "objid", None)
                if objid is not None:
                    return page_of.get(objid)
                if isinstance(head, int) and 0 <= head < len(self._pages):
                    return head
            return None

        entries: List[Tuple[int, str, int]] = []
        try:
            for level, title, dest, action, _se in self._doc.get_outlines():
                try:
                    idx = page_index(dest, action)
                except Exception:
                    idx = None
                if idx is not None and title:
                    entries.append((int(level), str(title).strip(), idx))
        except Exception:
            return []
        return entries

    # ── per-page extraction ──────────────────────────────────────────────

    def page_boxes(self, index: int) -> PdfPageText:
        """Layout boxes of one page (0-based). May raise on a damaged page: the caller decides."""
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        self._interp.process_page(self._pages[index])
        layout = self._device.get_result()
        from pdfminer.layout import LTTextBox

        boxes: List[TextBox] = []
        for el in layout:
            if isinstance(el, LTTextBox):
                raw = el.get_text().replace("\u2028", "\n").replace("\r", "")
                lines = [ln.strip() for ln in _line_fixes(raw).split("\n") if ln.strip()]
                if lines:
                    boxes.append(TextBox(lines=lines, y0=el.y0, y1=el.y1))
        page = PdfPageText(height=layout.height, boxes=boxes)
        if index < self._profile_pages:
            self._cache[index] = page
        return page

    # ── document profile ─────────────────────────────────────────────────

    def prepare(self, on_page: Optional[Callable[[int], None]] = None) -> None:
        """Sample the first pages to learn repeated headers/footers and vocabulary. Called lazily by
        clean_pages; call it up front to report progress or honour cancellation (`on_page(index)`)."""
        if self._profile is not None:
            return
        sample: List[PdfPageText] = []
        for i in range(min(self._profile_pages, len(self._pages))):
            if on_page:
                on_page(i)
            try:
                sample.append(self.page_boxes(i))
            except MemoryError:
                raise
            except Exception as exc:   # a damaged page just contributes nothing
                logger.warning(f"[PDF] page {i + 1} skipped while profiling: {exc}")
        threshold = max(3, math.ceil(0.2 * len(sample)))
        zone_keys: Counter = Counter()
        for pg in sample:
            seen = {_norm(" ".join(b.lines)) for b in pg.boxes if _in_zone(b, pg.height)}
            zone_keys.update(seen)
        repeated = {k for k, c in zone_keys.items() if c >= threshold}
        odd = sum(ln.count(c) for pg in sample for b in pg.boxes for ln in b.lines for c in _ODD_QUOTES)
        self._profile = _Profile(repeated=repeated, hyphenated=_compounds(sample), odd_quotes=odd >= 10)

    # ── cleanup ──────────────────────────────────────────────────────────

    def clean_pages(self, pages: Sequence[PdfPageText]) -> str:
        """Clean consecutive pages into plain text: paragraphs separated by blank lines.

        Paragraph joins and hyphen rejoining only happen within the pages passed in, so a caller that
        cleans in batches (one section at a time) keeps a paragraph that straddles two batches split."""
        self.prepare()
        profile = self._profile
        assert profile is not None
        hyphenated = profile.hyphenated | _compounds(pages)

        paragraphs: List[str] = []
        for pg in pages:
            for b in pg.boxes:
                text = " ".join(b.lines)
                if _in_zone(b, pg.height) and (_norm(text) in profile.repeated or _PAGE_NUMBER_RE.match(text.strip())):
                    continue
                paragraphs.append(_box_to_paragraph(b.lines, hyphenated))

        merged: List[str] = []
        for p in paragraphs:       # join a paragraph that continues across a box or page break
            p = p.strip()
            if not p:
                continue
            if merged and not merged[-1].endswith(_TERMINAL) and re.match(r"[a-z]", p):
                merged[-1] = _join_lines(merged[-1], p, hyphenated)
            else:
                merged.append(p)

        cleaned = [_hygiene(p, profile.odd_quotes) for p in merged]
        # chart labels and stray numbers ("57%", "03") are not sentences: a voice would read them aloud
        cleaned = [p for p in cleaned if p and (re.search(r"[A-Za-z]", p) or len(p) > 12)]
        return "\n\n".join(cleaned)
