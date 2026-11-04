"""Tests for the PDF path of cody_ai_shared_lib.fetcher (extract_pdf_text and the direct-download fetch)."""

import unittest
from unittest import mock

import pymupdf  # only used here to build fixtures; the fetcher reads with pdfminer.six

from cody_ai_shared_lib import fetcher
from cody_ai_shared_lib.fetcher import extract_pdf_text, fetch_article


def _pdf(pages, encrypt=False):
    """pages: list of paragraph strings, one page each (blank string = a page with no text)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(72, 100, 520, 700), text, fontsize=11)
    kwargs = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "user_pw": "open-me", "owner_pw": "owner"} if encrypt else {}
    data = doc.tobytes(**kwargs)
    doc.close()
    return data


def _sentence(n):
    return f"Sentence number {n} describes how the agents share one knowledge graph. " * 6


class _FakeResponse:
    def __init__(self, body, headers=None):
        self._body = body
        self.headers = headers or {}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


def _fetch_pdf_url(body, **kwargs):
    with mock.patch.object(fetcher, "safe_get", return_value=_FakeResponse(body)), \
            mock.patch.object(fetcher, "validate_public_url"):
        return fetch_article("https://example.com/report.pdf", **kwargs)


class TestExtractPdfText(unittest.TestCase):
    def test_returns_plain_text_and_page_counts(self):
        text, n_pages, converted = extract_pdf_text(_pdf([_sentence(1), _sentence(2)]))
        self.assertEqual((n_pages, converted), (2, 2))
        self.assertIn("Sentence number 1", text)
        self.assertIn("Sentence number 2", text)

    def test_stops_reading_once_the_character_budget_is_met(self):
        pages = [_sentence(i) for i in range(40)]
        _text, n_pages, converted = extract_pdf_text(_pdf(pages), max_chars=1000)
        self.assertEqual(n_pages, 40)
        self.assertLess(converted, n_pages)
        # the profile pages (10) are always read; a small budget must not read much beyond them
        self.assertLessEqual(converted, 12)

    def test_page_cap_drops_later_pages(self):
        text, n_pages, converted = extract_pdf_text(_pdf([_sentence(i) for i in range(5)]), max_pages=2)
        self.assertEqual((n_pages, converted), (5, 2))
        self.assertNotIn("Sentence number 4", text)

    def test_time_budget_returns_what_was_read_instead_of_failing(self):
        text, n_pages, converted = extract_pdf_text(_pdf([_sentence(i) for i in range(30)]), max_seconds=0)
        self.assertEqual(n_pages, 30)
        self.assertEqual(converted, 1)
        self.assertIn("Sentence number 0", text)

    def test_pdf_without_text_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            extract_pdf_text(_pdf(["", ""]), source_label="scan.pdf")
        self.assertIn("scan.pdf", str(ctx.exception))


class TestFetchPdfPath(unittest.TestCase):
    def test_output_is_framed_like_a_jina_response(self):
        out = _fetch_pdf_url(_pdf([_sentence(1), _sentence(2)]))
        self.assertTrue(out.startswith("URL Source: https://example.com/report.pdf\n\nNumber of Pages: 2\n\n"))
        self.assertIn("Markdown Content:\n", out)
        self.assertIn("Sentence number 2", out)
        self.assertNotIn("Pages Converted", out)

    def test_truncation_is_reported(self):
        out = _fetch_pdf_url(_pdf([_sentence(i) for i in range(5)]), max_pdf_pages=2)
        self.assertIn("Pages Converted: 2 of 5", out)

    def test_char_and_time_budgets_reach_the_extractor(self):
        out = _fetch_pdf_url(_pdf([_sentence(i) for i in range(40)]), max_pdf_chars=1000)
        self.assertIn("Number of Pages: 40", out)
        self.assertIn("Pages Converted:", out)
        out = _fetch_pdf_url(_pdf([_sentence(i) for i in range(30)]), max_pdf_seconds=0)
        self.assertIn("Pages Converted: 1 of 30", out)

    def test_password_protected_pdf_is_a_content_error(self):
        with self.assertRaises(ValueError) as ctx:
            _fetch_pdf_url(_pdf([_sentence(1)], encrypt=True))
        self.assertIn("password-protected", str(ctx.exception))
        self.assertIn("https://example.com/report.pdf", str(ctx.exception))

    def test_corrupt_pdf_is_a_content_error(self):
        with self.assertRaises(ValueError) as ctx:
            _fetch_pdf_url(b"%PDF-1.7 this is not really a pdf")
        self.assertIn("corrupt or unreadable", str(ctx.exception))

    def test_oversized_pdf_is_rejected_before_parsing(self):
        with mock.patch.object(fetcher, "safe_get", return_value=_FakeResponse(b"x", {"content-length": "99999999"})), \
                mock.patch.object(fetcher, "validate_public_url"):
            with self.assertRaises(ValueError) as ctx:
                fetch_article("https://example.com/report.pdf", max_pdf_bytes=1024)
        self.assertIn("limit", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
