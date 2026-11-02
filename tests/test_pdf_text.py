"""Tests for cody_ai_shared_lib.pdf_text, using small synthetic PDFs generated on the fly."""

import unittest

import pymupdf  # only used here to build fixtures; pdf_text itself reads with pdfminer.six

from cody_ai_shared_lib.pdf_text import (
    PdfEmptyError,
    PdfEncryptedError,
    PdfTextError,
    PdfTextReader,
    placeholder_page,
)


def _pdf(pages, toc=None):
    """pages: list of lists of (x, y, text) in top-left PDF points (page is 595 x 842)."""
    doc = pymupdf.open()
    for items in pages:
        page = doc.new_page()
        for x, y, text in items:
            page.insert_text((x, y), text, fontsize=11)
    if toc:
        doc.set_toc(toc)
    data = doc.tobytes()
    doc.close()
    return data


def _clean(data, upto=None):
    reader = PdfTextReader(data)
    try:
        n = len(reader) if upto is None else upto
        return reader.clean_pages([reader.page_boxes(i) for i in range(n)])
    finally:
        reader.close()


class TestPdfText(unittest.TestCase):
    def test_reads_paragraphs_and_reports_page_count(self):
        data = _pdf([[(72, 300, "First paragraph of the first page.")], [(72, 300, "Second page body text here.")]])
        reader = PdfTextReader(data)
        try:
            self.assertEqual(len(reader), 2)
        finally:
            reader.close()
        text = _clean(data)
        self.assertIn("First paragraph of the first page.", text)
        self.assertIn("Second page body text here.", text)

    def test_repeated_header_and_footer_and_page_numbers_are_dropped(self):
        pages = [
            [
                (72, 40, "ACME Quarterly Report"),                 # header, same on every page
                (72, 300, f"Unique body sentence number {n} ends here."),
                (72, 810, f"Page {n}"),                            # footer page number
            ]
            for n in range(1, 7)
        ]
        text = _clean(_pdf(pages))
        self.assertNotIn("ACME Quarterly Report", text)
        self.assertNotRegex(text, r"Page \d")
        for n in range(1, 7):
            self.assertIn(f"Unique body sentence number {n} ends here.", text)

    def test_line_end_hyphen_is_rejoined_but_known_compound_keeps_its_hyphen(self):
        pages = [[
            (72, 300, "We study red-teaming of the new manip-"),
            (72, 314, "ulate cases and also red-"),
            (72, 328, "teaming at scale."),
        ]]
        text = _clean(_pdf(pages))
        self.assertIn("manipulate", text)
        self.assertIn("red-teaming at scale", text)

    def test_paragraph_continuing_across_pages_is_joined(self):
        pages = [
            [(72, 300, "The model was trained for many days and then")],
            [(72, 300, "evaluated on a held-out set of tasks.")],
        ]
        text = _clean(_pdf(pages))
        self.assertIn("and then evaluated on a held-out set of tasks.", text)

    def test_number_only_labels_are_dropped(self):
        pages = [[(72, 300, "A real sentence about results."), (300, 400, "57%")]]
        text = _clean(_pdf(pages))
        self.assertIn("A real sentence about results.", text)
        self.assertNotIn("57%", text)

    def test_toc_resolves_to_zero_based_pages(self):
        pages = [[(72, 300, f"Body of page {n}.")] for n in range(1, 6)]
        reader = PdfTextReader(_pdf(pages, toc=[[1, "Alpha", 2], [1, "Beta", 4]]))
        try:
            self.assertEqual([(lvl, title, idx) for lvl, title, idx in reader.toc()], [(1, "Alpha", 1), (1, "Beta", 3)])
        finally:
            reader.close()

    def test_toc_is_empty_without_outline(self):
        reader = PdfTextReader(_pdf([[(72, 300, "No outline here at all.")]]))
        try:
            self.assertEqual(reader.toc(), [])
        finally:
            reader.close()

    def test_password_protected_pdf_raises_encrypted_error(self):
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 300), "Secret text.")
        data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="open-me", owner_pw="owner")
        doc.close()
        with self.assertRaises(PdfEncryptedError) as ctx:
            PdfTextReader(data)
        self.assertEqual(ctx.exception.code, "encrypted_pdf")

    def test_known_compound_split_across_boxes_keeps_its_hyphen(self):
        # the compound appears whole elsewhere, then breaks between two boxes on the next page
        pages = [
            [(72, 300, "Red-teaming is useful and red-teaming scales.")],
            [(72, 300, "We rely on red-")],
            [(72, 500, "teaming for coverage.")],
        ]
        text = _clean(_pdf(pages))
        self.assertIn("red-teaming for coverage", text)

    def test_corrupt_input_raises_pdf_text_error(self):
        with self.assertRaises(PdfTextError):
            PdfTextReader(b"%PDF-1.4 definitely not a real pdf")

    def test_blank_page_yields_empty_text(self):
        doc = pymupdf.open()
        doc.new_page()
        data = doc.tobytes()
        doc.close()
        # a blank page opens fine and yields no text; a zero-page PDF would raise PdfEmptyError
        reader = PdfTextReader(data)
        try:
            self.assertEqual(reader.clean_pages([reader.page_boxes(0)]), "")
        finally:
            reader.close()
        self.assertTrue(issubclass(PdfEmptyError, PdfTextError))

    def test_placeholder_page_survives_cleanup_as_visible_text(self):
        data = _pdf([[(72, 300, "Readable page text for the sample.")]])
        reader = PdfTextReader(data)
        try:
            pages = [reader.page_boxes(0), placeholder_page("[Page 2 could not be read.]")]
            text = reader.clean_pages(pages)
        finally:
            reader.close()
        self.assertIn("[Page 2 could not be read.]", text)

    def test_cleanup_normalises_ligatures_unicode_hyphens_and_cid_tokens(self):
        # built from escapes so the fixture itself carries no invisible characters
        raw = "The ﬁnal non‐existent (cid:12) result"
        data = _pdf([[(72, 300, "Placeholder")]])
        reader = PdfTextReader(data)
        try:
            from cody_ai_shared_lib.pdf_text import PdfPageText, TextBox, _line_fixes

            page = PdfPageText(height=842.0, boxes=[TextBox(lines=[_line_fixes(raw)], y0=400, y1=420)])
            text = reader.clean_pages([page])
        finally:
            reader.close()
        self.assertEqual(text, "The final non-existent result")


if __name__ == "__main__":
    unittest.main()
