"""PDF reader: orchestrates extraction, math reconstruction, layout and OCR."""

from __future__ import annotations

import itertools
import logging
import re
import time

import pypdfium2 as pdfium
import pypdfium2.raw as raw

from ...errors import CorruptFileError, EncryptedFileError, ResourceLimitError
from ...model import Block, BlockKind, Document, Placeholder
from .extract import PageContent, extract_page
from .geometry import Atom, median
from .glyphs.recover import FontRecovery, RecoveryConfig
from .layout import LineResult, analyse_line, build_lines, lines_to_blocks
from .mathengine import GeometricMathEngine
from .ocr import TesseractOcr

log = logging.getLogger("hebmath.pdf")

MAX_BOOKMARKS = 2000  # outlines larger than this are cut; they are navigation aids, not content
FIGURE_TEXT_MAX_SHARE = 0.5  # more of a page's glyphs than this inside figures: keep the text
MAX_GLYPHS_PER_PAGE = 60_000
_PAGE_NUMBER_RE = re.compile(
    r"^(?:[-–—]?\s*\d{1,4}\s*[-–—]?|\d{1,4}\s*/\s*\d{1,4}|(?:עמוד|עמ'|page)\s*\d{1,4}(?:\s*(?:מתוך|of)\s*\d{1,4})?)$",
    re.IGNORECASE,
)


class PdfReader:
    def __init__(self, config):
        self.config = config
        self.limits = config.limits
        self._engine_factory = (
            (lambda: config.pdf_math_engine) if config.pdf_math_engine else GeometricMathEngine
        )
        self.ocr = config.ocr_engine or TesseractOcr(
            languages=config.ocr_languages, timeout_s=config.limits.page_timeout_s
        )
        self.recovery = (
            FontRecovery(
                recognizer=getattr(config, "glyph_recognizer", None),
                ocr=self.ocr if config.enable_ocr else None,
                config=RecoveryConfig(
                    enabled=getattr(config, "recover_fonts", True),
                    max_ocr_pages=getattr(config, "recovery_ocr_pages", 6),
                ),
            )
            if getattr(config, "recover_fonts", True)
            else None
        )

    # ------------------------------------------------------------------
    def read(self, data: bytes) -> Document:
        self.engine = self._engine_factory()  # fresh per document (per-document state)
        pdf = self._open(data)
        doc = Document()
        try:
            doc.toc = self._bookmarks(pdf)
            n_pages = len(pdf)
            if n_pages == 0:
                raise CorruptFileError("PDF has no pages")
            if n_pages > self.limits.max_pdf_pages:
                raise ResourceLimitError(
                    f"PDF has {n_pages} pages (limit {self.limits.max_pdf_pages})"
                )
            pages: list[list[LineResult] | list[Block]] = []
            kinds: list[str] = []
            ocr_budget = self.limits.max_ocr_pages
            contents: list[PageContent | None] = []
            for index in range(n_pages):
                try:
                    page = pdf[index]
                except Exception as exc:  # noqa: BLE001 - a broken page must not kill the document
                    doc.warn(f"page {index + 1}: cannot be loaded ({type(exc).__name__})")
                    contents.append(None)
                    continue
                try:
                    content = extract_page(page, MAX_GLYPHS_PER_PAGE, index)
                    if content.truncated:
                        doc.warn(
                            f"page {index + 1}: more than {MAX_GLYPHS_PER_PAGE} glyphs, rest skipped"
                        )
                    contents.append(content)
                finally:
                    page.close()
            self._recover_fonts(pdf, contents, doc)
            default_rtl = self._document_direction(contents)

            for index, content in enumerate(contents):
                t0 = time.monotonic()
                if content is None:
                    pages.append([Block.paragraph([Placeholder("unreadable-page")])])
                    kinds.append("blocks")
                    continue
                if self._needs_ocr(content):
                    blocks = None
                    if self.config.enable_ocr and self.ocr.available() and ocr_budget > 0:
                        ocr_budget -= 1
                        blocks = self._ocr_page(pdf, index, doc)
                    if blocks is None:
                        doc.warn(f"page {index + 1}: no text layer and OCR is unavailable/disabled")
                        blocks = [Block.paragraph([Placeholder("unreadable-page")])]
                    pages.append(blocks)
                    kinds.append("blocks")
                    continue
                if content.rotated_chars:
                    doc.add_stat("rotated_glyphs_skipped", content.rotated_chars)
                lines = self._layout(content, default_rtl, doc, index)
                pages.append(lines)
                kinds.append("lines")
                doc.add_stat("pages_text_layer")
                if time.monotonic() - t0 > self.limits.page_timeout_s:
                    doc.warn(f"page {index + 1}: slow page ({time.monotonic() - t0:.0f}s)")

            votes = getattr(self.engine, "votes", None)
            if (
                votes
                and votes["mirrored"] > votes["as_is"]
                and not getattr(self.engine, "mirror_mode", True)
            ):
                # This producer writes mirrored code points for math: redo the text pages.
                self.engine.mirror_mode = True
                doc.warn("math glyphs were stored mirrored (RTL producer quirk); corrected")
                for index, kind in enumerate(kinds):
                    if kind == "lines":
                        pages[index] = self._layout(
                            contents[index], default_rtl, doc, index, count=False
                        )

            self._drop_running_headers(pages, kinds, contents)
            for index, (kind, payload) in enumerate(zip(kinds, pages)):
                if index:
                    doc.blocks.append(Block.page_break(index + 1))
                if kind == "lines":
                    content = contents[index]
                    blocks = lines_to_blocks(payload, 0.0, content.width if content else 0.0)
                else:
                    blocks = payload
                for b in blocks:
                    if b.kind is BlockKind.DISPLAY_MATH:
                        doc.add_stat("display_math")
                    elif b.kind is BlockKind.PARAGRAPH:
                        doc.add_stat(
                            "inline_math",
                            sum(1 for s in b.segments if type(s).__name__ == "InlineMath"),
                        )
                doc.blocks.extend(blocks)
            doc.stats["pages"] = n_pages
            return doc
        finally:
            pdf.close()

    # ------------------------------------------------------------------
    def _bookmarks(self, pdf: pdfium.PdfDocument) -> list[tuple[int, str, int]]:
        """Return the PDF's bookmarks as (level, title, 1-based page; 0 if unknown).

        An unreadable or odd outline gives an empty list: bookmarks are an extra, never a reason
        to fail the extraction.
        """
        try:
            found = []
            for item in itertools.islice(pdf.get_toc(), MAX_BOOKMARKS):
                dest = item.get_dest()
                index = dest.get_index() if dest is not None else None
                found.append(
                    (item.level + 1, item.get_title() or "", 0 if index is None else index + 1)
                )
            return found
        except Exception:  # noqa: BLE001 - see the docstring
            log.info("bookmarks could not be read", exc_info=True)
            return []

    def _open(self, data: bytes) -> pdfium.PdfDocument:
        try:
            return pdfium.PdfDocument(data)
        except pdfium.PdfiumError as exc:
            err = getattr(exc, "err_code", None)
            if err == raw.FPDF_ERR_PASSWORD or "password" in str(exc).lower():
                raise EncryptedFileError("password-protected PDF") from exc
            if err == raw.FPDF_ERR_SECURITY:
                raise EncryptedFileError("unsupported PDF security handler") from exc
            raise CorruptFileError(f"cannot open PDF: {exc}") from exc

    def _recover_fonts(self, pdf, contents, doc: Document) -> None:
        if self.recovery is None or not any(c is not None and c.cipher_fonts for c in contents):
            return

        def render_page(index: int, dpi: int) -> bytes:
            page = pdf[index]
            try:
                bitmap = page.render(scale=dpi / 72.0, grayscale=True, draw_annots=False)
                try:
                    return _bitmap_to_pgm(bitmap)
                finally:
                    bitmap.close()
            finally:
                page.close()

        try:
            report = self.recovery.run(pdf, contents, render_page)
        except Exception as exc:  # noqa: BLE001 - fall back to the previous behaviour (page OCR)
            log.exception("font recovery failed")
            doc.warn(
                f"glyph recovery for unmapped fonts failed ({type(exc).__name__}); pages may be OCR'd"
            )
            return
        if report.fonts:
            doc.add_stat("recovered_fonts", report.fonts)
            doc.add_stat("recovered_glyphs", report.glyphs - report.unresolved)
            if report.unresolved:
                doc.add_stat("unresolved_glyphs", report.unresolved)
            if report.ocr_pages:
                doc.add_stat("recovery_ocr_pages", report.ocr_pages)
            if report.assembled:
                doc.add_stat("assembled_delimiters", report.assembled)
            if report.restored:
                doc.add_stat("restored_glyphs", report.restored)
            doc.warn(
                f"{report.fonts} embedded font(s) had no Unicode mapping; "
                f"{report.glyphs - report.unresolved}/{report.glyphs} distinct glyphs were identified "
                f"by shape{' and OCR cross-check' if report.ocr_pages else ''}"
            )

    def _document_direction(self, contents) -> bool:
        heb = lat = 0
        for c in contents:
            if c is None:
                continue
            for a in c.atoms:
                if a.kind != "g" or not a.text:
                    continue
                ch = a.text[0]
                if a.is_hebrew:
                    heb += 1
                elif ch.isascii() and ch.isalpha() and not a.math_font:
                    lat += 1
        return heb >= lat if (heb or lat) else True

    def _needs_ocr(self, c: PageContent) -> bool:
        page_area = max(c.width * c.height, 1.0)
        if c.text_chars < 8:
            return c.image_area > 0.15 * page_area or c.text_chars == 0
        if c.unmapped_chars > 0.3 * c.text_chars:
            return True
        return False

    def _ocr_page(self, pdf, index: int, doc: Document) -> list[Block] | None:
        page = pdf[index]
        try:
            scale = self.limits.ocr_dpi / 72.0
            bitmap = page.render(scale=scale, grayscale=True, draw_annots=False)
            try:
                image = _bitmap_to_pgm(bitmap)
            finally:
                bitmap.close()
        finally:
            page.close()
        try:
            blocks = self.ocr.recognize(image, detect_math=self.config.detect_plain_math)
        except Exception as exc:  # noqa: BLE001
            doc.warn(f"page {index + 1}: OCR failed ({type(exc).__name__}: {exc})")
            return None
        doc.add_stat("pages_ocr")
        doc.warn(
            "some pages were OCR'd: text is approximate and formulas on them are only "
            "recognised when typed linearly"
        )
        return blocks

    def _page_atoms(
        self, content: PageContent, doc: Document, count: bool
    ) -> tuple[list[Atom], list]:
        """Glyph atoms plus atoms for vector-drawn math symbols, minus figure text."""
        atoms = list(content.atoms)
        rules = list(content.rules)
        size = median([a.size for a in atoms if a.kind == "g"], default=10.0)
        order = max((a.order for a in atoms), default=0) + 1
        strokes = [(x0, y0, x1, y1, "|") for x0, y0, x1, y1 in content.vbars] + list(content.angles)
        for x0, y0, x1, y1, char in strokes:
            h = y1 - y0
            s = min(h / 1.15, size) if h < 1.9 * size else size
            atoms.append(
                Atom(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    base=y0 + 0.2 * s if h < 1.9 * size else (y0 + y1) / 2 - 0.3 * size,
                    size=s,
                    text=char,
                    order=order,
                    kind="g",
                    math_font=True,
                    lx0=x0 - 0.1 * s,
                    lx1=x1 + 0.1 * s,
                    tall=h >= 1.9 * size,
                )
            )
            order += 1
        for r in content.radicals:
            h = r.y1 - r.y0
            s = min(h / 1.15, size) if h < 1.9 * size else size
            atoms.append(
                Atom(
                    x0=r.x0,
                    y0=r.y0,
                    x1=r.x1,
                    y1=r.y1,
                    base=r.y0 + 0.2 * s,
                    size=s,
                    text="√",
                    order=order,
                    kind="g",
                    math_font=True,
                    lx0=r.x0,
                    lx1=r.x1,
                )
            )
            order += 1
        if content.figures:

            def inside(a: Atom) -> bool:
                cx, cy = (a.x0 + a.x1) / 2, (a.y0 + a.y1) / 2
                return any(f[0] <= cx <= f[2] and f[1] <= cy <= f[3] for f in content.figures)

            kept = [a for a in atoms if not inside(a)]
            glyphs = sum(1 for a in atoms if a.kind == "g")
            glyphs_kept = sum(1 for a in kept if a.kind == "g")
            if glyphs and (glyphs - glyphs_kept) > FIGURE_TEXT_MAX_SHARE * glyphs:
                # most of the page's text sits inside figure boxes (a diagram page with its
                # headings, or a framed layout): dropping it would delete the page, so keep it
                if count:
                    doc.add_stat("figure_pages_text_kept")
                kept = atoms
            if count and len(kept) != len(atoms):
                doc.add_stat(
                    "figure_glyphs_dropped",
                    sum(1 for a in atoms if a.kind == "g") - sum(1 for a in kept if a.kind == "g"),
                )
            keep_figure_text = kept is atoms
            atoms = kept
            if not keep_figure_text:
                rules = [
                    r
                    for r in rules
                    if not any(
                        f[0] <= (r.x0 + r.x1) / 2 <= f[2] and f[1] <= r.y <= f[3]
                        for f in content.figures
                    )
                ]
        return atoms, rules

    def _layout(
        self, content: PageContent, default_rtl: bool, doc: Document, index: int, count: bool = True
    ) -> list[LineResult]:
        page_atoms, page_rules = self._page_atoms(content, doc, count)
        try:
            atoms = self.engine.build_structures(page_atoms, page_rules)
        except Exception as exc:  # noqa: BLE001 - fall back to flat text for this page
            log.exception("math structure detection failed")
            doc.warn(
                f"page {index + 1}: math structure detection failed ({type(exc).__name__}); flat text used"
            )
            atoms = page_atoms
        if count:
            doc.add_stat("math_structures", sum(1 for a in atoms if a.is_composite))
        lines = build_lines(atoms)
        column = (
            min((ln.x0 for ln in lines), default=0.0),
            max((ln.x1 for ln in lines), default=0.0),
        )
        results = []
        for ln in lines:
            try:
                results.append(
                    analyse_line(
                        ln, self.engine, default_rtl, column, rules=page_rules, page=index + 1
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("line analysis failed")
                doc.warn(f"page {index + 1}: a line could not be analysed ({type(exc).__name__})")
        return results

    def _drop_running_headers(self, pages, kinds, contents) -> None:
        """Remove page numbers and running headers/footers repeated across pages."""
        text_pages = [i for i, k in enumerate(kinds) if k == "lines"]
        if not text_pages:
            return
        signature_count: dict[str, int] = {}
        for i in text_pages:
            h = contents[i].height if contents[i] else 0
            seen = set()
            for ln in pages[i]:
                if not _in_margin(ln, h):
                    continue
                sig = _signature(ln.plain)
                if sig and sig not in seen:
                    seen.add(sig)
                    signature_count[sig] = signature_count.get(sig, 0) + 1
        repeated = {
            s
            for s, c in signature_count.items()
            if len(text_pages) >= 3 and c >= max(3, len(text_pages) // 2)
        }
        for i in text_pages:
            h = contents[i].height if contents[i] else 0
            lines = pages[i]
            kept = []
            for pos, ln in enumerate(lines):
                if ln.is_display:
                    kept.append(ln)
                    continue
                edge = pos == 0 or pos == len(lines) - 1
                is_number = bool(_PAGE_NUMBER_RE.match(ln.plain.strip()))
                # page numbers: alone on the first/last line of the page, separated from the body
                if is_number and edge and len(lines) > 1:
                    neighbour = lines[1] if pos == 0 else lines[-2]
                    if abs(neighbour.base - ln.base) > 1.8 * max(
                        ln.size, neighbour.size
                    ) or _in_margin(ln, h):
                        continue
                if _in_margin(ln, h) and _signature(ln.plain) in repeated:
                    continue
                kept.append(ln)
            pages[i] = kept


def _bitmap_to_pgm(bitmap) -> bytes:
    """Encode a PDFium bitmap as binary PGM (Tesseract/Leptonica read it natively).

    Avoids an imaging library dependency; grayscale bitmaps are 1 byte/pixel,
    colour ones (if a custom renderer returns them) are converted with integer luma.
    """
    w, h, stride = bitmap.width, bitmap.height, bitmap.stride
    channels = bitmap.n_channels
    buf = bytes(bitmap.buffer)
    rows = []
    for y in range(h):
        row = buf[y * stride : y * stride + w * channels]
        if channels == 1:
            rows.append(row)
        else:  # BGR(A)
            rows.append(
                bytes(
                    (row[i + 2] * 299 + row[i + 1] * 587 + row[i] * 114) // 1000
                    for i in range(0, len(row), channels)
                )
            )
    return b"P5\n%d %d\n255\n" % (w, h) + b"".join(rows)


def _in_margin(ln: LineResult, page_height: float) -> bool:
    if page_height <= 0:
        return False
    return ln.base > page_height * 0.92 or ln.base < page_height * 0.08


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip()
