"""Public entry point: bytes/path in -> ``ExtractionResult`` out.

``extract()`` never raises for bad *input*: corrupt, encrypted, unsupported
and oversized files produce a result with ``ok=False`` and a stable error
code. Programming errors in the extractor are also caught and reported as
``internal_error`` so a single bad upload cannot take down a pipeline worker.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import __version__
from .errors import ExtractionError, ResourceLimitError, UnsupportedFormatError
from .latex.omml import OmmlToLatex
from .model import BlockInfo, Document
from .safety import Limits
from .serialize import serialize_with_spans
from .structure import StructureInfo, build_structure, modal_style

log = logging.getLogger("hebmath")

PDF = "pdf"
DOCX = "docx"


@dataclass
class ExtractorConfig:
    limits: Limits = field(default_factory=Limits)
    enable_ocr: bool = True
    ocr_languages: str = "heb+eng"
    detect_plain_math: bool = True
    page_markers: bool = False
    # "formulas": a formula and the text around it go on separate lines (single
    # symbols such as $V$ stay in the sentence); "all": every formula; "none": format v1
    math_line_breaks: str = "formulas"
    # PDF fonts without a Unicode mapping: identify glyphs by shape (+ OCR cross-check)
    recover_fonts: bool = True
    recovery_ocr_pages: int = 6
    # Swappable components (see docs). ``None`` means the built-in default.
    omml_converter: OmmlToLatex | None = None
    pdf_math_engine: object | None = None
    ocr_engine: object | None = None
    glyph_recognizer: object | None = None


@dataclass
class ExtractionResult:
    ok: bool
    text: str
    format: str | None
    sha256: str
    extractor_version: str
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    engines: dict[str, str] = field(default_factory=dict)
    # Stage 1 (formatting metadata); the text above is unaffected by it.
    blocks: list[BlockInfo] = field(default_factory=list)
    doc_style: dict[str, float] = field(default_factory=dict)
    structure: StructureInfo | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "format": self.format,
            "sha256": self.sha256,
            "extractor_version": self.extractor_version,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "warnings": list(self.warnings),
            "stats": dict(sorted(self.stats.items())),
            "engines": dict(sorted(self.engines.items())),
        }


def sniff_format(data: bytes, filename: str | None = None) -> str:
    """Detect the format from content (magic bytes), not from the file name."""
    head = data[:1024]
    if b"%PDF-" in head:
        return PDF
    if data[:4] == b"PK\x03\x04":
        return DOCX
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return DOCX  # encrypted DOCX or legacy .doc -> the DOCX reader explains which
    name = (filename or "").lower()
    if name.endswith(".pdf") and data:
        return PDF  # let the PDF reader try to repair it
    raise UnsupportedFormatError("file is neither a PDF nor a DOCX document")


class Extractor:
    """Reusable, thread-compatible extractor (one instance per worker)."""

    def __init__(self, config: ExtractorConfig | None = None):
        self.config = config or ExtractorConfig()
        self._readers: dict[str, Callable[[bytes], Document]] = {}

    def _reader(self, fmt: str) -> Callable[[bytes], Document]:
        if fmt not in self._readers:
            if fmt == DOCX:
                from .readers.docx import DocxReader

                r = DocxReader(
                    omml=self.config.omml_converter,
                    limits=self.config.limits,
                    detect_plain_math=self.config.detect_plain_math,
                )
                self._readers[fmt] = r.read
            elif fmt == PDF:
                from .readers.pdf.reader import PdfReader

                r = PdfReader(self.config)
                self._readers[fmt] = r.read
            else:  # pragma: no cover
                raise UnsupportedFormatError(fmt)
        return self._readers[fmt]

    def extract_bytes(self, data: bytes, filename: str | None = None) -> ExtractionResult:
        digest = hashlib.sha256(data).hexdigest()
        fmt: str | None = None
        try:
            if not data:
                raise ExtractionError("empty file", code="empty_file")
            if len(data) > self.config.limits.max_file_bytes:
                raise ResourceLimitError(
                    f"file is larger than {self.config.limits.max_file_bytes} bytes"
                )
            fmt = sniff_format(data, filename)
            document = self._reader(fmt)(data)
            text, spans = serialize_with_spans(
                document,
                page_markers=self.config.page_markers,
                math_line_breaks=self.config.math_line_breaks,
            )
            blocks = [
                BlockInfo(
                    kind=document.blocks[index].kind.value,
                    start=start,
                    end=end,
                    line_styles=tuple(document.blocks[index].line_styles),
                )
                for index, start, end in spans
            ]
            structure = self._structure(document, text, spans)
            doc_style = (
                {"size": structure.body_size, "weight": structure.body_weight}
                if structure is not None and structure.body_size
                else modal_style(blocks)
            )
            warnings = list(document.warnings)
            if not text.strip():
                warnings.append("no text could be extracted")
            return ExtractionResult(
                ok=True,
                text=text,
                format=fmt,
                sha256=digest,
                extractor_version=__version__,
                warnings=warnings,
                stats=dict(document.stats),
                engines=self._engine_versions(fmt, document),
                blocks=blocks,
                doc_style=doc_style,
                structure=structure,
            )
        except ExtractionError as exc:
            log.info("extraction failed: %s (%s)", exc.code, exc)
            return ExtractionResult(
                ok=False,
                text="",
                format=fmt,
                sha256=digest,
                extractor_version=__version__,
                error_code=exc.code,
                error_message=str(exc),
            )
        except MemoryError:
            return ExtractionResult(
                ok=False,
                text="",
                format=fmt,
                sha256=digest,
                extractor_version=__version__,
                error_code="resource_limit",
                error_message="out of memory",
            )
        except Exception as exc:  # noqa: BLE001 - last line of defence for a pipeline worker
            log.exception("internal error while extracting")
            return ExtractionResult(
                ok=False,
                text="",
                format=fmt,
                sha256=digest,
                extractor_version=__version__,
                error_code="internal_error",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    def _structure(self, document: Document, text: str, spans) -> StructureInfo | None:
        """Build the stage-1 structure; a failure here must never fail the extraction."""
        try:
            info = build_structure(document, text, spans)
            if any(item["kind"] == "few-letters" for item in info.suspicious):
                document.warn(
                    "the text has almost no letters; the file's text layer is probably incomplete"
                    " (for example text drawn as shapes or images)"
                )
            return info
        except Exception:  # noqa: BLE001 - the text is the product; structure is an extra
            log.exception("structure building failed")
            document.warn("structure could not be built")
            return None

    def _engine_versions(self, fmt: str, document: Document) -> dict[str, str]:
        """Versions that determine the output (log them with the text for reproducibility)."""
        out = {"hebmath": __version__}
        if fmt == PDF:
            import pypdfium2

            out["pypdfium2"] = str(pypdfium2.PYPDFIUM_INFO)
            out["pdfium"] = str(pypdfium2.PDFIUM_INFO)
            reader = getattr(self._readers.get(PDF), "__self__", None)
            if document.stats.get("pages_ocr") or document.stats.get("recovery_ocr_pages"):
                ocr = getattr(reader, "ocr", None)
                version = getattr(ocr, "version", None)
                out["ocr"] = version() if callable(version) else getattr(ocr, "name", "custom")
            if document.stats.get("recovered_fonts"):
                rec = getattr(getattr(reader, "recovery", None), "recognizer", None)
                out["glyph_recognizer"] = getattr(rec, "version", getattr(rec, "name", "custom"))
        return out

    def extract_path(self, path: str | Path) -> ExtractionResult:
        p = Path(path)
        try:
            size = p.stat().st_size
        except OSError as exc:
            return ExtractionResult(
                ok=False,
                text="",
                format=None,
                sha256="",
                extractor_version=__version__,
                error_code="io_error",
                error_message=str(exc),
            )
        if size > self.config.limits.max_file_bytes:
            return ExtractionResult(
                ok=False,
                text="",
                format=None,
                sha256="",
                extractor_version=__version__,
                error_code="resource_limit",
                error_message=f"file is larger than {self.config.limits.max_file_bytes} bytes",
            )
        return self.extract_bytes(p.read_bytes(), filename=p.name)


_default: Extractor | None = None


def extract(
    source: bytes | str | Path, filename: str | None = None, config: ExtractorConfig | None = None
) -> ExtractionResult:
    """Convenience wrapper. ``source`` is raw bytes or a filesystem path."""
    global _default
    if config is not None:
        ex = Extractor(config)
    else:
        if _default is None:
            _default = Extractor()
        ex = _default
    if isinstance(source, (bytes, bytearray, memoryview)):
        return ex.extract_bytes(bytes(source), filename)
    return ex.extract_path(source)
