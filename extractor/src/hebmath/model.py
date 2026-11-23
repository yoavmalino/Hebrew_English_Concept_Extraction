"""Format-independent intermediate representation (IR).

Every reader (DOCX, PDF, OCR) produces a ``Document`` made of ``Block`` objects.
The serializer is the only component that knows about the output text format,
so readers never deal with delimiters or escaping.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class TextSpan:
    """Plain text in logical (reading) order."""

    text: str


@dataclass(frozen=True)
class InlineMath:
    """A formula that lives inside a line of prose. ``latex`` has no delimiters."""

    latex: str


@dataclass(frozen=True)
class Placeholder:
    """Content that exists in the source but could not be converted.

    ``kind`` is one of the values documented in docs/FORMAT.md
    (e.g. ``math-image`` for an equation stored only as a picture).
    """

    kind: str


Segment = Union[TextSpan, InlineMath, Placeholder]


@dataclass(frozen=True)
class StyledWord:
    """One word with the formatting of its glyphs, in logical (reading) order.

    Attributes:
        text: The word as the reader emitted it (before output normalization).
        face: Font name without the subset prefix; ``"docx"`` for Word runs.
        weight: PDFium font weight of the word's glyphs (0 or -1 when unknown); for Word runs 700
            when bold and 400 otherwise.
        size: Font size in points.
        name_bold: True when the font name or the Word run says bold.
        underline: True when the word is underlined.
        color: Text colour as 0xRRGGBB; -1 when unknown (Word "auto", or not reported).
    """

    text: str
    face: str
    weight: int
    size: float
    name_bold: bool
    underline: bool
    color: int = -1


@dataclass(frozen=True)
class LineStyle:
    """Formatting facts about one visual line of a PDF page, for pipeline stage 1.

    Attributes:
        page: 1-based page number.
        size: Mean font size of the line's glyphs, in points.
        weight: Median PDFium font weight of the glyphs (0 when unknown). The scale differs per
            font, so compare it with the document's most common weight, never with a constant.
        name_bold_share: Share of glyphs whose font name says bold (a second, name-based signal).
        underline_share: Share of glyphs that have a thin horizontal line just below the baseline.
        chars: Number of non-space glyphs in the line.
        x0: Left edge of the line's glyphs, in points (0 with x1 when unknown).
        x1: Right edge of the line's glyphs.
        top: Top of the line's glyphs; page y grows upwards, so top > bottom.
        bottom: Bottom of the line's glyphs.
    """

    page: int
    size: float
    weight: int
    name_bold_share: float
    underline_share: float
    chars: int
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0


class BlockKind(str, enum.Enum):
    PARAGRAPH = "paragraph"
    DISPLAY_MATH = "display_math"
    PAGE_BREAK = "page_break"


@dataclass
class Block:
    kind: BlockKind
    segments: list[Segment] = field(default_factory=list)
    # Only for DISPLAY_MATH:
    latex: str = ""
    tag: str | None = None  # equation number, e.g. "1.2"
    # Only for PAGE_BREAK:
    page_number: int | None = None
    # A table row ("cell | cell"): serialized on one line, formulas included
    table_row: bool = False
    # PARAGRAPH from a PDF: one entry per visual line merged into it (empty for DOCX and OCR)
    line_styles: list[LineStyle] = field(default_factory=list)
    # PARAGRAPH: the words with their formatting, in reading order (empty when unknown)
    words: list[StyledWord] = field(default_factory=list)
    # PARAGRAPH from a Word document with a heading style: its level (1 = main heading)
    heading_level: int | None = None

    @staticmethod
    def paragraph(segments: list[Segment]) -> "Block":
        return Block(kind=BlockKind.PARAGRAPH, segments=list(segments))

    @staticmethod
    def display_math(latex: str, tag: str | None = None) -> "Block":
        return Block(kind=BlockKind.DISPLAY_MATH, latex=latex, tag=tag)

    @staticmethod
    def page_break(page_number: int) -> "Block":
        return Block(kind=BlockKind.PAGE_BREAK, page_number=page_number)


@dataclass(frozen=True)
class BlockInfo:
    """Where a block sits in the output text, with the formatting of its lines (stage 1).

    Attributes:
        kind: The block kind (paragraph, display_math, page_break).
        start: Index in the text of the block's first character.
        end: Index just after its last character; ``text[start:end]`` is the block.
        line_styles: One entry per PDF line merged into a paragraph (empty for DOCX and OCR).
    """

    kind: str
    start: int
    end: int
    line_styles: tuple[LineStyle, ...] = ()


@dataclass
class Document:
    blocks: list[Block] = field(default_factory=list)
    # Non-fatal problems, in a stable order (they are part of the report).
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    # Bookmarks of a PDF as (level, title, 1-based page); level 1 is the top level
    toc: list[tuple[int, str, int]] = field(default_factory=list)

    def add_stat(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
