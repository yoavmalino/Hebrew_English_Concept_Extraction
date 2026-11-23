"""Geometric primitives shared by the PDF layout and math engines.

Coordinates are PDF user space after normalising text rotation: x grows to
the right, y grows *upwards*. ``Atom`` is either a single glyph or a composite
(fraction, radical, big operator with limits, accented symbol) produced by the
math engine.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


@dataclass(eq=False, slots=True)
class Atom:
    x0: float
    y0: float
    x1: float
    y1: float
    base: float  # baseline y
    size: float  # font size in points
    text: str = ""  # unicode for glyphs, "" for composites
    font: str = ""
    order: int = 0  # content-stream order (stable tie breaker)
    kind: str = "g"  # g=glyph, sp=space, frac, sqrt, op, acc, over, under
    latex: str | None = None
    math_font: bool = False
    italic: bool = False
    bold: bool = False
    code: int = 0  # raw unicode/char code reported by PDFium
    marks: str = ""  # combining marks (niqqud, accents) attached to this glyph
    lx0: float = 0.0  # loose (advance-based) horizontal extent
    lx1: float = 0.0
    axis: float | None = None  # math axis for fractions (rule y)
    fkey: str = ""  # font identity (see glyphs.fontid) - used to recover unmapped fonts
    tall: bool = False  # delimiter assembled from several pieces (⎛ ⎜ ⎝ ...)
    raw_code: bool = False  # PDFium had no Unicode for this glyph; ``code`` is the font's char code
    obj: int = 0  # identity of the PDF text object that drew the glyph
    weight: int = 0  # PDFium numeric font weight (not a 100..900 scale); 0 or -1 = unknown
    color: int = -1  # fill colour 0xRRGGBB; -1 = unknown
    # size as drawn: font size times the vertical scale of the text matrix (0 = same as size).
    # Formatting metadata only; math layout keeps using ``size`` so the text output is unchanged.
    style_size: float = 0.0
    children: list["Atom"] = field(default_factory=list)

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def is_composite(self) -> bool:
        return self.kind not in ("g", "sp")

    @property
    def is_space(self) -> bool:
        return self.kind == "sp"

    @property
    def is_hebrew(self) -> bool:
        return bool(self.text) and is_hebrew_char(self.text[0])

    def sort_key(self) -> tuple:
        return (round(self.x0, 2), -round(self.base, 2), self.order)


@dataclass(frozen=True)
class Rule:
    """A thin horizontal vector line (fraction bar, radical overline, underline, table border)."""

    x0: float
    x1: float
    y: float
    thickness: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0


def is_hebrew_char(ch: str) -> bool:
    cp = ord(ch)
    return 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F


def is_mark(ch: str) -> bool:
    return bool(ch) and unicodedata.category(ch[0]) in ("Mn", "Me")


def union(atoms: list[Atom]) -> tuple[float, float, float, float]:
    return (
        min(a.x0 for a in atoms),
        min(a.y0 for a in atoms),
        max(a.x1 for a in atoms),
        max(a.y1 for a in atoms),
    )


def x_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def median(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
