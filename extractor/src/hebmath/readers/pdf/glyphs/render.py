"""Render one glyph in isolation with PDFium and turn it into a shape descriptor.

Fonts whose glyphs carry no usable Unicode mapping (common in PDFs produced by
old Windows print drivers: PScript5 + Ghostscript, "TTE...t00" TrueType
subsets) still draw correctly, so the glyph *shape* is available. To look at
one glyph at a time - independently of its neighbours on the page - we place
it alone on a scratch page inside the same in-memory document (so the very
same font object is used), render that page and remove it again. The input
file is never written back.

The same routine renders the reference glyphs from which the bundled atlas
is built (tools/build_glyph_atlas.py), so unknown and reference glyphs go
through exactly one rasterisation path.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass

import pypdfium2.raw as raw

EM_PX = 64  # rendering size of one em, in pixels
GRID = 16  # side of the normalised shape grid
_CANVAS_EM = 3.0  # scratch page side, in em
_ORIGIN_EM = 1.0  # glyph origin (left, baseline) offset inside the canvas, in em
_INK = 160  # grey level below which a pixel counts as ink (0 = black)
_MAX_STRETCH = 4.0  # thin glyphs are stretched at most to this aspect ratio
TALL_STRETCH = 1e9  # no limit: used for assembled delimiters, compared only with each other


@dataclass(frozen=True, slots=True)
class Descriptor:
    """Scale-invariant description of a glyph shape.

    Metrics are in em units relative to the glyph origin (x to the right of the
    origin, y above the baseline). ``grid`` is the ink coverage of the glyph's
    ink box resampled to GRID x GRID cells (aspect preserved, centred), with
    values 0..15.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    grid: tuple[int, ...]
    components: int = 0  # connected ink regions (8-connectivity)
    holes: int = 0  # enclosed background regions (4-connectivity)

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def empty(self) -> bool:
        return not any(self.grid)

    def to_json(self) -> list:
        return [
            round(self.x0, 3),
            round(self.y0, 3),
            round(self.x1, 3),
            round(self.y1, 3),
            "".join("%x" % v for v in self.grid),
            self.components,
            self.holes,
        ]

    @staticmethod
    def from_json(item: list) -> "Descriptor":
        x0, y0, x1, y1, g, comps, holes = item
        return Descriptor(x0, y0, x1, y1, tuple(int(c, 16) for c in g), comps, holes)


EMPTY = Descriptor(0.0, 0.0, 0.0, 0.0, (0,) * (GRID * GRID))


def render_glyph(
    doc,
    font,
    *,
    charcode: int | None = None,
    text: str | None = None,
    max_stretch: float | None = None,
) -> Descriptor:
    """Render a single glyph of ``font`` (an FPDF_FONT handle owned by ``doc``).

    Exactly one of ``charcode`` (raw code, for fonts without a Unicode map) or
    ``text`` (one Unicode character) must be given.
    """
    item = (charcode, None, 0.0, 0.0) if charcode is not None else (None, text, 0.0, 0.0)
    return render_glyphs(doc, font, [item], max_stretch=max_stretch)


def render_glyphs(
    doc,
    font,
    items: list[tuple[int | None, str | None, float, float]],
    *,
    max_stretch: float | None = None,
) -> Descriptor:
    """Render several glyphs of one font together and describe the result.

    ``items`` are ``(charcode, text, dx, dy)`` with offsets in em relative to
    the first glyph's origin (y upwards). Used for delimiters that a producer
    assembled from pieces (⎛ ⎜ ⎝), which only make sense as a whole.
    """
    ys = [dy for *_, dy in items]
    span = (max(ys) - min(ys)) if ys else 0.0
    side_em = _CANVAS_EM + span
    side = int(EM_PX * side_em)
    index = raw.FPDF_GetPageCount(doc)
    page = raw.FPDFPage_New(doc, index, ctypes.c_double(side), ctypes.c_double(side))
    if not page:
        return EMPTY
    bitmap = None
    try:
        origin = EM_PX * _ORIGIN_EM
        base_y = origin - EM_PX * min(ys, default=0.0)
        placed = 0
        for charcode, text, dx, dy in items:
            obj = raw.FPDFPageObj_CreateTextObj(doc, font, ctypes.c_float(EM_PX))
            if not obj:
                continue
            if charcode is not None:
                arr = (ctypes.c_uint32 * 1)(charcode)
                ok = raw.FPDFText_SetCharcodes(obj, arr, 1)
            else:
                wide = (text or "").encode("utf-16-le") + b"\x00\x00"
                ok = raw.FPDFText_SetText(
                    obj, ctypes.cast(ctypes.c_char_p(wide), ctypes.POINTER(ctypes.c_ushort))
                )
            if not ok:
                raw.FPDFPageObj_Destroy(obj)
                continue
            raw.FPDFPageObj_Transform(
                obj,
                1,
                0,
                0,
                1,
                ctypes.c_double(origin + EM_PX * dx),
                ctypes.c_double(base_y + EM_PX * dy),
            )
            raw.FPDFPage_InsertObject(page, obj)  # page now owns obj
            placed += 1
        if not placed:
            return EMPTY
        bitmap = raw.FPDFBitmap_Create(side, side, 0)
        raw.FPDFBitmap_FillRect(bitmap, 0, 0, side, side, 0xFFFFFFFF)
        raw.FPDF_RenderPageBitmap(bitmap, page, 0, 0, side, side, 0, raw.FPDF_GRAYSCALE)
        stride = raw.FPDFBitmap_GetStride(bitmap)
        buf = ctypes.string_at(raw.FPDFBitmap_GetBuffer(bitmap), stride * side)
        return _describe(
            buf, stride, side, origin, base_y, _MAX_STRETCH if max_stretch is None else max_stretch
        )
    finally:
        if bitmap:
            raw.FPDFBitmap_Destroy(bitmap)
        raw.FPDF_ClosePage(page)
        raw.FPDFPage_Delete(doc, index)


def _describe(
    buf: bytes, stride: int, side: int, origin_x: float, origin_y: float, max_stretch: float
) -> Descriptor:
    """Bitmap (BGRx, top-down) -> Descriptor."""
    # grey value of a pixel = its blue channel (grayscale rendering: B=G=R)
    rows = [buf[y * stride : y * stride + side * 4 : 4] for y in range(side)]
    ys = [y for y in range(side) if min(rows[y]) < _INK]
    if not ys:
        return EMPTY
    xs_min, xs_max = side, -1
    for y in ys:
        row = rows[y]
        for x in range(side):
            if row[x] < _INK:
                xs_min = min(xs_min, x)
                break
        for x in range(side - 1, -1, -1):
            if row[x] < _INK:
                xs_max = max(xs_max, x)
                break
    top, bottom = ys[0], ys[-1]
    # page y grows upwards; bitmap rows grow downwards
    x0 = (xs_min - origin_x) / EM_PX
    x1 = (xs_max + 1 - origin_x) / EM_PX
    y1 = (side - top - origin_y) / EM_PX
    y0 = (side - bottom - 1 - origin_y) / EM_PX
    grid = _resample(rows, xs_min, top, xs_max + 1, bottom + 1, max_stretch)
    comps, holes = _topology(rows, xs_min, top, xs_max + 1, bottom + 1)
    return Descriptor(x0, y0, x1, y1, grid, comps, holes)


def _topology(rows: list[bytes], left: int, top: int, right: int, bottom: int) -> tuple[int, int]:
    """Count ink components and holes of the binarised glyph (tiny specks ignored).

    These are typeface-independent: '0' has one hole and 'θ' two, '⊕' four
    and 'Θ' two, '≠' has more strokes than '='.
    """
    pad = 2  # white frame, room for the dilation
    w, h = right - left + 2 * pad, bottom - top + 2 * pad
    core = bytearray(w * h)
    for y in range(top, bottom):
        row = rows[y]
        base = (y - top + pad) * w + pad
        for x in range(left, right):
            if row[x] < _INK:
                core[base + x - left] = 1
    # One-pixel dilation: strokes that a typeface leaves *almost* touching
    # (Symbol's ⊕ cross, broken serifs at small sizes) still count as connected.
    ink = bytearray(core)
    for p in range(w * h):
        if core[p]:
            px, py = p % w, p // w
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        ink[ny * w + nx] = 1
    min_area = max(2, (EM_PX // 32) ** 2)

    def regions(value: int, neighbours) -> list[tuple[int, bool]]:
        seen = bytearray(w * h)
        out = []
        for start in range(w * h):
            if ink[start] != value or seen[start]:
                continue
            seen[start] = 1
            stack = [start]
            area = 0
            border = False
            while stack:
                p = stack.pop()
                area += 1
                px, py = p % w, p // w
                if px == 0 or py == 0 or px == w - 1 or py == h - 1:
                    border = True
                for dx, dy in neighbours:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        q = ny * w + nx
                        if ink[q] == value and not seen[q]:
                            seen[q] = 1
                            stack.append(q)
            out.append((area, border))
        return out

    n8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    n4 = [(0, -1), (-1, 0), (1, 0), (0, 1)]
    comps = sum(1 for area, _ in regions(1, n8) if area >= min_area)
    holes = sum(1 for area, border in regions(0, n4) if not border and area >= min_area)
    return comps, holes


def _resample(
    rows: list[bytes], left: int, top: int, right: int, bottom: int, max_stretch: float = 4.0
) -> tuple[int, ...]:
    """Average ink coverage of the ink box into a GRID x GRID array.

    The box is stretched to the full grid so that the same symbol drawn with
    different proportions by different typefaces (Symbol's wide cap vs STIX's
    narrow one) still compares well; the proportions themselves are compared
    separately through the metrics. Very thin glyphs (aspect beyond
    _MAX_STRETCH) are only stretched up to that ratio, so a dot does not
    become a bar.
    """
    w, h = right - left, bottom - top
    sw, sh = float(w), float(h)
    if sw > max_stretch * sh:
        sh = sw / max_stretch
    elif sh > max_stretch * sw:
        sw = sh / max_stretch
    ox = left - (sw - w) / 2
    oy = top - (sh - h) / 2
    cw, ch = sw / GRID, sh / GRID
    out = []
    for gy in range(GRID):
        sy0, sy1 = oy + gy * ch, oy + (gy + 1) * ch
        for gx in range(GRID):
            sx0, sx1 = ox + gx * cw, ox + (gx + 1) * cw
            total = area = 0.0
            for y in range(max(top, math.floor(sy0)), min(bottom, math.ceil(sy1))):
                fy = min(sy1, y + 1) - max(sy0, y)
                if fy <= 0:
                    continue
                row = rows[y]
                for x in range(max(left, math.floor(sx0)), min(right, math.ceil(sx1))):
                    fx = min(sx1, x + 1) - max(sx0, x)
                    if fx <= 0:
                        continue
                    total += fx * fy * (255 - row[x]) / 255.0
                    area += fx * fy
            cov = total / (cw * ch) if cw > 0 and ch > 0 else 0.0
            out.append(min(15, int(round(cov * 15))))
    return tuple(out)
