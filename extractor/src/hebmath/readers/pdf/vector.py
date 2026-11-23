"""Classification of vector paths that carry mathematical meaning.

Equation editors draw part of a formula with paths instead of glyphs:

* thin horizontal lines - fraction bars, overlines (z̄), radical vincula;
* thin vertical lines - absolute value / norm / determinant bars (MathType
  draws |z| this way);
* radical signs - a polygon shaped like √, often with the vinculum as its top
  edge (MathType, Word's legacy Equation Editor).

Everything else that is clearly a drawing (curves, slanted strokes, arrows)
is recorded as *graphics*, from which figure regions are derived: text inside a
figure (axis labels of a diagram) is not part of the running text and is
dropped, per the extraction contract (images and diagrams are ignored).

Coordinates are page space (y up), before text-rotation normalisation; the
caller rotates them together with the glyphs.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

import pypdfium2.raw as raw

_SEG_LINE, _SEG_BEZIER, _SEG_MOVE = 0, 1, 2


@dataclass
class Radical:
    """A radical sign: the sign's own box plus the vinculum (if drawn with it)."""

    x0: float
    y0: float
    x1: float  # right end of the sign (start of the vinculum)
    y1: float
    bar_x1: float  # right end of the vinculum (== x1 when there is none)
    bar_y: float


@dataclass
class VectorContent:
    hrules: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )  # x0, x1, y, thickness
    vbars: list[tuple[float, float, float, float]] = field(default_factory=list)  # x0, y0, x1, y1
    radicals: list[Radical] = field(default_factory=list)
    angles: list[tuple[float, float, float, float, str]] = field(
        default_factory=list
    )  # ⟨ ⟩ strokes
    graphics: list[tuple[float, float, float, float]] = field(default_factory=list)  # boxes
    lines: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )  # other axis-aligned paths


def _points(obj) -> tuple[list[tuple[int, float, float]], bool]:
    """Segments (type, x, y) in page space, and whether the path is closed anywhere."""
    m = raw.FS_MATRIX()
    if not raw.FPDFPageObj_GetMatrix(obj, ctypes.byref(m)):
        m.a, m.b, m.c, m.d, m.e, m.f = 1, 0, 0, 1, 0, 0
    n = raw.FPDFPath_CountSegments(obj)
    out = []
    closed = False
    x, y = ctypes.c_float(), ctypes.c_float()
    for i in range(max(n, 0)):
        seg = raw.FPDFPath_GetPathSegment(obj, i)
        if not seg or not raw.FPDFPathSegment_GetPoint(seg, ctypes.byref(x), ctypes.byref(y)):
            continue
        px = m.a * x.value + m.c * y.value + m.e
        py = m.b * x.value + m.d * y.value + m.f
        out.append((raw.FPDFPathSegment_GetType(seg), px, py))
        closed = closed or bool(raw.FPDFPathSegment_GetClose(seg))
    return out, closed


def _is_axis_aligned(pts: list[tuple[int, float, float]], tol: float) -> bool:
    prev = None
    for typ, x, y in pts:
        if typ == _SEG_BEZIER:
            return False
        if typ == _SEG_LINE and prev is not None:
            if abs(x - prev[0]) > tol and abs(y - prev[1]) > tol:
                return False
        prev = (x, y)
    return True


def _radical(pts, x0, y0, x1, y1) -> Radical | None:
    """Recognise a √-shaped polygon (with or without the vinculum on top)."""
    w, h = x1 - x0, y1 - y0
    lines = [(x, y) for t, x, y in pts if t != _SEG_BEZIER]
    # a radical sign is one small polygon (a dashed axis with an arrow head is not)
    if (
        not 4 <= len(lines) <= 24
        or h <= 0
        or w <= 0
        or any(t == _SEG_BEZIER for t, _, _ in pts)
        or sum(1 for t, _, _ in pts if t == _SEG_MOVE) > 2
    ):
        return None
    lowest = min(lines, key=lambda p: (p[1], p[0]))
    leftmost = min(lines, key=lambda p: (p[0], p[1]))
    top_pts = [p for p in lines if p[1] >= y1 - 0.08 * h]
    if not top_pts:
        return None
    top_left = min(p[0] for p in top_pts)
    # the bottom vertex sits left of the vertical stroke's top, the tick starts
    # on the left at mid height, and the top edge runs to the right border
    if not (
        lowest[1] <= y0 + 0.08 * h and x0 + 0.15 * min(w, h) <= lowest[0] <= top_left + 0.25 * h
    ):
        return None
    if not (y0 + 0.25 * h <= leftmost[1] <= y0 + 0.8 * h and leftmost[0] <= x0 + 0.05 * w + 0.5):
        return None
    if max(p[0] for p in top_pts) < x1 - 0.05 * w - 0.5:
        return None
    sign_right = top_left + 0.1 * h
    if sign_right >= x1 - 0.5:  # no vinculum drawn with the sign
        sign_right = x1
    return Radical(x0=x0, y0=y0, x1=sign_right, y1=y1, bar_x1=x1, bar_y=y1 - 0.04 * h)


def _angle(pts, x0, y0, x1, y1) -> str | None:
    """An angle bracket drawn as two strokes: ⟨ (apex on the left) or ⟩."""
    lines = [(x, y) for t, x, y in pts if t != _SEG_BEZIER]
    if len(lines) != 3 or any(t == _SEG_BEZIER for t, _, _ in pts):
        return None
    (ax, ay), (mx, my), (bx, by) = lines
    h = y1 - y0
    if h <= 0 or abs(ax - bx) > 0.15 * h or abs(abs(ay - by) - h) > 0.25 * h:
        return None
    if not (min(ay, by) + 0.3 * h <= my <= min(ay, by) + 0.7 * h) or abs(mx - ax) < 0.12 * h:
        return None
    return "⟨" if mx < ax else "⟩"


def classify_path(
    obj, bounds: tuple[float, float, float, float], text_size: float, out: VectorContent
) -> None:
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    if w <= 0 and h <= 0:
        return
    pts, _closed = _points(obj)
    tol = max(0.6, 0.02 * max(w, h))
    aligned = _is_axis_aligned(pts, tol)
    if w >= 1.5 and 0 < h <= max(2.0, 0.08 * w) and h <= 3.0:
        out.hrules.append((x0, x1, (y0 + y1) / 2, h))
        return
    if aligned and w <= max(2.0, 0.15 * text_size) and 0.6 * text_size <= h <= 5.0 * text_size:
        out.vbars.append((x0, y0, x1, y1))
        return
    if 0.6 * text_size <= h <= 6.0 * text_size and w <= 0.6 * h:
        ang = _angle(pts, x0, y0, x1, y1)
        if ang is not None:
            out.angles.append((x0, y0, x1, y1, ang))
            return
    if 0.5 * text_size <= h <= 6.0 * text_size and w <= 40.0 * text_size:
        rad = _radical(pts, x0, y0, x1, y1)
        if rad is not None:
            # producers often fill the sign and stroke its outline: keep one
            if not any(
                abs(r.x0 - rad.x0) < 1.0
                and abs(r.y0 - rad.y0) < 1.0
                and abs(r.bar_x1 - rad.bar_x1) < 1.5
                for r in out.radicals
            ):
                out.radicals.append(rad)
            return
    if not aligned:
        out.graphics.append((x0, y0, x1, y1))
    else:
        out.lines.append((x0, y0, x1, y1))


def figure_regions(
    graphics: list[tuple[float, float, float, float]],
    text_size: float,
    min_paths: int = 3,
    lines: list[tuple[float, float, float, float]] = (),
) -> list[tuple[float, float, float, float]]:
    """Merge nearby drawing strokes into figure boxes; keep the substantial ones.

    Straight axis-aligned strokes (axes, frames) do not start a figure - a
    framed formula or a table must not be mistaken for one - but those that
    touch a figure extend it, and the box gets a margin of about one text
    line so that labels placed just outside the strokes ("Re", "Im") are
    inside it.
    """
    boxes = [list(b) + [1] for b in sorted(graphics)]
    pad = 0.8 * text_size
    merged = True
    while merged:
        merged = False
        out: list[list[float]] = []
        for b in boxes:
            for o in out:
                if (
                    b[0] <= o[2] + pad
                    and o[0] <= b[2] + pad
                    and b[1] <= o[3] + pad
                    and o[1] <= b[3] + pad
                ):
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    o[4] += b[4]
                    merged = True
                    break
            else:
                out.append(list(b))
        boxes = out
    # a figure needs at least one substantial drawn element (a curve, a long
    # arrow); a cluster of small strokes is part of the text (vector-drawn symbols)
    big = [g for g in graphics if max(g[2] - g[0], g[3] - g[1]) >= 3 * text_size]
    figs = [
        b
        for b in boxes
        if b[4] >= min_paths
        and b[2] - b[0] >= 4 * text_size
        and b[3] - b[1] >= 3 * text_size
        and any(
            b[0] - pad <= g[0] and g[2] <= b[2] + pad and b[1] - pad <= g[1] and g[3] <= b[3] + pad
            for g in big
        )
    ]
    for f in figs:
        for x0, y0, x1, y1 in sorted(lines):
            if x0 <= f[2] + pad and f[0] <= x1 + pad and y0 <= f[3] + pad and f[1] <= y1 + pad:
                f[0], f[1], f[2], f[3] = min(f[0], x0), min(f[1], y0), max(f[2], x1), max(f[3], y1)
    margin = 1.0 * text_size
    return [(f[0] - margin, f[1] - margin, f[2] + margin, f[3] + margin) for f in figs]
