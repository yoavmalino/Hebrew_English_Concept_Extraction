"""Low-level page content extraction with PDFium (via pypdfium2).

PDFium is used (rather than pdfminer) because it reports *ink* bounding boxes
of glyphs, which the math engine needs to see where a big integral sign or a
radical really extends, and because it repairs damaged files the same way
Chrome does.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field

import pypdfium2 as pdfium
import pypdfium2.raw as raw

from .fonts import (
    family,
    is_bold_font,
    is_cmex,
    is_italic_font,
    is_math_font,
    is_unmapped,
    map_cmex,
)
from .geometry import Atom, Rule, is_mark
from .glyphs.fontid import font_identity
from .vector import Radical, VectorContent, classify_path, figure_regions


@dataclass
class PageContent:
    width: float
    height: float
    atoms: list[Atom] = field(default_factory=list)  # in normalised orientation
    rules: list[Rule] = field(default_factory=list)
    image_area: float = 0.0
    text_chars: int = 0
    unmapped_chars: int = 0
    rotated_chars: int = 0
    angle: int = 0  # dominant text rotation (degrees, multiple of 90)
    truncated: bool = False
    page_index: int = 0
    vbars: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )  # thin vertical strokes
    radicals: list[Radical] = field(default_factory=list)
    angles: list[tuple[float, float, float, float, str]] = field(default_factory=list)
    figures: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )  # diagram regions
    # fonts (by identity) that have at least one glyph without a Unicode mapping:
    # every code of such a font is suspect, including those PDFium "mapped"
    # through a fallback encoding.
    cipher_fonts: set[str] = field(default_factory=set)
    obj_bounds: dict[int, tuple[float, float, float, float]] = field(
        default_factory=dict
    )  # text objects


def _rotate_point(x: float, y: float, angle: int, w: float, h: float) -> tuple[float, float]:
    """Rotate page coordinates so text with ``angle`` becomes horizontal."""
    if angle == 0:
        return x, y
    if angle == 90:
        return y, w - x
    if angle == 180:
        return w - x, h - y
    return h - y, x  # 270


def _rotate_box(x0, y0, x1, y1, angle, w, h):
    pts = [_rotate_point(x, y, angle, w, h) for x, y in ((x0, y0), (x1, y1), (x0, y1), (x1, y0))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def extract_page(page: pdfium.PdfPage, max_glyphs: int, page_index: int = 0) -> PageContent:
    w, h = page.get_size()
    content = PageContent(width=w, height=h, page_index=page_index)
    object_ids = text_object_ids(page)
    textpage = page.get_textpage()
    try:
        _extract_chars(textpage, content, max_glyphs, object_ids)
    finally:
        textpage.close()
    _extract_objects(page, content)
    return content


def text_object_ids(page: pdfium.PdfPage) -> dict[int, int]:
    """Handle address -> 1-based position of each text object in content order.

    Atoms refer to their text object by this position, never by the handle
    address: addresses change each time a page is loaded, and code that looks
    objects up again on a reloaded page (glyph restoration) must find the same
    ones on every run.
    """
    ids: dict[int, int] = {}
    for obj in page.get_objects(max_depth=8):
        if obj.type == raw.FPDF_PAGEOBJ_TEXT:
            addr = ctypes.cast(obj.raw, ctypes.c_void_p).value or 0
            ids.setdefault(addr, len(ids) + 1)
    return ids


def _extract_chars(
    textpage, content: PageContent, max_glyphs: int, object_ids: dict[int, int] | None = None
) -> None:
    object_ids = object_ids or {}
    tp = textpage.raw
    n = raw.FPDFText_CountChars(tp)
    left, r, b, t = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    ox, oy = ctypes.c_double(), ctypes.c_double()
    rect = raw.FS_RECTF()
    flags = ctypes.c_int()
    red, green, blue, alpha = (ctypes.c_uint() for _ in range(4))
    char_matrix = raw.FS_MATRIX()
    name_buf = ctypes.create_string_buffer(256)
    font_cache: dict[tuple[str, int], tuple] = {}
    ident_cache: dict[int, str] = {}

    raw_chars = []
    angles: dict[int, int] = {}
    limit = min(n, max_glyphs)
    pending_surrogate: tuple[int, int] | None = None  # (index, code)
    for i in range(limit):
        if raw.FPDFText_IsGenerated(tp, i) == 1:
            continue
        code = raw.FPDFText_GetUnicode(tp, i)
        map_error = raw.FPDFText_HasUnicodeMapError(tp, i) == 1
        # PDFium reports astral characters (e.g. U+1D465 MATHEMATICAL ITALIC X)
        # as two UTF-16 surrogates, in either order. Re-combine them.
        if 0xD800 <= code <= 0xDFFF:
            if pending_surrogate is None:
                pending_surrogate = (i, code)
                continue
            _, other = pending_surrogate
            pending_surrogate = None
            hi, lo = (other, code) if other < 0xDC00 else (code, other)
            if 0xD800 <= hi < 0xDC00 <= lo <= 0xDFFF:
                code = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
            else:
                code = 0
        elif pending_surrogate is not None:
            pending_surrogate = None
        ch = chr(code) if 0 < code < 0x110000 else ""
        if map_error:
            ch = "\x00" + ch  # raw char code without a Unicode mapping (resolved per font below)
        if ch in ("\r", "\n"):
            continue
        if not raw.FPDFText_GetCharBox(tp, i, left, r, b, t):
            continue
        raw.FPDFText_GetCharOrigin(tp, i, ox, oy)
        raw.FPDFText_GetLooseCharBox(tp, i, rect)
        size = raw.FPDFText_GetFontSize(tp, i)
        weight = raw.FPDFText_GetFontWeight(tp, i)
        # Some writers set the font to size 1 and scale it with the text matrix, so the font size
        # alone reads 1.0 everywhere; the size as drawn is the font size times the matrix's
        # vertical scale (spike 2026-11-14 on the ML-1 summary: cover 26, lectures 20, body 10).
        style_size = 0.0
        if raw.FPDFText_GetMatrix(tp, i, char_matrix):
            style_size = size * math.hypot(char_matrix.c, char_matrix.d)
        # fill colour as 0xRRGGBB (stage 1 heading look); -1 when PDFium cannot tell
        color = -1
        if raw.FPDFText_GetFillColor(tp, i, red, green, blue, alpha):
            color = (red.value << 16) | (green.value << 8) | blue.value
        nlen = raw.FPDFText_GetFontInfo(tp, i, name_buf, 256, ctypes.byref(flags))
        fontname = name_buf.value.decode("latin-1", "replace") if nlen else ""
        fkey = ""
        tobj = raw.FPDFText_GetTextObject(tp, i)
        addr_obj = ctypes.cast(tobj, ctypes.c_void_p).value or 0 if tobj else 0
        obj_id = object_ids.get(addr_obj, 0) if addr_obj else 0
        if addr_obj and not obj_id:
            obj_id = object_ids.setdefault(
                addr_obj, len(object_ids) + 1
            )  # not reachable by get_objects
        if tobj and obj_id not in content.obj_bounds:
            ol, ob, orr, ot = ctypes.c_float(), ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
            if raw.FPDFPageObj_GetBounds(tobj, ol, ob, orr, ot):
                content.obj_bounds[obj_id] = (ol.value, ob.value, orr.value, ot.value)
        if tobj:
            font = raw.FPDFTextObj_GetFont(tobj)
            addr = ctypes.cast(font, ctypes.c_void_p).value or 0
            fkey = ident_cache.get(addr)
            if fkey is None:
                fkey = font_identity(font)
                if fkey.endswith("#t3"):
                    # unhashable font: unique per page, numbered in drawing order
                    fkey = f"{fkey}-p{content.page_index}-{sum(1 for v in ident_cache.values() if '#t3' in v)}"
                ident_cache[addr] = fkey

        ang_rad = raw.FPDFText_GetCharAngle(tp, i)
        angle = int(round(math.degrees(ang_rad) / 90.0)) % 4 * 90 if ang_rad >= 0 else 0
        angles[angle] = angles.get(angle, 0) + (0 if ch.isspace() else 1)
        raw_chars.append(
            (
                i,
                ch,
                code,
                left.value,
                b.value,
                r.value,
                t.value,
                ox.value,
                oy.value,
                rect.left,
                rect.right,
                rect.top,
                rect.bottom,
                size,
                fontname,
                flags.value,
                angle,
                fkey,
                obj_id,
                weight,
                color,
                style_size,
            )
        )

    content.truncated = n > max_glyphs
    dominant = max(sorted(angles), key=lambda a: angles[a]) if angles else 0
    content.angle = dominant
    W, H = content.width, content.height

    prev_glyph: Atom | None = None
    for (
        i,
        ch,
        code,
        x0,
        y0,
        x1,
        y1,
        oxv,
        oyv,
        lx0,
        lx1,
        ltop,
        lbot,
        size,
        fontname,
        fl,
        angle,
        fkey,
        obj_id,
        weight,
        color,
        style_size,
    ) in raw_chars:
        if angle != dominant:
            content.rotated_chars += 1
            continue
        if size <= 0.5:
            size = max(y1 - y0, 1.0)
        if style_size <= 0.5:
            style_size = size
        key = (fontname, fl)
        if key not in font_cache:
            font_cache[key] = (
                family(fontname),
                is_math_font(fontname),
                is_italic_font(fontname, fl),
                is_bold_font(fontname),
                is_cmex(fontname),
            )
        fam, mathf, ital, bold, cmex = font_cache[key]

        text = ch
        unmapped = ch.startswith("\x00")
        if unmapped:
            text = ch[1:]
        if cmex and code < 0x80:
            mapped = map_cmex(code)
            if mapped is not None:
                text = mapped[0]
                unmapped = False
        elif unmapped:
            text = "\ufffd"
        if text.isspace():
            kind = "sp"
        else:
            kind = "g"
            content.text_chars += 1
            if is_unmapped(text):
                content.unmapped_chars += 1
                text = "�"
                if fkey and not cmex:
                    # no mapping even after the font-specific tables: every code
                    # of this font is suspect (see glyphs.recover)
                    content.cipher_fonts.add(fkey)

        bx0, by0, bx1, by1 = _rotate_box(x0, y0, x1, y1, dominant, W, H)
        ox_r, oy_r = _rotate_point(oxv, oyv, dominant, W, H)
        if cmex:
            # TeX extension glyphs hang from their origin; centre them on the math axis instead.
            oy_r = (by0 + by1) / 2 - 0.25 * size
        llx0, _, llx1, _ = _rotate_box(min(lx0, lx1), lbot, max(lx0, lx1), ltop, dominant, W, H)
        if bx1 - bx0 <= 0 and kind == "g":
            bx0, bx1 = llx0, llx1
        if by1 - by0 <= 0 and kind == "g":
            by0, by1 = oy_r, oy_r + 0.6 * size

        atom = Atom(
            x0=bx0,
            y0=by0,
            x1=bx1,
            y1=by1,
            base=oy_r,
            size=size,
            text=text,
            font=fam,
            order=i,
            kind=kind,
            math_font=mathf,
            italic=ital,
            bold=bold,
            code=code,
            lx0=min(llx0, bx0),
            lx1=max(llx1, bx1),
            fkey=fkey,
            raw_code=ch.startswith("\x00"),
            obj=obj_id,
            weight=weight,
            color=color,
            style_size=style_size,
        )
        if (
            kind == "g"
            and is_mark(text)
            and prev_glyph is not None
            and min(bx1, prev_glyph.x1) - max(bx0, prev_glyph.x0) > -0.1 * size
        ):
            # combining mark (niqqud / diacritic) belongs to the previous base glyph
            prev_glyph.marks += text
            continue
        content.atoms.append(atom)
        if kind == "g":
            prev_glyph = atom


def _extract_objects(page: pdfium.PdfPage, content: PageContent) -> None:
    W, H = content.width, content.height
    angle = content.angle
    sizes = sorted(a.size for a in content.atoms if a.kind == "g")
    text_size = sizes[len(sizes) // 2] if sizes else 10.0
    vec = VectorContent()
    for obj in page.get_objects(max_depth=8):
        typ = obj.type
        try:
            x0, y0, x1, y1 = obj.get_bounds()
        except Exception:  # noqa: BLE001 - object without bounds
            continue
        if typ == raw.FPDF_PAGEOBJ_IMAGE:
            content.image_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
        elif typ == raw.FPDF_PAGEOBJ_PATH:
            bx0, by0, bx1, by1 = _rotate_box(x0, y0, x1, y1, angle, W, H)
            if obj.level == 0 and angle == 0:
                classify_path(obj.raw, (bx0, by0, bx1, by1), text_size, vec)
                continue
            wd, ht = bx1 - bx0, by1 - by0
            if wd >= 1.5 and 0 < ht <= max(2.0, 0.08 * wd) and ht <= 3.0:
                vec.hrules.append((bx0, bx1, (by0 + by1) / 2, ht))
    content.rules = [Rule(x0, x1, y, t) for x0, x1, y, t in vec.hrules]
    content.vbars = vec.vbars
    content.radicals = vec.radicals
    content.angles = vec.angles
    content.figures = figure_regions(vec.graphics, text_size, lines=vec.lines)
    for r in vec.radicals:
        if r.bar_x1 - r.x1 > 0.5:
            content.rules.append(Rule(r.x1, r.bar_x1, r.bar_y, 0.5))
    content.rules.sort(key=lambda r: (round(r.w, 2), -round(r.y, 2), round(r.x0, 2)))
