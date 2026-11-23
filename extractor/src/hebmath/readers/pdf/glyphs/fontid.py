"""Stable identity of an embedded PDF font.

PDFium strips the subset prefix ("ABCDEF+") from font names, and producers
such as PScript5 give unrelated subsets the same name, so the name alone is
not a key. The identity combines the base font name with a digest of the
embedded font program.
"""

from __future__ import annotations

import ctypes
import hashlib

import pypdfium2.raw as raw

_NAME_BUF = 256


def font_identity(font) -> str:
    """``"<base name>#<12 hex digits>"`` for an FPDF_FONT handle ("" if unavailable)."""
    if not font:
        return ""
    buf = ctypes.create_string_buffer(_NAME_BUF)
    n = raw.FPDFFont_GetBaseFontName(font, buf, _NAME_BUF)
    name = buf.value.decode("latin-1", "replace") if n else ""
    size = ctypes.c_size_t(0)
    digest = ""
    if raw.FPDFFont_GetFontData(font, None, 0, ctypes.byref(size)) and size.value:
        data = (ctypes.c_uint8 * size.value)()
        if raw.FPDFFont_GetFontData(font, data, size.value, ctypes.byref(size)):
            digest = hashlib.sha1(bytes(data)).hexdigest()[:12]
    if not digest:
        # No font program to hash (Type3 fonts draw glyphs with content streams).
        # Such fonts cannot be told apart reliably (PDFium exposes no metrics for
        # them), so the caller makes the identity unique per page.
        return f"{name}#t3"
    return f"{name}#{digest}"
