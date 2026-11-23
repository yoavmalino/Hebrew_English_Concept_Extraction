"""Font-aware glyph interpretation for PDF math.

TeX's Computer Modern / Latin Modern Type1 fonts often reach the text layer
without a usable Unicode mapping (their glyph names such as
``summationdisplay`` are not in the Adobe Glyph List), so PDFium reports the
raw character code. This module maps those codes back to symbols, and derives
math styles (blackboard bold, calligraphic, ...) from the font family.
"""

from __future__ import annotations

import re

# cmex10 / lmex10 encoding (TeX math extension font), code -> base symbol.
_CMEX_NAMES = [
    "(",
    ")",
    "[",
    "]",
    "⌊",
    "⌋",
    "⌈",
    "⌉",
    "{",
    "}",
    "⟨",
    "⟩",
    "|",
    "‖",
    "/",
    "\\",  # 0x00
    "(",
    ")",
    "(",
    ")",
    "[",
    "]",
    "⌊",
    "⌋",
    "⌈",
    "⌉",
    "{",
    "}",
    "⟨",
    "⟩",
    "/",
    "\\",  # 0x10
    "(",
    ")",
    "[",
    "]",
    "⌊",
    "⌋",
    "⌈",
    "⌉",
    "{",
    "}",
    "⟨",
    "⟩",
    "/",
    "\\",
    "/",
    "\\",  # 0x20
    "(",
    ")",
    "[",
    "]",
    "[",
    "]",
    "[",
    "]",
    "{",
    "}",
    "{",
    "}",
    "{",
    "}",
    "{",
    "|",  # 0x30 pieces
    "(",
    ")",
    "(",
    ")",
    "⟨",
    "⟩",
    "⨆",
    "⨆",
    "∮",
    "∮",
    "⨀",
    "⨀",
    "⨁",
    "⨁",
    "⨂",
    "⨂",  # 0x40
    "∑",
    "∏",
    "∫",
    "⋃",
    "⋂",
    "⨄",
    "⋀",
    "⋁",
    "∑",
    "∏",
    "∫",
    "⋃",
    "⋂",
    "⨄",
    "⋀",
    "⋁",  # 0x50
    "∐",
    "∐",
    "^",
    "^",
    "^",
    "~",
    "~",
    "~",
    "[",
    "]",
    "⌊",
    "⌋",
    "⌈",
    "⌉",
    "{",
    "}",  # 0x60
    "√",
    "√",
    "√",
    "√",
    "√",
    "√",
    "√",
    "‖",
    "↑",
    "↓",
    "{",
    "}",
    "{",
    "}",
    "⇑",
    "⇓",  # 0x70
]
# Extension pieces that are only part of a bigger delimiter (drawn as several glyphs).
_CMEX_PIECES = set(range(0x30, 0x44)) | {
    0x74,
    0x75,
    0x76,
    0x77,
    0x78,
    0x79,
    0x7A,
    0x7B,
    0x7C,
    0x7D,
    0x7E,
    0x7F,
}

_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")


def family(fontname: str) -> str:
    return _SUBSET_RE.sub("", fontname or "")


_MATH_FONT_RE = re.compile(
    r"^(cmmi|cmsy|cmex|cmbsy|cmmib|msam|msbm|eufm|eufb|eusm|eusb|rsfs|stmary|wasy|"
    r"lmmathitalic|lmmathsymbols|lmmathextension|lmmi|lmsy|lmex|txmi|txsy|txex|pxmi|pxsy|pxex|"
    r"mathdesign|esint|bbold|dsrom|mnsymbol|cmmib|"
    r"cambriamath|cambria-math|latinmodernmath|latinmodern-math|lmmath|stixmath|stix2math|stixtwomath|"
    r"stixgeneral|stixsizes|stixintegrals|xitsmath|xits-math|asanamath|asana-math|texgyre\w*math|"
    r"garamondmath|libertinusmath|firamath|dejavumath|newcmmath|newcomputermodernmath|"
    r"symbol$|symbolmt|mtextra|mt extra|mtsymbol|euclid|mathjax|katex_math|katex_ams|katex_size|"
    r"opensymbol|mathematicalpi|mathpi|mathtimes|mtmi|mtsy|mtex|lucidanewmath|lucidabrightmath)",
    re.IGNORECASE,
)


def is_math_font(fontname: str) -> bool:
    f = family(fontname).replace(" ", "").lower()
    return bool(_MATH_FONT_RE.match(f)) or "math" in f and "mathematica" not in f


def is_cmex(fontname: str) -> bool:
    f = family(fontname).lower()
    return f.startswith(("cmex", "lmex", "lmmathextension", "txex", "pxex", "mtex", "esint"))


def is_italic_font(fontname: str, flags: int) -> bool:
    f = family(fontname).lower()
    if flags & 0x40:  # PDF font descriptor flag: Italic
        return True
    return bool(re.search(r"(italic|oblique|-it\b|cmmi|lmmi|mathitalic|txmi|pxmi|mtmi)", f))


def is_bold_font(fontname: str) -> bool:
    f = family(fontname).lower()
    return bool(re.search(r"(bold|cmbx|cmmib|cmbsy|-bd\b|black|heavy)", f))


def map_cmex(code: int) -> tuple[str, bool] | None:
    """Return (symbol, is_piece) for a cmex character code."""
    if 0 <= code < len(_CMEX_NAMES):
        return _CMEX_NAMES[code], code in _CMEX_PIECES
    return None


def style_for_font(fontname: str) -> str | None:
    """LaTeX alphabet command implied by a TeX font family (for ASCII letters)."""
    f = family(fontname).lower()
    if f.startswith(("msbm", "bbold", "dsrom")):
        return r"\mathbb"
    if f.startswith(("cmsy", "lmsy", "lmmathsymbols", "eusm", "eusb", "cmbsy")):
        return r"\mathcal"
    if f.startswith("rsfs"):
        return r"\mathcal"
    if f.startswith(("eufm", "eufb")):
        return r"\mathfrak"
    return None


def is_unmapped(ch: str) -> bool:
    if not ch:
        return True
    cp = ord(ch[0])
    return (
        cp in (0, 0xFFFD, 0xFFFE, 0xFFFF)
        or 0xE000 <= cp <= 0xF8FF
        or cp < 0x20
        and ch not in "\t\n\r"
    )
