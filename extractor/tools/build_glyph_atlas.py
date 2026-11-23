"""Build ``src/hebmath/data/glyph_atlas.json.gz`` from SIL-OFL reference fonts.

Development tool, not used at run time. Needs fontTools (MIT) only to read
each font's character coverage; rendering goes through the same PDFium path
as run-time glyph recognition (``hebmath.readers.pdf.glyphs.render``).

    python tools/build_glyph_atlas.py --stix STIXTwoMath.input.ttf \
        --fonts-dir /path/to/google-fonts/ofl

Font provenance is recorded in the atlas ("sources") and in
THIRD_PARTY_NOTICES.md. Output is deterministic for a given PDFium version.
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pypdfium2.raw as raw
from fontTools.ttLib import TTFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hebmath.readers.pdf.glyphs.render import TALL_STRETCH, render_glyph, render_glyphs  # noqa: E402

ATLAS_VERSION = "1"


def _range(a: int, b: int) -> list[str]:
    return [chr(c) for c in range(a, b + 1)]


SCRIPTS = list("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
ASCII = _range(0x21, 0x7E)
DIGITS_PUNCT = [c for c in ASCII if not c.isalpha()]
HEBREW = _range(0x05D0, 0x05EA) + ["־", "׳", "״"]
TEXT_EXTRA = list("–—‘’‚“”„…•·°§©®™€₪×÷±") + SCRIPTS
MATH = (
    ASCII
    + list("±×÷¬·°µ′″‴‖…⋯⋮⋰⋱∂∇∞∅")
    + _range(0x0391, 0x03A9)
    + _range(0x03B1, 0x03C9)
    + list("ϑϕϖϵϱςϒ")
    + list("ℂℍℕℙℚℝℤℓ℘ℵℏℑℜ")
    + _range(0x2190, 0x21FF)
    + _range(0x2200, 0x22FF)
    + list("⌈⌉⌊⌋⌠⌡⟨⟩⟦⟧⟵⟶⟷⟸⟹⟺⟼")
    + _range(0x239B, 0x23B3)
    + ["⎯", "⏐", "│", "‖"]
    + list("⨀⨁⨂⨄⨆⩽⩾⪯⪰□△▽◇○●★")
    + SCRIPTS
)

# (font file relative to --fonts-dir, label, character set)
# Several math designs, so that symbol shapes from Symbol/MT Extra/Cambria are
# close to at least one reference.
MATH_FONTS = [
    ("libertinusmath/LibertinusMath-Regular.ttf", "Libertinus Math", MATH),
    ("notosansmath/NotoSansMath-Regular.ttf", "Noto Sans Math", MATH),
]
TEXT_FONTS = [
    ("tinos/Tinos-Regular.ttf", "Tinos Regular", HEBREW + ASCII + TEXT_EXTRA),
    ("tinos/Tinos-Bold.ttf", "Tinos Bold", HEBREW + ASCII + TEXT_EXTRA),
    ("tinos/Tinos-Italic.ttf", "Tinos Italic", ASCII),
    ("arimo/Arimo[wght].ttf", "Arimo", HEBREW + ASCII + TEXT_EXTRA),
    ("davidlibre/DavidLibre-Regular.ttf", "David Libre Regular", HEBREW + DIGITS_PUNCT),
    ("davidlibre/DavidLibre-Bold.ttf", "David Libre Bold", HEBREW + DIGITS_PUNCT),
    ("frankruhllibre/FrankRuhlLibre[wght].ttf", "Frank Ruhl Libre", HEBREW + DIGITS_PUNCT),
    ("miriamlibre/MiriamLibre[wght].ttf", "Miriam Libre", HEBREW + DIGITS_PUNCT),
    ("alef/Alef-Regular.ttf", "Alef Regular", HEBREW + DIGITS_PUNCT),
    ("alef/Alef-Bold.ttf", "Alef Bold", HEBREW + DIGITS_PUNCT),
    ("notoserifhebrew/NotoSerifHebrew[wdth,wght].ttf", "Noto Serif Hebrew", HEBREW),
    ("notosanshebrew/NotoSansHebrew[wdth,wght].ttf", "Noto Sans Hebrew", HEBREW),
]


# Tall delimiters as fonts assemble them from pieces (top, extension..., middle, ..., bottom).
# ``*`` marks the extension piece, repeated to make shorter and longer variants.
TALL = {
    "(": "⎛*⎜⎝",
    ")": "⎞*⎟⎠",
    "[": "⎡*⎢⎣",
    "]": "⎤*⎥⎦",
    "{": "⎧*⎪⎨*⎪⎩",
    "}": "⎫*⎪⎬*⎪⎭",
    "⌈": "⎡*⎢",
    "⌉": "⎤*⎥",
    "⌊": "*⎢⎣",
    "⌋": "*⎥⎦",
    "|": "*⎪",
    "∫": "⌠*⎮⌡",
}


def _assemble(doc, font, recipe: str, repeat: int) -> list:
    """Stack the pieces of ``recipe`` so that their ink touches; return a descriptor (JSON)."""
    pieces: list[str] = []
    i = 0
    while i < len(recipe):
        if recipe[i] == "*":
            pieces.extend([recipe[i + 1]] * repeat)
            i += 2
        else:
            pieces.append(recipe[i])
            i += 1
    if len(pieces) < 2:
        pieces = pieces * 2
    items, dy, prev = [], 0.0, None
    for ch in pieces:
        d = render_glyph(doc, font, text=ch)
        if d.empty:
            return []
        if prev is not None:
            dy = dy + prev.y0 - d.y1
        items.append((None, ch, 0.0, dy))
        prev = d
    return render_glyphs(doc, font, items, max_stretch=TALL_STRETCH).to_json()


def _render_tall(path: Path) -> list[tuple[str, list]]:
    cmap = TTFont(str(path)).getBestCmap() or {}
    data = path.read_bytes()
    doc = raw.FPDF_CreateNewDocument()
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    font = raw.FPDFText_LoadFont(doc, buf, len(data), raw.FPDF_FONT_TRUETYPE, 1)
    out = []
    for ch, recipe in TALL.items():
        if not all(ord(c) in cmap for c in recipe if c != "*"):
            continue
        for repeat in (0, 1, 3):
            item = _assemble(doc, font, recipe, repeat)
            if item:
                out.append((ch, item))
    raw.FPDFFont_Close(font)
    raw.FPDF_CloseDocument(doc)
    return out


def _render_font(path: Path, chars: list[str]) -> list[tuple[str, list]]:
    cmap = TTFont(str(path)).getBestCmap() or {}
    data = path.read_bytes()
    doc = raw.FPDF_CreateNewDocument()
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    font = raw.FPDFText_LoadFont(doc, buf, len(data), raw.FPDF_FONT_TRUETYPE, 1)
    if not font:
        raise RuntimeError(f"PDFium cannot load {path}")
    out = []
    seen = set()
    for ch in chars:
        if ch in seen or ord(ch) not in cmap:
            continue
        seen.add(ch)
        d = render_glyph(doc, font, text=ch)
        if not d.empty:
            out.append((ch, d.to_json()))
    raw.FPDFFont_Close(font)
    raw.FPDF_CloseDocument(doc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stix", type=Path, required=True, help="STIXTwoMath .ttf/.otf")
    ap.add_argument("--fonts-dir", type=Path, required=True, help="google/fonts 'ofl' directory")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/hebmath/data/glyph_atlas.json.gz",
    )
    args = ap.parse_args()

    raw.FPDF_InitLibrary()
    jobs = [(args.stix, "STIX Two Math", MATH)] + [
        (args.fonts_dir / rel, label, cs) for rel, label, cs in MATH_FONTS + TEXT_FONTS
    ]
    sources, glyphs, tall = [], [], []
    for path, label, chars in jobs:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        src_index = len(sources)
        sources.append(f"{label} (sha256:{digest})")
        rendered = _render_font(path, chars)
        glyphs.extend([ch, src_index, item] for ch, item in rendered)
        if chars is MATH:
            assembled = _render_tall(path)
            tall.extend([ch, src_index, item] for ch, item in assembled)
        else:
            assembled = []
        print(
            f"{label}: {len(rendered)} glyphs, {len(assembled)} assembled delimiters",
            file=sys.stderr,
        )
    doc = {"version": ATLAS_VERSION, "sources": sources, "glyphs": glyphs, "tall": tall}
    payload = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    args.output.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    print(
        f"wrote {args.output} ({len(glyphs)} glyphs, {args.output.stat().st_size} bytes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
