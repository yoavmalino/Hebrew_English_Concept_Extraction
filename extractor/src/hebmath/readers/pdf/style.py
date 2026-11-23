"""Per-line formatting facts (size, weight, bold, underline, colour, position) for stage 1.

Sizes are the size as drawn (font size times the text-matrix scale), not the raw font size.

Reads only what PDFium already gave the layout step. It never changes the extracted text.
"""

from __future__ import annotations

from collections import Counter

from ...model import LineStyle, StyledWord
from .geometry import Atom, Rule, median

# An underline is a thin horizontal stroke just below the baseline. A fraction bar sits above the
# baseline and a table border far below it, so these limits (as fractions of the drawn size) keep
# them out. Starting values, tuned on the labelled set (design: stage 1 calibration).
# Drop limit: some writers (LaTeX \underline) put the stroke under the deepest letter of the
# line, about 0.35 of the size below the baseline with ק or ף. 0.40-0.60 gave the same synthetic
# result on 2026-11-14; 0.40 is its low end, since 0.45 added a false main heading on the real ארגון summary.
UNDERLINE_MAX_DROP = 0.40
# Thickness limit as a share of the drawn size. A real Google Docs underline (ארגון summary) is
# 0.18 of it; 0.20-0.40 gave the same dev result on 2026-11-14, 0.25 sits inside that plateau.
UNDERLINE_MAX_THICKNESS = 0.25
UNDERLINE_MIN_OVERLAP = 0.5


def _drawn_size(atom: Atom) -> float:
    """Return the glyph's size as drawn on the page (falls back to the font size)."""
    return atom.style_size or atom.size


def _is_underlined(atom: Atom, rules: list[Rule]) -> bool:
    """Return True when a thin stroke runs under at least half of the glyph's width."""
    width = atom.x1 - atom.x0
    if width <= 0:
        return False
    for rule in rules:
        drop = atom.base - rule.y
        if not 0 <= drop <= UNDERLINE_MAX_DROP * _drawn_size(atom):
            continue
        if rule.thickness > UNDERLINE_MAX_THICKNESS * _drawn_size(atom):
            continue
        overlap = min(rule.x1, atom.x1) - max(rule.x0, atom.x0)
        if overlap >= UNDERLINE_MIN_OVERLAP * width:
            return True
    return False


def line_style(atoms: list[Atom], rules: list[Rule], page: int) -> LineStyle | None:
    """Summarize the formatting of one line.

    Args:
        atoms: The atoms of the line; only real glyphs (not spaces or composites) are counted.
        rules: The thin horizontal strokes of the page.
        page: 1-based page number.

    Returns:
        The line's style, or None when it has no glyphs.
    """
    glyphs = [a for a in atoms if a.kind == "g" and a.text and not a.text.isspace()]
    if not glyphs:
        return None
    top = max(a.y1 for a in glyphs)
    bottom = min(a.y0 for a in glyphs)
    reach = max(_drawn_size(a) for a in glyphs)
    nearby = [r for r in rules if bottom - reach <= r.y <= top]
    weights = [a.weight for a in glyphs if a.weight > 0]
    count = len(glyphs)
    return LineStyle(
        page=page,
        size=sum(_drawn_size(a) for a in glyphs) / count,
        weight=int(median(weights, 0)) if weights else 0,
        name_bold_share=sum(1 for a in glyphs if a.bold) / count,
        underline_share=sum(1 for a in glyphs if _is_underlined(a, nearby)) / count,
        chars=count,
        x0=min(a.x0 for a in glyphs),
        x1=max(a.x1 for a in glyphs),
        top=top,
        bottom=bottom,
    )


def underlined_glyph_ids(atoms: list[Atom], rules: list[Rule]) -> set[int]:
    """Return ``id(atom)`` for every glyph of the line that has an underline stroke under it."""
    glyphs = [a for a in atoms if a.kind == "g" and a.text and not a.text.isspace()]
    if not glyphs or not rules:
        return set()
    top = max(a.y1 for a in glyphs)
    bottom = min(a.y0 for a in glyphs)
    reach = max(_drawn_size(a) for a in glyphs)
    nearby = [r for r in rules if bottom - reach <= r.y <= top]
    return {id(a) for a in glyphs if _is_underlined(a, nearby)}


def make_word(
    chars: list[str], atoms: list[Atom | None], underlined: set[int]
) -> StyledWord | None:
    """Build a styled word from its characters and the glyph atoms they came from.

    The style of the word is the most common style among its glyphs (a word split between two
    fonts takes the one with more characters).

    Args:
        chars: The word's characters in reading order.
        atoms: The source atom of each character, or None when it is not a plain glyph.
        underlined: The result of ``underlined_glyph_ids`` for the line.

    Returns:
        The word, or None when none of its characters came from a glyph.
    """
    real = [a for a in atoms if a is not None]
    if not real:
        return None
    votes = Counter(
        (a.font, a.weight, round(_drawn_size(a) * 2) / 2, a.bold, id(a) in underlined, a.color)
        for a in real
    )
    face, weight, size, bold, underline, color = min(votes, key=lambda k: (-votes[k], str(k)))
    return StyledWord("".join(chars), face, weight, size, bold, underline, color)
