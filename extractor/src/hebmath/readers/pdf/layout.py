"""Page layout analysis: atoms -> lines -> tokens -> (prose | math) -> blocks.

Steps for one page (all deterministic, total orderings everywhere):

1. Cluster atoms into text lines by baseline. Full-size glyphs define lines;
   composites are attached by their math axis; smaller glyphs (scripts) are
   attached to the line whose vertical band contains them.
2. Split each line into tokens at horizontal gaps / spaces, and at
   Hebrew/non-Hebrew boundaries.
3. Classify tokens (Hebrew prose, Latin prose, strong math, weak math) and
   group consecutive math tokens into *math regions*.
4. Convert each region to LaTeX with the math engine and reorder the line from
   visual to logical order with the bidi algorithm, treating each formula as
   one opaque left-to-right unit.
5. Merge lines into paragraphs; lines that contain only math become display
   formulas (with an equation number when one is detected).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...bidi import MIRROR, Unit, bidi_class, reorder
from ...latex.plaintext import is_mathish_token, is_seed_char
from ...latex.symbols import FUNCTION_NAMES, OPERATOR_NAMES, math_alphanumeric, split_function_names
from ...model import Block, InlineMath, LineStyle, Segment, StyledWord, TextSpan
from .geometry import Atom, Rule, is_hebrew_char, median, union
from .mathengine import PdfMathEngine
from .style import line_style, make_word, underlined_glyph_ids

_EQNUM_RE = re.compile(r"^\(\s*\d+(?:[.\-]\d+)*[a-z]?\s*\)$")
_PUNCT = set(".,;:!?،")
_DELIMS = set("()[]{}|‖⟨⟩") | set("∑∏∐∫∬∭∮⋃⋂⨁⨂")  # also big operators: drawn larger, lower
_FUNC_CALL_RE = re.compile(r"(?:^|[^A-Za-z])([A-Za-z]{2,8})\(")
_OPERATOR_TOKENS = set("=<>≤≥≠≈≡⇒⇐⇔→←↔⟹⟺∈∉⊂⊆⊃⊇∪∩+−±×⋅")
_HYPHENS = set("-־‐‑")
# A line whose underline or bold share differs from the line above by this much starts a new
# paragraph, even when the line above spans the column: an underlined or bold heading at body size
# was glued to the paragraph before it on the synthetic corpus (2026-11-14).
EMPHASIS_BREAK_SHARE = 0.5


@dataclass(eq=False)
class Line:
    atoms: list[Atom] = field(default_factory=list)
    base: float = 0.0
    size: float = 0.0
    y0: float = 0.0
    y1: float = 0.0
    x0: float = 0.0
    x1: float = 0.0
    main_count: int = 0

    def add_main(self, a: Atom) -> None:
        self.atoms.append(a)
        self.main_count += 1
        if self.main_count == 1:
            self.base, self.size = a.base, a.size
            self.y0, self.y1, self.x0, self.x1 = a.y0, a.y1, a.x0, a.x1
        else:
            k = self.main_count
            self.base += (a.base - self.base) / k
            self.size += (a.size - self.size) / k
            self._grow(a)

    def add_secondary(self, a: Atom) -> None:
        self.atoms.append(a)
        self._grow(a)

    def _grow(self, a: Atom) -> None:
        self.y0, self.y1 = min(self.y0, a.y0), max(self.y1, a.y1)
        self.x0, self.x1 = min(self.x0, a.x0), max(self.x1, a.x1)


@dataclass
class LineResult:
    """A line after math detection and bidi reordering."""

    segments: list[Segment]
    is_display: bool
    display_latex: str = ""
    tag: str | None = None
    base: float = 0.0
    size: float = 0.0
    x0: float = 0.0
    x1: float = 0.0
    rtl: bool = True
    plain: str = ""  # normalised text for header/footer detection
    marker: bool = False  # starts with a list number
    style: LineStyle | None = None  # formatting facts (stage 1); None when unknown
    words: list[StyledWord] = field(default_factory=list)  # reading order, with formatting


# ---------------------------------------------------------------------------
# 1. lines
# ---------------------------------------------------------------------------


def _emphasis_changes(above: LineStyle | None, below: LineStyle | None) -> bool:
    """Return True when two lines differ clearly in underline or bold (a heading starts)."""
    if above is None or below is None:
        return False
    return (
        abs(above.underline_share - below.underline_share) >= EMPHASIS_BREAK_SHARE
        or abs(above.name_bold_share - below.name_bold_share) >= EMPHASIS_BREAK_SHARE
    )


def build_lines(atoms: list[Atom]) -> list[Line]:
    glyphs = [a for a in atoms if a.kind == "g"]
    comps = [a for a in atoms if a.is_composite]
    spaces = [a for a in atoms if a.is_space]
    lines: list[Line] = []
    deferred: list[Atom] = []

    def main_match(a: Atom) -> Line | None:
        best, best_d = None, None
        for ln in lines:
            d = abs(a.base - ln.base)
            if d <= 0.22 * min(a.size, ln.size) and 0.72 <= a.size / ln.size <= 1.39:
                if best_d is None or d < best_d:
                    best, best_d = ln, d
        return best

    def secondary_match(a: Atom) -> Line | None:
        best, best_d = None, None
        for ln in lines:
            if a.size >= 0.92 * ln.size and not a.is_composite:
                continue
            if not (ln.y0 - 0.3 * ln.size <= a.cy <= ln.y1 + 0.3 * ln.size):
                continue
            if a.x0 > ln.x1 + 1.0 * ln.size or a.x1 < ln.x0 - 1.0 * ln.size:
                continue
            d = abs(a.cy - (ln.base + 0.3 * ln.size))
            if best_d is None or d < best_d:
                best, best_d = ln, d
        return best

    sizes = sorted(a.size for a in glyphs)
    body_size = sizes[len(sizes) // 2] if sizes else 0.0
    for a in sorted(glyphs, key=lambda a: (-round(a.size, 1), -round(a.base, 2), a.x0, a.order)):
        ln = main_match(a)
        if ln is not None:
            ln.add_main(a)
        elif secondary_match(a) is not None or a.size < 0.8 * body_size or a.text in _DELIMS:
            # small glyphs (scripts, limits) wait until composites have been placed:
            # a limit below a tall bracket belongs to the bracket's line
            deferred.append(a)
        else:
            nl = Line()
            nl.add_main(a)
            lines.append(nl)

    for c in sorted(comps, key=lambda a: (a.order, a.x0)):
        best, best_d = None, None
        for ln in lines:
            if c.x0 > ln.x1 + 3.0 * ln.size or c.x1 < ln.x0 - 3.0 * ln.size:
                continue
            if c.axis is not None:
                rel = (c.axis - ln.base) / ln.size
                if not (0.0 <= rel <= 0.55):
                    continue
                d = abs(rel - 0.25)
            elif c.kind == "op" and abs(c.base - ln.base) > 0.3 * ln.size:
                # a big operator with limits: its centre lies in the line's band
                rel = (c.cy - ln.base) / ln.size
                if not (-0.2 <= rel <= 0.75):
                    continue
                d = 0.5 + abs(rel - 0.3)
            else:
                if abs(c.base - ln.base) > 0.3 * ln.size:
                    continue
                d = abs(c.base - ln.base) / ln.size
            if best_d is None or d < best_d:
                best, best_d = ln, d
        if best is not None:
            c.base = best.base if c.axis is not None else c.base
            best.add_secondary(c)
        else:
            nl = Line()
            nl.add_main(c)
            lines.append(nl)

    def delimiter_match(a: Atom) -> Line | None:
        """Equation editors draw bigger brackets with a lowered baseline: attach by overlap."""
        best, best_ov = None, 0.0
        for ln in lines:
            if a.x0 > ln.x1 + 1.0 * ln.size or a.x1 < ln.x0 - 1.0 * ln.size:
                continue
            lo, hi = ln.base - 0.25 * ln.size, ln.base + 0.75 * ln.size
            ov = min(a.y1, hi) - max(a.y0, lo)
            if ov > 0.6 * (hi - lo) and ov > best_ov:
                best, best_ov = ln, ov
        return best

    for a in sorted(deferred, key=lambda a: (-round(a.size, 1), -round(a.base, 2), a.x0, a.order)):
        ln = main_match(a)
        if ln is not None and ln.size <= a.size * 1.1:
            ln.add_main(a)
            continue
        if a.text in _DELIMS and a.size >= 0.8 * body_size:
            ln = delimiter_match(a)
            if ln is not None:
                ln.add_secondary(a)
                continue
        ln = secondary_match(a)
        if ln is not None:
            ln.add_secondary(a)
        else:
            nl = Line()
            nl.add_main(a)
            lines.append(nl)

    for s in spaces:
        for ln in lines:
            if (
                abs(s.base - ln.base) <= 0.3 * ln.size
                and ln.x0 - 0.5 * ln.size <= s.x0 <= ln.x1 + 0.5 * ln.size
            ):
                ln.atoms.append(s)
                break

    lines = [ln for ln in lines if any(not a.is_space for a in ln.atoms)]
    lines.sort(key=lambda ln: (-round(ln.base, 1), ln.x0))
    return lines


# ---------------------------------------------------------------------------
# 2-4. tokens, math regions, bidi
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Token:
    atoms: list[Atom]
    kind: str = ""  # heb | prose | strong | weak | punct
    gap_before: bool = True

    @property
    def text(self) -> str:
        return "".join(a.text for a in self.atoms if not a.is_space)

    @property
    def x0(self) -> float:
        return min(a.x0 for a in self.atoms)

    @property
    def x1(self) -> float:
        return max(a.x1 for a in self.atoms)


def _tokens(line: Line) -> list[Token]:
    atoms = sorted(line.atoms, key=Atom.sort_key)
    size = line.size
    # Space glyphs alone are not trusted for word breaks (some producers place
    # them inaccurately); a text-font space that sits in a real (if narrow) gap
    # is, e.g. in tightly set bold headings.
    space_x = [(a.x0 + a.x1) / 2 for a in atoms if a.is_space and not a.math_font and a.fkey]
    toks: list[Token] = []
    cur: list[Atom] = []
    reach = None
    ink_reach = None
    for a in atoms:
        if a.is_space:
            continue
        start = min(a.lx0, a.x0) if a.lx1 > a.lx0 else a.x0
        gap = start - reach if reach is not None else 0.0
        # a space glyph between the *ink* of two glyphs (advance boxes may overlap it)
        spaced = (
            cur
            and ink_reach is not None
            and a.x0 - ink_reach > 0.15 * size
            and any(ink_reach <= x <= a.x0 for x in space_x)
        )
        if cur and reach is not None and (gap > 0.13 * size or spaced):
            toks.append(Token(cur))
            cur = []
        cur.append(a)
        end = max(a.lx1, a.x1)
        reach = end if reach is None else max(reach, end)
        ink_reach = a.x1 if ink_reach is None else max(ink_reach, a.x1)
    if cur:
        toks.append(Token(cur))
    # split at Hebrew / non-Hebrew boundaries (hyphen and geresh stick to Hebrew),
    # and split off punctuation typed in another font (";B" = Hebrew ';' + math B)
    out: list[Token] = []
    hebrew_fonts = {a.fkey for a in line.atoms if a.is_hebrew and a.fkey}
    for t in toks:
        parts = [q for p in _split_script(t) for q in _split_foreign_punct(p, hebrew_fonts)]
        for idx, p in enumerate(parts):
            p.gap_before = idx == 0
        out.extend(parts)
    for t in out:
        t.kind = _classify(t, line)
    _mark_text_brackets(out, hebrew_fonts)
    return out


def _split_foreign_punct(t: Token, hebrew_fonts: set[str] = frozenset()) -> list[Token]:
    """Split edge punctuation (and Hebrew-font brackets) typed in another font than its neighbour."""
    atoms = sorted(t.atoms, key=Atom.sort_key)
    if len(atoms) < 2 or not any(a.fkey for a in atoms):
        return [t]

    def foreign(a: Atom, nb: Atom) -> bool:
        if a.kind != "g" or a.fkey == nb.fkey:
            return False
        return a.text in _PUNCT or (a.text in "()[]" and a.fkey in hebrew_fonts)

    head, tail = 0, len(atoms)
    while head < tail - 1 and foreign(atoms[head], atoms[head + 1]):
        head += 1
    while tail - 1 > head and foreign(atoms[tail - 1], atoms[tail - 2]):
        tail -= 1
    if head == 0 and tail == len(atoms):
        return [t]
    parts = (
        [Token([a]) for a in atoms[:head]]
        + [Token(atoms[head:tail])]
        + [Token([a]) for a in atoms[tail:]]
    )
    return parts


def _mark_text_brackets(tokens: list[Token], hebrew_fonts: set[str]) -> None:
    """A bracket typed in the Hebrew text font between glyphs of other fonts is text.

    In "(A בחזקה זוגית)" next to a formula object the ")" belongs to the Hebrew
    parenthesis, not to the formula. (When text and formula share one font, as
    in "f(x)" typed as plain text, nothing changes.)
    """
    if not hebrew_fonts:
        return
    order = sorted(range(len(tokens)), key=lambda k: tokens[k].x0)
    for pos, k in enumerate(order):
        t = tokens[k]
        real = [a for a in t.atoms if not a.is_space]
        if (
            not real
            or not all(
                a.kind == "g" and (a.text in "()[]" or a.text in _PUNCT) and a.fkey in hebrew_fonts
                for a in real
            )
            or not any(a.text in "()[]" for a in real)
        ):
            continue
        neigh = [tokens[order[q]] for q in (pos - 1, pos + 1) if 0 <= q < len(order)]
        if neigh and all(
            not any(a.fkey == real[0].fkey for a in n.atoms if not a.is_space)
            for n in neigh
            if n.kind in ("strong", "weak")
        ):
            t.kind = "textpunct"


def _is_heb_side(atoms: list[Atom], i: int) -> bool:
    a = atoms[i]
    if a.is_hebrew:
        return True
    if a.kind == "g" and a.text in "()[]":
        # a bracket glued to a Hebrew letter belongs to the Hebrew text: "(מדומה טהור)"
        right = atoms[i + 1] if i + 1 < len(atoms) else None
        left = atoms[i - 1] if i > 0 else None
        if (right is not None and right.is_hebrew) or (left is not None and left.is_hebrew):
            return True
    if a.kind == "g" and (a.text in _HYPHENS or a.text in "'\"׳״"):
        right = atoms[i + 1] if i + 1 < len(atoms) else None
        left = atoms[i - 1] if i > 0 else None
        if right is not None and right.is_hebrew:
            return True
        if a.text in "'\"׳״" and left is not None and left.is_hebrew:
            return True
    return False


def _split_script(t: Token) -> list[Token]:
    atoms = sorted(t.atoms, key=Atom.sort_key)
    if not any(a.is_hebrew for a in atoms) or all(
        _is_heb_side(atoms, i) for i in range(len(atoms))
    ):
        return [Token(atoms)]
    parts: list[Token] = []
    cur = [atoms[0]]
    cur_h = _is_heb_side(atoms, 0)
    for i in range(1, len(atoms)):
        h = _is_heb_side(atoms, i)
        if h != cur_h:
            parts.append(Token(cur))
            cur, cur_h = [atoms[i]], h
        else:
            cur.append(atoms[i])
    parts.append(Token(cur))
    return parts


def _classify(t: Token, line: Line) -> str:
    real = [a for a in t.atoms if not a.is_space]
    if not real:
        return "punct"
    if any(a.is_hebrew for a in real):
        return "heb"
    text = t.text
    if re.fullmatch(r"[⋮⋱⋰]+", text):
        return "strong"  # vertical/diagonal dots between rows of formulas
    if re.fullmatch(r",?(?:\.{3}|…|⋯),?", text):
        return "weak"  # an ellipsis continues a formula ("A^3 = A·A·A, ...")
    if all(a.kind == "g" and a.text in _PUNCT for a in real):
        return "punct"
    if any(a.is_composite for a in real):
        return "strong"
    if any(a.kind == "g" and a.math_font and a.text not in _PUNCT and a.text != "�" for a in real):
        return "strong"
    if any(
        a.size < 0.86 * line.size and abs(a.base - line.base) > 0.1 * line.size for a in real
    ) and any(
        abs(a.base - line.base) <= 0.1 * line.size
        and a.kind == "g"
        and a.text not in _PUNCT
        and (a.text.isalnum() or a.text in ")]}|")
        for a in real
    ):
        return "strong"  # contains a super/subscript attached to a base
    m = _FUNC_CALL_RE.search(text)
    if (
        m
        and any(
            p in FUNCTION_NAMES or p in OPERATOR_NAMES for p in split_function_names(m.group(1))
        )
        and not any(a.is_hebrew for a in real)
    ):
        return "strong"  # "dim(kerT)", "sin(x)", "rank(A)"
    if any(is_seed_char(c) for c in text) or any(math_alphanumeric(c) for c in text):
        return (
            "strong"
            if is_mathish_token(_plain(text)) or any(math_alphanumeric(c) for c in text)
            else "prose"
        )
    italic = all(a.italic for a in real if a.text.isalpha())
    letters = [a for a in real if a.text.isalpha()]
    if (
        italic
        and letters
        and len(letters) <= 2
        and all(a.text.isascii() for a in letters)
        and all(a.text.isalpha() or a.text in "()[]|," for a in real)
        and any(a.is_hebrew for a in line.atoms)
    ):
        # in Hebrew text an italic Latin letter is a variable (an equation object)
        return "strong"
    if is_mathish_token(text, italic=italic):
        return "weak"
    return "prose"


_MARKER_RTL_RE = re.compile(r"^\.(?:\d{1,3}|[א-ת]{1,2})$")  # "1." / "א." drawn right-to-left
_MARKER_LTR_RE = re.compile(r"^(?:\d{1,3}|[a-z])[.)]$")


def _mark_list_marker(tokens: list[Token], line: Line, rtl: bool) -> None:
    """A list number at the start of a line ("1.", "א.") is text, not part of a formula.

    It is the rightmost token of a right-to-left line (leftmost of a
    left-to-right one) and is separated from the rest by a gap.
    """
    if len(tokens) < 2:
        return
    order = sorted(range(len(tokens)), key=lambda k: tokens[k].x0)
    if rtl:
        k, nb = order[-1], order[-2]
        gap = tokens[k].x0 - tokens[nb].x1
        ok = _MARKER_RTL_RE.match(tokens[k].text) and gap >= 0.25 * line.size
    else:
        k, nb = order[0], order[1]
        gap = tokens[nb].x0 - tokens[k].x1
        ok = _MARKER_LTR_RE.match(tokens[k].text) and gap >= 0.8 * line.size
    if ok:
        tokens[k].kind = "marker"


def _bridge(tokens: list[Token], line: Line) -> None:
    """A short Latin word sandwiched between math tokens is math ("i sin θ", "a mod n")."""
    tokens = [Token([], kind="none")] + tokens + [Token([], kind="none")]
    for k in range(1, len(tokens) - 1):
        t = tokens[k]
        if t.kind != "prose":
            continue
        left, right = tokens[k - 1], tokens[k + 1]
        text = t.text
        if not (text.isascii() and text.isalpha() and len(text) <= 8):
            continue
        math_left = left.kind in ("strong", "weak") and t.x0 - left.x1 <= 0.6 * line.size
        math_right = right.kind in ("strong", "weak") and right.x0 - t.x1 <= 0.6 * line.size
        if not (math_left and math_right):
            # touching a formula on one side only: just an operator name ("dim V", "Sp(Y)")
            if (math_left or math_right) and text in FUNCTION_NAMES | OPERATOR_NAMES:
                t.kind = "weak"
            elif (
                (math_left or math_right)
                and all(
                    p in FUNCTION_NAMES | OPERATOR_NAMES or len(p) == 1
                    for p in split_function_names(text)
                )
                and any(p in FUNCTION_NAMES | OPERATOR_NAMES for p in split_function_names(text))
                and sum(1 for p in split_function_names(text) if len(p) == 1) <= 1
            ):
                t.kind = "weak"  # "dimV": the name glued to its argument
            continue
        parts = split_function_names(text)
        named = [p for p in parts if p in FUNCTION_NAMES or p in OPERATOR_NAMES]
        singles = [p for p in parts if len(p) == 1 and p not in named]
        # "isin" (i·sin), "cos" - but not English words ("with", "and")
        if named and all(len(p) == 1 or p in named for p in parts) and len(singles) <= 2:
            t.kind = "weak"


def _plain(text: str) -> str:
    return "".join(c for c in text if not math_alphanumeric(c))


def _regions(tokens: list[Token], line: Line) -> list[tuple[int, int]]:
    """Index ranges [i, j) of tokens forming math regions."""
    regions = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i].kind not in ("strong", "weak"):
            i += 1
            continue
        j = i
        while j < n and tokens[j].kind in ("strong", "weak", "punct"):
            # a very wide gap (equation number at the margin, table column) ends the region
            if j > i and tokens[j].x0 - tokens[j - 1].x1 > 2.5 * line.size:
                break
            # a punctuation token only continues the region if math follows it
            if tokens[j].kind == "punct" and not (
                j + 1 < n and tokens[j + 1].kind in ("strong", "weak")
            ):
                break
            j += 1
        while j > i and tokens[j - 1].kind == "punct":
            j -= 1
        i, j = _trim_prose_dashes(tokens, i, j, line)
        strong = any(tokens[k].kind == "strong" for k in range(i, j)) or _is_interval(tokens[i:j])
        if strong and _has_operand(tokens[i:j]) and _has_main_atom(tokens[i:j], line):
            regions.append((i, j))
        i = max(j, i + 1)
    return regions


_DASHES = ("-", "–", "—")


def _trim_prose_dashes(tokens: list[Token], i: int, j: int, line: Line) -> tuple[int, int]:
    """Drop a text dash at the edge of a math region ("... B - הסבר ...").

    A dash in a text font, set off by a space from the formula and with
    Hebrew (or the line end) on its other side, is sentence punctuation; a
    minus sign sits against its operand. Without this the dash becomes a
    leading minus of the formula ("$-B$").
    """

    def is_text_dash(t: Token) -> bool:
        return t.text in _DASHES and all(not a.math_font for a in t.atoms if not a.is_space)

    def heb_or_edge(k: int) -> bool:
        return k < 0 or k >= len(tokens) or tokens[k].kind == "heb"

    gap = 0.2 * line.size
    if (
        j - i >= 2
        and is_text_dash(tokens[i])
        and heb_or_edge(i - 1)
        and tokens[i + 1].x0 - tokens[i].x1 >= gap
    ):
        i += 1
    if (
        j - i >= 2
        and is_text_dash(tokens[j - 1])
        and heb_or_edge(j)
        and tokens[j - 1].x0 - tokens[j - 2].x1 >= gap
    ):
        j -= 1
    return i, j


_INTERVAL_RE = re.compile(
    r"^[\[\]()]-?(?:\d+(?:\.\d+)?|∞|[a-z]),-?(?:\d+(?:\.\d+)?|∞|[a-z])[\[\]()]$"
)


def _is_interval(tokens: list[Token]) -> bool:
    """[0,1], (a,b), (-∞,3] ... written with plain characters."""
    text = "".join(t.text for t in sorted(tokens, key=lambda t: t.x0))
    return bool(_INTERVAL_RE.match(text)) and any(c in text for c in "[]")


def _has_operand(tokens: list[Token]) -> bool:
    for t in tokens:
        for a in t.atoms:
            if a.is_composite:
                return True
            if a.kind == "g" and (
                a.text.isalnum() or math_alphanumeric(a.text) or a.text in "∞∅ℝℕℤℚℂ"
            ):
                return True
    return False


def _has_main_atom(tokens: list[Token], line: Line) -> bool:
    return any(
        abs(a.base - line.base) <= 0.15 * line.size or a.is_composite
        for t in tokens
        for a in t.atoms
        if not a.is_space
    )


def _strip_edge_punct(tokens: list[Token]) -> tuple[list[Atom], list[Atom], list[Atom]]:
    """Split sentence punctuation glued to the visual edges of a math region.

    An ellipsis ("A·A·A, ...") is part of the formula and stays.
    """
    atoms = sorted([a for t in tokens for a in t.atoms], key=Atom.sort_key)
    left: list[Atom] = []
    right: list[Atom] = []

    def ellipsis_at(seq: list[Atom]) -> bool:
        return len(seq) >= 3 and all(a.kind == "g" and a.text == "." for a in seq[:3])

    while atoms and atoms[0].kind == "g" and atoms[0].text in _PUNCT and not ellipsis_at(atoms):
        left.append(atoms.pop(0))
    while (
        atoms and atoms[-1].kind == "g" and atoms[-1].text in _PUNCT and not ellipsis_at(atoms[-3:])
    ):
        right.insert(0, atoms.pop())
    return left, atoms, right


def _base_level_hint(
    line: Line, units: list[Unit], default_rtl: bool, column: tuple[float, float] | None
) -> int | None:
    """Paragraph direction for lines without Hebrew letters.

    In a right-to-left document a Latin-only line that is right-aligned
    (and not full width) belongs to an RTL paragraph: its trailing period is
    drawn on the left. Lines with Hebrew use the bidi majority rule.
    """
    if column is None or not default_rtl:
        return None
    if any(u.cls in ("R", "AL") for u in units):
        return None
    visible = [u for u in units if u.cls not in ("WS",)]
    if (
        len(visible) >= 3
        and isinstance(visible[0].payload, str)
        and visible[0].payload in ".!?:;,"
        and visible[1].cls in ("L", "EN")
        and visible[-1].cls in ("L", "EN")
    ):
        return 1  # sentence punctuation drawn at the left end: RTL paragraph
    c0, c1 = column
    width = max(c1 - c0, 1.0)
    right_aligned = c1 - line.x1 <= 0.03 * width and line.x0 - c0 > 0.15 * width
    if right_aligned:
        return 1
    return None


def _object_splits(tokens: list[Token], line: Line, i: int, j: int) -> list[tuple[int, int]]:
    """Split region [i, j) where a *text-font* space glyph separates two tokens.

    Equation editors (MathType, Word's legacy editor, LibreOffice Math) emit
    each formula as its own object; the author separates objects with ordinary
    spaces. Inside one formula, spacing is done by positioning, never with a
    text-font space glyph. Each object is then a separate unit for the bidi
    algorithm (an embedded object is a neutral, so a sequence of objects in a
    right-to-left paragraph reads right to left, as the word processor showed it).
    """
    spaces = [
        ((a.x0 + a.x1) / 2, a.fkey) for a in line.atoms if a.is_space and not a.math_font and a.fkey
    ]
    rtl = any(a.is_hebrew for a in line.atoms)

    def nearest_symbol(k: int, step: int) -> Atom | None:
        """Closest non-punctuation glyph of the region from token k in direction step."""
        while i <= k < j:
            atoms = sorted((a for a in tokens[k].atoms if not a.is_space), key=Atom.sort_key)
            if step < 0:
                atoms.reverse()
            for a in atoms:
                if not (a.kind == "g" and a.text in _PUNCT):
                    return a
            k += step
        return None

    def new_statement(k: int) -> bool:
        """'... (matrix) A = ...': a variable + relation after a gap starts a new object (RTL)."""
        if not rtl or k + 1 >= j or tokens[k].x0 - tokens[k - 1].x1 < 0.4 * line.size:
            return False
        return (
            tokens[k].kind == "weak"
            and tokens[k + 1].text[:1] in "=≠≤≥<>≈≡"
            and tokens[k - 1].text[-1:] not in _OPERATOR_TOKENS
            and tokens[k - 1].kind != "punct"
        )

    def cut_here(k: int) -> bool:
        if new_statement(k):
            return True
        left = nearest_symbol(k - 1, -1)
        right = nearest_symbol(k, +1)
        if left is None or right is None:
            return False
        for c, fkey in spaces:
            # the space must come from another font than the formula's own glyphs:
            # "a² + b² = c²" typed as plain text uses one font for letters and spaces
            if (
                tokens[k - 1].x1 - 0.05 * line.size <= c <= tokens[k].x0 + 0.05 * line.size
                and fkey != left.fkey
                and fkey != right.fkey
            ):
                return True
        # in right-to-left text, objects are also set apart by wide gaps
        return rtl and tokens[k].x0 - tokens[k - 1].x1 >= 1.5 * line.size

    cuts = [k for k in range(i + 1, j) if cut_here(k)]
    # a lone weak token (a variable) next to a relation is one formula: "B = 0"
    cuts = [
        k
        for k in cuts
        if not (
            (
                tokens[k].text[:1] in _OPERATOR_TOKENS
                and (k - 1 == i or k - 1 in cuts or tokens[k - 2].kind == "punct")
                and tokens[k - 1].kind == "weak"
            )
            or (
                tokens[k - 1].text[-1:] in _OPERATOR_TOKENS
                and tokens[k].kind == "weak"
                and (k + 1 == j or k + 1 in cuts)
            )
        )
    ]
    out, start = [], i
    for k in cuts:
        out.append((start, k))
        start = k
    out.append((start, j))
    return out


def analyse_line(
    line: Line,
    engine: PdfMathEngine,
    default_rtl: bool,
    column: tuple[float, float] | None = None,
    rules: list[Rule] | None = None,
    page: int = 0,
) -> LineResult:
    tokens = _tokens(line)
    line_rtl = any(a.is_hebrew for a in line.atoms)
    rtl_hint = default_rtl if not line_rtl else True
    _mark_list_marker(tokens, line, rtl_hint)
    _bridge(tokens, line)
    regions: list[tuple[int, int]] = []
    split_flags: list[bool] = []
    for i, j in _regions(tokens, line):
        parts = _object_splits(tokens, line, i, j)
        if len(parts) > 1:
            # a split-off piece made only of weak tokens ("A" before "(A ...") is text
            parts = [
                (a, b)
                for a, b in parts
                if any(tokens[k].kind == "strong" for k in range(a, b))
                or _is_interval(tokens[a:b])
                or _OPERATOR_TOKENS.issuperset(tokens[k].text for k in range(a, b))
            ]
        for part in parts:
            regions.append(part)
            split_flags.append(len(parts) > 1)
    region_of = {}
    for r_i, (i, j) in enumerate(regions):
        for k in range(i, j):
            region_of[k] = r_i

    units: list[Unit] = []
    glyph_of: dict[int, Atom] = {}  # id(unit) -> the glyph it came from (formatting facts)

    def push(unit: Unit, atom: Atom | None = None) -> None:
        units.append(unit)
        if atom is not None:
            glyph_of[id(unit)] = atom

    prev_x1: float | None = None
    k = 0
    math_latex: list[str] = []
    region_flags: list[bool] = []
    marker_text = ""
    while k < len(tokens):
        if tokens[k].kind == "marker" and k not in region_of:
            # list numbers are placed at the logical start of the line (bidi rule W7
            # would otherwise glue "9." to an adjacent left-to-right formula)
            vis = tokens[k].text
            marker_text = vis[1:] + vis[0] if rtl_hint and vis.startswith(".") else vis
            prev_x1 = tokens[k].x1
            k += 1
            continue
        if k in region_of:
            r_i = region_of[k]
            i, j = regions[r_i]
            left, body, right = _strip_edge_punct(tokens[i:j])
            if prev_x1 is not None and tokens[i].gap_before:
                units.append(Unit(" ", "WS"))
            for a in left:
                push(Unit(a.text, bidi_class(a.text)), a)
            latex = engine.to_latex(body) if body else ""
            if latex:
                # A formula is one opaque left-to-right unit. Towards its neighbours it
                # behaves like an embedded object, i.e. a bidi *neutral* (word
                # processors and browsers treat equation objects / <math> boxes so):
                # in a right-to-left line it follows the surrounding direction.
                units.append(
                    Unit(InlineMath(latex), "ON" if (split_flags[r_i] or line_rtl) else "L")
                )
                math_latex.append(latex)
            for a in right:
                push(Unit(a.text, bidi_class(a.text)), a)
            region_flags.append(True)
            prev_x1 = max(a.x1 for t in tokens[i:j] for a in t.atoms)
            k = j
            continue
        t = tokens[k]
        if prev_x1 is not None and t.gap_before:
            units.append(Unit(" ", "WS"))
        for a in sorted(t.atoms, key=Atom.sort_key):
            if a.is_space:
                continue
            txt = a.text + a.marks if a.kind == "g" else (a.latex or "")
            if a.is_composite:
                units.append(Unit(InlineMath(a.latex or ""), "L"))
                continue
            if txt == "�":
                continue
            # a lone raised/lowered glyph in prose (footnote marker) is kept as text
            push(Unit(txt, bidi_class(txt[0]) if txt else "ON"), a if a.kind == "g" else None)
        region_flags.append(False)
        prev_x1 = t.x1
        k += 1

    ordered, levels, base_level = reorder(
        units,
        default_rtl=default_rtl,
        base_level=_base_level_hint(line, units, default_rtl, column),
    )
    segments: list[Segment] = []
    buf: list[str] = []
    underlined = underlined_glyph_ids(line.atoms, rules or [])
    words: list[StyledWord] = []
    word_chars: list[str] = []
    word_atoms: list[Atom | None] = []

    def end_word() -> None:
        if word_chars:
            word = make_word(word_chars, word_atoms, underlined)
            if word is not None:
                words.append(word)
        word_chars.clear()
        word_atoms.clear()

    for u, lv in zip(ordered, levels):
        if isinstance(u.payload, InlineMath):
            end_word()
            if buf:
                segments.append(TextSpan("".join(buf)))
                buf = []
            segments.append(u.payload)
        else:
            ch = u.payload
            if lv % 2 and len(ch) == 1 and ch in MIRROR:
                ch = MIRROR[ch]
            buf.append(ch)
            if ch.isspace():
                end_word()
            else:
                word_chars.append(ch)
                word_atoms.extend([glyph_of.get(id(u))] * len(ch))
    end_word()
    if buf:
        segments.append(TextSpan("".join(buf)))
    segments = _join_operator_objects(segments)
    segments = _tidy_punctuation(segments)
    if marker_text:
        segments = [TextSpan(marker_text + " ")] + segments

    result = LineResult(
        segments=segments,
        is_display=False,
        base=line.base,
        size=line.size,
        x0=line.x0,
        x1=line.x1,
        rtl=base_level % 2 == 1,
    )
    result.plain = "".join(s.text if isinstance(s, TextSpan) else "§" for s in segments).strip()
    result.marker = bool(marker_text)
    result.style = line_style(line.atoms, rules or [], page)
    result.words = words
    _detect_display(result, tokens, regions, engine)
    return result


_OPERATOR_ONLY_RE = re.compile(
    r"^(?:\\(?:Leftrightarrow|Rightarrow|Leftarrow|leftrightarrow|rightarrow|leftarrow|iff|implies|to|"
    r"longrightarrow|Longrightarrow|Longleftrightarrow|mapsto|leq|geq|neq|approx|equiv|sim|cong|in|notin|"
    r"subset|subseteq|supset|supseteq|cdot|times|pm|mp|cup|cap|oplus|setminus)|[=<>+\-−,;:])$"
)


_PUNCT_BEFORE_HEB_RE = re.compile(r"^\s+([.,;:])\s*(?=[\u05D0-\u05EA])")
_PUNCT_AT_END_RE = re.compile(r"^\s+([.,;:!?])\s*$")


def _tidy_punctuation(segs: list[Segment]) -> list[Segment]:
    """Sentence punctuation right after a formula belongs to it: "$x$ .שלום" -> "$x$. שלום".

    Also: a closing bracket hugs the formula ("$x$ )" -> "$x$)") and a formula
    is separated from an adjacent Hebrew word by a space ("$x$אינו" -> "$x$ אינו").
    """
    out = list(segs)
    for k in range(len(out)):
        if not isinstance(out[k], TextSpan):
            continue
        t = out[k].text
        if k > 0 and isinstance(out[k - 1], InlineMath):
            t = _PUNCT_BEFORE_HEB_RE.sub(lambda m: m.group(1) + " ", t)
            t = _PUNCT_AT_END_RE.sub(lambda m: m.group(1), t)
            t = re.sub(r"^\s+(?=[)\]])", "", t)
            if t[:1] and "\u05d0" <= t[0] <= "\u05ea":
                t = " " + t
        if (
            k + 1 < len(out)
            and isinstance(out[k + 1], InlineMath)
            and t[-1:]
            and "\u05d0" <= t[-1] <= "\u05ea"
        ):
            t = t + " "
        out[k] = TextSpan(t)
    return out


def _join_operator_objects(segs: list[Segment]) -> list[Segment]:
    r"""Glue a formula that is only an operator ("⟺") to the formula after it.

    After bidi reordering, "A ⟺ B ⟺ C" typed as separate objects is
    logically [A] [⟺] [B] [⟺] [C]; a lone operator is not a formula of its
    own, so it is merged with its right neighbour ("$\Leftrightarrow B$").
    """
    out: list[Segment] = []
    k = 0
    while k < len(segs):
        s = segs[k]
        if isinstance(s, InlineMath) and _OPERATOR_ONLY_RE.match(s.latex.strip()):
            nxt = k + 1
            if nxt < len(segs) and isinstance(segs[nxt], TextSpan) and not segs[nxt].text.strip():
                nxt += 1
            if nxt < len(segs) and isinstance(segs[nxt], InlineMath):
                segs = (
                    segs[:nxt]
                    + [InlineMath(s.latex.strip() + " " + segs[nxt].latex)]
                    + segs[nxt + 1 :]
                )
                k = nxt
                continue
            # trailing operator: attach to the formula before it
            prv = len(out) - 1
            if prv >= 0 and isinstance(out[prv], TextSpan) and not out[prv].text.strip():
                prv -= 1
            if prv >= 0 and isinstance(out[prv], InlineMath):
                out[prv] = InlineMath(out[prv].latex + " " + s.latex.strip())
                out = out[: prv + 1]
                k += 1
                continue
        out.append(s)
        k += 1
    return out


def _detect_display(result: LineResult, tokens: list[Token], regions, engine) -> None:
    if not regions:
        return
    covered = set()
    for i, j in regions:
        covered.update(range(i, j))
    rest = [t for k, t in enumerate(tokens) if k not in covered]
    tag = None
    rest_text = [t for t in rest if t.kind != "punct"]
    if any(t.kind == "marker" for t in rest_text):
        return  # a numbered list item, not a stand-alone formula
    if len(rest_text) == 1 and _EQNUM_RE.match(rest_text[0].text.replace(" ", "")):
        tag = rest_text[0].text.strip("() ")
        rest_text = []
    elif (
        len(rest_text) <= 3
        and rest_text
        and _EQNUM_RE.match("".join(t.text for t in sorted(rest_text, key=lambda t: t.x0)))
    ):
        tag = "".join(t.text for t in sorted(rest_text, key=lambda t: t.x0)).strip("() ")
        rest_text = []
    if rest_text:
        return
    maths = [s for s in result.segments if isinstance(s, InlineMath)]
    if not maths:
        return
    # Equation number can also be a separate region-less token handled above;
    # visually ordered regions -> left to right.
    latex = r" \quad ".join(m.latex for m in maths) if len(maths) > 1 else maths[0].latex
    if tag:
        m = re.fullmatch(r"(.*?)(?:\\quad\s*)?\((\d+(?:[.\-]\d+)*[a-z]?)\)", latex)
        if m:
            latex = m.group(1)
    result.is_display = True
    result.display_latex = latex.strip()
    result.tag = tag


# ---------------------------------------------------------------------------
# 5. paragraphs
# ---------------------------------------------------------------------------


def lines_to_blocks(lines: list[LineResult], page_x0: float, page_x1: float) -> list[Block]:
    blocks: list[Block] = []
    if not lines:
        return blocks
    pitches = [
        lines[k].base - lines[k + 1].base
        for k in range(len(lines) - 1)
        if 0 < lines[k].base - lines[k + 1].base < 3 * lines[k].size
    ]
    pitch = median(pitches, default=1.3 * median([ln.size for ln in lines], 12.0))
    text_lines = [ln for ln in lines if not ln.is_display]
    col_x0 = min((ln.x0 for ln in text_lines), default=page_x0)
    col_x1 = max((ln.x1 for ln in text_lines), default=page_x1)
    width = max(col_x1 - col_x0, 1.0)

    current: list[Segment] = []
    current_styles: list[LineStyle] = []
    current_words: list[StyledWord] = []
    prev: LineResult | None = None

    def flush():
        nonlocal current, current_styles, current_words
        segs = [s for s in current]
        styles = current_styles
        words = current_words
        current = []
        current_styles = []
        current_words = []
        if any(not isinstance(s, TextSpan) or s.text.strip() for s in segs):
            block = Block.paragraph(repair_brackets(_merge(segs)))
            block.line_styles = styles
            block.words = words
            blocks.append(block)

    for ln in lines:
        if ln.is_display:
            flush()
            blocks.append(Block.display_math(ln.display_latex, ln.tag))
            prev = None
            continue
        if prev is not None:
            gap = prev.base - ln.base
            same_size = abs(prev.size - ln.size) <= 0.15 * max(prev.size, ln.size)
            # the previous line spans the column (it wrapped) -> same paragraph
            if prev.rtl:
                full = prev.x0 - col_x0 <= 0.15 * width and col_x1 - prev.x1 <= 0.05 * width
            else:
                full = col_x1 - prev.x1 <= 0.15 * width and prev.x0 - col_x0 <= 0.05 * width
            if (
                gap <= 1.45 * pitch
                and same_size
                and full
                and not ln.marker
                and not _emphasis_changes(prev.style, ln.style)
            ):
                _join(current, ln.segments)
                if ln.style:
                    current_styles.append(ln.style)
                current_words.extend(ln.words)
                prev = ln
                continue
            flush()
        current = list(ln.segments)
        current_styles = [ln.style] if ln.style else []
        current_words = list(ln.words)
        prev = ln
    flush()
    return blocks


def _join(current: list[Segment], nxt: list[Segment]) -> None:
    if current and isinstance(current[-1], TextSpan):
        t = current[-1].text.rstrip()
        if (
            len(t) >= 2
            and t[-1] == "-"
            and t[-2].isascii()
            and t[-2].isalpha()
            and nxt
            and isinstance(nxt[0], TextSpan)
            and nxt[0].text[:1].islower()
        ):
            current[-1] = TextSpan(t[:-1])
            current.extend(nxt)
            return
    current.append(TextSpan(" "))
    current.extend(nxt)


def _merge(segs: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        if isinstance(s, TextSpan) and out and isinstance(out[-1], TextSpan):
            out[-1] = TextSpan(out[-1].text + s.text)
        else:
            out.append(s)
    return out


_PAIRS = {"(": ")", "[": "]", "{": "}"}
_REV = {v: k for k, v in _PAIRS.items()}


def repair_brackets(segs: list[Segment]) -> list[Segment]:
    """Fix prose brackets whose direction was lost in the PDF text layer.

    RTL producers sometimes store both parentheses of "(טקסט)" as ")" after
    mirroring. If a bracket type is unbalanced with an even total, flip the
    surplus ones that cannot be matched (rightmost unmatched openers / leftmost
    unmatched closers) and keep the result only if it is fully balanced.
    """
    spans = [i for i, s in enumerate(segs) if isinstance(s, TextSpan)]
    chars: list[tuple[int, int, str]] = []
    for i in spans:
        for k, c in enumerate(segs[i].text):
            if c in _PAIRS or c in _REV:
                chars.append((i, k, c))
    if not chars:
        return segs
    flips: dict[tuple[int, int], str] = {}
    for op, cl in _PAIRS.items():
        seq = [(i, k, c) for i, k, c in chars if c in (op, cl)]
        opens = sum(1 for _, _, c in seq if c == op)
        closes = len(seq) - opens
        if opens == closes or (opens + closes) % 2:
            continue
        cand = [c for _, _, c in seq]
        need = abs(opens - closes) // 2
        if opens > closes:
            idx = [n for n in range(len(cand)) if cand[n] == op][::-1][:need]
            for n in idx:
                cand[n] = cl
        else:
            idx = [n for n in range(len(cand)) if cand[n] == cl][:need]
            for n in idx:
                cand[n] = op
        depth = 0
        ok = True
        for c in cand:
            depth += 1 if c == op else -1
            if depth < 0:
                ok = False
                break
        if ok and depth == 0:
            for n in idx:
                i, k, _ = seq[n]
                flips[(i, k)] = cand[n]
    if not flips:
        return segs
    out = list(segs)
    for i in spans:
        t = list(out[i].text)
        for k in range(len(t)):
            if (i, k) in flips:
                t[k] = flips[(i, k)]
        out[i] = TextSpan("".join(t))
    return out


def page_text_bounds(lines: list[LineResult]) -> tuple[float, float]:
    if not lines:
        return 0.0, 0.0
    return min(ln.x0 for ln in lines), max(ln.x1 for ln in lines)


__all__ = [
    "build_lines",
    "analyse_line",
    "lines_to_blocks",
    "LineResult",
    "union",
    "is_hebrew_char",
]
