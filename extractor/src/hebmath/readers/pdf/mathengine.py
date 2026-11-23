"""Deterministic geometric math reconstruction for PDF text layers.

The engine works on glyph positions only (no ML), in two phases:

1. ``build_structures`` - turns vertical structures into composite atoms,
   bottom-up (innermost first): fractions, radicals, over-lines (all anchored
   on thin vector rules), big operators with limits above/below, and accents
   (\\hat, \\bar, \\vec ...).
2. ``to_latex`` - a baseline-structure parser for a horizontal run of atoms:
   it finds the dominant baseline and attaches raised/lowered smaller atoms as
   superscripts/subscripts, recursively.

The engine is a plug-in: anything implementing ``PdfMathEngine`` can replace
it (for example an adapter for an image-based formula recogniser).
"""

from __future__ import annotations

import re
from typing import Protocol

from ...bidi import MIRROR, visual_to_logical
from ...latex.symbols import (
    ACCENTS,
    BIG_OPERATORS,
    FUNCTION_NAMES,
    LIMIT_FUNCTIONS,
    OPERATOR_NAMES,
    WIDE_ACCENTS,
    char_to_latex,
    join_latex,
    math_alphanumeric,
    split_function_names,
)
from .fonts import style_for_font
from .geometry import Atom, Rule, median, union, x_overlap


class PdfMathEngine(Protocol):
    def build_structures(self, atoms: list[Atom], rules: list[Rule]) -> list[Atom]: ...

    def to_latex(self, atoms: list[Atom]) -> str: ...


_ACCENT_CHARS = {
    "ˆ",
    "˜",
    "¯",
    "˙",
    "¨",
    "ˇ",
    "´",
    "`",
    "˘",
    "→",
    "⃗",
    "^",
    "~",
    "‾",
    "\u0302",
    "\u0303",
    "\u0304",
    "\u0307",
    "\u0308",
    "\u20d7",
    "ˉ",
    "˚",
}
_ACCENT_TO_COMBINING = {
    "ˆ": "\u0302",
    "^": "\u0302",
    "˜": "\u0303",
    "~": "\u0303",
    "¯": "\u0304",
    "ˉ": "\u0304",
    "‾": "\u0304",
    "˙": "\u0307",
    "¨": "\u0308",
    "ˇ": "\u030c",
    "´": "\u0301",
    "`": "\u0300",
    "˘": "\u0306",
    "→": "\u20d7",
    "⃗": "\u20d7",
    "˚": "\u030a",
}
_LIMIT_WORDS = sorted(LIMIT_FUNCTIONS, key=len, reverse=True)
_RADICALS = {"√", "∛", "∜"}
_OPEN = {"(": ")", "[": "]", "{": "}", "⟨": "⟩", "⌊": "⌋", "⌈": "⌉"}
_CLOSE = {v: k for k, v in _OPEN.items()}


def _is_letter(a: Atom) -> bool:
    return a.kind == "g" and len(a.text) == 1 and a.text.isascii() and a.text.isalpha()


class GeometricMathEngine:
    """Default PDF math engine. Deterministic.

    ``mirror_mode`` handles producers that write *mirrored* code points for
    math in right-to-left documents ("f)x(" and "0<ε" for "f(x)" and "ε>0").
    The reader enables it for a document when the bracket evidence collected
    in ``votes`` shows that this producer mirrors.
    """

    def __init__(self) -> None:
        self.mirror_mode = False
        self.votes = {"as_is": 0, "mirrored": 0}

    # ------------------------------------------------------------------ phase 1
    def build_structures(self, atoms: list[Atom], rules: list[Rule]) -> list[Atom]:
        pool = [a for a in atoms]
        consumed: set[int] = set()
        new: list[Atom] = []

        live = lambda: [a for a in pool if id(a) not in consumed]  # noqa: E731

        self._accents(live(), consumed, new)
        pool = live() + new
        new = []
        consumed = set()

        # Rules innermost (shortest) first, so nested structures are collapsed bottom-up.
        # A fraction whose numerator/denominator holds a tall bracket (a matrix
        # over a determinant) waits until the matrices exist.
        postponed: list[Rule] = []
        for rule in rules:
            cand = [a for a in pool if id(a) not in consumed and not a.is_space]
            if self._spans_tall(rule, cand):
                postponed.append(rule)
                continue
            comp = (
                self._radical(rule, cand)
                or self._fraction(rule, cand)
                or self._overline(rule, cand)
            )
            if comp is not None:
                for c in comp.children:
                    consumed.add(id(c))
                pool.append(comp)
        pool = [a for a in pool if id(a) not in consumed]

        consumed = set()
        new = []
        self._radicals_without_rule(pool, consumed, new)
        pool = [a for a in pool if id(a) not in consumed] + new

        consumed = set()
        new = []
        self._operators_with_limits(pool, consumed, new)
        pool = [a for a in pool if id(a) not in consumed] + new

        consumed = set()
        new = []
        self._tall_structures(pool, consumed, new)
        # inner groups absorbed by an outer one are in ``new`` *and* consumed
        pool = [a for a in pool + new if id(a) not in consumed]

        if postponed:
            consumed = set()
            for rule in postponed:
                cand = [a for a in pool if id(a) not in consumed and not a.is_space]
                comp = self._fraction(rule, cand) or self._overline(rule, cand)
                if comp is not None:
                    for c in comp.children:
                        consumed.add(id(c))
                    pool.append(comp)
            pool = [a for a in pool if id(a) not in consumed]
        pool.sort(key=lambda a: (a.order, a.x0))
        return pool

    def _spans_tall(self, rule: Rule, atoms: list[Atom]) -> bool:
        """A tall bracket directly above or below the rule, within its span."""
        for a in atoms:
            if (
                a.kind == "g"
                and a.tall
                and a.x0 >= rule.x0 - 0.5
                and a.x1 <= rule.x1 + 0.5
                and (0 <= a.y0 - rule.y <= 0.6 * a.size or 0 <= rule.y - a.y1 <= 0.6 * a.size)
            ):
                return True
        return False

    # accents --------------------------------------------------------------
    def _accents(self, atoms: list[Atom], consumed: set[int], out: list[Atom]) -> None:
        glyphs = [a for a in atoms if a.kind == "g"]
        for acc in glyphs:
            if acc.text not in _ACCENT_CHARS or id(acc) in consumed:
                continue
            best, best_ov = None, 0.0
            for b in glyphs:
                if b is acc or id(b) in consumed or b.text in _ACCENT_CHARS or b.is_hebrew:
                    continue
                if not (
                    b.text.isalnum()
                    or math_alphanumeric(b.text)
                    or b.text in "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ"
                ):
                    continue
                ov = x_overlap(acc.x0, acc.x1, b.x0, b.x1)
                if ov < 0.5 * min(acc.w, b.w) or ov <= 0:
                    continue
                if not (acc.y0 >= b.y1 - 0.3 * b.size and acc.y0 - b.y1 <= 0.6 * b.size):
                    continue
                if abs(acc.base - b.base) > 1.2 * b.size:
                    continue
                if ov > best_ov:
                    best, best_ov = b, ov
            if best is None:
                continue
            cmd = ACCENTS.get(_ACCENT_TO_COMBINING.get(acc.text, acc.text), r"\hat")
            inner = self.to_latex([best])
            if acc.w > 1.6 * best.w and cmd in WIDE_ACCENTS:
                cmd = WIDE_ACCENTS[cmd]
            comp = self._composite(
                "acc", [acc, best], f"{cmd}{{{inner}}}", base=best.base, size=best.size
            )
            comp.italic, comp.math_font = best.italic, True
            consumed.add(id(acc))
            consumed.add(id(best))
            out.append(comp)

    # fractions --------------------------------------------------------------
    def _stack(self, rule: Rule, atoms: list[Atom], above: bool) -> list[Atom]:
        """Atoms vertically chained directly above (or below) a rule, within its span."""
        span = [
            a for a in atoms if a.x0 >= rule.x0 - 0.12 * a.size and a.x1 <= rule.x1 + 0.12 * a.size
        ]
        if above:
            span = [a for a in span if a.y0 >= rule.y - 0.25 * a.size and a.cy > rule.y]
            span.sort(key=lambda a: (a.y0, a.x0, a.order))
            gap = lambda a, edge: a.y0 - edge  # noqa: E731
            edge = rule.y + rule.thickness / 2
        else:
            span = [a for a in span if a.y1 <= rule.y + 0.25 * a.size and a.cy < rule.y]
            span.sort(key=lambda a: (-a.y1, a.x0, a.order))
            gap = lambda a, edge: edge - a.y1  # noqa: E731
            edge = rule.y - rule.thickness / 2
        out: list[Atom] = []
        for a in span:
            if not out:
                if gap(a, edge) > 0.9 * a.size:
                    break
                out.append(a)
                edge = a.y1 if above else a.y0
                continue
            # measured against the smaller glyph: an enlarged bracket of the next
            # line must not chain onto a subscript
            if gap(a, edge) <= 0.35 * min(a.size, out[-1].size):
                out.append(a)
                edge = max(edge, a.y1) if above else min(edge, a.y0)
        return out

    def _closed(
        self, group_: list[Atom], rule: Rule, atoms: list[Atom], reach: float = 0.45
    ) -> bool:
        """No foreign atom on the same baseline right next to the group (underline/table test)."""
        if not group_:
            return False
        size = max(a.size for a in group_)
        main = [a for a in group_ if a.size >= 0.85 * size] or group_
        base = median([a.base for a in main])
        ids = {id(a) for a in group_}
        x0, _, x1, _ = union(group_)
        for a in atoms:
            if id(a) in ids or a.is_space or (a.kind == "g" and a.text in ".,;:!?"):
                continue
            if not a.is_composite and a.size < 0.8 * size:
                continue  # a neighbour's superscript/subscript, e.g. the n of (-1)^n
            if a.y0 < rule.y < a.y1:
                continue  # straddles the rule: an operator on the math axis (e.g. "1+\frac{1}{x}")
            if abs(a.base - base) > 0.3 * size:
                continue
            if rule.x0 - reach * size <= a.x1 <= rule.x0 + 0.05 * size and a.x1 <= x0:
                return False
            if rule.x1 - 0.05 * size <= a.x0 <= rule.x1 + reach * size and a.x0 >= x1:
                return False
        return True

    def _fraction(self, rule: Rule, atoms: list[Atom]) -> Atom | None:
        num = self._stack(rule, atoms, above=True)
        den = self._stack(rule, atoms, above=False)
        if not num or not den:
            return None
        if not (_has_operand(num) and _has_operand(den)):
            return None  # a lone bracket of the neighbouring line over an overline
        nx0, _, nx1, _ = union(num)
        dx0, _, dx1, _ = union(den)
        content_w = max(nx1 - nx0, dx1 - dx0)
        s = max(median([a.size for a in num]), median([a.size for a in den]))
        if rule.w < 0.7 * content_w - 0.5 or rule.w > content_w + 2.0 * s:
            return None
        # numerator and denominator are centred on the bar (an overline with a
        # bracket of the previous line above it is not a fraction)
        mid0, mid1 = rule.x0 + 0.2 * rule.w, rule.x1 - 0.2 * rule.w
        if not (
            mid0 - 0.5 <= (nx0 + nx1) / 2 <= mid1 + 0.5
            and mid0 - 0.5 <= (dx0 + dx1) / 2 <= mid1 + 0.5
        ):
            return None
        if min(a.y0 for a in num) - rule.y > 0.6 * s:
            return None
        if not (self._closed(num, rule, atoms) and self._closed(den, rule, atoms)):
            return None
        if any(a.is_hebrew for a in num + den):
            # Hebrew words over a line are usually an underlined heading above the
            # next line, not a fraction: demand that both parts are isolated rows
            # (nothing else on their baselines within two em of the rule) and that
            # the rule is not glued to the numerator's baseline like an underline.
            if not (
                self._closed(num, rule, atoms, reach=2.0)
                and self._closed(den, rule, atoms, reach=2.0)
            ):
                return None
            num_base = median([a.base for a in num])
            if num_base - rule.y < 0.22 * s:
                return None
        latex = r"\frac{" + self.to_latex(num) + "}{" + self.to_latex(den) + "}"
        comp = self._composite("frac", num + den, latex, base=None, size=s, rule=rule)
        comp.axis = rule.y
        return comp

    def _overline(self, rule: Rule, atoms: list[Atom]) -> Atom | None:
        under = self._stack(rule, atoms, above=False)
        if under:
            # one row only: the overlined symbol and its scripts. The atom nearest
            # the rule may be a superscript (the n of \overline{z^n}); the row is
            # anchored on the largest atom that overlaps it vertically.
            first = under[0]
            row = [a for a in under if a.y1 >= first.y0 - 0.2 * first.size]
            top = max(row, key=lambda a: (a.size, -a.order))
            under = [a for a in under if top.y0 - 0.45 * top.size <= a.cy]
        if not under or any(a.is_hebrew for a in under):
            return None
        ux0, _, ux1, _ = union(under)
        s = median([a.size for a in under])
        # the clearance is measured against the main symbol: with a subscript in
        # the group (\overline{z_0}) the median size would be too small
        main = max(a.size for a in under)
        if abs(rule.w - (ux1 - ux0)) > 0.6 * s or rule.y - max(a.y1 for a in under) > 0.6 * main:
            return None
        # something directly above, as close and as centred as a numerator would
        # be, makes it a fraction/table line
        above = [a for a in self._stack(rule, atoms, above=True) if a.y0 - rule.y <= 0.45 * a.size]
        if above and _has_operand(above):
            ax0, _, ax1, _ = union(above)
            if rule.x0 + 0.2 * rule.w - 0.5 <= (ax0 + ax1) / 2 <= rule.x1 - 0.2 * rule.w + 0.5:
                return None
        # (unlike a fraction, an overlined group sits on the main baseline, so
        # neighbours on that baseline are expected: no _closed() test here)
        base = median([a.base for a in under])
        latex = r"\overline{" + self.to_latex(under) + "}"
        comp = self._composite("over", under, latex, base=base, size=s, rule=rule)
        return comp

    # radicals ----------------------------------------------------------------
    def _radical(self, rule: Rule, atoms: list[Atom]) -> Atom | None:
        rads = [
            a
            for a in atoms
            if a.kind == "g"
            and a.text in _RADICALS
            and abs(rule.x0 - a.x1) <= 0.6 * a.size + 1.0
            and a.y1 - 0.35 * a.size - 1.0 <= rule.y <= a.y1 + 0.35 * a.size + 1.0
        ]
        if not rads:
            return None
        rad = min(rads, key=lambda a: (abs(rule.x0 - a.x1), a.order))
        body = [
            a
            for a in atoms
            if a is not rad
            and a.x0 >= rule.x0 - 0.2 * a.size
            and a.x1 <= rule.x1 + 0.15 * a.size
            and a.y1 <= rule.y + 0.2 * a.size
            and a.y0 >= rad.y0 - 0.3 * a.size
        ]
        index = [
            a
            for a in atoms
            if a is not rad
            and a not in body
            and rad.x0 - 0.4 * rad.size <= a.cx <= rad.x0 + 0.65 * rad.w
            and a.y0 > rad.y0 + 0.3 * rad.h
            and a.y1 <= rule.y + 0.5 * rad.size
            and a.size < 0.9 * max((b.size for b in body), default=rad.size)
        ]
        inner = self.to_latex(body) if body else ""
        root = rad.text
        if index:
            deg = self.to_latex(index)
            latex = r"\sqrt[" + deg + "]{" + inner + "}"
        elif root == "∛":
            latex = r"\sqrt[3]{" + inner + "}"
        elif root == "∜":
            latex = r"\sqrt[4]{" + inner + "}"
        else:
            latex = r"\sqrt{" + inner + "}"
        parts = [rad] + body + index
        s = median([a.size for a in body]) if body else rad.size
        base = median([a.base for a in body]) if body else rad.base
        return self._composite("sqrt", parts, latex, base=base, size=s, rule=rule)

    def _radicals_without_rule(
        self, atoms: list[Atom], consumed: set[int], out: list[Atom]
    ) -> None:
        """√2 typed as a character: the radicand is the adjacent operand."""
        ordered = sorted([a for a in atoms if not a.is_space], key=Atom.sort_key)
        i = 0
        while i < len(ordered):
            a = ordered[i]
            if a.kind == "g" and a.text in _RADICALS and id(a) not in consumed:
                j = i + 1
                body: list[Atom] = []
                if j < len(ordered) and ordered[j].kind == "g" and ordered[j].text == "(":
                    depth = 0
                    while j < len(ordered):
                        t = ordered[j].text
                        depth += t == "("
                        depth -= t == ")"
                        body.append(ordered[j])
                        j += 1
                        if depth == 0:
                            break
                    body = body[1:-1] if len(body) >= 2 and body[-1].text == ")" else body
                else:
                    while (
                        j < len(ordered)
                        and (ordered[j].is_composite or ordered[j].text.isalnum())
                        and ordered[j].x0 - (body[-1].x1 if body else a.x1) < 0.25 * a.size
                        and abs(ordered[j].base - a.base) < 0.5 * a.size
                    ):
                        body.append(ordered[j])
                        j += 1
                if body:
                    latex = (
                        {"√": r"\sqrt{", "∛": r"\sqrt[3]{", "∜": r"\sqrt[4]{"}[a.text]
                        + self.to_latex(body)
                        + "}"
                    )
                    parts = [a] + ordered[i + 1 : j]
                    comp = self._composite("sqrt", parts, latex, base=a.base, size=a.size)
                    for p in parts:
                        consumed.add(id(p))
                    out.append(comp)
                    i = j
                    continue
            i += 1

    # big operators -------------------------------------------------------------
    def _operators_with_limits(
        self, atoms: list[Atom], consumed: set[int], out: list[Atom]
    ) -> None:
        glyphs = sorted([a for a in atoms if not a.is_space], key=Atom.sort_key)
        ops: list[tuple[Atom | list[Atom], str]] = []
        for a in glyphs:
            # integrals too: equation editors put their limits above/below (side
            # limits, as TeX sets them, are outside the limit window and stay scripts)
            if a.kind == "g" and a.text in BIG_OPERATORS:
                ops.append((a, BIG_OPERATORS[a.text]))
        # multi-letter limit operators (lim, max, sup...) spelled with upright glyphs;
        # matched along one baseline so that limits drawn underneath do not interleave
        letters = [g for g in glyphs if _is_letter(g) and not g.italic]
        i = 0
        while i < len(letters):
            matched = False
            g0 = letters[i]
            row = [g0] + [
                g
                for g in letters[i + 1 :]
                if abs(g.base - g0.base) < 0.1 * g0.size and abs(g.size - g0.size) < 0.1 * g0.size
            ]
            for word in _LIMIT_WORDS:
                seq = row[: len(word)]
                if (
                    len(seq) == len(word)
                    and "".join(g.text for g in seq) == word
                    and all(_is_letter(g) and not g.italic for g in seq)
                    and all(
                        abs(seq[k + 1].base - seq[k].base) < 0.1 * seq[k].size
                        and seq[k + 1].x0 - seq[k].x1 < 0.2 * seq[k].size
                        for k in range(len(seq) - 1)
                    )
                ):
                    ops.append((seq, "\\" + word))
                    i = letters.index(seq[-1]) + 1
                    matched = True
                    break
            if not matched:
                i += 1

        for op, cmd in ops:
            parts = op if isinstance(op, list) else [op]
            if any(id(p) in consumed for p in parts):
                continue
            x0, y0, x1, y1 = union(parts)
            size = median([p.size for p in parts])
            pool = [a for a in atoms if id(a) not in consumed and a not in parts and not a.is_space]
            upper = self._limit_row(pool, x0, x1, size, edge=y1, above=True)
            lower = self._limit_row(pool, x0, x1, size, edge=y0, above=False)
            if not upper and not lower:
                continue
            latex = cmd
            if lower:
                latex += "_{" + self.to_latex(lower) + "}"
            if upper:
                latex += "^{" + self.to_latex(upper) + "}"
            base = median([p.base for p in parts])
            comp = self._composite("op", parts + upper + lower, latex, base=base, size=size)
            comp.math_font = True
            for p in comp.children:
                consumed.add(id(p))
            out.append(comp)

    def _limit_row(self, pool, x0, x1, size, edge, above) -> list[Atom]:
        seeds = []
        for a in pool:
            if not (x0 - 0.5 * size <= a.cx <= x1 + 0.5 * size):
                continue
            if a.size > 0.92 * size and not a.is_composite:
                continue
            g = (a.y0 - edge) if above else (edge - a.y1)
            if -0.15 * a.size <= g <= 0.8 * size:
                seeds.append(a)
        if not seeds:
            return []

        # only the row nearest to the operator: a limit of an operator on the
        # neighbouring line must not be taken (e.g. stacked ∑ and ∏ formulas)
        def gap_of(a: Atom) -> float:
            return (a.y0 - edge) if above else (edge - a.y1)

        nearest = min(seeds, key=lambda a: (gap_of(a), a.x0))
        seeds = [a for a in seeds if abs(a.base - nearest.base) <= 0.3 * nearest.size]
        base = median([a.base for a in seeds])
        row = list(seeds)
        # extend horizontally along the same baseline (e.g. "k=1" wider than the operator)
        changed = True
        while changed:
            changed = False
            rx0, _, rx1, _ = union(row)
            for a in pool:
                if a in row or abs(a.base - base) > 0.25 * a.size:
                    continue
                if (0 <= rx0 - a.x1 < 0.35 * a.size) or (0 <= a.x0 - rx1 < 0.35 * a.size):
                    row.append(a)
                    changed = True
        return row

    # matrices, cases, \left...\right ---------------------------------------------
    def _tall_structures(self, atoms: list[Atom], consumed: set[int], out: list[Atom]) -> None:
        """Tall delimiters spanning several text rows: matrices, vectors, systems, big groups.

        A delimiter is *tall* when the producer assembled it from pieces or drew
        it as a long vector stroke (``Atom.tall``), or when a single bracket glyph
        is drawn much larger than the text. Pairs are matched by height and
        vertical extent and processed innermost first; what lies between them is
        split into rows (by baseline) and columns (by horizontal gaps).
        """
        glyph_sizes = [a.size for a in atoms if a.kind == "g" and not a.tall and a.text.isalnum()]
        text_size = median(glyph_sizes, default=10.0)
        talls = []
        for a in atoms:
            if a.kind != "g" or a.text not in _TALL_PAIR and a.text not in _TALL_CLOSE:
                continue
            if a.tall or a.h >= 1.8 * text_size:
                talls.append(a)
        if not talls:
            return
        talls.sort(key=lambda a: (a.x0, a.order))
        pairs: list[tuple[Atom, Atom | None]] = []
        used: set[int] = set()
        for i, op in enumerate(talls):
            if id(op) in used or op.text not in _TALL_PAIR:
                continue
            want = _TALL_PAIR[op.text]
            close = None
            for c in talls[i + 1 :]:
                if id(c) in used or c.text != want:
                    continue
                ov = min(op.y1, c.y1) - max(op.y0, c.y0)
                if ov >= 0.7 * min(op.h, c.h) and abs(op.h - c.h) <= 0.3 * max(op.h, c.h):
                    close = c
                    break
            if close is None and op.text != "{":
                continue
            used.add(id(op))
            if close is not None:
                used.add(id(close))
            pairs.append((op, close))
        # innermost (narrowest) first, so nested groups are collapsed bottom-up
        pairs.sort(key=lambda p: ((p[1].x1 if p[1] else p[0].x1 + 1e6) - p[0].x0, p[0].x0))
        pool = list(atoms)
        for op, close in pairs:
            if id(op) in consumed or (close is not None and id(close) in consumed):
                continue
            band0, band1 = op.y0 - 0.1 * text_size, op.y1 + 0.1 * text_size
            live = [
                a
                for a in pool
                if id(a) not in consumed
                and a is not op
                and a is not close
                and not a.is_space
                and band0 <= a.cy <= band1
            ]
            if close is not None:
                content = [a for a in live if op.x1 - 0.2 <= a.cx <= close.x0 + 0.2]
            else:
                content = self._cases_body(op, live, text_size)
            if not content:
                continue
            rows = _rows(content, text_size)
            if len(rows) < 2 and not (op.tall or (close is not None and close.tall)):
                # one big bracket glyph around one row: the baseline parser already
                # handles it as a plain bracket (keeps "(1+\frac{1}{n})^{n}" simple)
                continue
            env = _TALL_ENV.get((op.text, close.text if close else ""))
            if close is None:
                body = r" \\ ".join(self._cells_latex(r, text_size, cases=True) for r in rows)
                latex = r"\begin{cases}" + body + r"\end{cases}"
            elif len(rows) >= 2 and env:
                cols = _columns(rows, text_size)
                body = r" \\ ".join(" & ".join(self.to_latex(cell) for cell in row) for row in cols)
                latex = r"\begin{" + env + "}" + body + r"\end{" + env + "}"
            else:
                left, right = (
                    _LEFT_RIGHT.get(op.text, op.text),
                    _LEFT_RIGHT.get(close.text, close.text),
                )
                latex = r"\left" + left + self.to_latex(content) + r"\right" + right
            parts = [op] + ([close] if close is not None else []) + content
            axis = (op.y0 + op.y1) / 2
            comp = self._composite(
                "matrix", parts, latex, base=axis - 0.25 * text_size, size=text_size
            )
            comp.axis = axis
            for p in parts:
                consumed.add(id(p))
            out.append(comp)
            pool.append(comp)

    def _cases_body(self, op: Atom, live: list[Atom], text_size: float) -> list[Atom]:
        """Everything right of an unmatched tall '{' until a wide horizontal gap."""
        right = sorted([a for a in live if a.cx > op.x1 - 0.2], key=lambda a: (a.x0, a.order))
        body: list[Atom] = []
        reach = op.x1
        for a in right:
            if a.x0 - reach > 3.0 * text_size:
                break
            body.append(a)
            reach = max(reach, a.x1)
        return body

    def _cells_latex(self, row: list[Atom], text_size: float, cases: bool = False) -> str:
        cells = _split_cells(row, text_size)
        if cases and len(cells) >= 2:
            return (
                self.to_latex(cells[0])
                + " & "
                + r"\quad ".join(self.to_latex(c) for c in cells[1:])
            )
        return r"\quad ".join(self.to_latex(c) for c in cells)

    def _composite(self, kind, parts, latex, *, base, size, rule: Rule | None = None) -> Atom:
        x0, y0, x1, y1 = union(parts)
        if rule is not None:
            x0, x1 = min(x0, rule.x0), max(x1, rule.x1)
        if base is None:
            base = (rule.y - 0.25 * size) if rule is not None else median([p.base for p in parts])
        order = min(p.order for p in parts)
        return Atom(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            base=base,
            size=size,
            kind=kind,
            latex=latex,
            order=order,
            math_font=True,
            children=list(parts),
            lx0=x0,
            lx1=x1,
        )

    # ------------------------------------------------------------------ phase 2
    def to_latex(self, atoms: list[Atom]) -> str:
        if not atoms:
            return ""
        latex = self._parse(atoms)
        return _postprocess(latex)

    def _dominant(self, atoms: list[Atom]) -> tuple[float, float]:
        real = [a for a in atoms if not a.is_space]
        if not real:
            return atoms[0].base, atoms[0].size

        # big operators (with or without limits) and assembled delimiters are drawn
        # larger than the text they belong to: they do not define the baseline
        # Big operators (with or without limits), assembled delimiters and brackets
        # that an equation editor scaled up are larger than the text they
        # belong to: they do not define the baseline.
        def sizing(a: Atom) -> bool:
            return (
                not (
                    a.kind == "g"
                    and (a.text in BIG_OPERATORS or a.text in _BRACKETS or a.text in "{}|‖")
                )
                and a.kind != "op"
                and not a.tall
            )

        sizes = [a.size for a in real if sizing(a)] or [a.size for a in real]
        s_max = max(sizes)
        main = [a for a in real if a.size >= 0.9 * s_max and sizing(a)] or [
            a for a in real if a.size >= 0.9 * s_max
        ]
        # most frequent baseline among full-size atoms
        buckets: list[list[Atom]] = []
        for a in sorted(main, key=lambda a: (-a.base, a.order)):
            for b in buckets:
                if abs(b[0].base - a.base) <= 0.12 * s_max:
                    b.append(a)
                    break
            else:
                buckets.append([a])
        best = max(buckets, key=lambda b: (len(b), -min(x.x0 for x in b)))
        return median([a.base for a in best]), median([a.size for a in best])

    def _script_kind(self, a: Atom, base: float, size: float) -> str | None:
        shift = a.base - base
        if a.is_composite and a.kind != "acc":
            if a.size < 0.85 * size and shift > 0.25 * size:
                return "sup"
            if a.size < 0.85 * size and shift < -0.2 * size:
                return "sub"
            return None
        if a.kind == "g" and a.size >= 0.95 * size and (a.text in _BRACKETS or a.text in "{}|‖"):
            return None  # an enlarged bracket drawn with a lowered baseline
        small = a.size <= 0.86 * size
        if small and shift > 0.1 * size:
            return "sup"
        if small and shift < -0.08 * size:
            return "sub"
        if not small and abs(shift) > 0.55 * size and not a.is_space:
            return "sup" if shift > 0 else "sub"
        return None

    def _parse(self, atoms: list[Atom]) -> str:
        atoms = sorted(atoms, key=Atom.sort_key)
        base, size = self._dominant(atoms)
        nodes: list[dict] = []
        i = 0
        while i < len(atoms):
            a = atoms[i]
            kind = self._script_kind(a, base, size)
            if kind is None:
                if not a.is_space:
                    nodes.append({"atom": a, "sup": [], "sub": []})
                elif nodes:
                    nodes.append({"atom": a, "sup": [], "sub": []})
                i += 1
                continue
            j = i
            sups: list[Atom] = []
            subs: list[Atom] = []
            while j < len(atoms):
                k = self._script_kind(atoms[j], base, size)
                if k is None:
                    break
                (sups if k == "sup" else subs).append(atoms[j])
                j += 1
            if not nodes or nodes[-1]["atom"].is_space:
                nodes.append({"atom": None, "sup": sups, "sub": subs})
            else:
                nodes[-1]["sup"].extend(sups)
                nodes[-1]["sub"].extend(subs)
            i = j
        return self._render(nodes, size)

    def _render(self, nodes: list[dict], size: float) -> str:
        override = self._mirror_overrides(nodes)
        parts: list[str] = []
        i = 0
        n = len(nodes)
        prev_x1: float | None = None
        while i < n:
            node = nodes[i]
            a = node["atom"]
            if a is not None and not a.is_space:
                if prev_x1 is not None and a.x0 - prev_x1 > 1.4 * size:
                    parts.append(r"\quad ")
            ext = [x.x1 for x in ([a] if a is not None else []) + node["sup"] + node["sub"]]
            if ext:
                prev_x1 = max(ext + ([prev_x1] if prev_x1 is not None else []))
            # (stack) without a rule between big parentheses -> \binom
            if (
                a is not None
                and a.kind == "g"
                and (override.get(id(a)) or a.text) == "("
                and node["sup"]
                and node["sub"]
                and i + 1 < n
                and nodes[i + 1]["atom"] is not None
                and (override.get(id(nodes[i + 1]["atom"])) or nodes[i + 1]["atom"].text) == ")"
            ):
                close = nodes[i + 1]
                parts.append(
                    r"\binom{"
                    + self._parse(node["sup"])
                    + "}{"
                    + self._parse(node["sub"])
                    + "}"
                    + self._scripts(close)
                )
                prev_x1 = close["atom"].x1
                i += 2
                continue
            # Hebrew words inside math -> \text{...} in logical order
            if a is not None and a.kind == "g" and a.is_hebrew:
                j = i
                vis = ""
                prev = None
                while (
                    j < n
                    and nodes[j]["atom"] is not None
                    and (
                        nodes[j]["atom"].is_hebrew
                        or nodes[j]["atom"].is_space
                        or (nodes[j]["atom"].kind == "g" and nodes[j]["atom"].text in "-'\"־")
                    )
                ):
                    cur = nodes[j]["atom"]
                    if cur.is_space:
                        vis += " "
                    else:
                        if (
                            prev is not None
                            and cur.x0 - prev.x1 > 0.2 * cur.size
                            and not vis.endswith(" ")
                        ):
                            vis += " "
                        vis += cur.text + cur.marks
                        prev = cur
                    j += 1
                text = visual_to_logical(vis.strip(), default_rtl=True)
                parts.append(r"\text{" + _escape_text(text) + "}" + self._scripts(nodes[j - 1]))
                prev_x1 = _extent(nodes[i:j], prev_x1)
                i = j
                continue
            if a is not None and a.is_space:
                i += 1
                continue
            # runs of upright ASCII letters -> function names / \mathrm
            if (
                a is not None
                and _is_letter(a)
                and not a.italic
                and not node["sup"]
                and not node["sub"]
            ):
                j = i
                run = []
                while (
                    j < n
                    and nodes[j]["atom"] is not None
                    and _is_letter(nodes[j]["atom"])
                    and not nodes[j]["atom"].italic
                    and nodes[j]["atom"].font == a.font
                    and (not run or nodes[j]["atom"].x0 - run[-1].x1 < 0.12 * size)
                ):
                    run.append(nodes[j]["atom"])
                    if nodes[j]["sup"] or nodes[j]["sub"]:
                        j += 1
                        break
                    j += 1
                if len(run) >= 2:
                    word = "".join(r.text for r in run)
                    pieces = (
                        split_function_names(word)
                        if not a.math_font
                        else (
                            [word]
                            if word in FUNCTION_NAMES or word in OPERATOR_NAMES
                            else split_function_names(word)
                        )
                    )
                    rendered = []
                    for p in pieces:
                        if p in FUNCTION_NAMES:
                            rendered.append("\\" + p)
                        elif p in OPERATOR_NAMES:
                            rendered.append(r"\operatorname{" + p + "}")
                        elif len(p) > 1:
                            rendered.append(r"\mathrm{" + p + "}")
                        else:
                            rendered.append(p)
                    if a.math_font and not any(
                        p in FUNCTION_NAMES or p in OPERATOR_NAMES for p in pieces
                    ):
                        rendered = [r"\mathrm{" + word + "}"]
                    parts.append(join_latex(rendered) + self._scripts(nodes[j - 1]))
                    prev_x1 = _extent(nodes[i:j], prev_x1)
                    i = j
                    continue
            base_latex = self._atom_latex(a, override.get(id(a))) if a is not None else "{}"
            scripts = self._scripts(node)
            if scripts and not _is_single_token(base_latex):
                base_latex = "{" + base_latex + "}"
            parts.append(base_latex + scripts)
            i += 1
        return join_latex(parts)

    def _scripts(self, node: dict) -> str:
        out = ""
        if node["sub"]:
            out += "_{" + self._parse(node["sub"]) + "}"
        if node["sup"]:
            out += "^{" + self._parse(node["sup"]) + "}"
        return out

    def _mirror_overrides(self, nodes: list[dict]) -> dict[int, str]:
        glyphs = [nd["atom"] for nd in nodes if nd["atom"] is not None and nd["atom"].kind == "g"]
        if self.mirror_mode:
            return {id(a): MIRROR[a.text] for a in glyphs if a.text in MIRROR}
        br = [a for a in glyphs if a.text in _BRACKETS]
        if not br:
            return {}
        texts = [a.text for a in br]
        if _valid_brackets(texts):
            if any(t in _OPEN for t in texts):
                self.votes["as_is"] += 1
            return {}
        swapped = [MIRROR.get(t, t) for t in texts]
        if _valid_brackets(swapped):
            self.votes["mirrored"] += 1
            return {id(a): t for a, t in zip(br, swapped)}
        # a corner piece of an assembled bracket left on its own: "[v⌋" is "[v]"
        squared = [_CORNER_TO_SQUARE.get(t, t) for t in texts]
        if _valid_brackets(squared):
            return {id(a): t for a, t, old in zip(br, squared, texts) if t != old}
        return {}

    def _atom_latex(self, a: Atom, override: str | None = None) -> str:
        if a.is_composite:
            return a.latex or ""
        ch = override or a.text
        if not ch or ch == "\ufffd":
            return ""
        if ch.isascii() and ch.isalpha():
            style = style_for_font(a.font)
            if style is not None:
                return f"{style}{{{ch}}}"
            if a.bold and a.math_font:
                return r"\mathbf{" + ch + "}"
            return ch
        if ch in BIG_OPERATORS:
            return BIG_OPERATORS[ch]
        latex = char_to_latex(ch)
        if a.marks:
            for m in a.marks:
                cmd = ACCENTS.get(m)
                if cmd:
                    latex = f"{cmd}{{{latex}}}"
        return latex


_BRACKETS = set("()[]⟨⟩⌊⌋⌈⌉")
_CORNER_TO_SQUARE = {"⌈": "[", "⌊": "[", "⌉": "]", "⌋": "]"}
_TALL_PAIR = {"(": ")", "[": "]", "{": "}", "|": "|", "‖": "‖", "⌈": "⌉", "⌊": "⌋", "⟨": "⟩"}
_TALL_CLOSE = set(_TALL_PAIR.values())
_TALL_ENV = {
    ("(", ")"): "pmatrix",
    ("[", "]"): "bmatrix",
    ("|", "|"): "vmatrix",
    ("‖", "‖"): "Vmatrix",
    ("{", "}"): "Bmatrix",
}
_LEFT_RIGHT = {
    "{": r"\{",
    "}": r"\}",
    "‖": r"\|",
    "⟨": r"\langle ",
    "⟩": r"\rangle ",
    "⌈": r"\lceil ",
    "⌉": r"\rceil ",
    "⌊": r"\lfloor ",
    "⌋": r"\rfloor ",
}


def _has_operand(atoms: list[Atom]) -> bool:
    return any(
        a.is_composite
        or (
            a.kind == "g"
            and (a.text.isalnum() or math_alphanumeric(a.text) or a.text in "∞∅∂∇ℝℕℤℚℂ…⋯⋮")
        )
        for a in atoms
    )


def _rows(atoms: list[Atom], text_size: float) -> list[list[Atom]]:
    """Group the content of a tall group into rows by baseline (top to bottom)."""
    main = [
        a
        for a in atoms
        if (a.size >= 0.85 * text_size and not a.is_composite) or a.kind in ("frac", "matrix", "op")
    ]
    if not main:
        main = list(atoms)
    rows: list[list[Atom]] = []
    for a in sorted(main, key=lambda a: (-a.base, a.x0, a.order)):
        for r in rows:
            if abs(r[0].base - a.base) <= 0.45 * text_size:
                r.append(a)
                break
        else:
            rows.append([a])
    bases = [median([a.base for a in r]) for r in rows]
    for a in atoms:
        if any(a is b for r in rows for b in r):
            continue
        k = min(range(len(rows)), key=lambda i: (abs(a.cy - (bases[i] + 0.3 * text_size)), i))
        rows[k].append(a)
    order = sorted(range(len(rows)), key=lambda i: -bases[i])
    return [sorted(rows[i], key=Atom.sort_key) for i in order]


def _split_cells(row: list[Atom], text_size: float) -> list[list[Atom]]:
    cells: list[list[Atom]] = []
    reach = None
    for a in sorted(row, key=Atom.sort_key):
        if cells and reach is not None and a.x0 - reach >= 0.9 * text_size:
            cells.append([a])
        elif cells:
            cells[-1].append(a)
        else:
            cells.append([a])
        reach = a.x1 if reach is None else max(reach, a.x1)
    return cells


def _columns(rows: list[list[Atom]], text_size: float) -> list[list[list[Atom]]]:
    """Cells of every row, aligned to common columns (missing cells are empty)."""
    split = [_split_cells(r, text_size) for r in rows]
    width = max(len(c) for c in split)
    if all(len(c) == width for c in split):
        return split
    ref = max(split, key=len)
    centres = [(min(a.x0 for a in c) + max(a.x1 for a in c)) / 2 for c in ref]
    out = []
    for cells in split:
        slots: list[list[Atom]] = [[] for _ in centres]
        for c in cells:
            cx = (min(a.x0 for a in c) + max(a.x1 for a in c)) / 2
            k = min(range(len(centres)), key=lambda i: (abs(centres[i] - cx), i))
            slots[k].extend(c)
        out.append(slots)
    return out


def _extent(nodes: list[dict], prev: float | None) -> float | None:
    xs = [
        x.x1
        for nd in nodes
        for x in ([nd["atom"]] if nd["atom"] is not None else []) + nd["sup"] + nd["sub"]
    ]
    if prev is not None:
        xs.append(prev)
    return max(xs) if xs else prev


def _valid_brackets(seq: list[str]) -> bool:
    stack: list[str] = []
    for c in seq:
        if c in _OPEN:
            stack.append(c)
        elif c in _CLOSE:
            if not stack or stack[-1] != _CLOSE[c]:
                return False
            stack.pop()
    return not stack


_SINGLE_TOKEN_RE = re.compile(
    r"(?:\\[A-Za-z]+\*?|\\.|[^\\{}\s])"  # one char or one control sequence
    r"(?:\[[^\[\]]*\]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*"  # its [opt] / {arg} groups
)


def _is_single_token(latex: str) -> bool:
    return latex == "{}" or bool(_SINGLE_TOKEN_RE.fullmatch(latex))


def _escape_text(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    return "".join(repl.get(c, c) for c in text)


_PRIME_RE = re.compile(r"\^\{('+)\}")
_PRIME_CMD_RE = re.compile(r"\^\{(\\prime)+\}")


def _postprocess(latex: str) -> str:
    latex = _PRIME_RE.sub(lambda m: m.group(1), latex)
    latex = _PRIME_CMD_RE.sub(lambda m: "'" * m.group(0).count("prime"), latex)
    latex = latex.replace(r"\sqrt{}", r"\sqrt{\ }")
    latex = re.sub(r"(?:\\cdot\s*){3}", r"\\cdots ", latex)
    latex = re.sub(r"\.\s*\.\s*\.", r"\\ldots ", latex)
    latex = re.sub(r"\\(cdots|ldots) (?=[^A-Za-z])", r"\\\1", latex)
    latex = re.sub(r"(\\quad\s*)+$", "", latex)
    return latex.strip()
