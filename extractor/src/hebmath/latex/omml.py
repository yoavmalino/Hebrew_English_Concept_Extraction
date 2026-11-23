"""Office Math Markup Language (OMML, ECMA-376 Part 1 §22.1) -> LaTeX.

A direct, deterministic tree walk. OMML is already a structural description
(fraction, radical, n-ary, delimiter...), so no heuristics are needed: each
element maps to one LaTeX construct. Unknown elements degrade gracefully by
converting their children.

This module is pure Python (standard library only) and was written from the
ECMA-376 specification; it does not use Microsoft's OMML2MML.XSL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..xmlutil import attr as _attr_any
from ..xmlutil import local as _xlocal
from ..xmlutil import ns as _xns

from .symbols import (
    ACCENTS,
    BIG_OPERATORS,
    FUNCTION_NAMES,
    INTEGRAL_OPERATORS,
    LIMIT_FUNCTIONS,
    OPERATOR_NAMES,
    WIDE_ACCENTS,
    char_to_latex,
    delimiter_to_latex,
    group,
    is_hebrew,
    join_latex,
    text_to_latex,
)

M_NAMESPACES = {
    "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "http://purl.oclc.org/ooxml/officeDocument/math",  # ISO/IEC 29500 Strict
}
W_NAMESPACES = {
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/wordprocessingml/main",
}

_TRUE = {"1", "on", "true"}
MAX_DEPTH = 256

_SCRIPT_STYLES = {
    "double-struck": r"\mathbb",
    "script": r"\mathcal",
    "fraktur": r"\mathfrak",
    "sans-serif": r"\mathsf",
    "monospace": r"\mathtt",
}


def _ns(el) -> str:
    return _xns(el)


def _local(el) -> str:
    return _xlocal(el)


def is_math(el: object) -> bool:
    return isinstance(el.tag, str) and _ns(el) in M_NAMESPACES


def is_word(el: object) -> bool:
    return isinstance(el.tag, str) and _ns(el) in W_NAMESPACES


def _child(el: object, name: str) -> object | None:
    for c in el:
        if isinstance(c.tag, str) and _local(c) == name and _ns(c) in M_NAMESPACES:
            return c
    return None


def _children(el: object, name: str) -> list[object]:
    return [
        c for c in el if isinstance(c.tag, str) and _local(c) == name and _ns(c) in M_NAMESPACES
    ]


def _val(el) -> str | None:
    return _attr_any(el, "val")


def _prop(el: object, pr_name: str, prop_name: str) -> object | None:
    pr = _child(el, pr_name)
    return _child(pr, prop_name) if pr is not None else None


def _prop_val(el: object, pr_name: str, prop_name: str, default: str | None = None) -> str | None:
    p = _prop(el, pr_name, prop_name)
    if p is None:
        return default
    v = _val(p)
    return default if v is None else v


def _prop_bool(el: object, pr_name: str, prop_name: str) -> bool:
    p = _prop(el, pr_name, prop_name)
    if p is None:
        return False
    v = _val(p)
    return v is None or v.lower() in _TRUE


@dataclass
class _Ctx:
    depth: int = 0
    in_eqarr: bool = False
    warnings: list[str] = field(default_factory=list)


class OmmlToLatex:
    """Converter object; stateless between calls, safe to reuse."""

    def convert(self, el: object) -> str:
        """Convert an ``m:oMath`` / ``m:oMathPara`` / any OMML subtree."""
        ctx = _Ctx()
        latex = self._node(el, ctx)
        return latex.strip()

    # ------------------------------------------------------------------ dispatch
    def _node(self, el: object, ctx: _Ctx) -> str:
        if not isinstance(el.tag, str):
            return ""
        if ctx.depth > MAX_DEPTH:
            raise RecursionError("OMML nesting too deep")
        ctx.depth += 1
        try:
            if is_math(el):
                handler = self._HANDLERS.get(_local(el))
                if handler is not None:
                    return handler(self, el, ctx)
                if _local(el).endswith("Pr"):
                    return ""  # property containers carry no content
                return self._seq(el, ctx)
            if is_word(el):
                return self._word_node(el, ctx)
            return self._seq(el, ctx)
        finally:
            ctx.depth -= 1

    def _seq(self, el: object | None, ctx: _Ctx) -> str:
        if el is None:
            return ""
        return join_latex([self._node(c, ctx) for c in el])

    def _arg(self, el: object, name: str, ctx: _Ctx) -> str:
        return self._seq(_child(el, name), ctx).strip()

    # ------------------------------------------------------------------ handlers
    def _omath_para(self, el, ctx) -> str:
        lines = [self._node(c, ctx) for c in _children(el, "oMath")]
        lines = [x for x in lines if x.strip()]
        if len(lines) > 1:
            return r"\begin{gathered} " + r" \\ ".join(lines) + r" \end{gathered}"
        return lines[0] if lines else ""

    def _run(self, el, ctx) -> str:
        text = "".join(
            t.text or "" for t in el.iter() if isinstance(t.tag, str) and _local(t) == "t"
        )
        if not text:
            return ""
        rpr = _child(el, "rPr")
        normal = (
            rpr is not None
            and _child(rpr, "nor") is not None
            and (_val(_child(rpr, "nor")) is None or _val(_child(rpr, "nor")).lower() in _TRUE)
        )
        sty = _val(_child(rpr, "sty")) if rpr is not None else None
        scr = _val(_child(rpr, "scr")) if rpr is not None else None

        if normal:
            return r"\text{" + _escape_text_mode(text) + "}"

        if ctx.in_eqarr and "&" in text:
            pieces = text.split("&")
            return "&".join(self._styled_text(p, sty, scr) for p in pieces)
        return self._styled_text(text, sty, scr)

    def _styled_text(self, text: str, sty: str | None, scr: str | None) -> str:
        if not text:
            return ""
        upright = sty == "p"
        latex = text_to_latex(text, upright=upright)
        if scr in _SCRIPT_STYLES and latex:
            return f"{_SCRIPT_STYLES[scr]}{{{latex}}}"
        if sty == "b":
            return r"\mathbf{" + latex + "}"
        if sty == "bi":
            return r"\boldsymbol{" + latex + "}"
        return latex

    def _frac(self, el, ctx) -> str:
        num = self._arg(el, "num", ctx)
        den = self._arg(el, "den", ctx)
        ftype = _prop_val(el, "fPr", "type", "bar")
        if ftype in ("lin", "skw"):
            return f"{group(num)}/{group(den)}"
        if ftype == "noBar":
            return r"\genfrac{}{}{0pt}{}{" + num + "}{" + den + "}"
        return r"\frac{" + num + "}{" + den + "}"

    def _ssup(self, el, ctx) -> str:
        return f"{_base(self._arg(el, 'e', ctx))}^{{{self._arg(el, 'sup', ctx)}}}"

    def _ssub(self, el, ctx) -> str:
        return f"{_base(self._arg(el, 'e', ctx))}_{{{self._arg(el, 'sub', ctx)}}}"

    def _ssubsup(self, el, ctx) -> str:
        return (
            f"{_base(self._arg(el, 'e', ctx))}"
            f"_{{{self._arg(el, 'sub', ctx)}}}^{{{self._arg(el, 'sup', ctx)}}}"
        )

    def _spre(self, el, ctx) -> str:
        return (
            f"{{}}_{{{self._arg(el, 'sub', ctx)}}}^{{{self._arg(el, 'sup', ctx)}}}"
            f"{_base(self._arg(el, 'e', ctx))}"
        )

    def _rad(self, el, ctx) -> str:
        body = self._arg(el, "e", ctx)
        deg = self._arg(el, "deg", ctx)
        if _prop_bool(el, "radPr", "degHide") or not deg:
            return r"\sqrt{" + body + "}"
        return r"\sqrt[" + deg + "]{" + body + "}"

    def _nary(self, el, ctx) -> str:
        chr_ = _prop_val(el, "naryPr", "chr", "∫") or "∫"
        op = BIG_OPERATORS.get(chr_, char_to_latex(chr_))
        lim_loc = _prop_val(el, "naryPr", "limLoc")
        sub = "" if _prop_bool(el, "naryPr", "subHide") else self._arg(el, "sub", ctx)
        sup = "" if _prop_bool(el, "naryPr", "supHide") else self._arg(el, "sup", ctx)
        out = op
        if op in INTEGRAL_OPERATORS and lim_loc == "undOvr":
            out += r"\limits"
        elif op not in INTEGRAL_OPERATORS and lim_loc == "subSup":
            out += r"\nolimits"
        if sub:
            out += "_{" + sub + "}"
        if sup:
            out += "^{" + sup + "}"
        body = self._arg(el, "e", ctx)
        return join_latex([out, " " + body if body else ""])

    def _delim(self, el, ctx) -> str:
        beg = _prop_val(el, "dPr", "begChr", "(")
        end = _prop_val(el, "dPr", "endChr", ")")
        sep = _prop_val(el, "dPr", "sepChr", "|")
        items = [self._seq(e, ctx).strip() for e in _children(el, "e")]
        if sep == "|":
            sep_latex = r" \middle| "
        elif sep in ("‖",):
            sep_latex = r" \middle\| "
        else:
            sep_latex = char_to_latex(sep or "")
        left = delimiter_to_latex(beg or "", side="left")
        right = delimiter_to_latex(end or "", side="right")
        tall = any(_TALL_RE.search(x) for x in items)
        uses_middle = len(items) > 1 and sep_latex.lstrip().startswith(r"\middle")
        if not tall and not uses_middle and left != "." and right != ".":
            # plain delimiters for ordinary content: "f(x)" rather than "f\left( x \right)"
            return join_latex([left, sep_latex.join(items), right])
        inner = sep_latex.join(items)
        return r"\left" + left + " " + inner + r" \right" + right

    def _func(self, el, ctx) -> str:
        fname_el = _child(el, "fName")
        name = self._fname(fname_el, ctx)
        body = self._arg(el, "e", ctx)
        return join_latex([name, (" " if body and not body.startswith("\\left") else "") + body])

    def _fname(self, fname_el, ctx) -> str:
        if fname_el is None:
            return ""
        # A plain text function name ("sin", "arg", "sgn") -> command form.
        plain = "".join(
            t.text or "" for t in fname_el.iter() if isinstance(t.tag, str) and _local(t) == "t"
        ).strip()
        only_runs = all(
            (_local(c) in ("r", "ctrlPr") or _local(c).endswith("Pr"))
            for c in fname_el
            if isinstance(c.tag, str)
        )
        if only_runs and plain:
            if plain in FUNCTION_NAMES:
                return "\\" + plain
            if plain in OPERATOR_NAMES or plain.isalpha():
                return r"\operatorname{" + plain + "}"
        return self._seq(fname_el, ctx)

    def _lim_low(self, el, ctx) -> str:
        base_el = _child(el, "e")
        base = self._seq(base_el, ctx).strip()
        lim = self._arg(el, "lim", ctx)
        plain = (
            "".join(
                t.text or "" for t in base_el.iter() if isinstance(t.tag, str) and _local(t) == "t"
            ).strip()
            if base_el is not None
            else ""
        )
        if plain in LIMIT_FUNCTIONS:
            return "\\" + plain + "_{" + lim + "}"
        if base.startswith(r"\underbrace") or base.startswith(r"\overbrace"):
            return base + "_{" + lim + "}"
        return r"\underset{" + lim + "}{" + base + "}"

    def _lim_upp(self, el, ctx) -> str:
        base = self._arg(el, "e", ctx)
        lim = self._arg(el, "lim", ctx)
        if base.startswith(r"\overbrace") or base.startswith(r"\underbrace"):
            return base + "^{" + lim + "}"
        return r"\overset{" + lim + "}{" + base + "}"

    def _acc(self, el, ctx) -> str:
        ch = _prop_val(el, "accPr", "chr", "̂") or "̂"
        body = self._arg(el, "e", ctx)
        cmd = ACCENTS.get(ch)
        if cmd is None:
            return r"\overset{" + char_to_latex(ch) + "}{" + body + "}"
        if len(body) > 1 and not body.startswith("\\") and cmd in WIDE_ACCENTS:
            cmd = WIDE_ACCENTS[cmd]
        return cmd + "{" + body + "}"

    def _bar(self, el, ctx) -> str:
        body = self._arg(el, "e", ctx)
        pos = _prop_val(el, "barPr", "pos", "bot")
        return (r"\overline{" if pos == "top" else r"\underline{") + body + "}"

    def _group_chr(self, el, ctx) -> str:
        body = self._arg(el, "e", ctx)
        ch = _prop_val(el, "groupChrPr", "chr", "⏟") or "⏟"
        pos = _prop_val(el, "groupChrPr", "pos", "bot")
        if ch in ("⏟", "︸"):
            return r"\underbrace{" + body + "}"
        if ch in ("⏞", "︷"):
            return r"\overbrace{" + body + "}"
        sym = char_to_latex(ch)
        return (r"\overset{" if pos == "top" else r"\underset{") + sym + "}{" + body + "}"

    def _border_box(self, el, ctx) -> str:
        return r"\boxed{" + self._arg(el, "e", ctx) + "}"

    def _box(self, el, ctx) -> str:
        return self._arg(el, "e", ctx)

    def _phant(self, el, ctx) -> str:
        body = self._arg(el, "e", ctx)
        if _prop_val(el, "phantPr", "show", "1") in ("0", "off", "false"):
            return r"\phantom{" + body + "}"
        return body

    def _eqarr(self, el, ctx) -> str:
        prev = ctx.in_eqarr
        ctx.in_eqarr = True
        try:
            rows = [self._seq(e, ctx).strip() for e in _children(el, "e")]
        finally:
            ctx.in_eqarr = prev
        # Word's equation numbering: "<equation>#(<n>)" inside an eqArr row.
        tagged = [_split_eq_number(r) for r in rows]
        if len(rows) == 1:
            body, num = tagged[0]
            return body + (r" \tag{" + num + "}" if num else "")
        rows = [body + (r" \qquad (" + num + ")" if num else "") for body, num in tagged]
        env = "aligned" if any("&" in r for r in rows) else "gathered"
        return r"\begin{" + env + "} " + r" \\ ".join(rows) + r" \end{" + env + "}"

    def _matrix(self, el, ctx) -> str:
        rows = []
        for mr in _children(el, "mr"):
            rows.append(" & ".join(self._seq(e, ctx).strip() for e in _children(mr, "e")))
        return r"\begin{matrix} " + r" \\ ".join(rows) + r" \end{matrix}"

    def _passthrough(self, el, ctx) -> str:
        return self._seq(el, ctx)

    def _word_node(self, el, ctx) -> str:
        """WordprocessingML inside math (w:r with w:t), e.g. text runs."""
        name = _local(el)
        if name == "t":
            text = el.text or ""
            if any(is_hebrew(c) or c.isalpha() for c in text):
                return r"\text{" + _escape_text_mode(text) + "}"
            return text_to_latex(text)
        if name in ("rPr", "pPr", "del", "delText", "proofErr", "bookmarkStart", "bookmarkEnd"):
            return ""
        return self._seq(el, ctx)

    _HANDLERS: dict[str, Callable[["OmmlToLatex", object, _Ctx], str]] = {
        "oMathPara": _omath_para,
        "oMath": _passthrough,
        "r": _run,
        "f": _frac,
        "sSup": _ssup,
        "sSub": _ssub,
        "sSubSup": _ssubsup,
        "sPre": _spre,
        "rad": _rad,
        "nary": _nary,
        "d": _delim,
        "func": _func,
        "limLow": _lim_low,
        "limUpp": _lim_upp,
        "acc": _acc,
        "bar": _bar,
        "groupChr": _group_chr,
        "borderBox": _border_box,
        "box": _box,
        "phant": _phant,
        "eqArr": _eqarr,
        "m": _matrix,
        "e": _passthrough,
        "num": _passthrough,
        "den": _passthrough,
        "sub": _passthrough,
        "sup": _passthrough,
        "deg": _passthrough,
        "lim": _passthrough,
        "fName": _passthrough,
    }


_TALL_RE = re.compile(
    r"\\(?:frac|dfrac|genfrac|begin|sum|int|iint|iiint|oint|prod|coprod|binom|sqrt|lim|"
    r"bigcup|bigcap|overbrace|underbrace|overset|underset|left)"
)


def _split_eq_number(row: str) -> tuple[str, str | None]:
    idx = row.rfind(r"\#")
    if idx < 0:
        return row, None
    body, num = row[:idx].strip(), row[idx + 2 :].strip()
    m = re.fullmatch(r"\\left\(\s*(.*?)\s*\\right\)|\((.*)\)|(.*)", num)
    if m:
        num = next(g for g in m.groups() if g is not None)
    num = re.sub(r"\\text\{(.*?)\}", r"\1", num).strip()
    return body, (num or None)


def _base(latex: str) -> str:
    latex = latex.strip()
    if not latex:
        return "{}"
    return group(latex) if not latex.startswith(r"\left") else "{" + latex + "}"


def _escape_text_mode(text: str) -> str:
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
