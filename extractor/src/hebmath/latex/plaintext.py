"""Detection and conversion of math written as *plain characters*.

Used where no structural math markup exists: typed formulas in DOCX runs
("a² + b² = c²"), OCR output, and as the token classifier of the PDF engine.

The detector is deliberately conservative: a span becomes math only when it
contains at least one unambiguous *seed* character (a relation, a math symbol,
a Greek letter, a super/subscript digit...). Prose is never converted because
it merely contains digits or Latin words.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .symbols import (
    BIG_OPERATORS,
    FUNCTION_NAMES,
    OPERATOR_NAMES,
    SUBSCRIPTS,
    SUPERSCRIPTS,
    SYMBOLS,
    char_to_latex,
    is_hebrew,
    join_latex,
    math_alphanumeric,
    split_function_names,
)

ASCII_SEEDS = set("=<>^")
# Characters that are math-like but common in prose; they never seed a span.
_NON_SEED = set("·•…°%‰′″’‘~–") | {"·"}
_BULLETS = set("•▪◦●○■□➢✓✔►–") | {chr(c) for c in range(0xF000, 0xF100)}

FUNC_WORDS = FUNCTION_NAMES | OPERATOR_NAMES
_MATHISH_ASCII = set("0123456789.,()[]{}+-*/^_=<>|!'")


def is_seed_char(ch: str) -> bool:
    if ch in _NON_SEED or ch in _BULLETS:
        return False
    if ch in ASCII_SEEDS:
        return True
    if ch.isascii():
        return False
    if ch in SUPERSCRIPTS or ch in SUBSCRIPTS or ch in BIG_OPERATORS:
        return True
    if math_alphanumeric(ch) is not None:
        return True
    cp = ord(ch)
    if 0x0391 <= cp <= 0x03C9 or ch in "ϑϕϵϱϖ":  # Greek letters
        return True
    if 0x2190 <= cp <= 0x22FF or 0x27C0 <= cp <= 0x27FF or 0x2980 <= cp <= 0x2AFF:
        return True
    if ch in "ℝℕℤℚℂℏℓ℘ℵ√∛∜±×÷":
        return True
    return unicodedata.category(ch) == "Sm"


def is_mathish_char(ch: str) -> bool:
    """Characters allowed inside a math span (beyond seeds)."""
    if is_seed_char(ch):
        return True
    if ch.isascii():
        return ch.isalnum() or ch in _MATHISH_ASCII
    return ch in SYMBOLS and not is_hebrew(ch)


def token_has_seed(token: str) -> bool:
    return any(is_seed_char(c) for c in token)


def is_mathish_token(token: str, *, italic: bool = False) -> bool:
    """Whether a whitespace-delimited token may belong to a math span.

    Pure-letter tokens are accepted only if they are a single letter, a known
    function name, or typeset in an italic/math font (``italic``), so English
    words adjacent to an equals sign ("if x = 2") stay prose.
    """
    if not token or any(is_hebrew(c) for c in token):
        return False
    if not all(is_mathish_char(c) for c in token):
        return False
    letters = re.findall(r"[A-Za-z]+", token)
    if token.isascii() and token.isalpha():
        return len(token) == 1 or token in FUNC_WORDS or (italic and len(token) <= 3)
    for run in letters:
        if len(run) <= 2:
            continue
        parts = split_function_names(run)
        singles = sum(1 for p in parts if len(p) == 1 and p not in FUNC_WORDS)
        if not any(p in FUNC_WORDS for p in parts) or singles > 2:
            # an ordinary word ("holds", "and") - unless typeset in a math italic font
            if not (italic and len(run) <= 3):
                return False
    return True


def split_script_boundaries(token: str) -> list[str]:
    """Split "ב-x" / "וה-f(x)" into Hebrew and non-Hebrew parts (logical order).

    A hyphen or maqaf adjacent to Hebrew stays with the Hebrew part.
    """
    if not token:
        return []
    parts: list[str] = []
    cur = token[0]
    cur_heb = is_hebrew(token[0])
    for i, ch in enumerate(token[1:], start=1):
        heb = is_hebrew(ch)
        if ch in "-־" and cur_heb:
            heb = True
        if heb != cur_heb and not (ch in "\"'" and cur_heb):
            parts.append(cur)
            cur, cur_heb = ch, heb
        else:
            cur += ch
    parts.append(cur)
    return parts


# --------------------------------------------------------------------------
# Linear (one-line) math -> LaTeX
# --------------------------------------------------------------------------

_TRAIL_PUNCT = ".,;:!?"


def linear_to_latex(expr: str) -> str:
    """Convert a one-line plain-text formula ("f(x)=x²+√2", "a^2+b^2") to LaTeX."""
    out: list[str] = []
    i, n = 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in SUPERSCRIPTS or ch in SUBSCRIPTS:
            table, mark = (SUPERSCRIPTS, "^") if ch in SUPERSCRIPTS else (SUBSCRIPTS, "_")
            j = i
            buf = ""
            while j < n and expr[j] in table:
                buf += table[expr[j]]
                j += 1
            out.append(mark + "{" + buf + "}")
            i = j
            continue
        if ch in "^_" and i + 1 < n:
            j = i + 1
            if expr[j] == "{":
                depth, k = 0, j
                while k < n:
                    depth += expr[k] == "{"
                    depth -= expr[k] == "}"
                    if depth == 0:
                        break
                    k += 1
                inner = expr[j + 1 : k]
                out.append(ch + "{" + linear_to_latex(inner) + "}")
                i = k + 1
            elif expr[j] == "(":
                depth, k = 0, j
                while k < n:
                    depth += expr[k] == "("
                    depth -= expr[k] == ")"
                    if depth == 0:
                        break
                    k += 1
                out.append(ch + "{" + linear_to_latex(expr[j + 1 : k]) + "}")
                i = k + 1
            else:
                k = j
                if expr[k] in "+-−":
                    k += 1
                while k < n and (expr[k].isalnum() and expr[k].isascii()):
                    k += 1
                if k == j:
                    k = j + 1
                out.append(ch + "{" + linear_to_latex(expr[j:k]) + "}")
                i = k
            continue
        if ch in "√∛∜":
            root = {"√": "", "∛": "[3]", "∜": "[4]"}[ch]
            j = i + 1
            if j < n and expr[j] == "(":
                depth, k = 0, j
                while k < n:
                    depth += expr[k] == "("
                    depth -= expr[k] == ")"
                    if depth == 0:
                        break
                    k += 1
                arg = linear_to_latex(expr[j + 1 : k])
                i = k + 1
            else:
                k = j
                while k < n and (expr[k].isalnum() or expr[k] == "."):
                    k += 1
                arg = linear_to_latex(expr[j:k])
                i = k
            out.append(r"\sqrt" + root + "{" + arg + "}")
            continue
        if ch.isascii() and ch.isalpha():
            j = i
            while j < n and expr[j].isascii() and expr[j].isalpha():
                j += 1
            for part in split_function_names(expr[i:j]):
                if part in FUNCTION_NAMES:
                    out.append("\\" + part)
                elif part in OPERATOR_NAMES:
                    out.append(r"\operatorname{" + part + "}")
                else:
                    out.append(part)
            i = j
            continue
        if ch == "*":
            out.append(r"\cdot")
            i += 1
            continue
        out.append(char_to_latex(ch))
        i += 1
    return join_latex(out)


@dataclass(frozen=True)
class TaggedPiece:
    text: str
    is_math: bool


def tag_plain_text(text: str) -> list[TaggedPiece]:
    """Split logical-order prose into prose and math pieces.

    Whitespace is preserved in the prose pieces so the caller can rebuild the
    original spacing.
    """
    if not any(is_seed_char(c) for c in text):
        return [TaggedPiece(text, False)] if text else []

    # Tokenize keeping whitespace, then split tokens at Hebrew boundaries.
    raw = re.findall(r"\s+|\S+", text)
    toks: list[str] = []
    for t in raw:
        toks.extend([t] if t.isspace() else split_script_boundaries(t))

    flags = [(not t.isspace()) and is_mathish_token(t) for t in toks]

    pieces: list[TaggedPiece] = []
    i = 0
    while i < len(toks):
        if not flags[i]:
            pieces.append(TaggedPiece(toks[i], False))
            i += 1
            continue
        # Grow a run of math-ish tokens, allowing whitespace in between.
        j = i
        last_math = i
        while j < len(toks) and (flags[j] or toks[j].isspace()):
            if flags[j]:
                last_math = j
            j += 1
        run = "".join(toks[i : last_math + 1])
        run, lead, tail = _balance(run)
        if lead:
            pieces.append(TaggedPiece(lead, False))
        if token_has_seed(run) and _has_operand(run) and _balanced(run):
            core, trail = _strip_trailing_punct(run)
            pieces.append(TaggedPiece(core, True))
            if trail:
                pieces.append(TaggedPiece(trail, False))
        else:
            pieces.append(TaggedPiece(run, False))
        if tail:
            pieces.append(TaggedPiece(tail, False))
        i = last_math + 1
    return _merge(pieces)


def _balanced(run: str) -> bool:
    """Brackets balance and the formula neither starts nor ends with an operator."""
    core = run.strip()
    if not core or core[-1] in "=<>+-−*/^_≤≥≠≈" or core[0] in "=<>≤≥≠≈":
        return False
    if core.count("|") % 2:
        return False
    depth = 0
    for c in run:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _balance(run: str) -> tuple[str, str, str]:
    """Detach unmatched brackets at the edges ("(x=2" inside "(כאשר x=2)")."""
    lead = tail = ""
    while (
        run and run[0] in "([" and run.count("(") + run.count("[") > run.count(")") + run.count("]")
    ):
        lead += run[0]
        run = run[1:]
    while (
        run
        and run[-1] in ")]"
        and run.count(")") + run.count("]") > run.count("(") + run.count("[")
    ):
        tail = run[-1] + tail
        run = run[:-1]
    return run, lead, tail


def _has_operand(run: str) -> bool:
    """Reject runs that are only operators (e.g. a lone "=" or "->" arrow)."""
    return any(
        c.isalnum()
        or math_alphanumeric(c)
        or c in SUPERSCRIPTS
        or c in SUBSCRIPTS
        or (0x0391 <= ord(c) <= 0x03C9)
        or c in "∞∅ℝℕℤℚℂ"
        for c in run
    )


def _strip_trailing_punct(run: str) -> tuple[str, str]:
    core = run.rstrip(_TRAIL_PUNCT)
    # keep a trailing "." that is part of a number like "3." only if followed by digit (never here)
    if core.count("(") > core.count(")") and run[len(core) :].startswith(")"):
        return run, ""
    return core, run[len(core) :]


def _merge(pieces: list[TaggedPiece]) -> list[TaggedPiece]:
    out: list[TaggedPiece] = []
    for p in pieces:
        if not p.text:
            continue
        if out and out[-1].is_math == p.is_math and not p.is_math:
            out[-1] = TaggedPiece(out[-1].text + p.text, False)
        else:
            out.append(p)
    return out
