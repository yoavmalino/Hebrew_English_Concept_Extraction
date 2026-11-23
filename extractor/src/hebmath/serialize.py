"""IR -> plain text serializer.

The output grammar is specified in docs/FORMAT.md. Summary (format v2):

* Blocks are separated by exactly one empty line; the file ends with ``\\n``.
* Inline math:   ``$<latex>$``; display math: ``$$<latex>$$`` (a block of its
  own, one line) with an optional ``\\tag{<n>}`` at the end of the LaTeX.
* Inside a paragraph, a formula and the ordinary text next to it are on
  separate lines (a single line break, never inside a formula). Kept on the
  text's line: formulas that are a single symbol ($V$, $A_{n\\times n}$) and
  the punctuation that closes a formula. See ``math_line_breaks``.
* Placeholders:  ``[[hebmath:<kind>]]`` - only for content that is lost
  (``math-object``, ``unreadable-page``); images and other embedded objects
  are ignored and leave no trace.
* In prose, ``\\`` is written as ``\\\\`` and ``$`` as ``\\$``; nothing else is escaped.
* Prose is NFC-normalized, Unicode bidi control characters are removed.
"""

from __future__ import annotations

import re
import unicodedata

from .model import Block, BlockKind, Document, InlineMath, Placeholder, TextSpan

PLACEHOLDER_PREFIX = "[[hebmath:"
PLACEHOLDER_SUFFIX = "]]"

# Invisible characters that only carry presentation hints. They are removed
# because the output is already in logical order and downstream parsers should
# not have to deal with them.
_INVISIBLE = dict.fromkeys(
    [
        0x200B,  # zero width space
        0x200E,  # LRM
        0x200F,  # RLM
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,  # embeddings / overrides
        0x2066,
        0x2067,
        0x2068,
        0x2069,  # isolates
        0x00AD,  # soft hyphen
        0xFEFF,  # BOM / ZWNBSP
    ]
)

_WS_RE = re.compile(r"[ \t  -   　]+")
_NEWLINES_RE = re.compile(r"[\r\n  \v\f]+")


def _fold_presentation_forms(text: str) -> str:
    """NFKC only the Alphabetic Presentation Forms block (U+FB00–U+FB4F).

    PDFs frequently contain ligatures (U+FB01 "fi") and Hebrew presentation
    forms (U+FB2A shin-with-dot, U+FB4F aleph-lamed). Applying NFKC to the whole
    text would also destroy meaningful characters such as superscripts, so the
    compatibility mapping is restricted to that block.
    """
    if not any(0xFB00 <= ord(c) <= 0xFB4F for c in text):
        return text
    return "".join(
        unicodedata.normalize("NFKC", c) if 0xFB00 <= ord(c) <= 0xFB4F else c for c in text
    )


_LEADER_RE = re.compile(r"\s*\.{6,}\s*")


def normalize_prose(text: str) -> str:
    text = text.translate(_INVISIBLE)
    text = _fold_presentation_forms(text)
    text = unicodedata.normalize("NFC", text)
    text = _NEWLINES_RE.sub(" ", text)
    text = _LEADER_RE.sub(" … ", text)  # table-of-contents dot leaders
    text = _WS_RE.sub(" ", text)
    return text


def escape_prose(text: str) -> str:
    return text.replace("\\", "\\\\").replace("$", "\\$")


def normalize_latex(latex: str) -> str:
    latex = latex.translate(_INVISIBLE)
    latex = unicodedata.normalize("NFC", latex)
    latex = _NEWLINES_RE.sub(" ", latex)
    latex = re.sub(r"\s+", " ", latex).strip()
    # A bare "$" can never appear inside our math (it would end the formula).
    latex = re.sub(r"(?<!\\)\$", r"\\$", latex)
    return latex


# Placeholders for content that is *ignored by contract* (not lost math).
IGNORED_PLACEHOLDERS = {"image", "object"}

BLOCK_SEPARATOR = "\n\n"  # between blocks in the output text
MATH_LINE_BREAKS = ("formulas", "all", "none")

# One symbol, possibly styled/accented and with scripts: stays inside the sentence.
_SYMBOL = (
    r"(?:[A-Za-z0-9]|\d+(?:\.\d+)?|\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|"
    r"vartheta|iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|tau|upsilon|phi|varphi|chi|"
    r"psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|infty|emptyset|varnothing|ell|"
    r"hbar|partial|nabla|aleph)(?![A-Za-z])|\\(?:mathbb|mathcal|mathbf|mathrm|mathfrak|boldsymbol|"
    r"overline|bar|vec|hat|tilde|dot|ddot|widehat|widetilde)\{[A-Za-z0-9]\})"
)
_GROUP = r"(?:\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|[A-Za-z0-9])"
_SIMPLE_RE = re.compile(rf"^{_SYMBOL}(?:[_^]{_GROUP})*'*$")
_LEADING_PUNCT_RE = re.compile(r"^\s*([.,;:!?%)\]}\u05F3\u05F4'\"]+)(?=\s|$)")
_TRAILING_OPEN_RE = re.compile(r"([(\[{]+)\s*$")
_MARKER_ONLY_RE = re.compile(r"^(?:\d{1,3}|[א-ת]{1,2}|[a-z])[.)]$")
_HEBREW_PREFIX_RE = re.compile(r"(?:^|\s)([\u05D0-\u05EA]{1,3}[-\u05BE])$")
# The same prefix when the PDF set a space before the hyphen ("ה -" + formula):
# only the prefix letters ו ה ב כ ל מ ש, so a real dash ("זה - x") is not glued.
_SPACED_PREFIX_RE = re.compile(
    r"(?:^|\s)((?:[\u05D5\u05D4\u05D1\u05DB\u05DC\u05DE\u05E9]){1,3}) ([-\u05BE])$"
)


def is_simple_math(latex: str) -> bool:
    """A lone symbol ($V$, $\\lambda$, $A_{n\\times n}$, $\\mathbb{R}$) rather than a formula."""
    return bool(_SIMPLE_RE.match(latex.replace(" ", "")))


def _paragraph_pieces(block: Block) -> list[tuple[str, str]]:
    """("text"|"math"|"marker", rendered) pieces of a paragraph."""
    pieces: list[tuple[str, str]] = []
    for seg in block.segments:
        if isinstance(seg, TextSpan):
            pieces.append(("text", escape_prose(normalize_prose(seg.text))))
        elif isinstance(seg, InlineMath):
            latex = re.sub(r"\s*\\tag\{[^{}]*\}", "", normalize_latex(seg.latex)).strip()
            if latex:
                pieces.append(("math", latex))
        elif isinstance(seg, Placeholder):
            if seg.kind not in IGNORED_PLACEHOLDERS:
                pieces.append(("marker", f"{PLACEHOLDER_PREFIX}{seg.kind}{PLACEHOLDER_SUFFIX}"))
    return pieces


def _serialize_paragraph(block: Block, math_line_breaks: str = "formulas") -> str:
    """One paragraph; formulas and the text around them go on separate lines.

    Rules (format v2, docs/FORMAT.md):
    * a formula gets its own line, unless it is a single symbol (policy
      "formulas") or the policy is "none"; with "all" every formula does;
    * punctuation that closes the sentence/formula right after it (".", ",",
      ")" ...) stays on the formula's line, an opening bracket just before
      it moves onto it - so a line never starts with "." or ends with "(";
    * a list number ("1.", "א.") directly before a formula stays in front of it,
      and so does a Hebrew prefix written with a hyphen ("ו-$x$", "כש-$n>1$");
    * nothing is ever split *inside* a formula, and a table row stays on one line.
    """
    if block.table_row:
        math_line_breaks = "none"  # a table row stays one line: "cell | cell"
    lines: list[str] = []
    cur = ""
    own_math_open = False  # ``cur`` holds a formula that wants its own line

    def flush() -> None:
        nonlocal cur
        text = re.sub(r" {2,}", " ", cur).strip()
        if text:
            lines.append(text)
        cur = ""

    for kind, value in _paragraph_pieces(block):
        if kind == "math":
            rendered = f"${value}$"
            own = math_line_breaks == "all" or (
                math_line_breaks == "formulas" and not is_simple_math(value)
            )
            if not own:
                if own_math_open:
                    flush()
                    own_math_open = False
                cur += rendered
                continue
            prefix = ""
            m = _TRAILING_OPEN_RE.search(cur)
            if m:
                prefix, cur = m.group(1), cur[: m.start()]
            m = _HEBREW_PREFIX_RE.search(cur.rstrip())
            if m and not prefix:
                # "ו-$x$", "כש-$n>1$": a Hebrew prefix is grammatically part of the formula
                prefix, cur = m.group(1), cur[: m.start(1)]
            m = _SPACED_PREFIX_RE.search(cur.rstrip())
            if m and not prefix:
                prefix, cur = m.group(1) + m.group(2), cur[: m.start(1)]
            stripped = re.sub(r" {2,}", " ", cur).strip()
            if _MARKER_ONLY_RE.match(stripped) and not lines:
                prefix, cur = stripped + " " + prefix, ""
            if own_math_open and not stripped:
                cur = cur.rstrip() + " " + prefix + rendered  # consecutive formulas share a line
            else:
                flush()
                cur = prefix + rendered
            own_math_open = True
            continue
        if own_math_open:
            m = _LEADING_PUNCT_RE.match(value)
            if m:
                cur += m.group(1)
                value = value[m.end() :]
            if value.strip():
                flush()
                own_math_open = False
                cur = value
            else:
                cur += value
            continue
        cur += value
    flush()
    return "\n".join(lines)


def _serialize_display(block: Block) -> str:
    latex = normalize_latex(block.latex)
    if not latex:
        return ""
    if block.tag:
        latex = f"{latex} \\tag{{{normalize_latex(block.tag)}}}"
    return f"$${latex}$$"


def serialize(
    document: Document, *, page_markers: bool = False, math_line_breaks: str = "formulas"
) -> str:
    """Render the document to the documented text format. Pure and deterministic.

    ``math_line_breaks``: "formulas" (default) puts every formula that is more
    than a single symbol on its own line; "all" does so for every formula;
    "none" keeps formulas inside the text line (format v1 behaviour).
    """
    return serialize_with_spans(
        document, page_markers=page_markers, math_line_breaks=math_line_breaks
    )[0]


def serialize_with_spans(
    document: Document, *, page_markers: bool = False, math_line_breaks: str = "formulas"
) -> tuple[str, list[tuple[int, int, int]]]:
    """Render the document and report where each block landed in the text.

    Returns:
        The same text as ``serialize``, and one ``(block_index, start, end)`` per block that
        produced output, with ``text[start:end]`` equal to that block's rendering.
    """
    if math_line_breaks not in MATH_LINE_BREAKS:
        raise ValueError(f"math_line_breaks must be one of {MATH_LINE_BREAKS}")
    out: list[str] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for index, block in enumerate(document.blocks):
        if block.kind is BlockKind.PARAGRAPH:
            text = _serialize_paragraph(block, math_line_breaks)
        elif block.kind is BlockKind.DISPLAY_MATH:
            text = _serialize_display(block)
        elif block.kind is BlockKind.PAGE_BREAK:
            text = (
                f"{PLACEHOLDER_PREFIX}page={block.page_number}{PLACEHOLDER_SUFFIX}"
                if page_markers
                else ""
            )
        else:  # pragma: no cover - exhaustive enum
            text = ""
        if text:
            if out:
                offset += len(BLOCK_SEPARATOR)
            spans.append((index, offset, offset + len(text)))
            offset += len(text)
            out.append(text)
    if not out:
        return "", []
    return BLOCK_SEPARATOR.join(out) + "\n", spans
