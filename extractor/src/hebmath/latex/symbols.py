"""Unicode -> LaTeX mapping shared by all math engines.

Only standard LaTeX + amsmath/amssymb commands are emitted, so the output
compiles with a stock TeX distribution and renders in KaTeX/MathJax.
"""

from __future__ import annotations

import re
import unicodedata

# Characters with a dedicated LaTeX command (math mode).
SYMBOLS: dict[str, str] = {
    # Greek lowercase
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\varepsilon",
    "ϵ": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ϑ": r"\vartheta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "µ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "ο": "o",
    "π": r"\pi",
    "ϖ": r"\varpi",
    "ρ": r"\rho",
    "ϱ": r"\varrho",
    "σ": r"\sigma",
    "ς": r"\varsigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\varphi",
    "ϕ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    # Greek uppercase (those that differ from Latin)
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Υ": r"\Upsilon",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Χ": "X",
    # Binary operators
    "±": r"\pm",
    "∓": r"\mp",
    "×": r"\times",
    "÷": r"\div",
    "·": r"\cdot",
    "⋅": r"\cdot",
    "∙": r"\cdot",
    "∗": "*",
    "−": "-",
    "–": "-",
    "∘": r"\circ",
    "⊕": r"\oplus",
    "⊗": r"\otimes",
    "⊖": r"\ominus",
    "⊙": r"\odot",
    "∪": r"\cup",
    "∩": r"\cap",
    "∧": r"\wedge",
    "∨": r"\vee",
    "∖": r"\setminus",
    "⋆": r"\star",
    "†": r"\dagger",
    "⊎": r"\uplus",
    "⊓": r"\sqcap",
    "⊔": r"\sqcup",
    "≀": r"\wr",
    "⨯": r"\times",
    # Relations
    "≤": r"\le",
    "≥": r"\ge",
    "⩽": r"\leqslant",
    "⩾": r"\geqslant",
    "≠": r"\neq",
    "≈": r"\approx",
    "≡": r"\equiv",
    "≅": r"\cong",
    "∼": r"\sim",
    "≃": r"\simeq",
    "∝": r"\propto",
    "≪": r"\ll",
    "≫": r"\gg",
    "∈": r"\in",
    "∉": r"\notin",
    "∋": r"\ni",
    "⊂": r"\subset",
    "⊃": r"\supset",
    "⊆": r"\subseteq",
    "⊇": r"\supseteq",
    "⊄": r"\not\subset",
    "⊊": r"\subsetneq",
    "⊋": r"\supsetneq",
    "⊥": r"\perp",
    "∥": r"\|",
    "∣": r"\mid",
    "∤": r"\nmid",
    "⊢": r"\vdash",
    "⊨": r"\models",
    "≔": r"\coloneqq",
    "≜": r"\triangleq",
    "≺": r"\prec",
    "≻": r"\succ",
    "≍": r"\asymp",
    "≐": r"\doteq",
    "≮": r"\not<",
    "≯": r"\not>",
    "≰": r"\nleq",
    "≱": r"\ngeq",
    "⊲": r"\lhd",
    "⊳": r"\rhd",
    "∽": r"\backsim",
    # Arrows
    "→": r"\to",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "⇔": r"\Leftrightarrow",
    "↦": r"\mapsto",
    "⟶": r"\longrightarrow",
    "⟵": r"\longleftarrow",
    "⟹": r"\Longrightarrow",
    "⟸": r"\Longleftarrow",
    "⟺": r"\Longleftrightarrow",
    "↑": r"\uparrow",
    "↓": r"\downarrow",
    "⇑": r"\Uparrow",
    "⇓": r"\Downarrow",
    "↗": r"\nearrow",
    "↘": r"\searrow",
    "↪": r"\hookrightarrow",
    "⇀": r"\rightharpoonup",
    "⇌": r"\rightleftharpoons",
    "⟼": r"\longmapsto",
    # Logic / sets / misc
    "∀": r"\forall",
    "∃": r"\exists",
    "∄": r"\nexists",
    "¬": r"\neg",
    "∅": r"\emptyset",
    "∞": r"\infty",
    "∂": r"\partial",
    "∇": r"\nabla",
    "ℵ": r"\aleph",
    "ℏ": r"\hbar",
    "ℓ": r"\ell",
    "℘": r"\wp",
    "ℜ": r"\Re",
    "ℑ": r"\Im",
    "∴": r"\therefore",
    "∵": r"\because",
    "…": r"\ldots",
    "⋯": r"\cdots",
    "⋮": r"\vdots",
    "⋱": r"\ddots",
    "′": "'",
    "″": "''",
    "‴": "'''",
    "°": r"^{\circ}",
    "∠": r"\angle",
    "△": r"\triangle",
    "□": r"\square",
    "■": r"\blacksquare",
    "◦": r"\circ",
    "♯": r"\sharp",
    "♭": r"\flat",
    "⊤": r"\top",
    "∎": r"\blacksquare",
    "⌀": r"\varnothing",
    "∡": r"\measuredangle",
    "√": r"\sqrt{}",
    "∛": r"\sqrt[3]{}",
    "∜": r"\sqrt[4]{}",
    # Double-struck / script letterlike symbols
    "ℝ": r"\mathbb{R}",
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}",
    "ℂ": r"\mathbb{C}",
    "ℙ": r"\mathbb{P}",
    "ℍ": r"\mathbb{H}",
    "ℒ": r"\mathcal{L}",
    "ℱ": r"\mathcal{F}",
    "ℋ": r"\mathcal{H}",
    "ℐ": r"\mathcal{I}",
    "ℛ": r"\mathcal{R}",
    "ℬ": r"\mathcal{B}",
    "ℰ": r"\mathcal{E}",
    "ℳ": r"\mathcal{M}",
    "ℯ": "e",
    "ℊ": "g",
    "ℴ": "o",
    "ℭ": r"\mathfrak{C}",
    "ℌ": r"\mathfrak{H}",
    "ℨ": r"\mathfrak{Z}",
    "ⅅ": r"\mathbb{D}",
    "ⅆ": "d",
    "ⅇ": "e",
    "ⅈ": "i",
    "ⅉ": "j",
    # Delimiters
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "〈": r"\langle",
    "〉": r"\rangle",
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
    "‖": r"\|",
    "{": r"\{",
    "}": r"\}",
    # Characters that are special in LaTeX
    "#": r"\#",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
    "~": r"\sim",
    "\\": r"\backslash",
    "$": r"\$",
    "^": r"\hat{}",
    # Large operators (when they occur as plain characters)
    "∑": r"\sum",
    "∏": r"\prod",
    "∐": r"\coprod",
    "∫": r"\int",
    "∬": r"\iint",
    "∭": r"\iiint",
    "∮": r"\oint",
    "∯": r"\oiint",
    "⋃": r"\bigcup",
    "⋂": r"\bigcap",
    "⋁": r"\bigvee",
    "⋀": r"\bigwedge",
    "⨁": r"\bigoplus",
    "⨂": r"\bigotimes",
    "⨀": r"\bigodot",
    "⨄": r"\biguplus",
    # Quotes that sometimes leak into equations
    "’": "'",
    "‘": "`",
}

BIG_OPERATORS = {
    "∑": r"\sum",
    "∏": r"\prod",
    "∐": r"\coprod",
    "∫": r"\int",
    "∬": r"\iint",
    "∭": r"\iiint",
    "∮": r"\oint",
    "∯": r"\oiint",
    "⋃": r"\bigcup",
    "⋂": r"\bigcap",
    "⋁": r"\bigvee",
    "⋀": r"\bigwedge",
    "⨁": r"\bigoplus",
    "⨂": r"\bigotimes",
    "⨀": r"\bigodot",
    "⨄": r"\biguplus",
}

# Operators whose limits LaTeX places as sub/superscripts (\int_a^b).
INTEGRAL_OPERATORS = {r"\int", r"\iint", r"\iiint", r"\oint", r"\oiint"}

SUPERSCRIPTS = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ⁿ": "n",
    "ⁱ": "i",
    "ᵀ": "T",
    "ᵃ": "a",
    "ᵇ": "b",
    "ᶜ": "c",
    "ᵈ": "d",
    "ᵉ": "e",
    "ᵏ": "k",
    "ᵐ": "m",
    "ᵒ": "o",
    "ᵖ": "p",
    "ʳ": "r",
    "ˢ": "s",
    "ᵗ": "t",
    "ᵘ": "u",
    "ᵛ": "v",
    "ʷ": "w",
    "ˣ": "x",
    "ʸ": "y",
    "ᶻ": "z",
}
SUBSCRIPTS = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
    "ₐ": "a",
    "ₑ": "e",
    "ₒ": "o",
    "ₓ": "x",
    "ₕ": "h",
    "ₖ": "k",
    "ₗ": "l",
    "ₘ": "m",
    "ₙ": "n",
    "ₚ": "p",
    "ₛ": "s",
    "ₜ": "t",
    "ᵢ": "i",
    "ⱼ": "j",
    "ᵣ": "r",
    "ᵤ": "u",
    "ᵥ": "v",
}

# Combining accents (used by OMML m:acc and by some PDFs).
ACCENTS = {
    "̂": r"\hat",
    "̃": r"\tilde",
    "̄": r"\bar",
    "̅": r"\overline",
    "̇": r"\dot",
    "̈": r"\ddot",
    "̌": r"\check",
    "́": r"\acute",
    "̀": r"\grave",
    "̆": r"\breve",
    "⃗": r"\vec",
    "⃖": r"\overleftarrow",
    "⃡": r"\overleftrightarrow",
    "⃛": r"\dddot",
    "̊": r"\mathring",
    "^": r"\hat",
    "~": r"\tilde",
    "¯": r"\bar",
    "→": r"\vec",
    "˙": r"\dot",
    "¨": r"\ddot",
    "ˇ": r"\check",
    "˜": r"\tilde",
    "ˆ": r"\hat",
}
WIDE_ACCENTS = {r"\hat": r"\widehat", r"\tilde": r"\widetilde", r"\bar": r"\overline"}

# Function names typeset upright with a dedicated command.
FUNCTION_NAMES = {
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "coth",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "log",
    "ln",
    "lg",
    "exp",
    "lim",
    "liminf",
    "limsup",
    "max",
    "min",
    "sup",
    "inf",
    "det",
    "dim",
    "ker",
    "deg",
    "gcd",
    "arg",
    "hom",
    "Pr",
}
# Function names that take limits underneath in display style.
LIMIT_FUNCTIONS = {"lim", "liminf", "limsup", "max", "min", "sup", "inf", "det", "gcd", "Pr"}
# Not standard LaTeX commands -> \operatorname.
OPERATOR_NAMES = {
    "sgn",
    "rank",
    "tr",
    "span",
    "Var",
    "Cov",
    "lcm",
    "Im",
    "Re",
    "diag",
    "mod",
    "Sp",
    "Span",
    "trace",
    "adj",
    "cis",
    "Ker",
    "id",
    "argmax",
    "argmin",
    "Tr",
    "sign",
}

_FUNC_RE = re.compile("|".join(sorted(FUNCTION_NAMES | OPERATOR_NAMES, key=lambda s: (-len(s), s))))

_STYLE_WORDS = [
    ("BOLD ITALIC", r"\boldsymbol"),
    ("BOLD SCRIPT", r"\mathcal"),
    ("BOLD FRAKTUR", r"\mathfrak"),
    ("SANS-SERIF BOLD ITALIC", r"\mathsf"),
    ("SANS-SERIF BOLD", r"\mathsf"),
    ("SANS-SERIF ITALIC", r"\mathsf"),
    ("SANS-SERIF", r"\mathsf"),
    ("DOUBLE-STRUCK ITALIC", r"\mathbb"),
    ("DOUBLE-STRUCK", r"\mathbb"),
    ("MONOSPACE", r"\mathtt"),
    ("FRAKTUR", r"\mathfrak"),
    ("SCRIPT", r"\mathcal"),
    ("BOLD", r"\mathbf"),
    ("ITALIC", None),  # math italic is the default for letters
]

_GREEK_NAMES = {
    "ALPHA": "α",
    "BETA": "β",
    "GAMMA": "γ",
    "DELTA": "δ",
    "EPSILON": "ε",
    "ZETA": "ζ",
    "ETA": "η",
    "THETA": "θ",
    "IOTA": "ι",
    "KAPPA": "κ",
    "LAMDA": "λ",
    "MU": "μ",
    "NU": "ν",
    "XI": "ξ",
    "OMICRON": "ο",
    "PI": "π",
    "RHO": "ρ",
    "SIGMA": "σ",
    "TAU": "τ",
    "UPSILON": "υ",
    "PHI": "φ",
    "CHI": "χ",
    "PSI": "ψ",
    "OMEGA": "ω",
    "NABLA": "∇",
    "PARTIAL DIFFERENTIAL": "∂",
    "FINAL SIGMA": "ς",
    "EPSILON SYMBOL": "ϵ",
    "THETA SYMBOL": "ϑ",
    "PHI SYMBOL": "ϕ",
    "RHO SYMBOL": "ϱ",
    "PI SYMBOL": "ϖ",
    "KAPPA SYMBOL": "ϰ",
}


def math_alphanumeric(ch: str) -> tuple[str, str | None] | None:
    """Decompose a U+1D400 block character into (base character, style command)."""
    cp = ord(ch)
    if not (0x1D400 <= cp <= 0x1D7FF):
        return None
    name = unicodedata.name(ch, "")
    if not name.startswith("MATHEMATICAL "):
        return None
    rest = name[len("MATHEMATICAL ") :]
    style: str | None = None
    for words, cmd in _STYLE_WORDS:
        if rest.startswith(words + " "):
            style = cmd
            rest = rest[len(words) + 1 :]
            break
    m = re.fullmatch(r"(CAPITAL|SMALL) ([A-Z])", rest)
    if m:
        base = m.group(2) if m.group(1) == "CAPITAL" else m.group(2).lower()
        return base, style
    m = re.fullmatch(r"DIGIT (ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE)", rest)
    if m:
        digits = "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE".split()
        return str(digits.index(m.group(1))), style
    m = re.fullmatch(r"(CAPITAL|SMALL) (.+)", rest)
    if m and m.group(2) in _GREEK_NAMES:
        g = _GREEK_NAMES[m.group(2)]
        return (g.upper() if m.group(1) == "CAPITAL" else g), style
    if rest in _GREEK_NAMES:
        return _GREEK_NAMES[rest], style
    return None


def is_hebrew(ch: str) -> bool:
    cp = ord(ch)
    return 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F


def char_to_latex(ch: str) -> str:
    """Map a single character to math-mode LaTeX."""
    if ch in SYMBOLS:
        return SYMBOLS[ch]
    if ch.isascii():
        if ch in "{}":
            return "\\" + ch
        return ch
    alnum = math_alphanumeric(ch)
    if alnum is not None:
        base, style = alnum
        inner = SYMBOLS.get(base, base)
        return f"{style}{{{inner}}}" if style else inner
    if ch in SUPERSCRIPTS:
        return "^{" + SUPERSCRIPTS[ch] + "}"
    if ch in SUBSCRIPTS:
        return "_{" + SUBSCRIPTS[ch] + "}"
    if ch.isspace():
        return " "
    if is_hebrew(ch) or unicodedata.category(ch).startswith("L"):
        return "\\text{" + ch + "}"
    # Unknown symbol: keep the character itself (valid in unicode-math / KaTeX)
    return ch


def text_to_latex(text: str, *, upright: bool = False) -> str:
    """Map a run of characters (one OMML m:t or one PDF glyph group) to LaTeX.

    Multi-letter upright runs become ``\\sin``/``\\operatorname{...}``/``\\mathrm{...}``.
    Hebrew (and other non-Latin letters) are wrapped in ``\\text{}``.
    """
    if not text:
        return ""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # Hebrew / non-math words -> one \text{} group (keeps inner spaces)
        if is_hebrew(ch):
            j = i
            while j < n and (
                is_hebrew(text[j]) or (text[j] in " '\"-־" and j + 1 < n and is_hebrew(text[j + 1]))
            ):
                j += 1
            out.append("\\text{" + text[i:j] + "}")
            i = j
            continue
        if ch.isascii() and ch.isalpha():
            j = i
            while j < n and text[j].isascii() and text[j].isalpha():
                j += 1
            out.append(_latin_run_to_latex(text[i:j], upright=upright))
            i = j
            continue
        if ch.isspace():
            i += 1
            continue
        out.append(char_to_latex(ch))
        i += 1
    return join_latex(out)


def _latin_run_to_latex(word: str, *, upright: bool) -> str:
    if word in FUNCTION_NAMES:
        return "\\" + word
    if word in OPERATOR_NAMES:
        return "\\operatorname{" + word + "}"
    if upright and len(word) > 1:
        return "\\mathrm{" + word + "}"
    # Split embedded function names: "xsinx" stays as letters unless upright.
    if upright:
        return "\\mathrm{" + word + "}"
    return word


def split_function_names(word: str) -> list[str]:
    """Greedy split of a letter run into function names and single letters."""
    parts: list[str] = []
    i = 0
    while i < len(word):
        m = _FUNC_RE.match(word, i)
        if m and (
            m.end() == len(word) or not word[m.end()].islower() or m.group(0) in FUNCTION_NAMES
        ):
            parts.append(m.group(0))
            i = m.end()
        else:
            parts.append(word[i])
            i += 1
    return parts


_CMD_TAIL = re.compile(r"\\[A-Za-z]+$")


def join_latex(parts: list[str]) -> str:
    """Concatenate LaTeX fragments, inserting a space only where required
    (after a control word that would otherwise absorb the next letter)."""
    out = ""
    for p in parts:
        if not p:
            continue
        if out and _CMD_TAIL.search(out) and p[0].isalpha():
            out += " "
        out += p
    return out


def group(latex: str) -> str:
    """Brace a fragment for use as a script/argument unless it is atomic."""
    if re.fullmatch(r"[A-Za-z0-9]|\\[A-Za-z]+", latex):
        return latex
    return "{" + latex + "}"


def delimiter_to_latex(ch: str, *, side: str) -> str:
    """Delimiter for \\left / \\right; empty string means an invisible fence."""
    if ch == "":
        return "."
    mapping = {
        "(": "(",
        ")": ")",
        "[": "[",
        "]": "]",
        "{": r"\{",
        "}": r"\}",
        "|": "|",
        "‖": r"\|",
        "⟨": r"\langle",
        "⟩": r"\rangle",
        "〈": r"\langle",
        "〉": r"\rangle",
        "⌈": r"\lceil",
        "⌉": r"\rceil",
        "⌊": r"\lfloor",
        "⌋": r"\rfloor",
        "/": "/",
        "\\": r"\backslash",
        "↑": r"\uparrow",
        "↓": r"\downarrow",
    }
    return mapping.get(ch, ".")
