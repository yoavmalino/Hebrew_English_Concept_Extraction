"""Visual -> logical reordering for lines extracted from PDF.

PDF stores glyphs at positions on the page; sorting a line's glyphs by x gives
the *visual* order. To recover the *logical* (reading/storage) order of a
right-to-left line we apply the reordering phase of the Unicode Bidirectional
Algorithm (UAX #9) to the visual sequence. For the single-level embeddings
found in real documents (RTL paragraph with LTR runs of Latin, numbers and
formulas) the UBA reordering is an involution, so applying it to the visual
sequence yields the logical one.

This is a self-contained implementation of UAX #9 rules W1-W7, N1-N2, I1-I2
and L2 (no explicit embeddings: PDFs do not contain them), using only the
bidi classes from Python's ``unicodedata``. It deliberately avoids
python-bidi (LGPL) and ICU bindings.

Units can be single characters or opaque objects (a formula): opaque objects
are treated as a strong LTR character.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

T = TypeVar("T")

_STRONG_R = {"R", "AL"}
_NEUTRAL = {"B", "S", "WS", "ON", "BN", "LRI", "RLI", "FSI", "PDI"}

# Bidi_Mirroring_Glyph for the common pairs (UCD BidiMirroring.txt subset).
MIRROR = {
    "(": ")",
    ")": "(",
    "[": "]",
    "]": "[",
    "{": "}",
    "}": "{",
    "<": ">",
    ">": "<",
    "«": "»",
    "»": "«",
    "‹": "›",
    "›": "‹",
    "⟨": "⟩",
    "⟩": "⟨",
    "≤": "≥",
    "≥": "≤",
    "⊂": "⊃",
    "⊃": "⊂",
    "⊆": "⊇",
    "⊇": "⊆",
    "∈": "∋",
    "∋": "∈",
    "⌈": "⌉",
    "⌉": "⌈",
    "⌊": "⌋",
    "⌋": "⌊",
    "〈": "〉",
    "〉": "〈",
    "≪": "≫",
    "≫": "≪",
    "⁅": "⁆",
    "⁆": "⁅",
    "⦃": "⦄",
    "⦄": "⦃",
    "⟦": "⟧",
    "⟧": "⟦",
}


def bidi_class(ch: str) -> str:
    if not ch:
        return "ON"
    return unicodedata.bidirectional(ch[0]) or "L"


@dataclass
class Unit(Generic[T]):
    payload: T
    cls: str  # UAX #9 bidi class


def text_units(text: str) -> list[Unit[str]]:
    return [Unit(c, bidi_class(c)) for c in text]


def base_level_for(classes: Sequence[str], default_rtl: bool) -> int:
    """P2/P3 heuristic for visual input: majority of strong characters.

    In visual order the 'first strong character' rule is meaningless (it would
    be the *last* one in logical order for RTL), so we use the majority,
    falling back to the document default.
    """
    r = sum(1 for c in classes if c in _STRONG_R)
    ltr_count = sum(1 for c in classes if c == "L")
    if r == 0 and ltr_count == 0:
        return 1 if default_rtl else 0
    if default_rtl and r and ltr_count <= 4 * r:
        return 1  # a Hebrew line in a Hebrew document ("הגדרה – adjoint:")
    if r == ltr_count:
        return 1 if default_rtl else 0
    return 1 if r > ltr_count else 0


def resolve_levels(classes: list[str], base: int) -> list[int]:
    n = len(classes)
    if n == 0:
        return []
    types = list(classes)
    sos = "R" if base % 2 else "L"

    # W1: NSM takes the type of the previous character.
    prev = sos
    for i, t in enumerate(types):
        if t == "NSM":
            types[i] = prev if prev not in ("LRI", "RLI", "FSI", "PDI") else "ON"
        prev = types[i]
    # W2: EN after AL -> AN.
    last_strong = sos
    for i, t in enumerate(types):
        if t in ("L", "R", "AL"):
            last_strong = t
        elif t == "EN" and last_strong == "AL":
            types[i] = "AN"
    # W3: AL -> R.
    types = ["R" if t == "AL" else t for t in types]
    # W4: single ES between EN, single CS between same numbers.
    for i in range(1, n - 1):
        if types[i] == "ES" and types[i - 1] == "EN" and types[i + 1] == "EN":
            types[i] = "EN"
        elif types[i] == "CS" and types[i - 1] == types[i + 1] and types[i - 1] in ("EN", "AN"):
            types[i] = types[i - 1]
    # W5: ET adjacent to EN -> EN.
    i = 0
    while i < n:
        if types[i] == "ET":
            j = i
            while j < n and types[j] == "ET":
                j += 1
            if (i > 0 and types[i - 1] == "EN") or (j < n and types[j] == "EN"):
                for k in range(i, j):
                    types[k] = "EN"
            i = j
        else:
            i += 1
    # W6: remaining separators/terminators -> ON.
    types = ["ON" if t in ("ET", "ES", "CS") else t for t in types]
    # W7: EN preceded by strong L -> L.
    last_strong = sos
    for i, t in enumerate(types):
        if t in ("L", "R"):
            last_strong = t
        elif t == "EN" and last_strong == "L":
            types[i] = "L"

    # N1/N2: neutrals take the surrounding direction (EN/AN count as R).
    def strong_dir(t: str) -> str | None:
        if t == "L":
            return "L"
        if t in ("R", "EN", "AN"):
            return "R"
        return None

    i = 0
    eos = sos
    while i < n:
        if types[i] in _NEUTRAL or types[i] == "NSM":
            j = i
            while j < n and (types[j] in _NEUTRAL or types[j] == "NSM"):
                j += 1
            before = sos
            for k in range(i - 1, -1, -1):
                d = strong_dir(types[k])
                if d:
                    before = d
                    break
            after = eos
            for k in range(j, n):
                d = strong_dir(types[k])
                if d:
                    after = d
                    break
            fill = before if before == after else ("R" if base % 2 else "L")
            for k in range(i, j):
                types[k] = fill
            i = j
        else:
            i += 1
    # I1/I2: implicit levels.
    levels = []
    for t in types:
        if base % 2 == 0:
            levels.append(base + (1 if t == "R" else 2 if t in ("AN", "EN") else 0))
        else:
            levels.append(base + (1 if t in ("L", "EN", "AN") else 0))
    # L1: trailing whitespace goes to the base level.
    for k in range(n - 1, -1, -1):
        if classes[k] in ("WS", "S", "B", "BN"):
            levels[k] = base
        else:
            break
    return levels


def reorder(
    units: list[Unit[T]], *, default_rtl: bool = True, base_level: int | None = None
) -> tuple[list[Unit[T]], list[int], int]:
    """Return (reordered units, their resolved levels, base level)."""
    if not units:
        return [], [], base_level if base_level is not None else (1 if default_rtl else 0)
    classes = [u.cls for u in units]
    base = base_level if base_level is not None else base_level_for(classes, default_rtl)
    levels = resolve_levels(classes, base)
    idx = list(range(len(units)))
    max_level = max(levels)
    min_odd = min((lv for lv in levels if lv % 2), default=max_level + 1)
    for level in range(max_level, min_odd - 1, -1):
        i = 0
        while i < len(idx):
            if levels[idx[i]] >= level:
                j = i
                while j < len(idx) and levels[idx[j]] >= level:
                    j += 1
                idx[i:j] = reversed(idx[i:j])
                i = j
            else:
                i += 1
    return [units[k] for k in idx], [levels[k] for k in idx], base


def visual_to_logical(text: str, *, default_rtl: bool = True, mirror: bool = True) -> str:
    units, levels, _ = reorder(text_units(text), default_rtl=default_rtl)
    out = []
    for u, lv in zip(units, levels):
        ch = u.payload
        if mirror and lv % 2 and ch in MIRROR:
            ch = MIRROR[ch]
        out.append(ch)
    return "".join(out)
