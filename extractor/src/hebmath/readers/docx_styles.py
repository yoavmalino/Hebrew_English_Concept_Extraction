"""Word styles for pipeline stage 1: bold, underline, size, colour and heading level.

A run's formatting comes from several layers: the document defaults, the paragraph style (and the
styles it is based on), the character style, and direct formatting. This module resolves them in
that order. Word has no numeric weight, so bold is a yes or no here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..latex.omml import W_NAMESPACES
from ..xmlutil import attr, local, ns

DEFAULT_SIZE_POINTS = 11.0  # Word's own default when a document states none
HALF_POINTS_PER_POINT = 2.0  # w:sz is in half-points
HEX_COLOR_DIGITS = 6  # w:color is RRGGBB in hex
MAX_STYLE_DEPTH = 20  # a basedOn chain longer than this is treated as broken
MAX_HEADING_LEVEL = 9  # Word has nine built-in heading levels
_HEADING_NAME_RE = re.compile(r"^(?:heading|כותרת)\s*(\d)$", re.IGNORECASE)
_OFF_VALUES = ("0", "false", "off", "none")


@dataclass(frozen=True)
class RunFormat:
    """The resolved formatting of a run."""

    bold: bool = False
    underline: bool = False
    size: float = DEFAULT_SIZE_POINTS
    color: int = -1  # 0xRRGGBB; -1 for "auto" or not set (the reader's default text colour)


@dataclass
class _Overrides:
    """Formatting that a style sets; None means "inherit"."""

    bold: bool | None = None
    underline: bool | None = None
    size: float | None = None
    color: int | None = None
    heading_level: int | None = None
    based_on: str | None = None


@dataclass
class Styles:
    """The styles part of a Word document, resolved on demand."""

    default: _Overrides = field(default_factory=_Overrides)
    by_id: dict[str, _Overrides] = field(default_factory=dict)

    @classmethod
    def parse(cls, root) -> "Styles":
        """Read ``styles.xml``; a missing or odd part gives Word's defaults.

        Args:
            root: The parsed XML root of the styles part, or None.
        """
        styles = cls()
        if root is None:
            return styles
        for el in root:
            if not _is_w(el):
                continue
            if local(el) == "docDefaults":
                rpr = _child(_child(el, "rPrDefault"), "rPr")
                styles.default = _read_run_properties(rpr)
            elif local(el) == "style":
                style_id = attr(el, "styleId")
                if not style_id:
                    continue
                over = _read_run_properties(_child(el, "rPr"))
                over.based_on = _val(_child(el, "basedOn"))
                over.heading_level = _heading_level(_val(_child(el, "name")), _child(el, "pPr"))
                styles.by_id[style_id] = over
        return styles

    def heading_level(self, style_id: str | None) -> int | None:
        """Return the heading level (1 = main) of a paragraph style, or None."""
        for over in self._chain(style_id):
            if over.heading_level:
                return over.heading_level
        return None

    def run_format(self, paragraph_style: str | None, rpr) -> RunFormat:
        """Resolve a run's formatting: defaults, paragraph style, character style, direct."""
        layers = [self.default]
        layers += list(reversed(self._chain(paragraph_style)))
        direct = _read_run_properties(rpr)
        char_style = _val(_child(rpr, "rStyle")) if rpr is not None else None
        layers += list(reversed(self._chain(char_style)))
        layers.append(direct)
        bold = underline = False
        size = DEFAULT_SIZE_POINTS
        color = -1
        for layer in layers:
            if layer.bold is not None:
                bold = layer.bold
            if layer.underline is not None:
                underline = layer.underline
            if layer.size is not None:
                size = layer.size
            if layer.color is not None:
                color = layer.color
        return RunFormat(bold, underline, size, color)

    def _chain(self, style_id: str | None) -> list[_Overrides]:
        """Return the style and the styles it is based on, most specific first."""
        chain: list[_Overrides] = []
        seen: set[str] = set()
        while style_id and style_id in self.by_id and style_id not in seen:
            if len(chain) >= MAX_STYLE_DEPTH:
                break
            seen.add(style_id)
            over = self.by_id[style_id]
            chain.append(over)
            style_id = over.based_on
        return chain


def direct_heading_level(ppr) -> int | None:
    """Return the outline level set directly on a paragraph (1 = main), or None."""
    if ppr is None:
        return None
    return _outline_level(_child(ppr, "outlineLvl"))


def _is_w(el) -> bool:
    return isinstance(el.tag, str) and ns(el) in W_NAMESPACES


def _child(el, name: str):
    if el is None:
        return None
    for c in el:
        if _is_w(c) and local(c) == name:
            return c
    return None


def _val(el) -> str | None:
    return attr(el, "val") if el is not None else None


def _on(el) -> bool | None:
    """Read a Word on/off element: absent -> None, present without value -> on."""
    if el is None:
        return None
    value = _val(el)
    return not (value is not None and value.lower() in _OFF_VALUES)


def _read_run_properties(rpr) -> _Overrides:
    over = _Overrides()
    if rpr is None:
        return over
    over.bold = _on(_child(rpr, "b"))
    underline = _child(rpr, "u")
    if underline is not None:
        over.underline = (_val(underline) or "single").lower() not in _OFF_VALUES
    over.color = _color(_child(rpr, "color"))
    size = _child(rpr, "sz")
    if size is not None:
        try:
            over.size = float(_val(size) or "") / HALF_POINTS_PER_POINT
        except ValueError:
            pass  # an unreadable size falls back to the inherited one
    return over


def _color(el) -> int | None:
    """Read ``w:color``: 0xRRGGBB, -1 for "auto", None when absent or unreadable."""
    value = _val(el)
    if value is None:
        return None
    if value.lower() == "auto":
        return -1
    try:
        return int(value, 16) if len(value) == HEX_COLOR_DIGITS else None
    except ValueError:
        return None  # a malformed colour inherits the style's colour


def _outline_level(el) -> int | None:
    value = _val(el)
    try:
        level = int(value) + 1 if value is not None else 0
    except ValueError:
        return None
    return level if 1 <= level <= MAX_HEADING_LEVEL else None


def _heading_level(name: str | None, ppr) -> int | None:
    match = _HEADING_NAME_RE.match((name or "").strip())
    if match:
        return int(match.group(1))
    return direct_heading_level(ppr)
