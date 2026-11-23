"""Reference parser for the ``hebmath text`` output format, v1 and v2 (docs/FORMAT.md).

Downstream pipeline stages should use this instead of ad-hoc regexes::

    from hebmath.format import parse
    for block in parse(text):
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PLACEHOLDER_RE = re.compile(r"\[\[hebmath:([a-z0-9=\-]+)\]\]")
_TAG_RE = re.compile(r"\s*\\tag\{([^{}]*)\}\s*$")


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Math:
    latex: str
    display: bool = False
    tag: str | None = None


@dataclass(frozen=True)
class Marker:
    kind: str


@dataclass
class ParsedBlock:
    kind: str  # "paragraph" | "display_math"
    parts: list = field(default_factory=list)  # the whole paragraph (lines joined by a space)
    lines: list = field(default_factory=list)  # per output line: list of parts


def parse(text: str) -> list[ParsedBlock]:
    """Parse format v1 or v2 text. A paragraph may span several lines (v2):
    line breaks inside a paragraph only separate formulas from text and are
    equivalent to a space."""
    blocks: list[ParsedBlock] = []
    for raw in text.split("\n\n"):
        line = raw.strip("\n")
        if not line:
            continue
        if "\n" in line:
            per_line = [parse_line(ln) for ln in line.split("\n") if ln.strip()]
            joined: list = []
            for k, parts in enumerate(per_line):
                if k:
                    joined.append(Text(" "))
                joined.extend(parts)
            blocks.append(ParsedBlock("paragraph", _merge_text(joined), per_line))
            continue
        if line.startswith("$$") and line.endswith("$$") and len(line) >= 4:
            latex = line[2:-2]
            tag = None
            m = _TAG_RE.search(latex)
            if m:
                tag = m.group(1)
                latex = latex[: m.start()]
            parts = [Math(latex, display=True, tag=tag)]
            blocks.append(ParsedBlock("display_math", parts, [parts]))
            continue
        parts = parse_line(line)
        blocks.append(ParsedBlock("paragraph", parts, [parts]))
    return blocks


def _merge_text(parts: list) -> list:
    out: list = []
    for p in parts:
        if isinstance(p, Text) and out and isinstance(out[-1], Text):
            out[-1] = Text(out[-1].text + p.text)
        else:
            out.append(p)
    return out


def parse_line(line: str) -> list:
    """Split one paragraph line into Text / Math / Marker parts (unescaping prose)."""
    parts: list = []
    buf: list[str] = []
    i, n = 0, len(line)

    def flush():
        if buf:
            parts.append(Text("".join(buf)))
            buf.clear()

    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n and line[i + 1] in "\\$":
            buf.append(line[i + 1])
            i += 2
            continue
        if c == "$":
            j = i + 1
            while j < n:
                if line[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if line[j] == "$":
                    break
                j += 1
            flush()
            parts.append(Math(line[i + 1 : j]))
            i = j + 1
            continue
        if c == "[" and line.startswith("[[hebmath:", i):
            m = _PLACEHOLDER_RE.match(line, i)
            if m:
                flush()
                parts.append(Marker(m.group(1)))
                i = m.end()
                continue
        buf.append(c)
        i += 1
    flush()
    return parts
