"""Hardened XML parsing on the standard library only.

lxml is deliberately *not* used: its binary wheels statically bundle GNU
libiconv (LGPL-2.1). The stdlib parser (expat, MIT) is sufficient for OOXML.

OOXML parts never contain a DTD, so any ``<!DOCTYPE``/``<!ENTITY`` is rejected
up front - this closes XXE and entity-expansion ("billion laughs") attacks
regardless of the expat version in use.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from .errors import CorruptFileError

Element = ET.Element

_DTD_RE = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


def parse_xml(data: bytes, *, what: str) -> ET.Element:
    head = data[:4096]
    if _DTD_RE.search(head) or (len(data) > 4096 and _DTD_RE.search(data)):
        raise CorruptFileError(f"{what}: DTD/entity declarations are not allowed in OOXML")
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise CorruptFileError(f"{what}: invalid XML ({exc})") from exc


def split_tag(tag) -> tuple[str, str]:
    """'{ns}local' -> (ns, local). Non-string tags (comments, PIs) -> ('', '')."""
    if not isinstance(tag, str):
        return "", ""
    if tag[:1] == "{":
        ns, _, local = tag[1:].partition("}")
        return ns, local
    return "", tag


def ns(el) -> str:
    return split_tag(el.tag)[0]


def local(el) -> str:
    return split_tag(el.tag)[1]


def attr(el, name: str) -> str | None:
    """Attribute by local name, whatever its namespace prefix."""
    if el is None:
        return None
    v = el.attrib.get(name)
    if v is not None:
        return v
    for k, val in el.attrib.items():
        if k.endswith("}" + name):
            return val
    return None
