"""DOCX reader: WordprocessingML + OMML -> IR.

Implemented directly on ``zipfile`` + the stdlib XML parser (no python-docx) because we need
the raw XML tree to see OMML, tracked changes, fields, text boxes and
``mc:AlternateContent`` - things high-level APIs hide.

Text in DOCX is stored in logical order, so no bidi reordering is required.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from dataclasses import dataclass, field

from ..errors import (
    CorruptFileError,
    EncryptedFileError,
    ResourceLimitError,
    UnsupportedFormatError,
)
from ..latex.omml import M_NAMESPACES, W_NAMESPACES, OmmlToLatex
from ..model import Block, Document, InlineMath, Placeholder, Segment, StyledWord, TextSpan
from ..safety import Limits, check_zip
from ..xmlutil import attr as _attr
from ..xmlutil import local as _xlocal
from ..xmlutil import ns as _xns
from ..xmlutil import parse_xml
from .docx_styles import Styles, direct_heading_level
from ..text_math import merge_text_spans, promote_lone_math, tag_segments

REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_DOC_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
    "http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument",
)
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
V_NS = "urn:schemas-microsoft-com:vml"
O_NS = "urn:schemas-microsoft-com:office:office"

# Elements we never descend into (their text is not part of the visible document).
_SKIP_W = {
    "instrText",
    "delText",
    "delInstrText",
    "rPr",
    "pPr",
    "tblPr",
    "trPr",
    "tcPr",
    "sectPr",
    "proofErr",
    "bookmarkStart",
    "bookmarkEnd",
    "commentRangeStart",
    "commentRangeEnd",
    "commentReference",
    "annotationRef",
    "del",
    "moveFrom",
    "fldData",
    "permStart",
    "permEnd",
    "lastRenderedPageBreak",
}

_EQ_NUMBER_RE = re.compile(r"^\s*\(?\s*(\d+(?:[.\-]\d+)*[a-z]?)\s*\)?\s*$")
_TAG_IN_LATEX_RE = re.compile(r"\s*\\tag\{([^{}]*)\}\s*$")


def _local(el) -> str:
    return _xlocal(el)


def _ns(el) -> str:
    return _xns(el)


def _is_w(el, name: str | None = None) -> bool:
    return _ns(el) in W_NAMESPACES and (name is None or _local(el) == name)


def _is_m(el, name: str | None = None) -> bool:
    return _ns(el) in M_NAMESPACES and (name is None or _local(el) == name)


def _w_child(el, name: str):
    for c in el:
        if _is_w(c, name):
            return c
    return None


# ---------------------------------------------------------------------------
# Numbering (lists)
# ---------------------------------------------------------------------------

_HEBREW_LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"


def _hebrew_number(n: int) -> str:
    vals = [
        (400, "ת"),
        (300, "ש"),
        (200, "ר"),
        (100, "ק"),
        (90, "צ"),
        (80, "פ"),
        (70, "ע"),
        (60, "ס"),
        (50, "נ"),
        (40, "מ"),
        (30, "ל"),
        (20, "כ"),
        (10, "י"),
        (9, "ט"),
        (8, "ח"),
        (7, "ז"),
        (6, "ו"),
        (5, "ה"),
        (4, "ד"),
        (3, "ג"),
        (2, "ב"),
        (1, "א"),
    ]
    out = ""
    while n > 0:
        if n % 100 in (15, 16):
            out += "טו" if n % 100 == 15 else "טז"
            n -= n % 100
            continue
        for v, s in vals:
            if n >= v:
                out += s
                n -= v
                break
    return out


def _roman(n: int) -> str:
    vals = [
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def _format_number(n: int, fmt: str) -> str:
    if fmt in ("decimal", "decimalZero"):
        return str(n) if fmt == "decimal" else f"{n:02d}"
    if fmt == "lowerLetter":
        return chr(ord("a") + (n - 1) % 26)
    if fmt == "upperLetter":
        return chr(ord("A") + (n - 1) % 26)
    if fmt == "lowerRoman":
        return _roman(n)
    if fmt == "upperRoman":
        return _roman(n).upper()
    if fmt in ("hebrew1",):
        return _hebrew_number(n)
    if fmt in ("hebrew2",):
        return _HEBREW_LETTERS[(n - 1) % len(_HEBREW_LETTERS)]
    return str(n)


@dataclass
class _Level:
    fmt: str = "decimal"
    text: str = "%1."
    start: int = 1


@dataclass
class Numbering:
    abstract: dict[str, dict[int, _Level]] = field(default_factory=dict)
    num_to_abstract: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, dict[int, int]] = field(default_factory=dict)
    counters: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def parse(cls, root) -> "Numbering":
        nb = cls()
        if root is None:
            return nb
        for an in root:
            if _is_w(an, "abstractNum"):
                aid = _attr(an, "abstractNumId") or ""
                levels: dict[int, _Level] = {}
                for lvl in an:
                    if not _is_w(lvl, "lvl"):
                        continue
                    ilvl = int(_attr(lvl, "ilvl") or 0)
                    L = _Level()
                    for c in lvl:
                        if _is_w(c, "numFmt"):
                            L.fmt = _attr(c, "val") or "decimal"
                        elif _is_w(c, "lvlText"):
                            L.text = _attr(c, "val") or ""
                        elif _is_w(c, "start"):
                            try:
                                L.start = int(_attr(c, "val") or 1)
                            except ValueError:
                                pass
                    levels[ilvl] = L
                nb.abstract[aid] = levels
        for num in root:
            if _is_w(num, "num"):
                nid = _attr(num, "numId") or ""
                a = _w_child(num, "abstractNumId")
                if a is not None:
                    nb.num_to_abstract[nid] = _attr(a, "val") or ""
                for ov in num:
                    if _is_w(ov, "lvlOverride"):
                        so = _w_child(ov, "startOverride")
                        if so is not None:
                            try:
                                nb.overrides.setdefault(nid, {})[int(_attr(ov, "ilvl") or 0)] = int(
                                    _attr(so, "val") or 1
                                )
                            except ValueError:
                                pass
        return nb

    def label(self, num_id: str, ilvl: int) -> str:
        if num_id in ("", "0") or num_id not in self.num_to_abstract:
            return ""
        levels = self.abstract.get(self.num_to_abstract[num_id], {})
        lvl = levels.get(ilvl, _Level())
        counters = self.counters.setdefault(num_id, [0] * 9)
        ilvl = max(0, min(8, ilvl))
        for k in range(9):
            if counters[k] == 0:
                L = levels.get(k, _Level())
                counters[k] = self.overrides.get(num_id, {}).get(k, L.start) - 1
        counters[ilvl] += 1
        for k in range(ilvl + 1, 9):
            L = levels.get(k, _Level())
            counters[k] = self.overrides.get(num_id, {}).get(k, L.start) - 1
        if lvl.fmt == "bullet":
            return "•"
        if lvl.fmt == "none":
            return ""

        def repl(m):
            k = int(m.group(1)) - 1
            L = levels.get(k, _Level())
            return _format_number(max(counters[k], 1), L.fmt)

        return re.sub(r"%(\d)", repl, lvl.text).strip()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

WORD_FACE = "docx"  # Word has no per-glyph fonts here; formatting is per run
BOLD_WEIGHT = 700  # Word has no numeric weight; these two stand for bold and regular
REGULAR_WEIGHT = 400


class DocxReader:
    """Reads a DOCX byte string into a ``Document``."""

    def __init__(
        self,
        omml: OmmlToLatex | None = None,
        limits: Limits | None = None,
        detect_plain_math: bool = True,
    ):
        self.omml = omml or OmmlToLatex()
        self.limits = limits or Limits()
        self.detect_plain_math = detect_plain_math

    # ---- container -----------------------------------------------------
    def read(self, data: bytes) -> Document:
        if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            if "EncryptedPackage".encode("utf-16-le") in data:
                raise EncryptedFileError("password-protected DOCX")
            raise UnsupportedFormatError("legacy binary .doc is not supported; convert to .docx")
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except (zipfile.BadZipFile, ValueError, OSError, EOFError, NotImplementedError) as exc:
            raise CorruptFileError(f"not a valid DOCX/ZIP container: {exc}") from exc
        with zf:
            check_zip(zf, self.limits, len(data))
            names = set(zf.namelist())
            self._zf = zf
            self._names = names
            main = self._main_part(zf, names)
            self._main = main
            self._doc = Document()
            doc_root = parse_xml(self._read(main), what=main)
            rels = self._rels(main)
            self._numbering = Numbering.parse(self._part_by_type(rels, "numbering", main))
            self._styles = Styles.parse(self._part_by_type(rels, "styles", main))
            self._par_style: str | None = None
            self._footnotes = self._notes(self._part_by_type(rels, "footnotes", main), "footnote")
            self._endnotes = self._notes(self._part_by_type(rels, "endnotes", main), "endnote")
            self._note_order: list[tuple[str, str]] = []
            body = next((c for c in doc_root if _is_w(c, "body")), None)
            if body is None:
                raise CorruptFileError("DOCX has no w:body")
            for block in self._blocks(body):
                self._doc.blocks.append(block)
            self._append_notes()
            return self._doc

    def _read(self, name: str) -> bytes:
        info = self._zf.getinfo(name) if name in self._names else None
        if info is not None and info.file_size > self.limits.max_zip_uncompressed_bytes:
            raise ResourceLimitError(f"part {name} is too large")
        try:
            return self._zf.read(name)
        except (
            Exception
        ) as exc:  # BadZipFile, zlib.error, CRC errors, bad seeks, encrypted entries ...
            raise CorruptFileError(f"cannot read part {name}: {type(exc).__name__}: {exc}") from exc

    def _main_part(self, zf: zipfile.ZipFile, names: set[str]) -> str:
        if "_rels/.rels" in names:
            root = parse_xml(self._read("_rels/.rels"), what="_rels/.rels")
            for rel in root:
                if _attr(rel, "Type") in OFFICE_DOC_REL:
                    target = (_attr(rel, "Target") or "").lstrip("/")
                    if target in names:
                        return target
        if "word/document.xml" in names:
            return "word/document.xml"
        if any(n.startswith("xl/") for n in names) or any(n.startswith("ppt/") for n in names):
            raise UnsupportedFormatError("this is an Excel/PowerPoint file, not a Word document")
        raise CorruptFileError("no main document part found in DOCX")

    def _rels(self, part: str) -> dict[str, tuple[str, str]]:
        d, f = posixpath.split(part)
        rels_name = posixpath.join(d, "_rels", f + ".rels")
        if rels_name not in self._names:
            return {}
        root = parse_xml(self._read(rels_name), what=rels_name)
        out = {}
        for rel in root:
            rid = _attr(rel, "Id") or ""
            out[rid] = (_attr(rel, "Type") or "", _attr(rel, "Target") or "")
        return out

    def _part_by_type(self, rels, suffix: str, base: str):
        for typ, target in rels.values():
            if typ.endswith("/" + suffix):
                path = posixpath.normpath(posixpath.join(posixpath.dirname(base), target)).lstrip(
                    "/"
                )
                if path in self._names:
                    try:
                        return parse_xml(self._read(path), what=path)
                    except CorruptFileError as exc:
                        self._doc.warn(f"ignored unreadable part: {exc}")
                        return None
        return None

    def _notes(self, root, kind: str) -> dict[str, object]:
        out = {}
        if root is None:
            return out
        for note in root:
            if _is_w(note, kind):
                typ = _attr(note, "type")
                if typ in ("separator", "continuationSeparator", "continuationNotice"):
                    continue
                out[_attr(note, "id") or ""] = note
        return out

    # ---- block level -----------------------------------------------------
    def _blocks(self, container):
        for el in container:
            if _is_w(el, "p"):
                yield from self._paragraph(el)
            elif _is_w(el, "tbl"):
                yield from self._table(el)
            elif _is_w(el, "sdt"):
                content = _w_child(el, "sdtContent")
                if content is not None:
                    yield from self._blocks(content)
            elif (
                _is_w(el, "customXml")
                or _is_w(el, "ins")
                or _is_w(el, "moveTo")
                or _is_w(el, "smartTag")
            ):
                yield from self._blocks(el)
            elif _ns(el) == MC_NS and _local(el) == "AlternateContent":
                yield from self._blocks(self._alt_choice(el))
            elif _is_m(el, "oMathPara"):
                yield self._display(el)

    def _alt_choice(self, el):
        for c in el:
            if _ns(c) == MC_NS and _local(c) == "Choice":
                return c
        for c in el:
            if _ns(c) == MC_NS and _local(c) == "Fallback":
                return c
        return el

    def _table(self, tbl):
        for tr in tbl:
            if not _is_w(tr, "tr"):
                continue
            cells: list[list[Segment]] = []
            extra: list[Block] = []
            row_words: list[StyledWord] = []
            for tc in tr:
                if _is_w(tc, "sdt"):
                    content = _w_child(tc, "sdtContent")
                    tcs = [x for x in (content if content is not None else []) if _is_w(x, "tc")]
                else:
                    tcs = [tc] if _is_w(tc, "tc") else []
                for cell in tcs:
                    segs: list[Segment] = []
                    for b in self._blocks(cell):
                        if b.kind.value == "paragraph":
                            if segs:
                                segs.append(TextSpan(" "))
                            segs.extend(b.segments)
                            row_words.extend(b.words)
                        elif b.kind.value == "display_math":
                            if segs:
                                segs.append(TextSpan(" "))
                            segs.append(InlineMath(b.latex))
                        else:
                            extra.append(b)
                    cells.append(segs)
            row: list[Segment] = []
            non_empty = [
                c for c in cells if any(not isinstance(s, TextSpan) or s.text.strip() for s in c)
            ]
            for i, c in enumerate(non_empty):
                if i:
                    row.append(TextSpan(" | "))
                row.extend(c)
            if row:
                self._doc.add_stat("table_rows")
                block = Block.paragraph(merge_text_spans(row))
                block.table_row = True
                block.words = row_words
                yield block
            yield from extra

    def _display(self, el) -> Block:
        latex = self._convert_math(el)
        tag = None
        m = _TAG_IN_LATEX_RE.search(latex)
        if m:
            tag = m.group(1)
            latex = latex[: m.start()]
        self._doc.add_stat("display_math")
        return Block.display_math(latex, tag)

    def _convert_math(self, el) -> str:
        try:
            latex = self.omml.convert(el)
        except RecursionError:
            self._doc.warn("an equation was too deeply nested and was skipped")
            return ""
        except Exception as exc:  # converter bug must not lose the document
            self._doc.warn(f"equation conversion failed: {type(exc).__name__}")
            return ""
        self._doc.add_stat("omml_equations")
        return latex

    def _paragraph(self, p):
        ppr = _w_child(p, "pPr")
        label = ""
        if ppr is not None:
            numpr = _w_child(ppr, "numPr")
            if numpr is not None:
                num_id_el, ilvl_el = _w_child(numpr, "numId"), _w_child(numpr, "ilvl")
                num_id = _attr(num_id_el, "val") if num_id_el is not None else ""
                try:
                    ilvl = int(_attr(ilvl_el, "val") or 0) if ilvl_el is not None else 0
                except ValueError:
                    ilvl = 0
                label = self._numbering.label(num_id or "", ilvl)

        # A paragraph is a sequence of "lines"; each line is a list of items,
        # where display math splits the paragraph.
        pieces: list[object] = []  # Segment | Block (display) | "BR" | StyledWord
        floating: list[Block] = []
        style_el = _w_child(ppr, "pStyle") if ppr is not None else None
        style_id = _attr(style_el, "val") if style_el is not None else None
        heading = self._styles.heading_level(style_id) or direct_heading_level(ppr)
        saved_style = self._par_style
        self._par_style = style_id  # runs resolve their formatting against it
        try:
            self._inline(p, pieces, floating)
        finally:
            self._par_style = saved_style  # a text box inside the paragraph changed it

        blocks: list[Block] = []
        current: list[Segment] = []
        current_words: list[StyledWord] = []
        first = True

        def flush():
            nonlocal current, current_words, first
            segs = current
            words = current_words
            current = []
            current_words = []
            if not any(not isinstance(s, TextSpan) or s.text.strip() for s in segs):
                return
            if first and label:
                segs = [TextSpan(label + " ")] + segs
            first = False
            if self.detect_plain_math:
                segs = tag_segments(segs)
            blk = Block.paragraph(merge_text_spans(segs))
            blk.words = words
            blk.heading_level = heading
            blocks.append(promote_lone_math(blk))

        for item in pieces:
            if item == "BR":
                flush()
            elif isinstance(item, Block):
                flush()
                blocks.append(item)
            elif isinstance(item, StyledWord):
                current_words.append(item)
            else:
                current.append(item)  # type: ignore[arg-type]
        flush()

        # Lone math in a paragraph -> display. Equation number left behind -> tag.
        blocks = self._attach_equation_numbers(blocks)
        if len(blocks) == 1 and blocks[0].kind.value == "paragraph":
            segs = [
                s
                for s in blocks[0].segments
                if not (isinstance(s, TextSpan) and not s.text.strip())
            ]
            if len(segs) == 1 and isinstance(segs[0], InlineMath) and not label:
                self._doc.add_stat("display_math")
                blocks = [Block.display_math(segs[0].latex)]
        yield from blocks
        yield from floating

    def _attach_equation_numbers(self, blocks: list[Block]) -> list[Block]:
        out: list[Block] = []
        for b in blocks:
            if (
                b.kind.value == "paragraph"
                and out
                and out[-1].kind.value == "display_math"
                and out[-1].tag is None
            ):
                text = "".join(s.text for s in b.segments if isinstance(s, TextSpan))
                only_text = all(isinstance(s, TextSpan) for s in b.segments)
                m = _EQ_NUMBER_RE.match(text)
                if only_text and m:
                    out[-1].tag = m.group(1)
                    continue
            if (
                b.kind.value == "display_math"
                and out
                and out[-1].kind.value == "paragraph"
                and b.tag is None
            ):
                text = "".join(s.text for s in out[-1].segments if isinstance(s, TextSpan))
                if all(isinstance(s, TextSpan) for s in out[-1].segments) and _EQ_NUMBER_RE.match(
                    text
                ):
                    # number before the equation (RTL documents put it on the right = first)
                    b.tag = _EQ_NUMBER_RE.match(text).group(1)
                    out.pop()
            out.append(b)
        return out

    def _inline(self, el, out: list, floating: list[Block]) -> None:
        for c in el:
            if not isinstance(c.tag, str):
                continue
            if _is_m(c, "oMathPara"):
                out.append(self._display(c))
            elif _is_m(c, "oMath"):
                latex = self._convert_math(c)
                if latex:
                    out.append(InlineMath(latex))
                    self._doc.add_stat("inline_math")
            elif _is_w(c):
                name = _local(c)
                if name in _SKIP_W:
                    continue
                if name == "r":
                    self._run(c, out, floating)
                elif name in (
                    "hyperlink",
                    "ins",
                    "moveTo",
                    "smartTag",
                    "customXml",
                    "fldSimple",
                    "dir",
                    "bdo",
                    "sdtContent",
                ):
                    self._inline(c, out, floating)
                elif name == "sdt":
                    self._inline(
                        _w_child(c, "sdtContent") if _w_child(c, "sdtContent") is not None else c,
                        out,
                        floating,
                    )
                elif name == "pPr":
                    continue
                else:
                    self._inline(c, out, floating)
            elif _ns(c) == MC_NS and _local(c) == "AlternateContent":
                self._inline(self._alt_choice(c), out, floating)
            else:
                self._inline(c, out, floating)

    def _run(self, r, out: list, floating: list[Block]) -> None:
        rpr = _w_child(r, "rPr")
        vert = None
        hidden = False
        if rpr is not None:
            va = _w_child(rpr, "vertAlign")
            if va is not None:
                vert = _attr(va, "val")
            vanish = _w_child(rpr, "vanish")
            if vanish is not None and (_attr(vanish, "val") or "1") not in ("0", "false", "off"):
                hidden = True
        if hidden:
            return
        buf: list[str] = []
        fmt = self._styles.run_format(self._par_style, rpr)

        def emit_text():
            if not buf:
                return
            text = "".join(buf)
            buf.clear()
            out.extend(
                StyledWord(
                    w,
                    WORD_FACE,
                    BOLD_WEIGHT if fmt.bold else REGULAR_WEIGHT,
                    fmt.size,
                    fmt.bold,
                    fmt.underline,
                    fmt.color,
                )
                for w in text.split()
            )
            if (
                vert in ("superscript", "subscript")
                and text.strip()
                and not any("֐" <= ch <= "׿" for ch in text)
            ):
                mark = "^" if vert == "superscript" else "_"
                out.append(TextSpan(mark + "{" + text.strip() + "}"))
            else:
                out.append(TextSpan(text))

        for c in r:
            if not isinstance(c.tag, str):
                continue
            if _is_w(c):
                name = _local(c)
                if name == "t":
                    buf.append(c.text or "")
                elif name in ("tab", "ptab"):
                    buf.append(" ")
                elif name == "br":
                    emit_text()
                    if (_attr(c, "type") or "textWrapping") in ("textWrapping", "page", "column"):
                        out.append("BR")
                elif name == "cr":
                    emit_text()
                    out.append("BR")
                elif name in ("noBreakHyphen",):
                    buf.append("-")
                elif name == "softHyphen":
                    continue
                elif name == "sym":
                    ch = _attr(c, "char")
                    font = (_attr(c, "font") or "").lower()
                    if ch and "symbol" not in font and "wingdings" not in font:
                        try:
                            code = int(ch, 16)
                            if code >= 0xF000:
                                code -= 0xF000
                            buf.append(chr(code))
                        except ValueError:
                            pass
                elif name in ("footnoteReference", "endnoteReference"):
                    emit_text()
                    nid = _attr(c, "id") or ""
                    kind = "footnote" if name == "footnoteReference" else "endnote"
                    self._note_order.append((kind, nid))
                    out.append(TextSpan(f"[{len(self._note_order)}]"))
                elif name in ("drawing", "pict", "object"):
                    emit_text()
                    self._object(c, out, floating)
                elif name in _SKIP_W:
                    continue
            elif _ns(c) == MC_NS and _local(c) == "AlternateContent":
                emit_text()
                choice = self._alt_choice(c)
                for cc in choice:
                    if _is_w(cc) and _local(cc) in ("drawing", "pict", "object"):
                        self._object(cc, out, floating)
            elif _is_m(c, "oMath"):
                emit_text()
                latex = self._convert_math(c)
                if latex:
                    out.append(InlineMath(latex))
        emit_text()

    def _object(self, el, out: list, floating: list[Block]) -> None:
        """Drawings, VML pictures, OLE objects and text boxes."""
        # Text boxes: their paragraphs follow the host paragraph.
        txbx = [x for x in el.iter() if _is_w(x, "txbxContent")]
        if txbx:
            for t in txbx:
                floating.extend(self._blocks(t))
            return
        # Legacy equation objects (Equation Editor 3.0 / MathType) are OLE.
        for x in el.iter():
            if not isinstance(x.tag, str):
                continue
            if _ns(x) == O_NS and _local(x) == "OLEObject":
                prog = _attr(x, "ProgID") or ""
                if prog.startswith("Equation") or "MathType" in prog or prog.startswith("Mathtype"):
                    self._doc.add_stat("equation_objects")
                    self._doc.warn(
                        "document contains legacy Equation Editor/MathType objects; "
                        "they are emitted as [[hebmath:math-object]] placeholders"
                    )
                    out.append(Placeholder("math-object"))
                else:
                    out.append(Placeholder("object"))
                return
        self._doc.add_stat("images")
        out.append(Placeholder("image"))

    def _append_notes(self) -> None:
        if not self._note_order:
            return
        self._doc.blocks.append(Block.paragraph([TextSpan("———")]))
        for idx, (kind, nid) in enumerate(self._note_order, start=1):
            note = (self._footnotes if kind == "footnote" else self._endnotes).get(nid)
            if note is None:
                continue
            blocks = list(self._blocks(note))
            first = True
            for b in blocks:
                if first and b.kind.value == "paragraph":
                    b.segments = [TextSpan(f"[{idx}] ")] + b.segments
                    first = False
                self._doc.blocks.append(b)
