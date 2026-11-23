"""Document structure for pipeline stage 1: styles, runs, headings, sections, units, signals.

Input is the reader's ``Document`` (blocks with styled words, optional bookmarks) and the output
text with each block's offsets. Everything here is pure and deterministic, and it never changes
the text: the result only describes it with character offsets into ``text``.

Every threshold below is a named starting value. The design (stage 1 calibration) tunes them on
the labelled set; do not treat them as measured.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .bidi import visual_to_logical
from .model import BlockInfo, BlockKind, Document, StyledWord

SIZE_STEP = 0.5  # font sizes are grouped to the nearest half point

# --- bold, size and signals ---------------------------------------------------------------
BOLD_RATIO = 1.4  # a face this much heavier than its reference weight counts as bold
LARGE_SIZE_RATIO = 1.15  # design stage 5, V_s: at least 15% above the most common size
# A line whose main size is this much above the body may be a heading even when plain: one point
# over an 11-point body (Word 11 -> 12, LaTeX \large) is a common sub-heading look, measured on
# the synthetic corpus (2026-11-14). Importance (V_s) keeps LARGE_SIZE_RATIO.
HEADING_SIZE_RATIO = 1.08
MAJORITY = 0.5  # a signal on more than this share of words cannot mark a heading

# --- headings ------------------------------------------------------------------------------
MAX_HEADING_CHARS = 120  # a heading is a short line, not a paragraph
MAX_HEADING_WORDS = 14
HEADING_FORMAT_SHARE = 0.8  # share of the line's words that must carry the heading format
MAX_HEADING_LEVELS = 4
MAX_LECTURE_TOKENS = (
    12  # a line longer than this that mentions a lecture is a sentence, not a title
)
LOW_LETTER_SHARE = (
    0.3  # prose with fewer letters is not text (measured: 0.02-0.04 broken, >= 0.62 normal)
)
MIN_CHARS_FOR_LETTER_CHECK = 500  # shorter documents are too small to judge
MAIN_LEVEL = 1
MAX_BULLET_CONTENTS = 80  # a hand-written contents list is cut after this many lines
HEADING_WRAP_MIN_CHARS = 35  # a heading line this long may have wrapped onto the next line
MIN_LETTER_SHARE = 0.5  # text that is mostly digits and symbols is formula debris, not a heading
TITLE_SIZE_RATIO = 1.6  # a first-page-only format this large is the title, not a heading
MAX_TITLE_BLOCKS = 4
# A main heading opens a large unit (owner's gold definition 2026-11-14). On page 1, a main-level
# line whose section holds fewer characters than this before the next main-level heading is the
# cover or a front note, not a lecture. Measured on the 9 reviewed summaries: covers and notes
# 70-94 characters, real lecture sections thousands.
MIN_MAIN_SECTION_CHARS = 200
MAX_BOOKMARK_BLOCK_CHARS = 2 * MAX_HEADING_CHARS  # longer blocks are paragraphs, never headings
MIN_TITLE_MATCH_CHARS = 4  # shorter headings must match a bookmark title exactly
BOOKMARK_MIN_ENTRIES = 2  # bookmarks override format inference when there are at least this many
HEADING_END_CHARS = (".", ",", ";", "?", "!")  # a heading does not end like a sentence
BULLET_CHARS = ("•", "-", "–", "*", "·")

# --- ranking heading looks by structure (where they occur), not by an assumed order of formats --
MIN_MAIN_OCCURRENCES = 2  # a look seen once cannot be the main level (a cover, a stray line)
MIN_MAIN_SPREAD = 0.5  # main headings run through at least half the text (first to last)
LEAD_TIE = 0.02  # first appearances closer than this share of the text do not decide an order
NEAR_BLACK = 0x30  # a colour whose channels are all at most this is black for the "coloured" test

# --- numbered series (an optional, language-free booster) ---------------------------------------
USE_NUMBERED_SERIES = True  # "X 1", "X 2", "X 3" lines through the text form a series
MIN_SERIES_MEMBERS = 3
SERIES_EDGE_TOKENS = 3  # the number must sit within this many words of a line edge
# Language pack: words that name a lecture-level part. Used only to let a plain line become a
# heading candidate; never to set a level. Empty it to run without any language-specific rule.
LECTURE_WORDS = frozenset(
    {"הרצאה", "תרגול", "פרק", "שיעור", "lecture", "chapter", "week", "tutorial"}
)

# --- position (recorded for calibration; not used for levels yet) ------------------------------
CENTER_TOLERANCE = 0.08  # share of the column width the line's centre may be off the column's
CENTER_MAX_WIDTH = 0.8  # a line wider than this share of the column is not "centred"

# --- other roles and signals -----------------------------------------------------------------
MAX_STANDALONE_CHARS = 80  # design signal L: a short line that stands alone
ALIGN_WINDOW = 80  # how far ahead (characters) a word may be found when aligning to the text

# Printed table of contents: titles followed by a leader and a page number.
_TOC_ENTRY_RE = re.compile(r"(.+?)\s*(?:\.{3,}|…|\. \. \.|·{3,})\s*(\d+)")
_NUMBER_TOKEN_RE = re.compile(r"^\d+[.:\-–]?$")
_TOKEN_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)+)(?:\s|\.|$)")
_TOC_LEADERS = ("...", "…", ". . .", "···")
_TOC_TITLE_RE = re.compile(
    r"^\s*(תוכן(?:\s+ה?עניינ?י?ים?)?|table of contents|contents)\s*:?\s*$", re.IGNORECASE
)
# A contents list written by hand: bullet lines with no page numbers ("∢ הקדמה", "o נושא").
_TOC_BULLET_RE = re.compile(r"^\s*(?:[•·∢▪◦\-–*]|o(?=\s))")
_FORMULA_RE = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"\[\[hebmath:([a-z0-9-]+)[^\]]*\]\]")
_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)

# Unit types by the word that opens the block (Hebrew and English), and their starting weights
# (design section 3, omega_u: definition and theorem 2, paragraph 1, example and exercise 0.5;
# the weights are rescaled so the mean over a document is 1).
_UNIT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("definition", re.compile(r"^\W*(הגדרה|definition|def\b)", re.I)),
    (
        "theorem",
        re.compile(r"^\W*(משפט|למה|טענה|מסקנה|theorem|lemma|proposition|corollary|claim)", re.I),
    ),
    ("example", re.compile(r"^\W*(דוגמ[הא]|דוגמאות|example)", re.I)),
    ("exercise", re.compile(r"^\W*(תרגיל|שאלה|exercise|problem)", re.I)),
    # Added by the owner on 2026-11-14 so "הוכחה:" and "הערה:" lines are labels, not headings.
    ("proof", re.compile(r"^\W*(הוכחה|proof)", re.I)),
    ("remark", re.compile(r"^\W*(הערה|הערות|remark|note)", re.I)),
)
UNIT_WEIGHTS = {
    "definition": 2.0,
    "theorem": 2.0,
    "paragraph": 1.0,
    "example": 0.5,
    "exercise": 0.5,
    "proof": 1.0,  # initial: a proof explains a theorem but defines nothing new
    "remark": 0.5,  # initial: side notes, like examples
}
# A label may carry its number before the colon ("למה 2.1:", "משפט א'"); it is not a name.
_UNIT_NUMBER_RE = re.compile(r"^[\d.]+$|^[א-ת]'$|^\(?[\dא-ת]{1,3}\)?$")

_FAMILY_STYLE_RE = re.compile(
    r"[-,_ ]?(bold|italic|oblique|regular|roman|book|medium|light|semibold|demibold|black|heavy|"
    r"condensed|psmt|ps|mt)$",
    re.I,
)


# =========================================================================================
# Results
# =========================================================================================


@dataclass(frozen=True)
class StyleRecord:
    """One distinct word style of the document."""

    face: str
    weight: int
    size: float
    bold: bool
    underline: bool
    large: bool  # size at least LARGE_SIZE_RATIO times the most common size
    color: int = -1  # 0xRRGGBB; -1 when unknown
    colored: bool = False  # a colour other than the body text's


@dataclass(frozen=True)
class Run:
    """A stretch of ``text[start:end]`` whose words share one style (index into ``styles``)."""

    start: int
    end: int
    style: int


@dataclass(frozen=True)
class BlockRecord:
    """A block of the text with its role and formatting summary."""

    kind: str
    start: int
    end: int
    page: int  # 1-based; 0 when the format has no pages
    lines: int
    role: str  # body, heading, title, toc, math, page_marker, unit_label
    heading_level: int  # 0 unless role is heading
    unit: int  # index into ``units``; -1 when the block is not part of a unit
    section: int  # index into ``sections`` of the main (level 1) section; -1 when none
    subsection: int  # index of the nearest heading of any level; -1 when none
    standalone: bool  # a short line that stands alone (design signal L)
    starts_page: bool  # the first block of its page
    gap_before: float  # space above it, in body-size units; -1 when unknown (no geometry)
    centered: bool  # centred in its column and clearly narrower than it
    words: int
    bold_share: float
    underline_share: float
    large_share: float
    named_unit: str = ""  # unit type when a heading names one ("משפט פיתגורס"); else empty


@dataclass(frozen=True)
class Section:
    """A part of the document that starts at a heading (or at a page when there are none)."""

    title: str
    level: int  # 1 = main heading
    start: int
    end: int
    source: str  # bookmark, style, format, toc, pages


@dataclass(frozen=True)
class Unit:
    """A logical block of knowledge: a definition, theorem, example, exercise or paragraph."""

    start: int
    end: int
    type: str
    omega: float  # weight, rescaled so the mean over the document is 1
    section: int


@dataclass
class StructureInfo:
    """Everything stage 1 reports about a document, as offsets into the output text."""

    body_size: float = 0.0
    body_weight: int = 0
    body_color: int = 0
    styles: list[StyleRecord] = field(default_factory=list)
    runs: list[Run] = field(default_factory=list)
    blocks: list[BlockRecord] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)
    signal_share: dict[str, float] = field(default_factory=dict)
    suspicious: list[dict] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable form (the shape stored in ``learning_materials.structure``)."""
        return {
            "body_size": self.body_size,
            "body_weight": self.body_weight,
            "body_color": self.body_color,
            "styles": [vars(s) for s in self.styles],
            "runs": [[r.start, r.end, r.style] for r in self.runs],
            "blocks": [vars(b) for b in self.blocks],
            "sections": [vars(s) for s in self.sections],
            "units": [vars(u) for u in self.units],
            "signal_share": dict(self.signal_share),
            "suspicious": list(self.suspicious),
            "stats": dict(sorted(self.stats.items())),
        }


# =========================================================================================
# Line-level helper kept for callers that only have line styles
# =========================================================================================


def modal_style(blocks: Iterable[BlockInfo]) -> dict[str, float]:
    """Return the most common font size and weight of the line styles, weighted by characters.

    Args:
        blocks: The blocks of an extraction result.

    Returns:
        ``{"size": points, "weight": PDFium weight}``; a value is 0 when no line carried it.
    """
    sizes: Counter[float] = Counter()
    weights: Counter[int] = Counter()
    for block in blocks:
        for line in block.line_styles:
            sizes[round(line.size / SIZE_STEP) * SIZE_STEP] += line.chars
            if line.weight > 0:
                weights[line.weight] += line.chars
    return {"size": _most_common(sizes), "weight": _most_common(weights)}


def _most_common(counts: Counter) -> float:
    """Return the key with the highest count, the smaller key on a tie; 0 when empty."""
    if not counts:
        return 0
    return min(counts, key=lambda key: (-counts[key], key))


# =========================================================================================
# Bold classification
# =========================================================================================


def _base_family(face: str) -> str:
    """Return the font family without its style suffix ("Arial-BoldMT" -> "arial")."""
    name = face.lower().split("+")[-1]
    previous = None
    while previous != name:
        previous = name
        name = _FAMILY_STYLE_RE.sub("", name)
    return name


def classify_bold(words: Iterable[StyledWord]) -> dict[tuple[str, int, bool], bool]:
    """Decide for each font variant whether it is bold.

    PDFium's numeric weight has no common scale: regular text measured 345 to 555 depending on
    the font, bold 545 to 752. So a variant is bold when its name or Word run says so, or when its
    weight is BOLD_RATIO times the lightest weight of the same family, or, for a family seen with
    a single weight, of the body text. A single-weight family that is the body is never bold.

    Args:
        words: All styled words of the document.

    Returns:
        A map from ``(face, weight, name_bold)`` to the decision.
    """
    chars: Counter[tuple[str, int, bool]] = Counter()
    for word in words:
        chars[(word.face, word.weight, word.name_bold)] += len(word.text)
    positive = {k: n for k, n in chars.items() if k[1] > 0}
    body_weight = 0
    if positive:
        body_weight = max(positive, key=lambda k: (positive[k], -k[1]))[1]
    lightest: dict[str, int] = {}
    for (face, weight, _), _n in positive.items():
        family = _base_family(face)
        lightest[family] = min(weight, lightest.get(family, weight))
    result: dict[tuple[str, int, bool], bool] = {}
    for key in chars:
        face, weight, name_bold = key
        if name_bold:
            result[key] = True
        elif weight <= 0:
            result[key] = False
        else:
            family_variants = {k[1] for k in positive if _base_family(k[0]) == _base_family(face)}
            reference = lightest[_base_family(face)] if len(family_variants) > 1 else body_weight
            result[key] = weight >= BOLD_RATIO * reference
    return result


# =========================================================================================
# Alignment of words to the output text
# =========================================================================================


def align_words(
    text: str, start: int, end: int, words: list[StyledWord]
) -> list[tuple[int, int, StyledWord]]:
    """Find each word in ``text[start:end]``, in order, and return its offsets.

    The serializer normalizes whitespace, escapes characters and joins hyphenated words, so a
    word is searched only a short way ahead. A word that is not found is skipped.

    Args:
        text: The output text.
        start: Start of the block.
        end: End of the block.
        words: The block's words in reading order.

    Returns:
        ``(start, end, word)`` for each word that was found.
    """
    found: list[tuple[int, int, StyledWord]] = []
    cursor = start
    for word in words:
        if not word.text:
            continue
        index = text.find(word.text, cursor, min(end, cursor + ALIGN_WINDOW + len(word.text)))
        if index < 0:
            continue
        found.append((index, index + len(word.text), word))
        cursor = index + len(word.text)
    return found


# =========================================================================================
# Building the structure
# =========================================================================================


def _normalize(text: str) -> str:
    """Return text lowercased without spaces and punctuation, for comparing titles."""
    return _NORMALIZE_RE.sub("", text).lower()


def _same_title(here: str, target: str) -> bool:
    """Compare a heading with a bookmark title (both normalized).

    Equal, or one contains the other (bookmarks often drop a number or a trailing colon), or the
    same characters in another order: some producers store the title in visual order, and the
    bidi conversion cannot always restore a title that mixes digits and punctuation.
    """
    if here == target:
        return True
    if len(here) < MIN_TITLE_MATCH_CHARS:
        return False
    if here in target or target in here:
        return True
    # the reordered-characters test is only for short texts: sorting a paragraph is wasted work
    return len(here) <= MAX_HEADING_CHARS and sorted(here) == sorted(target)


def _toc_entries(text: str) -> list[tuple[str, str]]:
    """Return the (title, page) entries of a printed table-of-contents block, or [] if it is not one.

    The regex backtracks badly on long paragraphs, so a block is only scanned when it contains a
    leader at all.
    """
    if not any(leader in text for leader in _TOC_LEADERS):
        return []
    entries = _TOC_ENTRY_RE.findall(text)
    return entries if len(entries) >= 2 else []


def _toc_titles(text: str) -> list[str]:
    """Return the entry titles of a printed table-of-contents block (empty if it is not one)."""
    return [title.strip() for title, _page in _toc_entries(text)]


def build_structure(
    document: Document, text: str, spans: list[tuple[int, int, int]]
) -> StructureInfo:
    """Describe the text: styles, runs, headings, sections, units and signal shares.

    Args:
        document: The reader's document (blocks with styled words, optional bookmarks).
        text: The output text.
        spans: ``(block_index, start, end)`` for each block that produced text.

    Returns:
        The structure, with all offsets into ``text``.
    """
    info = StructureInfo()
    aligned: dict[int, list[tuple[int, int, StyledWord]]] = {}
    all_words: list[StyledWord] = []
    total_words = 0
    for index, start, end in spans:
        block = document.blocks[index]
        if block.kind is BlockKind.PARAGRAPH and block.words:
            total_words += len(block.words)
            aligned[index] = align_words(text, start, end, block.words)
            all_words.extend(w for _s, _e, w in aligned[index])
    info.stats = {"words": total_words, "words_aligned": len(all_words)}
    bold_of = classify_bold(all_words)

    size_chars: Counter[float] = Counter()
    weight_chars: Counter[int] = Counter()
    for word in all_words:
        size_chars[round(word.size / SIZE_STEP) * SIZE_STEP] += len(word.text)
        if word.weight > 0:
            weight_chars[word.weight] += len(word.text)
    info.body_size = _most_common(size_chars)
    info.body_weight = int(_most_common(weight_chars))
    color_chars: Counter[int] = Counter()
    for word in all_words:
        color_chars[_plain_color(word.color)] += len(word.text)
    info.body_color = int(_most_common(color_chars))

    style_ids: dict[tuple, int] = {}

    def style_of(word: StyledWord) -> int:
        bold = bold_of[(word.face, word.weight, word.name_bold)]
        size = round(word.size / SIZE_STEP) * SIZE_STEP
        large = info.body_size > 0 and size >= LARGE_SIZE_RATIO * info.body_size
        color = _plain_color(word.color)
        key = (word.face, word.weight, size, bold, word.underline, large, color)
        key = key + (color != info.body_color,)
        if key not in style_ids:
            style_ids[key] = len(info.styles)
            info.styles.append(StyleRecord(*key))
        return style_ids[key]

    word_styles: dict[int, list[tuple[int, int, int]]] = {}
    for index, found in aligned.items():
        word_styles[index] = [(s, e, style_of(w)) for s, e, w in found]
    info.runs = _merge_runs(
        text, [item for index in sorted(word_styles) for item in word_styles[index]]
    )

    _build_blocks(document, text, spans, info, word_styles)
    _shares_and_flags(info, text)
    info.suspicious = _suspicious(text, document)
    return info


def _plain_color(color: int) -> int:
    """Return a colour with "unknown" and near-black folded into black (0).

    Word's "auto" and PDFs that draw text in a very dark grey are the same as black to a reader,
    so they must not count as a coloured heading.
    """
    if color < 0:
        return 0
    channels = ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF)
    return 0 if max(channels) <= NEAR_BLACK else color


def _merge_runs(text: str, word_spans: list[tuple[int, int, int]]) -> list[Run]:
    """Merge consecutive words of the same style (separated only by whitespace) into runs."""
    runs: list[Run] = []
    for start, end, style in word_spans:
        if runs and runs[-1].style == style and not text[runs[-1].end : start].strip():
            runs[-1] = Run(runs[-1].start, end, style)
        else:
            runs.append(Run(start, end, style))
    return runs


def _block_summary(
    word_styles: list[tuple[int, int, int]], styles: list[StyleRecord]
) -> tuple[int, float, float, float]:
    """Return (word count, bold share, underline share, large share) of a block's words."""
    count = len(word_styles)
    if not count:
        return 0, 0.0, 0.0, 0.0
    recs = [styles[s] for _a, _b, s in word_styles]
    return (
        count,
        sum(r.bold for r in recs) / count,
        sum(r.underline for r in recs) / count,
        sum(r.large for r in recs) / count,
    )


def _build_blocks(document, text, spans, info, word_styles) -> None:
    """Fill blocks, sections and units (the order matters: roles, headings, sections, units)."""
    records: list[dict] = []
    for index, start, end in spans:
        block = document.blocks[index]
        summary = _block_summary(word_styles.get(index, []), info.styles)
        block_text = text[start:end]
        lines = len(block.line_styles) or 1
        page = block.line_styles[0].page if block.line_styles else 0
        role = "body"
        if block.kind is BlockKind.DISPLAY_MATH:
            role = "math"
        elif block.kind is BlockKind.PAGE_BREAK:
            role = "page_marker"
        elif _toc_entries(block_text):
            role = "toc"
        stripped = block_text.strip()
        dominant = Counter(s for _a, _b, s in word_styles.get(index, []))
        format_key = (0.0, False, False, False, True)
        if dominant:
            top = info.styles[min(dominant, key=lambda s: (-dominant[s], s))]
            # a line ending with a colon introduces something, so it ranks below a plain heading
            # of the same look
            format_key = (
                top.size,
                top.bold,
                top.underline,
                top.colored,
                not stripped.endswith(":"),
            )
        standalone = (
            block.kind is BlockKind.PARAGRAPH
            and lines == 1
            and len(stripped) <= MAX_STANDALONE_CHARS
            and not stripped.endswith(".")
        )
        records.append(
            {
                "index": index,
                "kind": block.kind.value,
                "start": start,
                "end": end,
                "page": page,
                "lines": lines,
                "role": role,
                "level": 0,
                "standalone": standalone,
                "summary": summary,
                "format": format_key,
                "text": block_text,
                "block": block,
                "geometry": _block_geometry(block),
            }
        )
    _mark_positions(records, info.body_size)
    for i, rec in enumerate(records):
        if rec["role"] == "body" and _TOC_TITLE_RE.match(rec["text"]):
            following = records[i + 1] if i + 1 < len(records) else None
            if following is not None and following["role"] == "toc":
                rec["role"] = "toc"
            elif following is not None and _TOC_BULLET_RE.match(following["text"]):
                rec["role"] = "toc"
                _mark_bullet_contents(records, i + 1)
    toc_titles = [t for rec in records if rec["role"] == "toc" for t in _toc_titles(rec["text"])]
    sources = _assign_headings(document, records, info, toc_titles, text)
    sections = _make_sections(records, sources, info, text)
    info.sections = sections
    units = _make_units(records, info)
    info.units = units
    for rec in records:
        words, bold, underline, large = rec["summary"]
        info.blocks.append(
            BlockRecord(
                kind=rec["kind"],
                start=rec["start"],
                end=rec["end"],
                page=rec["page"],
                lines=rec["lines"],
                role=rec["role"],
                heading_level=rec["level"],
                unit=rec.get("unit", -1),
                section=rec.get("section", -1),
                subsection=rec.get("subsection", -1),
                standalone=rec["standalone"],
                starts_page=rec["starts_page"],
                gap_before=rec["gap_before"],
                centered=rec["centered"],
                words=words,
                bold_share=bold,
                underline_share=underline,
                large_share=large,
                named_unit=rec.get("named_unit", ""),
            )
        )


def _block_geometry(block) -> tuple[float, float, float, float] | None:
    """Return a block's (x0, x1, top, bottom) on its page, or None when it has no geometry."""
    lines = [ls for ls in block.line_styles if ls.x1 > ls.x0]
    if not lines:
        return None
    return (
        min(ls.x0 for ls in lines),
        max(ls.x1 for ls in lines),
        lines[0].top,
        lines[-1].bottom,
    )


def _mark_positions(records: list[dict], body_size: float) -> None:
    """Record where each block sits: first on its page, space above it, centred in its column.

    The column of a page is the span from its leftmost to its rightmost text. These facts are
    stored for calibration (owner's decision 2026-11-14: position may separate heading levels);
    the level ranking does not use them yet.
    """
    columns: dict[int, tuple[float, float]] = {}
    for rec in records:
        geometry = rec["geometry"]
        if geometry and rec["page"]:
            left, right = columns.get(rec["page"], (geometry[0], geometry[1]))
            columns[rec["page"]] = (min(left, geometry[0]), max(right, geometry[1]))
    previous: dict | None = None
    for rec in records:
        geometry = rec["geometry"]
        rec["starts_page"] = bool(rec["page"]) and (
            previous is None or previous["page"] != rec["page"]
        )
        rec["gap_before"] = -1.0
        rec["centered"] = False
        if geometry and rec["page"] in columns:
            left, right = columns[rec["page"]]
            width = right - left
            line_width = geometry[1] - geometry[0]
            offset = abs((geometry[0] + geometry[1]) / 2 - (left + right) / 2)
            rec["centered"] = (
                width > 0
                and line_width <= CENTER_MAX_WIDTH * width
                and offset <= CENTER_TOLERANCE * width
            )
            last = previous["geometry"] if previous else None
            if last and previous["page"] == rec["page"] and body_size > 0:
                rec["gap_before"] = round(max(0.0, last[3] - geometry[2]) / body_size, 3)
        if rec["role"] != "page_marker":
            previous = rec


def _mark_bullet_contents(records: list[dict], start: int) -> None:
    """Mark the bullet lines after a "contents" title as part of the table of contents.

    A summary often lists its chapters by hand, with no page numbers. Those lines are the
    author's own outline: they are not headings and they are left out of the signal shares.
    """
    for rec in records[start : start + MAX_BULLET_CONTENTS]:
        if rec["role"] != "body" or not _TOC_BULLET_RE.match(rec["text"]):
            break
        rec["role"] = "toc"


def _is_format_heading(rec: dict, doc_shares: dict[str, float], body_size: float = 0.0) -> bool:
    """Return True when a short standalone block is formatted unlike the body text."""
    if rec["role"] != "body" or rec["kind"] != BlockKind.PARAGRAPH.value:
        return False
    text = rec["text"].strip()
    words, bold, underline, large = rec["summary"]
    if (
        not text
        or rec["lines"] > 1
        or len(text) > MAX_HEADING_CHARS
        or words > MAX_HEADING_WORDS
        or words == 0
        or text.endswith(HEADING_END_CHARS)
        or text.startswith(BULLET_CHARS)
        or text.startswith("$")
        or _letter_share(text) < MIN_LETTER_SHARE
    ):
        return False
    if large >= HEADING_FORMAT_SHARE:
        return True
    if body_size > 0 and rec["format"][0] >= HEADING_SIZE_RATIO * body_size:
        return True
    if bold >= HEADING_FORMAT_SHARE and doc_shares["bold"] <= MAJORITY:
        return True
    return underline >= HEADING_FORMAT_SHARE and doc_shares["underline"] <= MAJORITY


def _letter_share(text: str) -> float:
    """Return the share of non-space characters that are letters (formula debris scores low)."""
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def names_a_lecture(text: str) -> bool:
    """Return True for a short line that names a lecture, chapter or tutorial with its number.

    The word may stand at the start ("הרצאה 3 - ..."), after one word of course name ("למידה הרצאה
    10"), or at the end ("... - 4 הרצאה": a line mixing Hebrew and English is often stored in
    visual order). The number may be next to it, or at the other end ("הרצאה Spectral Clustering -
    12"). Only lines of at most ``MAX_LECTURE_TOKENS`` words are considered.
    """
    tokens = [_TOKEN_EDGE_RE.sub("", t) for t in text.split()]
    tokens = [t for t in tokens if t] if len(text.split()) <= MAX_LECTURE_TOKENS else []
    if not tokens:
        return False
    edge = tokens[:2] + tokens[-2:]
    if not any(t.lower() in LECTURE_WORDS for t in edge):
        return False
    return any(_NUMBER_TOKEN_RE.match(t) for t in tokens)


def _is_pattern_heading(rec: dict) -> bool:
    """Return True for a short standalone line that names a lecture, chapter or the like."""
    text = rec["text"].strip()
    return (
        rec["role"] == "body"
        and rec["kind"] == BlockKind.PARAGRAPH.value
        and rec["lines"] == 1
        and 0 < len(text) <= MAX_HEADING_CHARS
        and not text.endswith(HEADING_END_CHARS)
        and names_a_lecture(text)
    )


def _pattern_level(text: str) -> int:
    """Return the level a heading's numbering gives it ("2.1 ..." is level 2), or 0.

    Only the structure of the number counts. Words such as "הרצאה" never set a level (owner's
    decision 2026-11-14): a level comes from where a look occurs, not from vocabulary.
    """
    text = text.strip()
    numbered = _NUMBERED_RE.match(text)
    if numbered:
        return min(numbered.group(1).count(".") + 1, MAX_HEADING_LEVELS)
    return 0


def _doc_shares(records: list[dict]) -> dict[str, float]:
    """Return the share of words that are bold and underlined across the body blocks."""
    total = bold = underline = 0.0
    for rec in records:
        if rec["role"] == "toc":
            continue
        words, b, u, _large = rec["summary"]
        total += words
        bold += words * b
        underline += words * u
    if not total:
        return {"bold": 0.0, "underline": 0.0}
    return {"bold": bold / total, "underline": underline / total}


def _assign_headings(document, records, info, toc_titles, text) -> dict[int, str]:
    """Mark heading blocks and their levels; return the source of each heading's level."""
    sources: dict[int, str] = {}
    styled = [r for r in records if r["block"].heading_level]
    if styled:
        for rec in styled:
            rec["role"] = "heading"
            rec["level"] = min(rec["block"].heading_level, MAX_HEADING_LEVELS)
            sources[rec["index"]] = "style"
        return sources
    bookmarks = document.toc
    if len(bookmarks) >= BOOKMARK_MIN_ENTRIES:
        used: set[int] = set()
        # a bookmark names a heading, so only short blocks can match; indexing them by page keeps
        # a long outline over a long document fast
        short = [
            r for r in records if r["role"] == "body" and len(r["text"]) <= MAX_BOOKMARK_BLOCK_CHARS
        ]
        by_page: dict[int, list[dict]] = {}
        for rec in short:
            by_page.setdefault(rec["page"], []).append(rec)
        for level, title, page in bookmarks:
            # some producers store bookmark titles in visual order; try the logical form too
            targets = {t for t in (_normalize(title), _normalize(visual_to_logical(title))) if t}
            if not targets:
                continue
            pool = short if not page else by_page.get(page, []) + by_page.get(page + 1, [])
            if not page:
                pool = short
            for rec in pool:
                if rec["index"] in used:
                    continue
                here = rec.setdefault("norm", _normalize(rec["text"]))
                if here and any(_same_title(here, t) for t in targets):
                    rec["role"] = "heading"
                    rec["level"] = min(level, MAX_HEADING_LEVELS)
                    sources[rec["index"]] = "bookmark"
                    used.add(rec["index"])
                    break
        info.stats["bookmarks"] = len(bookmarks)
        info.stats["bookmarks_matched"] = len(used)
        if used:
            return sources
    shares = _doc_shares(records)
    toc_keys = {_normalize(t) for t in toc_titles}
    series = _numbered_series(records, len(text)) if USE_NUMBERED_SERIES else set()
    info.stats["series_lines"] = len(series)
    candidates = []
    for rec in records:
        if _is_format_heading(rec, shares, info.body_size):
            candidates.append(rec)
            sources[rec["index"]] = "format"
        elif rec["role"] == "body" and rec["text"].strip() and _normalize(rec["text"]) in toc_keys:
            candidates.append(rec)
            sources[rec["index"]] = "toc"
        elif id(rec) in series and _is_short_line(rec):
            candidates.append(rec)
            sources[rec["index"]] = "series"
        elif _is_pattern_heading(rec):
            candidates.append(rec)
            sources[rec["index"]] = "pattern"
    for rec in candidates:
        rec["in_series"] = id(rec) in series
    candidates = _merge_wrapped_headings(records, candidates, text)
    _rank_levels(candidates, info, len(text))
    for rec in candidates:
        if rec["role"] != "title":
            rec["role"] = "heading"
            forced = _pattern_level(rec["text"])
            if forced:
                rec["level"] = forced
    _mark_front_matter(candidates, info, len(text))
    _mark_unit_labels(candidates, info)
    return sources


def _unit_heading(text: str) -> tuple[str | None, bool]:
    """Return the unit type a line opens with and whether a name follows the unit word.

    "משפט:", "למה 2.1" and "Example 3 -" carry no name; "משפט פיתגורס" and "הגדרה: וקטור עצמי"
    do. The whole first word must be the unit word, so "משפטים חשובים" (a topic) is neither.

    Returns:
        ``(unit type or None, has a name)``.
    """
    words = text.split()
    first = _TOKEN_EDGE_RE.sub("", words[0]) if words else ""
    unit_type = next((name for name, pattern in _UNIT_PATTERNS if pattern.fullmatch(first)), None)
    if unit_type is None:
        return None, False
    rest = [_TOKEN_EDGE_RE.sub("", word) for word in words[1:]]
    rest = [word for word in rest if word and not _UNIT_NUMBER_RE.match(word)]
    return unit_type, bool(rest)


def _mark_unit_labels(candidates: list[dict], info: StructureInfo) -> None:
    """Turn headings that only name a unit ("משפט:", "הוכחה:") into labels; flag named units.

    A label says what kind of text follows; it opens no sub-topic, so it must not enter the
    heading path or count as a heading (owner's decision 2026-11-14). It still types the
    paragraph under it (see ``_make_units``). A heading that names its unit ("משפט פיתגורס")
    stays a heading and gets ``named_unit``: the owner wants the name scored as important
    (stage 5). Level-1 headings never become labels: their gold is measured and none was one.
    """
    labels = named = 0
    for rec in candidates:
        if rec["role"] != "heading":
            continue
        unit_type, has_name = _unit_heading(rec["text"])
        if unit_type is None:
            continue
        if has_name:
            rec["named_unit"] = unit_type
            named += 1
        elif rec["level"] >= 2:
            rec["role"] = "unit_label"
            rec["level"] = 0
            labels += 1
    info.stats["unit_labels"] = labels
    info.stats["named_units"] = named


def _mark_front_matter(candidates: list[dict], info: StructureInfo, text_length: int) -> None:
    """Turn main-level lines on page 1 that open almost nothing into the cover (role title).

    A cover line often has the very look of the lecture headings, so format cannot separate it;
    what does is that its section is empty: the next main-level heading follows within
    ``MIN_MAIN_SECTION_CHARS``. Only page 1 of a document with more pages is checked, so a short
    real section later in the text is never dropped.
    """
    main = sorted(
        (rec for rec in candidates if rec["role"] == "heading" and rec["level"] == 1),
        key=lambda rec: rec["start"],
    )
    if not main or max(rec["page"] for rec in candidates) <= 1:
        return
    count = 0
    for rec, following in zip(main, main[1:] + [None]):
        if rec["page"] != 1:
            break
        section_end = following["start"] if following else text_length
        if section_end - rec["start"] < MIN_MAIN_SECTION_CHARS:
            rec["role"] = "title"
            rec["level"] = 0
            count += 1
    info.stats["front_matter_titles"] = count


def _is_short_line(rec: dict) -> bool:
    """Return True for a one-line body paragraph short enough to be a heading."""
    text = rec["text"].strip()
    return (
        rec["role"] == "body"
        and rec["kind"] == BlockKind.PARAGRAPH.value
        and rec["lines"] == 1
        and 0 < len(text) <= MAX_HEADING_CHARS
        and not text.endswith(HEADING_END_CHARS)
    )


def _series_key(text: str) -> tuple[str, int] | None:
    """Return (template word, number) for a line like "X 3 - ..." or "... 3 X", else None.

    The number must stand within ``SERIES_EDGE_TOKENS`` words of an edge of the line (lines that
    mix Hebrew and English are often stored in visual order, so the end counts too). The template
    is the word next to the number on the side away from the edge, or the line's first word. No
    vocabulary is involved: "הרצאה 3", "Week 3" and "Kapitel 3" all qualify.
    """
    raw_tokens = text.split()
    if not raw_tokens or len(raw_tokens) > MAX_LECTURE_TOKENS:
        return None
    tokens = [_TOKEN_EDGE_RE.sub("", t) for t in raw_tokens]
    tokens = [t for t in tokens if t]
    # isdecimal, not isdigit: "⁷" (a superscript) is a digit that int() cannot read.
    numbers = [i for i, t in enumerate(tokens) if t.isdecimal()]
    if not numbers:
        return None
    position = numbers[0]
    words = [t.lower() for t in tokens if not t.isdecimal()]
    if not words:
        return None
    if position < SERIES_EDGE_TOKENS:
        return words[0], int(tokens[position])
    position = numbers[-1]
    if len(tokens) - 1 - position < SERIES_EDGE_TOKENS:
        return words[-1], int(tokens[position])
    return None


def _numbered_series(records: list[dict], text_length: int) -> set[int]:
    """Return ``id(rec)`` of the short lines that form a numbered series through the text.

    A series: at least ``MIN_SERIES_MEMBERS`` lines with the same template word, numbers that
    never go down (no restart), spread over at least ``MIN_MAIN_SPREAD`` of the text. Unit labels
    ("למה 1", "Example 2") restart per chapter and name units, so a template that opens a unit is
    never a series. It is a booster: it can make a plain line a candidate and break a tie between
    looks, but it never sets a level by itself.
    """
    groups: dict[str, list[tuple[int, dict]]] = {}
    for rec in records:
        if rec["role"] != "body" or rec["kind"] != BlockKind.PARAGRAPH.value or rec["lines"] > 1:
            continue
        key = _series_key(rec["text"].strip())
        if key is None or _unit_type(key[0]):
            continue
        groups.setdefault(key[0], []).append((key[1], rec))
    found: set[int] = set()
    for members in groups.values():
        numbers = [number for number, _rec in members]
        if len(members) < MIN_SERIES_MEMBERS or numbers != sorted(numbers):
            continue
        first, last = members[0][1]["start"], members[-1][1]["start"]
        if text_length and (last - first) / text_length >= MIN_MAIN_SPREAD:
            found.update(id(rec) for _number, rec in members)
    return found


def _merge_wrapped_headings(records: list[dict], candidates: list[dict], text: str) -> list[dict]:
    """Join the continuation lines of a heading that wrapped onto a second block.

    A long heading is cut into one block per line. The next candidate is a continuation when it
    follows directly, on the same page, in the same look, and the first line is long enough to
    have wrapped and does not end like a sentence. The first block then spans both lines and the
    second gets the role ``heading_cont``.
    """
    chosen = {id(rec) for rec in candidates}
    merged: list[dict] = []
    previous: dict | None = None
    previous_pos = -2
    for pos, rec in enumerate(records):
        if id(rec) not in chosen:
            continue
        continues = (
            previous is not None
            and pos == previous_pos + 1
            and rec["page"] == previous["page"]
            and rec["format"][:4] == previous["format"][:4]
            and len(previous["text"].strip()) >= HEADING_WRAP_MIN_CHARS
            and not previous["text"].strip().endswith(HEADING_END_CHARS + (":",))
            and _pattern_level(rec["text"]) == 0
        )
        if continues:
            previous["end"] = rec["end"]
            previous["text"] = text[previous["start"] : previous["end"]]
            first_words, *shares = previous["summary"]
            extra_words = rec["summary"][0]
            previous["summary"] = (first_words + extra_words, *shares)
            rec["role"] = "heading_cont"
        else:
            merged.append(rec)
            previous = rec
        previous_pos = pos
    return merged


def _rank_levels(candidates: list[dict], info: StructureInfo, text_length: int) -> None:
    """Give each candidate a level from where its look occurs, not from an assumed format order.

    A look is the heading's visual format (size, bold, underline, colour). Main headings are few,
    run through the whole text and open it, while sub-headings occur inside them. So a look can be
    the main level only when it occurs at least ``MIN_MAIN_OCCURRENCES`` times over at least
    ``MIN_MAIN_SPREAD`` of the text. Among such looks a larger size comes first (size is the one
    format fact every corpus file ranked consistently), then more stacked emphasis (bold,
    underline, colour count equally: no one of them outranks another), then a numbered series,
    then the look that appears first, then the rarer look. Looks that still tie share a level and
    are counted in ``stats["heading_level_ties"]``: the system does not guess an order it cannot
    see. A look that ends with a colon sits one level below the same look without it, unless one
    of the two occurs only once (then it is the same heading style with stray punctuation).

    Args:
        candidates: Heading candidates with ``format``, ``start``, ``page`` and ``in_series``.
        info: The structure being built (body size, stats).
        text_length: Length of the output text, for spread and first appearance.
    """
    looks = _look_groups(candidates, text_length)
    ordered = sorted(looks, key=lambda key: looks[key]["order"])
    main = next((key for key in ordered if looks[key]["eligible"]), None)
    info.stats["main_look_found"] = int(main is not None)
    info.stats["heading_level_ties"] = 0
    _mark_titles(candidates, looks, ordered, main, info)
    level = 0
    previous: dict | None = None
    for key in ordered:
        group = [rec for rec in looks[key]["records"] if rec["role"] != "title"]
        if not group:
            continue
        if _ties(previous, looks[key]):
            info.stats["heading_level_ties"] += 1
        else:
            level += 1
        previous = looks[key] if looks[key]["eligible"] else None
        plain = [rec for rec in group if rec["format"][-1]]
        colon = [rec for rec in group if not rec["format"][-1]]
        split = _splits_by_colon(len(plain), len(colon))
        for rec in plain if split else group:
            rec["level"] = min(level, MAX_HEADING_LEVELS)
        if split:
            level += 1
            for rec in colon:
                rec["level"] = min(level, MAX_HEADING_LEVELS)


def _splits_by_colon(plain: int, colon: int) -> bool:
    """Return True when the colon form of a look is its own, lower level.

    A form seen once next to a form seen ``MIN_MAIN_OCCURRENCES`` times or more is stray
    punctuation on the same heading style, so the two stay one level.
    """
    if not plain or not colon:
        return False
    stray = min(plain, colon) == 1 and max(plain, colon) >= MIN_MAIN_OCCURRENCES
    return not stray


def _ties(previous: dict | None, look: dict) -> bool:
    """Return True when two eligible looks differ in nothing the ranking can see."""
    return (
        previous is not None
        and look["eligible"]
        and previous["tie_rank"] == look["tie_rank"]
        and abs(previous["lead"] - look["lead"]) < LEAD_TIE
    )


def _look_groups(candidates: list[dict], text_length: int) -> dict[tuple, dict]:
    """Group candidates by visual look and measure where each look occurs."""
    groups: dict[tuple, list[dict]] = {}
    for rec in candidates:
        groups.setdefault(rec["format"][:4], []).append(rec)
    looks: dict[tuple, dict] = {}
    for key, records in groups.items():
        starts = sorted(rec["start"] for rec in records)
        lead = starts[0] / text_length if text_length else 0.0
        spread = (starts[-1] - starts[0]) / text_length if text_length else 0.0
        eligible = len(records) >= MIN_MAIN_OCCURRENCES and spread >= MIN_MAIN_SPREAD
        size, bold, underline, colored = key
        emphasis = int(bold) + int(underline) + int(colored)
        series = any(rec.get("in_series") for rec in records)
        tie_rank = (size, emphasis, series)
        looks[key] = {
            "records": records,
            "eligible": eligible,
            "tie_rank": tie_rank,
            "order": (not eligible, -size, -emphasis, not series, lead, len(records)),
            "size": size,
            "lead": lead,
        }
    return looks


def _mark_titles(candidates, looks, ordered, main, info) -> None:
    """Mark a cover: a look seen only on page 1 that outranks the main look, or is clearly large.

    A cover must not take level 1. Without a main look the older test applies: the look is at
    least ``TITLE_SIZE_RATIO`` times the body size, or the largest look of all.
    """
    pages = {rec["page"] for rec in candidates if rec["page"]}
    if len(pages) <= 1 and max(pages, default=0) <= 1:
        return
    largest = max((looks[key]["size"] for key in looks), default=0.0)
    main_size = looks[main]["size"] if main is not None else None
    for key in ordered:
        group = looks[key]["records"]
        if len(group) > MAX_TITLE_BLOCKS or any(rec["page"] != 1 for rec in group):
            continue
        size = looks[key]["size"]
        above_main = main_size is not None and size > main_size
        clearly_large = info.body_size > 0 and size >= TITLE_SIZE_RATIO * info.body_size
        if above_main or clearly_large or (main_size is None and size == largest):
            for rec in group:
                rec["role"] = "title"


def _make_sections(records, sources, info, text) -> list[Section]:
    """Build sections from the heading blocks; fall back to pages when there are none."""
    headings = [r for r in records if r["role"] == "heading"]
    sections: list[Section] = []
    if headings:
        for i, rec in enumerate(headings):
            end = len(text)
            for later in headings[i + 1 :]:
                if later["level"] <= rec["level"]:
                    end = later["start"]
                    break
            sections.append(
                Section(
                    title=rec["text"].strip(),
                    level=rec["level"],
                    start=rec["start"],
                    end=end,
                    source=sources.get(rec["index"], "format"),
                )
            )
        sub = -1
        main_index = -1
        for rec in records:
            if rec["role"] == "heading":
                sub += 1
                if rec["level"] == 1:
                    main_index = sub
            rec["section"] = main_index
            rec["subsection"] = sub if sub >= 0 else -1
        return sections
    by_page: dict[int, list[dict]] = {}
    for rec in records:
        if rec["role"] in ("body", "math"):
            by_page.setdefault(rec["page"], []).append(rec)
    for page in sorted(by_page):
        group = by_page[page]
        sections.append(
            Section(
                title=f"page {page}" if page else "document",
                level=1,
                start=group[0]["start"],
                end=group[-1]["end"],
                source="pages",
            )
        )
    page_index = {page: i for i, page in enumerate(sorted(by_page))}
    for rec in records:
        i = page_index.get(rec["page"], -1)
        rec["section"] = i
        rec["subsection"] = i
    return sections


def _unit_type(text: str) -> str | None:
    """Return the unit type named by the first word of a block, or None for plain text."""
    for name, pattern in _UNIT_PATTERNS:
        if pattern.match(text):
            return name
    return None


def _make_units(records, info) -> list[Unit]:
    """Cut the body blocks into units and give each its relative weight omega.

    A heading or unit label that names a unit type ("הגדרה — פרישה ליניארית:") types the first
    paragraph under it.
    """
    raw: list[dict] = []
    pending: str | None = None
    for rec in records:
        if rec["role"] in ("heading", "unit_label"):
            pending = _unit_type(rec["text"])
            continue
        if rec["role"] == "math":
            if raw and raw[-1]["section"] == rec.get("section", -1):
                raw[-1]["end"] = rec["end"]
                rec["unit"] = len(raw) - 1
            continue
        if rec["role"] != "body" or rec["kind"] != BlockKind.PARAGRAPH.value:
            continue
        unit_type = _unit_type(rec["text"]) or pending or "paragraph"
        pending = None
        raw.append(
            {
                "start": rec["start"],
                "end": rec["end"],
                "type": unit_type,
                "section": rec.get("section", -1),
            }
        )
        rec["unit"] = len(raw) - 1
    if not raw:
        return []
    mean = sum(UNIT_WEIGHTS[u["type"]] for u in raw) / len(raw)
    return [
        Unit(u["start"], u["end"], u["type"], UNIT_WEIGHTS[u["type"]] / mean, u["section"])
        for u in raw
    ]


def _shares_and_flags(info: StructureInfo, text: str) -> None:
    """Compute the share of words with each signal (design stage 5, p_f)."""
    total = bold = underline = large = line = 0
    for rec in info.blocks:
        if rec.role == "toc":
            continue
        total += rec.words
        bold += rec.words * rec.bold_share
        underline += rec.words * rec.underline_share
        large += rec.words * rec.large_share
        if rec.standalone:
            line += rec.words
    if total:
        info.signal_share = {
            "bold": bold / total,
            "underline": underline / total,
            "size": large / total,
            "line": line / total,
        }


def letter_share(text: str) -> float:
    """Return the share of non-space characters that are letters, ignoring LaTeX formulas."""
    prose = _FORMULA_RE.sub("", text)
    chars = [c for c in prose if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def _suspicious(text: str, document: Document) -> list[dict]:
    """List places the reader could not convert, for the hard-segment step (stage 3).

    Besides placeholders and unresolved glyphs, a document whose text has hardly any letters is
    flagged: its text layer is incomplete (for example Hebrew drawn as shapes), and every later
    stage would work on nonsense.
    """
    found = [
        {"kind": m.group(1), "start": m.start(), "end": m.end()}
        for m in _PLACEHOLDER_RE.finditer(text)
    ]
    unresolved = document.stats.get("unresolved_glyphs", 0)
    if unresolved:
        found.append({"kind": "unresolved-glyphs", "count": unresolved})
    share = letter_share(text)
    if len(text) >= MIN_CHARS_FOR_LETTER_CHECK and share < LOW_LETTER_SHARE:
        found.append({"kind": "few-letters", "share": round(share, 3)})
    return found
