"""Split stage-1 blocks into word segments: runs of words that no punctuation or formula interrupts.

A candidate phrase never crosses a segment edge (design stage 2: n-grams do not cross punctuation
or a formula). Offsets are character positions in the extraction's flat text. Formulas are
returned separately for the operator lexicon. Nothing here decides what a concept is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Blocks whose text is not prose: the table of contents repeats headings, page markers are noise,
# and a wrapped heading's continuation is already inside the heading block's span.
SKIPPED_ROLES = frozenset({"toc", "page_marker", "heading_cont"})
MATH_ROLE = "math"

# hebmath marks what it could not read ("[[hebmath:unreadable-page]]"); it is markup, not words.
_PLACEHOLDER_RE = re.compile(r"\[\[hebmath:[^\]]*\]\]")
# One alternative per token kind. A Hebrew word may hold a geresh or gershayim inside ("אם"ם",
# "ע"י"); a hyphen or maqaf between letters joins two words without breaking the phrase
# ("תת-מרחב" equals "תת מרחב"). Anything else that is not a space breaks the segment.
# A formula that holds only one word of at least this many letters is read as that word; a
# shorter one ("$x$", "$AB$", "$det$") is a symbol.
_MATH_WORD_RE = re.compile(r"[A-Za-z]{4,}")
# The extractor sometimes drops the spaces of English it reads as math ("$biasvariancetradeoff$",
# found 2026-11-17 in חישובית 2). A math word this long that the document never writes is split
# into words the document does write; parts shorter than `MIN_GLUED_PART` are not trusted.
MIN_GLUED_LETTERS = 8
MIN_GLUED_PART = 2
_TOKEN_RE = re.compile(
    r"(?P<math>\$\$.*?\$\$|\$[^$]*\$)"
    r"|(?P<he>[א-ת]+(?:[\"'׳״][א-ת]+)*)"
    r"|(?P<en>[A-Za-z]+(?:'[A-Za-z]+)?)"
    r"|(?P<join>(?<=[א-תA-Za-z])[-־](?=[א-תA-Za-z]))"
    r"|(?P<space>\s+)"
    r"|(?P<other>.)",
    re.DOTALL,
)


@dataclass
class Token:
    """One word: its surface text, normal form (set later), and span in the flat text."""

    text: str
    start: int
    end: int
    norm: str = ""


@dataclass
class Segment:
    """Consecutive words of one block with no break between them."""

    block: int  # index into structure.blocks
    tokens: list[Token] = field(default_factory=list)


@dataclass
class Formula:
    """A formula's LaTeX and span, with the block it belongs to."""

    block: int
    latex: str
    start: int
    end: int


@dataclass
class BlockText:
    """Words and formulas of one block, in reading order."""

    index: int
    segments: list[Segment]
    formulas: list[Formula]


def clean_text(text: str) -> str:
    """Return the text with hebmath placeholders blanked out, keeping every offset in place."""
    return _PLACEHOLDER_RE.sub(lambda match: " " * len(match.group()), text)


def split_glued(word: str, vocabulary: set[str]) -> list[str] | None:
    """Return the fewest words of `vocabulary` that spell `word`, or None when none do.

    Only parts the document writes elsewhere are used, so a split never invents a word. Among
    splits with as few parts, the one with the longest first part wins (deterministic).
    """
    best: list[list[str] | None] = [None] * (len(word) + 1)
    best[len(word)] = []
    for i in range(len(word) - 1, -1, -1):
        for j in range(len(word), i + MIN_GLUED_PART - 1, -1):
            rest = best[j]
            if rest is None or word[i:j] not in vocabulary:
                continue
            if best[i] is None or len(rest) + 1 < len(best[i]):
                best[i] = [word[i:j], *rest]
    parts = best[0]
    return parts if parts and len(parts) > 1 else None


def _math_words(match: re.Match, vocabulary: set[str] | None) -> list[Token] | None:
    """Return the word tokens a one-word formula stands for, or None when it is a symbol."""
    inner = match.group().strip("$").strip()
    if not _MATH_WORD_RE.fullmatch(inner):
        return None
    word = inner.lower()
    offset = match.start() + match.group().index(inner)
    if vocabulary and len(word) >= MIN_GLUED_LETTERS and word not in vocabulary:
        parts = split_glued(word, vocabulary)
        if parts:
            tokens, position = [], offset
            for part in parts:
                tokens.append(Token(part, position, position + len(part)))
                position += len(part)
            return tokens
    return [Token(word, match.start(), match.end())]


def _split_block(
    text: str, index: int, start: int, end: int, vocabulary: set[str] | None = None
) -> BlockText:
    """Tokenize one block's span of the cleaned text into segments and formulas.

    `vocabulary` (the document's Latin words) lets a glued math word split into words.
    """
    segments: list[Segment] = []
    formulas: list[Formula] = []
    current = Segment(block=index)
    for match in _TOKEN_RE.finditer(text, start, end):
        kind = match.lastgroup
        if kind in ("he", "en"):
            word = match.group().lower() if kind == "en" else match.group()
            # A single Hebrew letter is a list marker or a detached prefix ("א.", "ב $x$").
            if kind == "he" and len(word) == 1:
                kind = "other"
            else:
                current.tokens.append(Token(word, match.start(), match.end()))
                continue
        if kind in ("join", "space"):
            continue
        if kind == "math":
            # "$Robustness$": the extractor put a plain English word in math mode.
            words = _math_words(match, vocabulary)
            if words:
                current.tokens.extend(words)
                continue
            formulas.append(Formula(index, match.group(), match.start(), match.end()))
        if current.tokens:
            segments.append(current)
        current = Segment(block=index)
    if current.tokens:
        segments.append(current)
    return BlockText(index, segments, formulas)


def words_of(text: str) -> list[str]:
    """Return the words of a short phrase as the tokenizer reads them (Latin lowercased).

    Used for names written outside a document, such as a judge's concept name, so they split
    exactly like the document's words ("תת-מרחב" gives two words).
    """
    block = _split_block(clean_text(text), -1, 0, len(text))
    return [token.text for segment in block.segments for token in segment.tokens]


def split_blocks(result) -> list[BlockText]:
    """Return the words and formulas of every prose and formula block of an extraction.

    Args:
        result: A hebmath ``ExtractionResult`` with ``structure``.

    Returns:
        One ``BlockText`` per block that is not skipped, in document order. A formula block
        has formulas and no segments.
    """
    text = clean_text(result.text)
    records = [
        (index, record)
        for index, record in enumerate(result.structure.blocks)
        if record.role not in SKIPPED_ROLES
    ]
    vocabulary = {
        match.group().lower()
        for _, record in records
        for match in _TOKEN_RE.finditer(text, record.start, record.end)
        if match.lastgroup == "en" and len(match.group()) >= MIN_GLUED_PART
    }
    blocks = []
    for index, record in records:
        block = _split_block(text, index, record.start, record.end, vocabulary)
        if record.role == MATH_ROLE and not block.formulas:
            latex = text[record.start : record.end].strip()
            block.formulas.append(Formula(index, latex, record.start, record.end))
            block.segments = []
        blocks.append(block)
    return blocks
