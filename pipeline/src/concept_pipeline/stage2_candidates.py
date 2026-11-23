"""Stage 2 of the concept pipeline: candidate terms with their sources and counts per unit.

Input is a hebmath extraction (flat text plus the stage-1 structure). Output: one candidate per
normal form, with the sources that proposed it (design stage 2, six sources plus a label-line source) and every mention
with its unit, main section and strength a_cu. Recall comes first, because a concept lost here
never returns. There is no model call yet: normalization uses `PrefixNormalizer`.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field

from .lexicon import FUNCTION_WORDS, OPERATORS, STRUCTURAL_WORDS, UNIT_LABEL_WORDS
from .normalize import LemmaNormalizer, PrefixNormalizer
from .tokens import SKIPPED_ROLES, BlockText, Segment, Token, clean_text, split_blocks

# Bump when the output for the same input changes; later stages store it with their snapshots.
STAGE2_VERSION = "2.5-prefix"

# Starting values, calibrated against the must-concept gold once it exists (plan, section יא).
MAX_PHRASE_WORDS = 4  # design: noun phrases of one to four words
MIN_PHRASE_COUNT = 2  # an n-gram seen once is noise unless another source proposes it
MAX_HEADING_WORDS = 6  # a longer heading is a sentence, not a name
MAX_EMPHASIS_WORDS = 4
# Emphasis that covers most of a block (or of the document) marks layout, not a term.
EMPHASIS_MAX_SHARE = 0.5
MAX_GLOSS_WORDS = 5

# Mention strength a_cu (design section 3).
DEFINED = 1.0  # the mention is the term a definition pattern introduces
SUBJECT = 0.7  # the unit's subject: in its heading or label, or emphasized in its opening block
MENTION = 0.4  # any other mention in prose
OPERATOR = 0.2  # only through an operator in a formula

# "label" was added on 2026-11-16 after the committee gold showed one-time label lines missed.
SOURCES = ("definition", "heading", "emphasis", "ngram", "gloss", "operator", "label")

# The term follows these words: "נקרא הפרישה הליניארית", "הגדרת סכום ישר", "is called a group".
AFTER_TRIGGERS = frozenset(
    "נקרא נקראת נקראים נקראות תקרא תיקרא יקרא ייקרא יקראו ייקראו נגדיר מגדירים הגדרת called"
    " מכונה מכונים מכונות".split()
)
# A trigger often carries ש or ו ("שנקרא", "שמכונה template strand"), which the prefix
# normalizer strips only when the bare word is frequent enough; these are checked directly.
TRIGGER_PREFIXES = ("וש", "ש", "ו")
# "ביותר" closes a superlative inside a term ("עץ מסלולים קצרים ביותר"), so it may end a term
# but never begins one (found in the 2026-11-16 dev diagnosis).
SUPERLATIVE_WORDS = frozenset({"ביותר"})
# A heading that compares two terms names both ("Random controlled trial vs observational study").
COMPARISON_WORDS = frozenset({"vs", "vs.", "versus", "לעומת"})
# The term comes right before these: "X מוגדר כ...", "X is defined as".
BEFORE_TRIGGERS = frozenset("מוגדר מוגדרת מוגדרים מוגדרות defined".split())
# A label that opens a block and names what follows: "הגדרה — מטריצה הפיכה:".
LABEL_TRIGGERS = frozenset({"הגדרה", "definition"})
# "X הוא ..." is a definition only inside a definition unit, else it is any sentence.
COPULAS = frozenset({"הוא", "היא", "הם", "הן", "is", "are"})
# Words skipped right after a trigger before the term starts ("נגדיר את X", "called a group").
LEADING_FILLERS = frozenset({"את", "the", "a", "an", "as", "to", "be", "by"})
# A condition that closes a definition whose subject a formula splits: "מטריצה $A$ סקלרית אם".
CONDITION_WORDS = frozenset({"אם", "if", "iff"})
STOP_WORDS = FUNCTION_WORDS | UNIT_LABEL_WORDS | STRUCTURAL_WORDS
# Words that open a clause: a heading or label name holding one is a sentence, not a name
# ("בשיעור הנוכחי נתאר את X כאשר"; found 2026-11-17 in חישובית 2, dev).
CLAUSE_WORDS = frozenset(
    {"את", "כאשר", "אשר", "כי", "אם", "כדי", "לכן", "כלומר", "אזי", "שבו", "שבה", "שבהם"}
    | {"when", "where", "which", "that", "if", "then", "because"}
)
# A word written before a number this often, and in at least this share of its occurrences, is a
# unit label of this file ("שבוע 3", "הרצאה 4"), handled like `UNIT_LABEL_WORDS`, so no list has
# to name every course's unit words. Owner request 2026-11-17 ("שבוע" ranked high in חישובית 2).
UNIT_NUMBER_MIN_COUNT = 2
UNIT_NUMBER_SHARE = 0.5
_NUMBER_AFTER_RE = re.compile(r"[ \t]*[-–־]?[ \t]*\d")
# What may precede a label's name on its line: list numbers or letters and bullets. "o" counts
# as a bullet only before a space, because Word exports a hollow bullet as the letter o.
# A short formula at the start of a line is a bullet glyph the extractor read as math ("$0$").
_LABEL_PREFIX_RE = re.compile(
    r"(?:\s*(?:\d+(?:\.\d+)*[.)]?|[א-ת][.)]|[•·*▪●○◦\-–]|o(?=\s)|\$[^$\n]{1,3}\$))*\s*"
)
# Ends a label's name: a colon, after an optional remark in parentheses ("(שלושה בנאים):"), or
# a dash with a space on at least one side ("Boxing – ...", "רב מחלקתי- המסווג"). A dash
# between two letters joins one word ("תת-מרחב") and is not a separator.
# The maqaf "־" is a Hebrew dash; Word summaries use it as a separator ("קבוצה סופית ־ קבוצה ש").
_LABEL_END_RE = re.compile(r"\s*(?:\([^)\n]{0,40}\)\s*)?:|\s[-–—־]|[-–—־]\s")
# A line with no separator is a label only when this short and marked or alone in its block.
MAX_STANDALONE_WORDS = 4
# A trailing bullet that right-to-left extraction moved after the name ("Template method o –").
_TRAILING_BULLETS = frozenset({"o"})
# A sentence end inside a label line: a period, question or exclamation mark or semicolon followed
# by a space ("x.y" and "ע"י" are not sentence ends).
_SENTENCE_BREAK_RE = re.compile(r"[.!?;](?=\s)")
_GLOSS_RE = re.compile(r"\(\s*([A-Za-z][A-Za-z\s'\-]*?)\s*\)")


@dataclass(frozen=True)
class Mention:
    """One occurrence of a candidate: its span, unit and main section (-1 for none), a_cu."""

    start: int
    end: int
    unit: int
    section: int
    strength: float
    form: str  # the surface words as written


@dataclass
class Candidate:
    """A normal form with the sources that proposed it and all of its mentions."""

    key: str
    sources: list[str]
    aliases: list[str]
    mentions: list[Mention]

    @property
    def count(self) -> int:
        """Return the number of mentions."""
        return len(self.mentions)

    @property
    def form(self) -> str:
        """Return the most common surface form (ties broken alphabetically)."""
        forms = Counter(m.form for m in self.mentions)
        return min(forms, key=lambda form: (-forms[form], form))

    def by_unit(self) -> dict[int, int]:
        """Return mention counts per unit; -1 collects mentions outside any unit."""
        return dict(sorted(Counter(m.unit for m in self.mentions).items()))

    def by_section(self) -> dict[int, int]:
        """Return mention counts per main section; -1 collects mentions before the first one."""
        return dict(sorted(Counter(m.section for m in self.mentions).items()))

    def strength_by_unit(self) -> dict[int, float]:
        """Return a_cu per unit: the strongest mention of the candidate in that unit."""
        strengths: dict[int, float] = {}
        for m in self.mentions:
            strengths[m.unit] = max(strengths.get(m.unit, 0.0), m.strength)
        return dict(sorted(strengths.items()))


@dataclass
class Stage2Result:
    """Candidates of one document, the normalizer that produced their keys, and its word count."""

    candidates: list[Candidate]
    normalizer: PrefixNormalizer
    words: int
    language: str  # "he" or "en", by the letters of the prose


@dataclass
class _Pool:
    """Candidate keys found so far, with sources, aliases and strength at specific spans."""

    sources: dict[tuple[str, ...], set[str]] = field(default_factory=dict)
    aliases: dict[tuple[str, ...], set[str]] = field(default_factory=dict)
    strength_at: dict[tuple[int, int], float] = field(default_factory=dict)
    operator_mentions: dict[tuple[str, ...], list[Mention]] = field(default_factory=dict)

    def add(self, tokens: list[Token], source: str, strength: float = MENTION) -> None:
        """Record a phrase proposed by a source; a strength above MENTION is kept for its span."""
        if not tokens:
            return
        key = tuple(token.norm for token in tokens)
        self.sources.setdefault(key, set()).add(source)
        span = (tokens[0].start, tokens[-1].end)
        if strength > MENTION:
            self.strength_at[span] = max(self.strength_at.get(span, 0.0), strength)


def _is_stop(token: Token, words: frozenset[str] = FUNCTION_WORDS) -> bool:
    """Return True when the word, as written or normalized, is in the given list."""
    return token.text in words or token.norm in words


def _is_trigger(token: Token) -> bool:
    """Return True when the word is a definition trigger, also with an attached ש or ו."""
    if _is_stop(token, AFTER_TRIGGERS):
        return True
    for prefix in TRIGGER_PREFIXES:
        if token.text.startswith(prefix) and token.text[len(prefix) :] in AFTER_TRIGGERS:
            return True
    return False


def _trim(tokens: list[Token], words: frozenset[str] = FUNCTION_WORDS) -> list[Token]:
    """Drop listed words from both ends of a phrase."""
    start, end = 0, len(tokens)
    while start < end and _is_stop(tokens[start], words):
        start += 1
    while end > start and _is_stop(tokens[end - 1], words):
        is_superlative = tokens[end - 1].text in SUPERLATIVE_WORDS
        if is_superlative and end - 1 > start and not _is_stop(tokens[end - 2], words):
            break
        end -= 1
    return tokens[start:end]


def _is_latin(token: Token) -> bool:
    """Return True when the word is written in Latin letters."""
    return token.text[0].isascii()


def _ends_term(term: list[Token], token: Token) -> bool:
    """Return True when the word cannot continue the term that has been read so far.

    A term ends at a function word, at its length limit, where the script changes ("support
    vectors ואלו הנקודות") and at a word whose attested ו prefix opens a new clause ("ומסומן").
    """
    if len(term) == MAX_PHRASE_WORDS:
        return True
    if token.text in SUPERLATIVE_WORDS:
        return not term
    if _is_stop(token):
        return True
    if not term:
        return False
    starts_clause = token.text.startswith("ו") and token.norm != token.text
    return starts_clause or _is_latin(token) != _is_latin(term[-1])


def _term_after(tokens: list[Token]) -> list[Token]:
    """Return the term that starts a word list: fillers skipped, cut where `_ends_term` says."""
    position = 0
    while position < len(tokens) and _is_stop(tokens[position], LEADING_FILLERS):
        position += 1
    term: list[Token] = []
    for token in tokens[position:]:
        if _ends_term(term, token):
            break
        term.append(token)
    return term


def _term_before(tokens: list[Token]) -> list[Token]:
    """Return the term that ends a word list, read backwards up to the first function word."""
    term: list[Token] = []
    for token in reversed(tokens):
        if _is_stop(token) or len(term) == MAX_PHRASE_WORDS:
            break
        if term and _is_latin(token) != _is_latin(term[0]):
            break
        term.insert(0, token)
    return term


def _following_words(block: BlockText, segment_index: int, position: int) -> list[Token]:
    """Return the words after a position, continuing into the next segment past punctuation.

    "הגדרה — מטריצה הפיכה:" puts the term in the next segment. A formula between the two
    segments stops it: the term would be a symbol, not a name.
    """
    segment = block.segments[segment_index]
    if position + 1 < len(segment.tokens):
        return segment.tokens[position + 1 :]
    if segment_index + 1 >= len(block.segments):
        return []
    following = block.segments[segment_index + 1]
    gap_start, gap_end = segment.tokens[-1].end, following.tokens[0].start
    if any(gap_start <= f.start < gap_end for f in block.formulas):
        return []
    return following.tokens


def _head_noun_term(subject: list[Token], term: list[Token]) -> list[Token]:
    """Return the subject's head noun joined to a one-word defined term, else nothing.

    "מטריצה מדורגת תקרא קנונית" defines "מטריצה קנונית": the summary names the term by its
    adjective alone. The phrase counts only where it also appears whole in the text.
    """
    subject = _trim(subject, STOP_WORDS)
    if len(term) != 1 or not subject or _is_latin(subject[0]) != _is_latin(term[0]):
        return []
    return [subject[0], term[0]]


def _split_subject_term(block: BlockText, s: int) -> list[Token]:
    """Return noun + adjective of a definition that a formula splits, else nothing.

    "מטריצה $A_{n\\times n}$ סקלרית אם ..." gives "מטריצה סקלרית": segment `s` starts with
    up to two words and a condition word, a formula separates it from the segment before, and
    that segment ends with the noun.
    """
    segment, previous = block.segments[s], block.segments[s - 1]
    gap_start, gap_end = previous.tokens[-1].end, segment.tokens[0].start
    if not any(gap_start <= f.start < gap_end for f in block.formulas):
        return []
    condition = next(
        (p for p, t in enumerate(segment.tokens[:3]) if p and _is_stop(t, CONDITION_WORDS)), None
    )
    noun = previous.tokens[-1]
    if condition is None or _is_stop(noun, STOP_WORDS):
        return []
    words = segment.tokens[:condition]
    if any(_is_stop(t) or _is_latin(t) != _is_latin(noun) for t in words):
        return []
    return [noun, *words]


def _definition_terms(blocks: list[BlockText], records, units, pool: _Pool) -> None:
    """Source 1: terms that a definition pattern introduces (strength DEFINED)."""
    for block in blocks:
        record = records[block.index]
        in_definition = record.unit >= 0 and units[record.unit].type == "definition"
        for s, segment in enumerate(block.segments):
            for p, token in enumerate(segment.tokens):
                is_label = s == 0 and p == 0 and _is_stop(token, LABEL_TRIGGERS)
                if is_label or _is_trigger(token):
                    term = _term_after(_following_words(block, s, p))
                    # The end of a term is a guess ("נקרא דיאגרמת וורונוי המחלק"), so its
                    # shorter beginnings are proposed too.
                    for length in range(2, len(term) + 1):
                        pool.add(term[:length], "definition", DEFINED)
                    pool.add(_head_noun_term(segment.tokens[:p], term), "definition", DEFINED)
                elif _is_stop(token, BEFORE_TRIGGERS):
                    before = segment.tokens[:p]
                    if token.text == "defined" and before and before[-1].text in COPULAS:
                        before = before[:-1]
                    pool.add(_term_before(before), "definition", DEFINED)
            if s > 0:
                pool.add(_split_subject_term(block, s), "definition", DEFINED)
            if in_definition and s == 0:
                copula = next(
                    (
                        p
                        for p, t in enumerate(segment.tokens[: MAX_PHRASE_WORDS + 1])
                        if p and _is_stop(t, COPULAS)
                    ),
                    None,
                )
                if copula is not None:
                    pool.add(_trim(segment.tokens[:copula], STOP_WORDS), "definition", DEFINED)


def _add_name(tokens: list[Token], source: str, pool: _Pool, keep_raw: bool = True) -> None:
    """Add a heading or label name whole, trimmed, and without a leading "הגדרת".

    "משפט המימדים" stays whole (the theorem's name) and also gives "המימדים". A leading
    "הגדרת" usually announces the name ("הגדרת סכום של תת-מרחבים") but is sometimes part of
    it ("הגדרת חיים"), so both forms are kept. `keep_raw` adds the name with its edge words
    ("No Path", "רציפות במידה שווה"); it is off for a bare short line, where an edge word is
    more often a sentence fragment ("של תכונות") than part of a name.
    """
    for i, token in enumerate(tokens):
        if token.text.lower() in COMPARISON_WORDS:
            _add_name(tokens[:i], source, pool, keep_raw)
            _add_name(tokens[i + 1 :], source, pool, keep_raw)
    announced = tokens
    while announced and _is_stop(announced[0], AFTER_TRIGGERS | LABEL_TRIGGERS):
        announced = announced[1:]
    forms = [tokens, announced] if keep_raw else []
    forms += [
        _trim(tokens),
        _trim(tokens, STOP_WORDS),
        _trim(announced),
        _trim(announced, STOP_WORDS),
    ]
    # A name that mixes scripts ("בעיית Rod Cutting") also gives each one-script run.
    forms += [_trim(run, STOP_WORDS) for run in _script_runs(tokens) if len(run) < len(tokens)]
    for name in forms:
        if any(t.text.lower() in CLAUSE_WORDS for t in name):
            continue
        if 0 < len(name) <= MAX_HEADING_WORDS and not all(_is_stop(t, STOP_WORDS) for t in name):
            pool.add(name, source, SUBJECT)


def _script_runs(tokens: list[Token]) -> list[list[Token]]:
    """Split words into maximal runs written in one script (Hebrew or Latin)."""
    runs: list[list[Token]] = []
    for token in tokens:
        if runs and _is_latin(runs[-1][-1]) == _is_latin(token):
            runs[-1].append(token)
        else:
            runs.append([token])
    return runs


def _by_line(tokens: list[Token], text: str) -> list[list[Token]]:
    """Split a run of words where a line break falls between two of them."""
    lines: list[list[Token]] = []
    for token in tokens:
        if lines and "\n" not in text[lines[-1][-1].end : token.start]:
            lines[-1].append(token)
        else:
            lines.append([token])
    return lines


def _heading_terms(blocks: list[BlockText], records, text: str, pool: _Pool) -> None:
    """Source 2: non-generic headings and the names of named units (strength SUBJECT).

    Each line is also a name on its own: stage 1 sometimes joins a heading with the next line
    ("9.3 רציפות במידה שווה" + "הגדרה"), which would make the whole too long.
    """
    for block in blocks:
        if records[block.index].role not in ("heading", "title", "unit_label"):
            continue
        for segment in block.segments:
            _add_name(segment.tokens, "heading", pool)
            lines = _by_line(segment.tokens, text)
            for line in lines if len(lines) > 1 else []:
                _add_name(line, "heading", pool)


def _label_terms(blocks: list[BlockText], records, text: str, pool: _Pool) -> None:
    """Source 7: the name on a label line, "5. נוסחאות וייטה:" (strength SUBJECT).

    Read line by line, because stage 1 may put several lines in one block. A line names what
    follows when, after optional list numbers or bullets, a short run of words ends at a colon
    or a dash ("▪ Boxing – מהטיפוס..."), or when the whole line is a few words and is either
    marked ("• העתקה רדודה") or in a block whose lines are all that short. Such names often appear once, below the
    n-gram threshold, yet name exam material (committee gold, 2026-11-16).
    """
    for block in blocks:
        tokens = [token for segment in block.segments for token in segment.tokens]
        if not tokens:
            continue
        record = records[block.index]
        lines = _line_spans(text, record.start, record.end)
        # A block of short lines reads like a heading ("הכללה תחת יציבות ההחלטה" over
        # "$Robustness$."), so each of its lines may name a term, as a line alone does.
        is_all_short = all(
            sum(1 for t in tokens if start <= t.start < end) <= MAX_STANDALONE_WORDS
            for start, end in lines
        )
        for line_start, line_end in lines:
            name, is_separated = _label_on_line(
                block, tokens, text, line_start, line_end, is_all_short
            )
            _add_name(name, "label", pool, keep_raw=is_separated)


def _line_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return the (start, end) offsets of each line between two offsets."""
    spans, position = [], start
    for line in text[start:end].split("\n"):
        spans.append((position, position + len(line)))
        position += len(line) + 1
    return spans


def _words_between(
    block: BlockText, tokens: list[Token], start: int, end: int, text: str = ""
) -> list[Token]:
    """Return the words inside a span, or nothing when a formula or a sentence end is inside it.

    A sentence end between two of the words means the line is prose ("שיעור נוכחי. שיעור
    נוכחי."), not a name (fixed 2026-11-17).
    """
    if any(start <= f.start < end for f in block.formulas):
        return []
    words = [t for t in tokens if start <= t.start and t.end <= end]
    while words and words[-1].text in _TRAILING_BULLETS:
        words.pop()
    if words and _SENTENCE_BREAK_RE.search(text, words[0].end, words[-1].start):
        return []
    return words


def _label_on_line(
    block, tokens, text, start: int, end: int, is_all_short: bool
) -> tuple[list[Token], bool]:
    """Return the name a label line gives (or nothing) and whether a colon or dash ended it.

    "מסקנה – No Path:" names "No Path": a first part made only of listed words is skipped.
    """
    prefix_end = _LABEL_PREFIX_RE.match(text, start, end).end()
    is_marked = bool(text[start:prefix_end].strip())
    separators = list(_LABEL_END_RE.finditer(text, prefix_end, end))[:2]
    if not separators:
        words = _words_between(block, tokens, prefix_end, end, text)
        is_short = 0 < len(words) <= MAX_STANDALONE_WORDS
        return (words if is_short and (is_marked or is_all_short) else []), False
    bounds = [prefix_end, *(s for m in separators for s in (m.start(), m.end()))]
    for part_start, part_end in zip(bounds[0::2], bounds[1::2]):
        words = _words_between(block, tokens, part_start, part_end, text)
        if words and not all(_is_stop(t, STOP_WORDS) for t in words):
            return words, True
    return [], False


def _emphasized(token: Token, runs, run_starts: list[int], allowed: list[bool]) -> bool:
    """Return True when the word starts inside a run of an allowed emphasized style."""
    i = bisect.bisect_right(run_starts, token.start) - 1
    return i >= 0 and runs[i].start <= token.start < runs[i].end and allowed[runs[i].style]


def _emphasis_style_mask(info) -> list[bool]:
    """Return, per style, whether it is an emphasis signal still meaningful in this document."""
    share = info.signal_share
    use_bold = share.get("bold", 0.0) <= EMPHASIS_MAX_SHARE
    use_underline = share.get("underline", 0.0) <= EMPHASIS_MAX_SHARE
    return [
        (style.bold and use_bold) or (style.underline and use_underline) or style.colored
        for style in info.styles
    ]


def _emphasis_terms(blocks: list[BlockText], info, pool: _Pool) -> None:
    """Source 3: bold, underlined or coloured phrases in prose blocks.

    Skipped in a block where emphasis covers more than half of the words: there it marks a
    whole line, not a term. In the opening block of a unit the phrase is its subject.
    """
    runs = sorted(info.runs, key=lambda run: run.start)
    run_starts = [run.start for run in runs]
    allowed = _emphasis_style_mask(info)
    for block in blocks:
        record = info.blocks[block.index]
        if record.role not in ("body", "unit_label"):
            continue
        flags = [
            [_emphasized(t, runs, run_starts, allowed) for t in segment.tokens]
            for segment in block.segments
        ]
        words = sum(len(f) for f in flags)
        if not words or sum(sum(f) for f in flags) > EMPHASIS_MAX_SHARE * words:
            continue
        previous = info.blocks[block.index - 1] if block.index else None
        opens_unit = record.unit >= 0 and (previous is None or previous.unit != record.unit)
        for segment, segment_flags in zip(block.segments, flags):
            for phrase in _flagged_runs(segment.tokens, segment_flags):
                phrase = _trim(phrase)
                if 0 < len(phrase) <= MAX_EMPHASIS_WORDS:
                    pool.add(phrase, "emphasis", SUBJECT if opens_unit else MENTION)


def _flagged_runs(tokens: list[Token], flags: list[bool]) -> list[list[Token]]:
    """Return the maximal runs of consecutive flagged tokens."""
    phrases: list[list[Token]] = []
    current: list[Token] = []
    for token, flag in zip(tokens, flags):
        if flag:
            current.append(token)
        elif current:
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    return phrases


def _phrases(segment: Segment):
    """Yield every 1-4 word phrase of a segment that neither starts nor ends with a function word."""
    tokens = segment.tokens
    for i in range(len(tokens)):
        if _is_stop(tokens[i]):
            continue
        for n in range(1, MAX_PHRASE_WORDS + 1):
            if i + n > len(tokens):
                break
            if not _is_stop(tokens[i + n - 1]):
                yield tokens[i : i + n]


def _ngram_terms(blocks: list[BlockText], pool: _Pool, min_count: int) -> None:
    """Source 4: phrases of one to four words seen at least `min_count` times."""
    counts: Counter[tuple[str, ...]] = Counter()
    first_seen: dict[tuple[str, ...], list[Token]] = {}
    for block in blocks:
        for segment in block.segments:
            for phrase in _phrases(segment):
                key = tuple(t.norm for t in phrase)
                counts[key] += 1
                first_seen.setdefault(key, phrase)
    for key, count in counts.items():
        if count >= min_count:
            pool.add(first_seen[key], "ngram")


def _gloss_terms(blocks: list[BlockText], text: str, pool: _Pool) -> None:
    """Source 5: an English term in parentheses after a Hebrew one ("מרחב וקטורי (vector space)").

    Both sides become candidates and aliases of each other. The Hebrew side is the longest
    ending of the preceding words that is already a candidate, else as many words as the English.
    """
    for block in blocks:
        for s, segment in enumerate(block.segments):
            following = block.segments[s + 1] if s + 1 < len(block.segments) else None
            if following is None or not _is_hebrew_segment(segment):
                continue
            between = text[segment.tokens[-1].end : following.tokens[-1].end + 2]
            match = _GLOSS_RE.match(between.lstrip())
            english = following.tokens
            if not match or len(english) > MAX_GLOSS_WORDS or _is_hebrew_segment(following):
                continue
            if match.group(1).lower().split() != [t.text for t in english]:
                continue
            hebrew = _gloss_hebrew_side(segment.tokens, len(english), pool)
            if not hebrew:
                continue
            pool.add(hebrew, "gloss")
            pool.add(english, "gloss")
            hebrew_key = " ".join(t.norm for t in hebrew)
            english_key = " ".join(t.norm for t in english)
            pool.aliases.setdefault(tuple(t.norm for t in hebrew), set()).add(english_key)
            pool.aliases.setdefault(tuple(t.norm for t in english), set()).add(hebrew_key)


def _is_hebrew_segment(segment: Segment) -> bool:
    """Return True when the segment's last word is Hebrew."""
    return "א" <= segment.tokens[-1].text[0] <= "ת"


def _gloss_hebrew_side(tokens: list[Token], english_words: int, pool: _Pool) -> list[Token]:
    """Return the Hebrew words that an English gloss translates (see `_gloss_terms`)."""
    for length in range(min(MAX_PHRASE_WORDS, len(tokens)), 0, -1):
        ending = tokens[-length:]
        if _is_stop(ending[0]) or all(_is_stop(t, STOP_WORDS) for t in ending):
            continue
        if tuple(t.norm for t in ending) in pool.sources:
            return ending
    return _trim(tokens[-min(MAX_PHRASE_WORDS, max(1, english_words)) :], STOP_WORDS)


def _operator_terms(blocks: list[BlockText], records, language: str, normalizer, pool) -> None:
    """Source 6: concepts named by LaTeX operators, one weak mention per formula and operator."""
    for block in blocks:
        record = records[block.index]
        for formula in block.formulas:
            for pattern, hebrew, english in OPERATORS:
                if not pattern.search(formula.latex):
                    continue
                name = hebrew if language == "he" else english
                key = normalizer.normalize_phrase(tuple(name.split()))
                pool.sources.setdefault(key, set()).add("operator")
                mention = Mention(
                    formula.start, formula.end, record.unit, record.section, OPERATOR, name
                )
                pool.operator_mentions.setdefault(key, []).append(mention)


def _numbered_words(blocks: list[BlockText], text: str) -> set[str]:
    """Return the normal forms written before a number often enough to be unit labels."""
    numbered: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for block in blocks:
        for segment in block.segments:
            for token in segment.tokens:
                total[token.norm] += 1
                if _NUMBER_AFTER_RE.match(text, token.end):
                    numbered[token.norm] += 1
    return {
        word
        for word, count in numbered.items()
        if count >= UNIT_NUMBER_MIN_COUNT and count >= UNIT_NUMBER_SHARE * total[word]
    }


def _modifier_words(keys: set[tuple[str, ...]]) -> set[str]:
    """Return the words that appear in a phrase other than as its head.

    The head is the first word of a Hebrew phrase and the last of an English one. A structural
    word that heads a phrase keeps its structural sense ("שיעור נוכחי", "חלק ראשון"); as a
    modifier it is content ("מרחב התוצאות"). Fixed 2026-11-17: "שיעור" passed the stop list
    because junk heading phrases began with it.
    """
    words = set()
    for key in keys:
        # A phrase that begins or ends with a function word is a sentence piece ("למד שבוע
        # עבר"), and must not vouch for its words.
        if key[0] in FUNCTION_WORDS or key[-1] in FUNCTION_WORDS:
            continue
        head = len(key) - 1 if all(word.isascii() for word in key) else 0
        words |= {word for i, word in enumerate(key) if i != head}
    return words


# Unit-label heads that open a named result, so "משפט X" is a name and "הגדרה X" is not.
NAMED_RESULT_WORDS = frozenset(
    {"משפט", "למה", "למת", "אקסיומת", "אקסיומה", "עקרון", "עיקרון", "חוק", "נוסחת"}
    | {"theorem", "lemma", "axiom", "principle", "law"}
)
# Function words that may sit inside a name ("מרחב של מטריצות", "theory of computation").
NAME_LINK_WORDS = frozenset({"של", "of"})


def is_name_like(words: list[str]) -> bool:
    """Return whether a phrase reads as a term's name, not a sentence piece or a labelled line.

    No function word at an edge, none inside except `NAME_LINK_WORDS`, no comparison word ("A vs
    B" names two things), and the head is not a unit
    label or structural word ("הגדרה אלגברית", "דרך atpase"). Used by stage 4 to choose which
    longer concepts may absorb a fragment.
    """
    if not words or words[0] in FUNCTION_WORDS or words[-1] in FUNCTION_WORDS:
        return False
    if any(word in FUNCTION_WORDS and word not in NAME_LINK_WORDS for word in words):
        return False
    if any(word in COMPARISON_WORDS for word in words):
        return False
    head = words[-1] if all(word.isascii() for word in words) else words[0]
    # A named result is a name: "משפט ויירשטראס", "למת צורן", "Bayes theorem" (2026-11-17,
    # "היינה" stayed apart because "משפט היינה" was not a name).
    if len(words) > 1 and head in NAMED_RESULT_WORDS:
        return True
    return head not in UNIT_LABEL_WORDS and head not in STRUCTURAL_WORDS


def _kept_keys(pool: _Pool, numbered: set[str] = frozenset()) -> set[tuple[str, ...]]:
    """Apply the academic stop list and its exception; return the candidate keys that stay.

    A unit label alone ("משפט", or a word of `numbered`) stays only with a definition pattern or
    an English gloss. A structural word alone ("תוצאה") also stays when a kept phrase holds it
    as a modifier ("מרחב התוצאות"). A phrase made only of listed words is dropped.
    """
    multiword = {
        key for key in pool.sources if len(key) > 1 and not all(word in STOP_WORDS for word in key)
    }
    in_phrases = _modifier_words(multiword)
    kept = set(multiword)
    for key, sources in pool.sources.items():
        if len(key) != 1:
            continue
        word = key[0]
        evidence = bool(sources & {"definition", "gloss"})
        # One Latin letter is a variable or a list marker ("x", "e"), not a term.
        if word in FUNCTION_WORDS or (len(word) == 1 and word.isascii()):
            continue
        if (word in UNIT_LABEL_WORDS or word in numbered) and not evidence:
            continue
        if word in STRUCTURAL_WORDS and not (evidence or word in in_phrases):
            continue
        kept.add(key)
    return kept


def _scan_mentions(blocks, records, keys, pool: _Pool) -> dict[tuple[str, ...], list[Mention]]:
    """Find every occurrence of every kept key in prose, plus the operator mentions."""
    longest = max((len(key) for key in keys), default=0)
    found: dict[tuple[str, ...], list[Mention]] = {}
    for block in blocks:
        record = records[block.index]
        for segment in block.segments:
            tokens = segment.tokens
            for i in range(len(tokens)):
                for n in range(1, min(longest, len(tokens) - i) + 1):
                    key = tuple(t.norm for t in tokens[i : i + n])
                    if key not in keys:
                        continue
                    span = (tokens[i].start, tokens[i + n - 1].end)
                    strength = pool.strength_at.get(span, MENTION)
                    form = " ".join(t.text for t in tokens[i : i + n])
                    mention = Mention(*span, record.unit, record.section, strength, form)
                    found.setdefault(key, []).append(mention)
    for key, mentions in pool.operator_mentions.items():
        if key in keys:
            found.setdefault(key, []).extend(mentions)
    return found


def _language(blocks: list[BlockText]) -> str:
    """Return "he" when Hebrew letters outnumber Latin ones in the prose, else "en"."""
    hebrew = latin = 0
    for block in blocks:
        for segment in block.segments:
            for token in segment.tokens:
                if "א" <= token.text[0] <= "ת":
                    hebrew += len(token.text)
                else:
                    latin += len(token.text)
    return "he" if hebrew >= latin else "en"


def extract_candidates(
    result, min_phrase_count: int = MIN_PHRASE_COUNT, lemmas: dict[str, str] | None = None
) -> Stage2Result:
    """Return the stage-2 candidates of one extraction.

    Args:
        result: A successful hebmath ``ExtractionResult`` with ``structure``.
        min_phrase_count: The n-gram frequency threshold; a parameter so calibration can trace
            recall against pool size.
        lemmas: Accepted model lemmas (`concept_pipeline.stage2_model_normalizer`); None keeps the prefix
            fallback alone, the output before model normalization.

    Returns:
        A ``Stage2Result``; candidates are sorted by key and every list inside is in text order,
        so the same input always gives the same output.

    Raises:
        ValueError: The extraction failed or has no structure.
    """
    if not result.ok or result.structure is None:
        raise ValueError("stage 2 needs a successful extraction with a stage-1 structure")
    info = result.structure
    blocks = split_blocks(result)
    tokens = [t for b in blocks for s in b.segments for t in s.tokens]
    counts = Counter(t.text for t in tokens)
    normalizer = LemmaNormalizer(counts, lemmas) if lemmas else PrefixNormalizer(counts)
    for token in tokens:
        token.norm = normalizer.normalize_word(token.text)
    language = _language(blocks)

    pool = _Pool()
    _definition_terms(blocks, info.blocks, info.units, pool)
    text = clean_text(result.text)
    _heading_terms(blocks, info.blocks, text, pool)
    _emphasis_terms(blocks, info, pool)
    _ngram_terms(blocks, pool, min_phrase_count)
    _gloss_terms(blocks, result.text, pool)
    _operator_terms(blocks, info.blocks, language, normalizer, pool)
    _label_terms(blocks, info.blocks, text, pool)

    keys = _kept_keys(pool, _numbered_words(blocks, text))
    mentions = _scan_mentions(blocks, info.blocks, keys, pool)
    candidates = [
        Candidate(
            key=" ".join(key),
            sources=[source for source in SOURCES if source in pool.sources[key]],
            aliases=sorted(pool.aliases.get(key, ())),
            mentions=sorted(found, key=lambda m: (m.start, m.end, m.strength)),
        )
        for key, found in mentions.items()
    ]
    candidates.sort(key=lambda candidate: candidate.key)
    return Stage2Result(candidates, normalizer, len(tokens), language)


def to_snapshot(stage2: Stage2Result) -> dict:
    """Return the stage-2 output as plain JSON-ready data (the stage snapshot, plan section 1).

    Mentions are stored as ``[start, end, unit, section, strength]``; counts per unit and section
    use string keys because JSON object keys are strings.
    """
    return {
        "stage": 2,
        "version": STAGE2_VERSION,
        "language": stage2.language,
        "words": stage2.words,
        "candidates": [
            {
                "key": c.key,
                "form": c.form,
                "sources": c.sources,
                "aliases": c.aliases,
                "count": c.count,
                "by_unit": {str(unit): n for unit, n in c.by_unit().items()},
                "by_section": {str(section): n for section, n in c.by_section().items()},
                "strength_by_unit": {str(unit): a for unit, a in c.strength_by_unit().items()},
                "mentions": [[m.start, m.end, m.unit, m.section, m.strength] for m in c.mentions],
            }
            for c in stage2.candidates
        ],
    }


def check_candidates(stage2: Stage2Result, result) -> list[str]:
    """Return every stage-2 gate invariant the candidates break (empty when all hold).

    The algorithmic gate (plan, section יא): counts per unit and per section add up to the
    total; each mention lies inside one block and carries that block's unit and section, and a
    unit's mention lies inside the unit; normalizing a key again leaves it unchanged; keys are
    unique. Determinism is checked by running twice (tests).
    """
    info = result.structure
    records = [r for r in info.blocks if r.role not in SKIPPED_ROLES]
    starts = [r.start for r in records]
    problems = []
    if len({c.key for c in stage2.candidates}) != len(stage2.candidates):
        problems.append("two candidates share a key")
    for c in stage2.candidates:
        if sum(c.by_unit().values()) != c.count or sum(c.by_section().values()) != c.count:
            problems.append(f"{c.key!r}: counts per unit or section do not add up to the total")
        words = tuple(c.key.split())
        if stage2.normalizer.normalize_phrase(words) != words:
            problems.append(f"{c.key!r}: normalizing the key again changes it")
        for m in c.mentions:
            i = bisect.bisect_right(starts, m.start) - 1
            record = records[i] if i >= 0 else None
            if record is None or m.end > record.end:
                problems.append(f"{c.key!r}: mention {m.start}-{m.end} is not inside one block")
            elif (m.unit, m.section) != (record.unit, record.section):
                problems.append(f"{c.key!r}: mention {m.start}-{m.end} has the wrong unit")
            elif m.unit >= 0 and not info.units[m.unit].start <= m.start < info.units[m.unit].end:
                problems.append(f"{c.key!r}: mention {m.start}-{m.end} is outside its unit")
    return problems
