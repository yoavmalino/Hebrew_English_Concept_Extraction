"""Stage 4 of the concept-graph pipeline: merge pairs, their evidence, and grouping by decisions.

The algorithm only proposes pairs and collects evidence; a model or the user decides whether two
candidates name the same concept (design stage 4). Grouping is deterministic: the same decisions
give the same groups in any input order. Pure: no I/O and no model calls (the decider is
`concept_pipeline.stage4_merge_decider`).
"""

from __future__ import annotations

import bisect
import itertools
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .normalize import FINAL_TO_PLAIN, MIN_LETTERS_FOR_TYPO, _edit_distance
from .stage2_candidates import (
    MENTION,
    NAME_LINK_WORDS,
    NAMED_RESULT_WORDS,
    STOP_WORDS,
    SUBJECT,
    Candidate,
    Mention,
    Stage2Result,
    is_name_like,
)
from .tokens import clean_text

STAGE4_VERSION = "4.2"
# Pair finders, in report order. "gloss" is the author's own evidence (an English term in
# parentheses); the others only find pairs for the decider (design stage 4, layer 3).
# "translation" comes from a model (`concept_pipeline.stage4_translation_finder`) and is passed in, because
# this module makes no model calls (owner decision 2026-11-16, instead of embeddings).
# "named" pairs a named result with its bare name ("משפט ויירשטראס" / "ויירשטראס") and is decided
# by rule, never by the model (owner decision 2026-11-20; design stage 4, "fragments and names").
FINDERS = ("gloss", "acronym", "spelling", "typo", "glyph", "named", "translation", "synonym")
RULE_FINDERS = frozenset({"named"})
# English result words that may end a named result ("bayes theorem"); "principle" is left out
# because "principle component" (a misspelt "principal") paired with "component" on dev.
NAMED_RESULT_SUFFIXES = frozenset({"theorem", "lemma", "law"})
# A named result is a name only when the author writes it as one; an n-gram alone is prose
# ("המשפט תקף", "משפט ערך", dev 2026-11-20).
NAMING_SOURCES = frozenset({"definition", "heading", "label", "emphasis", "gloss"})
# A bare name with its own definition or operator is a concept of its own ("מימד", dim), not
# a short form of the result ("משפט המימדים").
OWN_CONCEPT_SOURCES = frozenset({"definition", "operator"})
# An acronym has this many letters; a longer one is a word, a shorter one a variable.
MIN_ACRONYM_LETTERS, MAX_ACRONYM_LETTERS = 2, 5
# One changed letter is a spelling variant only in a long word (the gold's rule too).
MIN_TYPO_LETTERS = MIN_LETTERS_FOR_TYPO
# An English word has no prefixes to confuse with a typo, so a shorter one is trusted
# ("bais" / "bias", 2026-11-17, חישובית 2). Two swapped neighbouring letters count as one typo.
MIN_LATIN_TYPO_LETTERS = 4
# A fragment: a shorter concept whose mentions sit inside one longer concept's mentions
# ("אמינו" in "חומצה אמינו"). Absorbed only when at least this share of its mentions is inside
# that one concept and at most `FRAGMENT_MAX_ALONE` stand alone, so a concept used on its own or
# inside many phrases ("גבול") stays. Owner decision 2026-11-17: absorbed, never an alias, so a
# later file that uses it alone still gets it as a concept.
FRAGMENT_SHARE = 0.8
FRAGMENT_MAX_ALONE = 1
# A block of keys that share a spelling skeleton larger than this is a common root ("עבר",
# "עובר", "עברי", ...), not variants of one term; it is skipped and counted in the report.
MAX_KEYS_PER_BLOCK = 8
# Hebrew letters that one font maps to the other when its Unicode table is broken (ש/ט,
# root CLAUDE.md, text extractor item).
GLYPH_CONFUSIONS = str.maketrans("ט", "ש")
# Evidence per candidate: its strongest mentions, each as a short window around the mention.
MAX_CONTEXTS = 2
CONTEXT_CHARS = 90
# Decision strength, weakest first (design stage 4: an explicit "different" breaks a chain at
# its weakest link).
MODEL_STRENGTH, EVIDENCE_STRENGTH, USER_STRENGTH = 1, 2, 3
_SENTENCE_END_RE = re.compile(r"[.!?\n]")


@dataclass(frozen=True)
class MergePair:
    """Two candidate keys (in sorted order) that a finder proposes as one concept."""

    first: str
    second: str
    finders: tuple[str, ...]

    @property
    def has_direct_evidence(self) -> bool:
        """Return True when the author wrote one as the other's name in parentheses."""
        return "gloss" in self.finders


@dataclass(frozen=True)
class Decision:
    """A same/different verdict on a pair, with its strength and who gave it."""

    first: str
    second: str
    is_same: bool
    strength: int
    by: str  # "model:<name>", "user", "test"


@dataclass
class Concept:
    """Candidates merged into one concept: the most mentioned key names it."""

    key: str
    members: list[str]
    sources: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    # The most mentioned member. Stage 5's T reads its words, not the key's, so a group named by a
    # longer member ("זיזה" as "פעולה זיזה") keeps the specificity of the word the text uses
    # (2026-11-20: the class word pulled "זיזה" from rank 4 to 12 and "ראדמאכר" from 12 to 63).
    core: str = ""

    @property
    def count(self) -> int:
        """Return the number of mentions of all members."""
        return len(self.mentions)

    @property
    def written(self) -> str:
        """Return the concept's name as the text writes it.

        The key is built from normal forms and may read oddly; the name shown is the key when the
        text writes it exactly, else the written form closest in length to the key (so the bare
        form beats one with prefixes), then the most frequent.
        """
        forms = Counter(m.form.lower() for m in self.mentions if m.form)
        if not forms or self.key in forms:
            return self.key
        return min(forms, key=lambda form: (abs(len(form) - len(self.key)), -forms[form], form))


def _is_hebrew(word: str) -> bool:
    """Return True when the word's first letter is Hebrew."""
    return bool(word) and "א" <= word[0] <= "ת"


def _ordered(first: str, second: str) -> tuple[str, str]:
    """Return the two keys in sorted order, the identity of a pair."""
    return (first, second) if first <= second else (second, first)


def _gloss_pairs(candidates: list[Candidate], keys: set[str]) -> set[tuple[str, str]]:
    """Return pairs the author linked with an English term in parentheses (stage-2 aliases)."""
    return {
        _ordered(c.key, alias)
        for c in candidates
        for alias in c.aliases
        if alias != c.key and alias in keys
    }


def _acronym_letters(key: str) -> str | None:
    """Return the letters of a one-word acronym ("df", 'ט"ל'), or None when it is not one."""
    if " " in key:
        return None
    if key.isascii():
        letters = key
    elif '"' in key or "״" in key:
        letters = key.replace('"', "").replace("״", "").translate(FINAL_TO_PLAIN)
    else:
        return None
    is_acronym = letters.isalpha() and MIN_ACRONYM_LETTERS <= len(letters) <= MAX_ACRONYM_LETTERS
    return letters if is_acronym else None


def _acronym_pairs(keys: set[str]) -> set[tuple[str, str]]:
    """Return (acronym, phrase) pairs whose initials spell the acronym ("df" = document frequency)."""
    by_initials: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        words = key.split()
        if MIN_ACRONYM_LETTERS <= len(words) <= MAX_ACRONYM_LETTERS:
            by_initials["".join(word[0] for word in words).translate(FINAL_TO_PLAIN)].append(key)
    pairs = set()
    for key in keys:
        letters = _acronym_letters(key)
        for phrase in by_initials.get(letters, []) if letters else []:
            pairs.add(_ordered(key, phrase))
    return pairs


def _skeleton(word: str) -> str:
    """Return a Hebrew word without ו and י after its first letter, finals plain ("חיסרון")."""
    if not _is_hebrew(word):
        return word
    plain = word.translate(FINAL_TO_PLAIN)
    return plain[0] + plain[1:].replace("ו", "").replace("י", "")


def _pairs_in_blocks(blocks: dict, skipped: list[int]) -> set[tuple[str, str]]:
    """Return every pair inside each block; a block above `MAX_KEYS_PER_BLOCK` is skipped."""
    pairs = set()
    for members in blocks.values():
        if len(members) > MAX_KEYS_PER_BLOCK:
            skipped.append(len(members))
            continue
        pairs |= set(itertools.combinations(sorted(members), 2))
    return pairs


def _spelling_pairs(keys: set[str], skipped: list[int]) -> set[tuple[str, str]]:
    """Return keys that differ only in full or defective spelling ("חיסרון" / "חסרון")."""
    blocks: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for key in keys:
        blocks[tuple(_skeleton(word) for word in key.split())].add(key)
    return _pairs_in_blocks(blocks, skipped)


def _glyph_pairs(keys: set[str], skipped: list[int]) -> set[tuple[str, str]]:
    """Return keys that differ only in a letter pair a broken font confuses (ש/ט)."""
    blocks: dict[str, set[str]] = defaultdict(set)
    for key in keys:
        if _is_hebrew(key):
            blocks[key.translate(GLYPH_CONFUSIONS)].add(key)
    return _pairs_in_blocks(blocks, skipped)


def _is_swap(first: str, second: str) -> bool:
    """Return True when the words differ only by two neighbouring letters swapped."""
    if len(first) != len(second):
        return False
    diff = [i for i, (a, b) in enumerate(zip(first, second)) if a != b]
    return (
        len(diff) == 2
        and diff[1] == diff[0] + 1
        and first[diff[0]] == second[diff[1]]
        and first[diff[1]] == second[diff[0]]
    )


def _is_one_typo(first: str, second: str) -> bool:
    """Return True when two keys differ by one typo: a changed, added or lost letter, or a swap."""
    distance = sum(_edit_distance(x, y) for x, y in zip(first.split(), second.split()))
    if distance == 1:
        return True
    changed = [(x, y) for x, y in zip(first.split(), second.split()) if x != y]
    return len(changed) == 1 and _is_swap(*changed[0])


def _typo_pairs(keys: set[str], skipped: list[int]) -> set[tuple[str, str]]:
    """Return keys equal but for one typo in one long word (`MIN_TYPO_LETTERS`, Latin shorter)."""
    blocks: dict[tuple, set[str]] = defaultdict(set)
    for key in keys:
        words = key.split()
        for i, word in enumerate(words):
            minimum = MIN_LATIN_TYPO_LETTERS if word.isascii() else MIN_TYPO_LETTERS
            if len(word) < minimum:
                continue
            others = (len(words), i, tuple(words[:i] + words[i + 1 :]))
            # Blocked by the other words and by the word with one letter deleted: a changed
            # letter gives one shared deletion, an added letter the shorter word itself. The
            # old block by word length put every one-word key of a length together, so the
            # block was skipped and "אנטרופיה" / "אנתרופיה" never met (2026-11-16).
            for variant in {word} | {word[:j] + word[j + 1 :] for j in range(len(word))}:
                blocks[(*others, variant)].add(key)
    near = _pairs_in_blocks(blocks, skipped)
    return {(a, b) for a, b in near if _is_one_typo(a, b)}


def _named_tail(key: str) -> str | None:
    """Return the bare name inside a named result ("למה של היינה בורל" -> "היינה בורל"), or None.

    Hebrew opens with the result word ("משפט פרמה"); English may also end with it ("bayes
    theorem"). A link word after the result word ("של", "of") is dropped.
    """
    words = key.split()
    if len(words) < 2:
        return None
    is_latin = key.isascii()
    # English puts the result word first only before "of" ("law of large numbers"); without it
    # the word is a modifier ("principle component").
    if words[0] in NAMED_RESULT_WORDS and (not is_latin or words[1] in NAME_LINK_WORDS):
        tail = words[2:] if words[1] in NAME_LINK_WORDS else words[1:]
    elif is_latin and words[-1] in NAMED_RESULT_SUFFIXES:
        tail = words[:-1]
    else:
        return None
    if not tail or tail[0] in NAMED_RESULT_WORDS or all(word in STOP_WORDS for word in tail):
        return None
    return " ".join(tail)


def _named_pairs(candidates: list[Candidate]) -> set[tuple[str, str]]:
    """Return (bare name, named result) pairs whose bare name is a candidate of its own.

    Guards (dev replay 2026-11-20): the result must be written as a name (`NAMING_SOURCES`), so
    prose ("המשפט תקף", "משפט ערך") gets no pair; a bare name that is a concept in its own right
    (`OWN_CONCEPT_SOURCES`: "מימד" next to "משפט המימדים") stays apart; and a bare name that two
    different results share exactly ("משפט X" and "למה X") is ambiguous and gets no pair.
    """
    by_key = {c.key: c for c in candidates}
    results_by_tail: dict[str, set[str]] = defaultdict(set)
    for key, candidate in by_key.items():
        tail = _named_tail(key)
        if tail:
            results_by_tail[tail].add(key)
    pairs = set()
    for tail, results in results_by_tail.items():
        if tail not in by_key or len(results) != 1:
            continue
        result = next(iter(results))
        if not NAMING_SOURCES & set(by_key[result].sources):
            continue
        if OWN_CONCEPT_SOURCES & set(by_key[tail].sources):
            continue
        pairs.add(_ordered(tail, result))
    return pairs


def decide_by_rule(pairs: list[MergePair]) -> tuple[list[Decision], list[MergePair]]:
    """Split pairs into decisions made by rule and the pairs the model still has to decide.

    A pair a `RULE_FINDERS` finder proposed is "same" with `EVIDENCE_STRENGTH`, so only a
    user's "different" outweighs it; it is never sent to the model, which saves the call.

    Returns:
        ``(decisions, model_pairs)``.
    """
    decisions, model_pairs = [], []
    for pair in pairs:
        rule = sorted(RULE_FINDERS & set(pair.finders))
        if rule:
            decisions.append(
                Decision(pair.first, pair.second, True, EVIDENCE_STRENGTH, f"rule:{rule[0]}")
            )
        else:
            model_pairs.append(pair)
    return decisions, model_pairs


def find_pairs(
    candidates: list[Candidate],
    translations: set[tuple[str, str]] | None = None,
    synonyms: set[tuple[str, str]] | None = None,
) -> tuple[list[MergePair], dict]:
    """Return the merge pairs of a file, each with the finders that proposed it, and a report.

    Args:
        candidates: The stage-2 candidates of one file.
        translations: Pairs from the model translation finder; None when it was not run.
        synonyms: Pairs from the model synonym finder (`concept_pipeline.stage4_synonym_finder`); None when
            it was not run.

    Returns:
        ``(pairs, report)``: pairs sorted by keys; the report counts pairs per finder and the
        blocks skipped as too large (`MAX_KEYS_PER_BLOCK`).
    """
    keys = {c.key for c in candidates}
    skipped: list[int] = []
    found = {
        "gloss": _gloss_pairs(candidates, keys),
        "acronym": _acronym_pairs(keys),
        "spelling": _spelling_pairs(keys, skipped),
        "typo": _typo_pairs(keys, skipped),
        "glyph": _glyph_pairs(keys, skipped),
        "named": _named_pairs(candidates),
        "translation": {_ordered(a, b) for a, b in translations or () if a in keys and b in keys},
        "synonym": {_ordered(a, b) for a, b in synonyms or () if a in keys and b in keys},
    }
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for finder in FINDERS:
        for pair in found[finder]:
            if pair[0] != pair[1]:
                by_pair[pair].append(finder)
    pairs = [MergePair(a, b, tuple(finders)) for (a, b), finders in sorted(by_pair.items())]
    report = {
        "pairs": len(pairs),
        "by_finder": {finder: len(found[finder]) for finder in FINDERS},
        "skipped_blocks": len(skipped),
    }
    return pairs, report


def contexts(candidate: Candidate, result) -> list[str]:
    """Return short windows around a candidate's strongest mentions, with their heading.

    The decider gets the sentence and the heading path, never a bare form (plan section ב).
    """
    text = clean_text(result.text)
    sections = result.structure.sections
    chosen = sorted(candidate.mentions, key=lambda m: (-m.strength, m.start))[:MAX_CONTEXTS]
    windows = []
    for mention in sorted(chosen, key=lambda m: m.start):
        start = max(0, mention.start - CONTEXT_CHARS)
        end = min(len(text), mention.end + CONTEXT_CHARS)
        before = [m.end() for m in _SENTENCE_END_RE.finditer(text, start, mention.start)]
        after = _SENTENCE_END_RE.search(text, mention.end, end)
        window = text[before[-1] if before else start : after.start() if after else end]
        heading = sections[mention.section].title if 0 <= mention.section < len(sections) else ""
        window = " ".join(window.split())
        windows.append(f"[{' '.join(heading.split())}] {window}" if heading else window)
    return windows


def _strongest(decisions: list[Decision]) -> dict[tuple[str, str], Decision]:
    """Return one decision per pair: the strongest; on a tie "different" wins."""
    best: dict[tuple[str, str], Decision] = {}
    for decision in decisions:
        pair = _ordered(decision.first, decision.second)
        current = best.get(pair)
        rank = (decision.strength, not decision.is_same)
        if current is None or rank > (current.strength, not current.is_same):
            best[pair] = decision
    return best


def group_keys(keys: list[str], decisions: list[Decision]) -> list[list[str]]:
    """Return the keys grouped by "same" decisions without joining any explicit "different".

    "Same" decisions are applied from the strongest to the weakest, then by key; a join that
    would put two sides of a "different" into one group is skipped, so a chain breaks at its
    weakest link (design stage 4). The result does not depend on the order of the inputs.

    Returns:
        Groups as sorted key lists, sorted by their first key.
    """
    best = _strongest(decisions)
    group_of = {key: key for key in sorted(set(keys))}
    members = {key: {key} for key in group_of}
    apart: dict[str, set[str]] = defaultdict(set)
    for (a, b), decision in best.items():
        if not decision.is_same and a in group_of and b in group_of:
            apart[a].add(b)
            apart[b].add(a)
    joins = sorted(
        (pair for pair, decision in best.items() if decision.is_same),
        key=lambda pair: (-best[pair].strength, pair),
    )
    for a, b in joins:
        if a not in group_of or b not in group_of:
            continue
        left, right = group_of[a], group_of[b]
        if left == right or any(apart[key] & members[right] for key in members[left]):
            continue
        keep, gone = min(left, right), max(left, right)
        for key in members[gone]:
            group_of[key] = keep
        members[keep] |= members.pop(gone)
    return sorted(sorted(group) for group in members.values())


def merge(stage2: Stage2Result, decisions: list[Decision]) -> list[Concept]:
    """Return the concepts of a file: candidates joined by `group_keys`, statistics added up.

    A merged concept's counts are the sum of its members' counts (design stage 2: merging is
    addition). It takes the key of `_group_name`.
    """
    by_key = {c.key: c for c in stage2.candidates}
    concepts = []
    for group in group_keys(list(by_key), decisions):
        parts = [by_key[key] for key in group]
        name = _group_name(parts)
        core = _most_mentioned(parts).key
        sources = sorted({s for c in parts for s in c.sources})
        aliases = sorted({a for c in parts for a in c.aliases} | set(group) - {name})
        mentions = sorted(
            (m for c in parts for m in c.mentions), key=lambda m: (m.start, m.end, m.strength)
        )
        concepts.append(Concept(name, group, sources, aliases, mentions, core))
    return sorted(concepts, key=lambda c: c.key)


def _group_name(parts: list[Candidate]) -> str:
    """Return the key a merged group is named by: its fullest name, else the most mentioned member.

    First a named result whose bare name holds every word of the most mentioned member, the one
    with the most words ("משפט בולצאנו ויירשטראס" over "משפט בולצאנו" and "בולצאנו"). Then a
    longer member holding those words that the author marks more strongly, at least as a
    heading subject or label ("אי רציפות קפיצה" over "קפיצה"; owner decisions 2026-11-20). Any
    longer member without that evidence was tried and gave worse names ("label set לסמן",
    "mrna אמצעות"; 258 renames on dev against 85 with the evidence rule).
    """
    common = _most_mentioned(parts)
    words = set(common.key.split())
    named = [c for c in parts if words <= set((_named_tail(c.key) or "").split())]
    if named:
        return min(named, key=lambda c: (-len(c.key.split()), -c.count, c.key)).key
    floor = max(_strongest_mention(common), SUBJECT)
    marked = [
        c
        for c in parts
        if words < set(c.key.split())
        and is_name_like(c.key.split())
        and _strongest_mention(c) >= floor
        and _strongest_mention(c) > _strongest_mention(common)
    ]
    if not marked:
        return common.key
    return min(
        marked, key=lambda c: (-_strongest_mention(c), len(c.key.split()), -c.count, c.key)
    ).key


def _most_mentioned(parts: list[Candidate]) -> Candidate:
    """Return the member with the most mentions, ties broken by key."""
    return min(parts, key=lambda c: (-c.count, c.key))


def _strongest_mention(candidate: Candidate) -> float:
    """Return the strength of a candidate's strongest mention (0 when it has none)."""
    return max((m.strength for m in candidate.mentions), default=0.0)


def _inside(mention: Mention, outer: Mention) -> bool:
    """Return True when `outer` is a longer span that holds `mention`."""
    return (
        outer.start <= mention.start
        and mention.end <= outer.end
        and (outer.end - outer.start) > (mention.end - mention.start)
    )


def _has_own_evidence(concept: Concept, alone: list[Mention]) -> bool:
    """Return True when the author marks the concept itself, not only the phrase holding it.

    Evidence is a definition, a gloss, or a mention alone stronger than plain text (a heading,
    a label, an emphasis).
    """
    if {"definition", "gloss"} & set(concept.sources):
        return True
    return any(m.strength > MENTION for m in alone)


def absorb_fragments(concepts: list[Concept]) -> tuple[list[Concept], dict[str, str]]:
    """Return the concepts without fragments, and each absorbed fragment's parent key.

    A fragment's mentions inside its parent are already the parent's mentions, so absorbing it
    only removes it; it does not become an alias (see `FRAGMENT_SHARE`).
    """
    by_start: dict[int, list[tuple[Mention, str]]] = defaultdict(list)
    # Only a parent that reads as a name may absorb: a fragment of a sentence piece ("compatible"
    # inside "גרף אז הוא compatible") is a concept the junk phrase must not swallow.
    for concept in (c for c in concepts if is_name_like(c.key.split())):
        for mention in concept.mentions:
            by_start[mention.start].append((mention, concept.key))
    starts = sorted(by_start)
    longest = max((m.end - m.start for c in concepts for m in c.mentions), default=0)
    absorbed: dict[str, str] = {}
    for concept in concepts:
        if not concept.mentions:
            continue
        holders: Counter[str] = Counter()
        alone = []
        for mention in concept.mentions:
            # Only a span that starts at most `longest` characters earlier can hold it.
            low = bisect.bisect_left(starts, mention.start - longest)
            high = bisect.bisect_right(starts, mention.start)
            outer = {
                key
                for start in starts[low:high]
                for other, key in by_start[start]
                if key != concept.key and _inside(mention, other)
            }
            holders.update(outer)
            if not outer:
                alone.append(mention)
        if not holders or len(alone) > FRAGMENT_MAX_ALONE or _has_own_evidence(concept, alone):
            continue
        parent, inside = min(holders.items(), key=lambda item: (-item[1], item[0]))
        if inside >= FRAGMENT_SHARE * concept.count:
            absorbed[concept.key] = parent
    return [c for c in concepts if c.key not in absorbed], absorbed


def check_groups(keys: list[str], decisions: list[Decision], shuffles: int = 20) -> list[str]:
    """Return the algorithmic gate's problems; an empty list means it passes (plan, stage 4).

    Every key is in exactly one group, no explicit "different" pair (as finally decided) shares
    a group, and `shuffles` shuffled orders of keys and decisions give the same groups.
    """
    problems = []
    groups = group_keys(keys, decisions)
    placed = [key for group in groups for key in group]
    if sorted(placed) != sorted(set(keys)):
        problems.append("a key is missing or in more than one group")
    group_of = {key: i for i, group in enumerate(groups) for key in group}
    for (a, b), decision in _strongest(decisions).items():
        if not decision.is_same and a in group_of and group_of.get(a) == group_of.get(b):
            problems.append(f"{a!r} and {b!r} were decided different but share a group")
    for turn in range(1, shuffles + 1):
        # A seeded shuffle, so a failure can be reproduced from the turn number.
        shuffler = random.Random(turn)
        shuffled_keys, shuffled_decisions = list(keys), list(decisions)
        shuffler.shuffle(shuffled_keys)
        shuffler.shuffle(shuffled_decisions)
        if group_keys(shuffled_keys, shuffled_decisions) != groups:
            problems.append(f"input order {turn} gives other groups")
            break
    return problems
