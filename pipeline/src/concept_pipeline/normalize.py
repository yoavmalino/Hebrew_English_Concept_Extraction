"""Word normalization for stage 2: a cautious prefix fallback and the guard for model answers.

The design normalizes with one cheap model call per file (design stage 2). Until that call is
approved, `PrefixNormalizer` removes Hebrew prefixes only when the bare word is attested in the
same file; plurals and construct forms stay apart. `is_close_normalization` is the guard that a
model's answer must pass, and `LemmaNormalizer` applies the accepted answers. Nothing here reads
files or calls a model.
"""

from __future__ import annotations

from collections import Counter

from .lexicon import FUNCTION_WORDS

# One-letter prefixes that attach to a Hebrew word: ו ה ב כ ל מ ש (design stage 2).
PREFIX_LETTERS = frozenset("והבכלמש")
# "והמטריצה" carries three prefix letters; four in a row is almost always part of the word.
MAX_PREFIX_LETTERS = 3
# A bare word shorter than this is too ambiguous to merge into ("מה", "לה").
MIN_STEM_LETTERS = 2
# ב כ ל מ ש begin many roots ("מרחב", "משפט", "בסיס"). Removing one of them is allowed only when
# the bare word is at least as frequent, so "מרחב" does not merge into a rare "רחב".
RISKY_PREFIX_LETTERS = frozenset("בכלמש")
# A model answer may change at most this many letters after removing prefixes (plan, section יא).
MAX_EDIT_LETTERS = 2
# Final letters are the same letter at a word's end; without this "ערכים" -> "ערך" counts three
# changes instead of two (found by the 2026-11-16 normalization accuracy check on dev).
FINAL_TO_PLAIN = str.maketrans("ךםןףץ", "כמנפצ")
# Plural endings, longest first. After removing one, the singular may add one ending letter
# ("דוגמאות" -> "דוגמ" + "ה"), which the plain edit limit cannot cover. Only an added ending is
# allowed, never a changed letter, so "ערכים" -> "ערב" stays refused.
PLURAL_SUFFIXES = ("אות", "יות", "ים", "ות")
SINGULAR_ENDINGS = ("", "ה", "ת", "י")
# One changed letter is a spelling variant only in a long word ("אורתוגונאלי" / "אורתוגונלי");
# in a short word it is another word ("use" / "used"), which the first gold match rule accepted.
MIN_LETTERS_FOR_TYPO = 6


def _is_hebrew(word: str) -> bool:
    """Return True when the word's first letter is Hebrew."""
    return bool(word) and "א" <= word[0] <= "ת"


def _prefix_splits(word: str) -> list[tuple[str, str]]:
    """Return every (prefix, rest) split with a valid prefix chain, longest prefix first."""
    splits = []
    for length in range(1, MAX_PREFIX_LETTERS + 1):
        prefix, rest = word[:length], word[length:]
        if len(rest) < MIN_STEM_LETTERS or any(ch not in PREFIX_LETTERS for ch in prefix):
            break
        splits.append((prefix, rest))
    return splits[::-1]


class PrefixNormalizer:
    """Map a word to its bare form when the bare form also appears in the same document.

    Deterministic and idempotent: stripping repeats until nothing changes, against a fixed count
    table, so normalizing a normal form returns it unchanged.

    Args:
        word_counts: How often each surface word occurs in the document.
    """

    def __init__(self, word_counts: Counter[str]):
        self._counts = word_counts
        self._cache: dict[str, str] = {}

    def _strip_once(self, word: str) -> str:
        """Return the word with one prefix chain removed, or the word itself when none applies."""
        for prefix, rest in _prefix_splits(word):
            if not self._counts.get(rest) or rest in FUNCTION_WORDS:
                continue
            risky = any(ch in RISKY_PREFIX_LETTERS for ch in prefix)
            if risky and self._counts[rest] < self._counts.get(word, 0):
                continue
            return rest
        return word

    def normalize_word(self, word: str) -> str:
        """Return the normal form of one word (lowercase Latin, prefixes removed when attested)."""
        if word in self._cache:
            return self._cache[word]
        normal = word.lower()
        if _is_hebrew(normal):
            while (stripped := self._strip_once(normal)) != normal:
                normal = stripped
        self._cache[word] = normal
        return normal

    def normalize_phrase(self, words: tuple[str, ...]) -> tuple[str, ...]:
        """Return the normal form of a phrase, word by word."""
        return tuple(self.normalize_word(word) for word in words)


class LemmaNormalizer(PrefixNormalizer):
    """Map words through accepted model lemmas first, then fall back to `PrefixNormalizer`.

    Stays idempotent: chains are resolved ("א" -> "ב" -> "ג" becomes "א" -> "ג"), every lemma is
    its own normal form, and the prefix fallback's result also goes through the lemma map.

    The key is an internal identity for counting and merging, so a lemma the text never writes
    ("משותף") is allowed here; what is shown is a written form (`stage4_merging.Concept.written`). Owner
    decision 2026-11-17, after an attested-only rule cost 0.012 nDCG on dev by leaving
    inflections apart.

    Args:
        word_counts: How often each surface word occurs in the document.
        lemmas: Lowercase word -> accepted lemma, as `is_close_normalization` let them through.
    """

    def __init__(self, word_counts: Counter[str], lemmas: dict[str, str]):
        super().__init__(word_counts)
        self._lemmas = _resolve_chains(lemmas)
        self._targets = set(self._lemmas.values())

    def normalize_word(self, word: str) -> str:
        """Return the lemma of a word, or its prefix-normal form mapped through the lemmas."""
        lower = word.lower()
        if lower in self._targets:
            return lower
        if lower in self._lemmas:
            return self._lemmas[lower]
        fallback = super().normalize_word(word)
        return self._lemmas.get(fallback, fallback)


def _resolve_chains(lemmas: dict[str, str]) -> dict[str, str]:
    """Return the map with every chain followed to its end; a cycle stops at its first repeat."""
    resolved = {}
    for word in lemmas:
        seen, current = {word}, lemmas[word]
        while current in lemmas and current not in seen:
            seen.add(current)
            current = lemmas[current]
        resolved[word] = current
    return resolved


def _edit_distance(first: str, second: str) -> int:
    """Return the Levenshtein distance between two short strings."""
    previous = list(range(len(second) + 1))
    for i, first_char in enumerate(first, start=1):
        current = [i]
        for j, second_char in enumerate(second, start=1):
            cost = 0 if first_char == second_char else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _is_close_word(original: str, proposed: str) -> bool:
    """Return True when `proposed` is `original` minus prefixes and a plural ending, nearly unchanged."""
    if original == proposed:
        return True
    if len(proposed) < MIN_STEM_LETTERS:
        return False
    stems = [original] + [rest for _, rest in _prefix_splits(original)]
    target = proposed.translate(FINAL_TO_PLAIN)
    for stem in stems:
        plain = stem.translate(FINAL_TO_PLAIN)
        if _edit_distance(plain, target) <= MAX_EDIT_LETTERS:
            return True
        if _is_close_after_plural(plain, target):
            return True
    return False


def _is_close_after_plural(stem: str, target: str) -> bool:
    """Return True when `target` is `stem` minus a plural ending, plus at most one ending letter."""
    # Callers compare with final letters made plain, so "ים" arrives as "ימ".
    for suffix in {s.translate(FINAL_TO_PLAIN) for s in PLURAL_SUFFIXES} | set(PLURAL_SUFFIXES):
        bare = stem[: -len(suffix)]
        if stem.endswith(suffix) and len(bare) >= MIN_STEM_LETTERS:
            if any(target == bare + ending for ending in SINGULAR_ENDINGS):
                return True
    return False


def is_close_normalization(original: str, proposed: str) -> bool:
    """Return True when a proposed normal form is close enough to the original to accept.

    The guard against a far answer (design stage 2): each word may lose a prefix chain (ו ה ב כ
    ל מ ש) and change at most MAX_EDIT_LETTERS letters, which covers plural and construct forms
    ("מטריצות" -> "מטריצה", "מרחבי" -> "מרחב"). The word count must not change.

    Args:
        original: The surface form, words separated by spaces.
        proposed: The normal form a model suggested.

    Returns:
        True to accept the proposal; False keeps the original form.
    """
    original_words = original.lower().split()
    proposed_words = proposed.lower().split()
    if len(original_words) != len(proposed_words):
        return False
    return all(_is_close_word(o, p) for o, p in zip(original_words, proposed_words))


def _is_word_match(gold_word: str, key_word: str) -> bool:
    """Return True when two words are the same word up to prefixes, plural and final letters."""
    if gold_word == key_word:
        return True
    target = key_word.translate(FINAL_TO_PLAIN)
    for stem in [gold_word] + [rest for _, rest in _prefix_splits(gold_word)]:
        plain = stem.translate(FINAL_TO_PLAIN)
        if plain == target or _is_close_after_plural(plain, target):
            return True
        if _is_close_after_plural(target, plain):
            return True
        if min(len(plain), len(target)) >= MIN_LETTERS_FOR_TYPO:
            if _edit_distance(plain, target) <= 1:
                return True
    return False


def is_name_match(form: str, key: str) -> bool:
    """Return True when a gold name form and a candidate key name the same thing, word by word."""
    form_words, key_words = form.lower().split(), key.lower().split()
    if len(form_words) != len(key_words):
        return False
    return all(_is_word_match(g, k) for g, k in zip(form_words, key_words))
