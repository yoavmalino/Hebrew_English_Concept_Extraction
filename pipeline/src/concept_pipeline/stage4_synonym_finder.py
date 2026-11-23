"""Stage-4 pair finder for synonyms no rule can see ("מעגל קרבס" / "מעגל החומצה הציטרית").

One model call per file (owner decision 2026-11-16): the model gets the file's whole candidate
list, numbered in order of first mention so that terms of one passage sit together, and returns
groups of entries that name one concept. One call per file keeps the cost near half a cent; a call
per term against the list (as the translation finder does for Latin terms) would cost about 25
times more, against the design's ceiling of $1 per student per month. It only proposes pairs: the
decider (`concept_pipeline.stage4_merge_decider`) still decides them. No network code: the cache holds the sender.
"""

from __future__ import annotations

import itertools
import json
import re

from .llm_cache import LlmCache, LlmCall
from .stage2_candidates import Candidate, Stage2Result

SYNONYM_PROMPT_VERSION = "1"
# A group larger than this is a topic, not one concept; it is cut to its first entries.
MAX_GROUP_ENTRIES = 5
SYSTEM_PROMPT = """You help merge terms in a university course summary written in Hebrew with
English terms. You get the summary's terms, numbered in the order they first appear. Find groups
of entries that are interchangeable names of one concept: Hebrew synonyms ("מעגל קרבס" and
"מעגל החומצה הציטרית"), a Hebrew word and its loanword ("ריאקציה" and "תגובה"), an English
inflection or acronym ("wobble" and "wobbling", "k nn" and "k nearest neighbor"), or a
translation. Do not group a broader and a narrower concept ("מטריצה" and "מטריצה הפיכה"), related
concepts ("גרף קשיר" and "קשירות"), a process and its enzyme, or a term and its property. Each
group has two to five entries. Leave out every entry with no synonym in the list.
Answer with JSON only: {"groups": [[<entry number>, <entry number>, ...], ...]}."""


def ordered_entries(stage2: Stage2Result) -> list[Candidate]:
    """Return every candidate in order of its first mention (ties by key), numbered from 1."""
    return sorted(stage2.candidates, key=lambda c: (min(m.start for m in c.mentions), c.key))


def build_prompt(entries: list[Candidate]) -> str:
    """Return the user prompt: the numbered entries by their most common surface form."""
    return "Terms:\n" + "\n".join(f"{n} {entry.form}" for n, entry in enumerate(entries, 1))


def parse_groups(text: str, entry_count: int) -> list[list[int]]:
    """Return the answered groups as lists of valid, distinct entry numbers (two or more each).

    A number outside the list or a non-number is dropped; a group is cut to
    `MAX_GROUP_ENTRIES`; an answer that is not JSON gives no groups.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            data = json.loads(match.group()) if match else {}
        except json.JSONDecodeError:
            data = {}
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, list):
        return []
    found = []
    for group in groups:
        if not isinstance(group, list):
            continue
        numbers = [
            n
            for n in group
            if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= entry_count
        ]
        numbers = list(dict.fromkeys(numbers))[:MAX_GROUP_ENTRIES]
        if len(numbers) >= 2:
            found.append(numbers)
    return found


def synonym_pairs(
    stage2: Stage2Result, cache: LlmCache, model: str
) -> tuple[set[tuple[str, str]], int]:
    """Ask the model, through the cache, for groups of synonyms; return their pairs.

    Returns:
        ``(pairs, calls)``: every pair of keys inside an answered group (sorted keys), and the
        number of calls made (one, or none for a file without two candidates).
    """
    entries = ordered_entries(stage2)
    if len(entries) < 2:
        return set(), 0
    call = LlmCall("synonyms", SYNONYM_PROMPT_VERSION, model, SYSTEM_PROMPT, build_prompt(entries))
    pairs: set[tuple[str, str]] = set()
    for group in parse_groups(cache.answer(call).text, len(entries)):
        keys = sorted({entries[n - 1].key for n in group})
        pairs |= set(itertools.combinations(keys, 2))
    return pairs, 1
