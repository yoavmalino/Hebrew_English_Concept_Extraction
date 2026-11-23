"""Stage-4 pair finder for translations the author did not put in parentheses, through a model.

The model gets the file's Latin-script candidates and a numbered list of its Hebrew candidates,
and picks for each English term the Hebrew entries that name the same concept. Choosing from the
file's own list (owner decision 2026-11-16, version 2) replaced guessing a translation and
matching it locally, which missed every term the author translated with another word ("וקטור
תומך" for the guessed "וקטור תמיכה"). It only proposes pairs: the decider
(`concept_pipeline.stage4_merge_decider`) still decides them. No network code: the cache holds the sender.
"""

from __future__ import annotations

import json
import re

from .llm_cache import LlmCache, LlmCall
from .stage2_candidates import Candidate, Stage2Result

TRANSLATE_PROMPT_VERSION = "2"
# Latin terms per call; the whole Hebrew list goes with every call, so fewer calls cost less.
# A typical course has 100-320 Latin terms, so one to three calls.
MAX_TERMS_PER_CALL = 150
# Entries chosen per term: a term has one Hebrew name, sometimes a spelling or plural variant.
MAX_CHOICES_PER_TERM = 3
SYSTEM_PROMPT = """You help merge terms in a university course summary written in Hebrew with
English terms. You get English terms and a numbered list of the Hebrew terms of the same summary.
For each English term, choose the Hebrew entries that name the same concept: its translation or
transliteration as the author wrote it ("support vector" -> "וקטור תומך", "eigenvalue" -> "ערך
עצמי"). Do not choose a broader or narrower concept, a related concept, or a term that only
contains the translation ("cell" is not "תא אאוקריוטי"). Choose at most three entries per term.
Leave out a term with no such entry.
Answer with JSON only: {"terms": {"<english term>": [<entry number>, ...], ...}}."""


def latin_terms(stage2: Stage2Result) -> list[str]:
    """Return the file's Latin-script candidate keys worth translating, sorted."""
    return sorted(
        c.key
        for c in stage2.candidates
        if c.key.isascii() and sum(ch.isalpha() for ch in c.key) > 1
    )


def hebrew_entries(stage2: Stage2Result) -> list[Candidate]:
    """Return the file's non-Latin candidates in a fixed order (by key), numbered from 1."""
    return sorted((c for c in stage2.candidates if not c.key.isascii()), key=lambda c: c.key)


def build_prompt(terms: list[str], entries: list[Candidate]) -> str:
    """Return the user prompt: the English terms, then the Hebrew entries by surface form."""
    hebrew = "\n".join(f"{number} {entry.form}" for number, entry in enumerate(entries, 1))
    return "English terms:\n" + "\n".join(terms) + "\n\nHebrew entries:\n" + hebrew


def parse_choices(text: str, terms: list[str], entry_count: int) -> dict[str, list[int]]:
    """Return ``{term: [entry number, ...]}`` for terms answered with valid entry numbers.

    A number outside the list, a non-number, or an unknown term is dropped; at most
    `MAX_CHOICES_PER_TERM` numbers are kept per term.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            data = json.loads(match.group()) if match else {}
        except json.JSONDecodeError:
            data = {}
    answers = data.get("terms") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        return {}
    found = {}
    for term in terms:
        choices = answers.get(term)
        if not isinstance(choices, list):
            continue
        valid = [
            number
            for number in choices
            if isinstance(number, int)
            and not isinstance(number, bool)
            and 1 <= number <= entry_count
        ]
        if valid:
            found[term] = list(dict.fromkeys(valid))[:MAX_CHOICES_PER_TERM]
    return found


def translation_pairs(
    stage2: Stage2Result, cache: LlmCache, model: str
) -> tuple[set[tuple[str, str]], int]:
    """Ask the model, through the cache, which Hebrew candidates name each Latin one.

    Returns:
        ``(pairs, calls)``: (Latin key, Hebrew key) pairs, and the number of calls made.
    """
    terms = latin_terms(stage2)
    entries = hebrew_entries(stage2)
    pairs: set[tuple[str, str]] = set()
    calls = 0
    if not entries:
        return pairs, calls
    for start in range(0, len(terms), MAX_TERMS_PER_CALL):
        batch = terms[start : start + MAX_TERMS_PER_CALL]
        call = LlmCall(
            "translate",
            TRANSLATE_PROMPT_VERSION,
            model,
            SYSTEM_PROMPT,
            build_prompt(batch, entries),
        )
        choices = parse_choices(cache.answer(call).text, batch, len(entries))
        calls += 1
        for term, numbers in choices.items():
            pairs |= {(term, entries[number - 1].key) for number in numbers}
    return pairs, calls
