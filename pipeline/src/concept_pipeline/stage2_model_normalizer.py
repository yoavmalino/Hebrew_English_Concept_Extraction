"""Model normalization for stage 2: ask a cheap model for each word's base form, then guard it.

Design stage 2: one cheap call per file, each form sent once, and an answer is kept only when
`is_close_normalization` accepts it, otherwise the word stays as it is. A long summary has
thousands of words, so they go in batches of `MAX_WORDS_PER_CALL`; each batch is one call. Calls
go through `LlmCache`, so a rerun sends nothing and tests use a fake sender.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .lexicon import FUNCTION_WORDS
from .llm_cache import LlmCache, LlmCall
from .normalize import is_close_normalization
from .stage2_candidates import Stage2Result, extract_candidates

# Bump when the prompt or the batching changes, so old answers are not replayed for new prompts.
NORMALIZE_PROMPT_VERSION = "1"
# Keeps each JSON answer short enough that a cheap model does not cut it at its output limit.
MAX_WORDS_PER_CALL = 400
# One letter is never normalized ("ה", "x"); the guard refuses stems this short anyway.
MIN_WORD_LETTERS = 2
_DIGITS_RE = re.compile(r"\d")

SYSTEM_PROMPT = """You normalize words from a university course summary, usually in Hebrew.
For each word, give its dictionary base form as used in the example phrase:
- Hebrew: remove attached prefixes (ו ה ב כ ל מ ש) only when they are prefixes and not part of the
  word; use the singular; use the absolute state, not the construct state ("מטריצות" -> "מטריצה",
  "והמרחבים" -> "מרחב", "מרחבי" -> "מרחב" when it means "spaces of").
- English or other Latin words: lowercase singular ("vectors" -> "vector").
- A name, an abbreviation, or a word you are unsure about: return it unchanged.
Answer with JSON only: {"lemmas": {"<word>": "<base form>", ...}} with every word given."""


@dataclass
class NormalizationReport:
    """Accepted lemmas of one file, the answers the guard refused, and the calls made."""

    lemmas: dict[str, str] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)
    calls: int = 0


def words_to_normalize(stage2: Stage2Result) -> list[tuple[str, str]]:
    """Return each word of the candidates once, with the shortest candidate form holding it.

    Function words, digits and one-letter words are skipped. The example phrase gives the model
    the sense of the word ("מרחבי" in "מרחבי וקטורים" is a construct plural).

    Returns:
        ``(word, example)`` pairs sorted by word, so the same file always builds the same calls.
    """
    examples: dict[str, str] = {}
    for candidate in stage2.candidates:
        form = candidate.form
        for word in form.lower().split():
            if len(word) < MIN_WORD_LETTERS or word in FUNCTION_WORDS or _DIGITS_RE.search(word):
                continue
            best = examples.get(word)
            if best is None or (len(form), form) < (len(best), best):
                examples[word] = form
    return sorted(examples.items())


def build_prompt(items: list[tuple[str, str]]) -> str:
    """Return the user prompt for one batch: one ``word<TAB>example`` line per word."""
    lines = [f"{word}\t{example}" for word, example in items]
    return "Words (word, then an example phrase from the summary):\n" + "\n".join(lines)


def parse_lemmas(text: str, words: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return the accepted and the refused lemmas from one answer.

    A lemma is accepted when it is one word, differs from the word and passes
    `is_close_normalization`. A missing or malformed entry leaves the word as it is; an answer
    that is not JSON at all leaves every word of the batch as it is.

    Returns:
        ``(accepted, refused)``, each mapping a word to the model's proposal.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            data = json.loads(match.group()) if match else {}
        except json.JSONDecodeError:
            data = {}
    proposals = data.get("lemmas") if isinstance(data, dict) else None
    if not isinstance(proposals, dict):
        return {}, {}
    accepted, refused = {}, {}
    for word in words:
        proposal = proposals.get(word)
        if not isinstance(proposal, str):
            continue
        proposal = proposal.strip().lower()
        if not proposal or proposal == word:
            continue
        if len(proposal.split()) == 1 and is_close_normalization(word, proposal):
            accepted[word] = proposal
        else:
            refused[word] = proposal
    return accepted, refused


def model_lemmas(stage2: Stage2Result, cache: LlmCache, model: str) -> NormalizationReport:
    """Ask the model, through the cache, for the lemmas of a file's candidate words."""
    items = words_to_normalize(stage2)
    report = NormalizationReport()
    for start in range(0, len(items), MAX_WORDS_PER_CALL):
        batch = items[start : start + MAX_WORDS_PER_CALL]
        call = LlmCall(
            "normalize", NORMALIZE_PROMPT_VERSION, model, SYSTEM_PROMPT, build_prompt(batch)
        )
        answer = cache.answer(call)
        report.calls += 1
        accepted, refused = parse_lemmas(answer.text, [word for word, _ in batch])
        report.lemmas.update(accepted)
        report.refused.update(refused)
    return report


def extract_with_model(
    result, cache: LlmCache, model: str, **options
) -> tuple[Stage2Result, NormalizationReport]:
    """Return stage-2 candidates whose keys use the model's lemmas, with the normalization report.

    Two passes: the first, with the prefix fallback alone, finds the words to normalize; the
    second builds the candidates again with the accepted lemmas.
    """
    first = extract_candidates(result, **options)
    report = model_lemmas(first, cache, model)
    return extract_candidates(result, lemmas=report.lemmas, **options), report
