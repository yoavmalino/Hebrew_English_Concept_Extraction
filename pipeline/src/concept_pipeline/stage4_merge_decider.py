"""The model that decides stage-4 merge pairs: same concept or not, through the response cache.

Builds the prompt from each pair's contexts, parses the answer, and turns it into decisions.
A pair the model does not answer stays undecided, so its two candidates stay apart (a duplicate
is caught later; a wrong merge deletes a concept). No network code: the cache holds the sender.
"""

from __future__ import annotations

import json
import re

from .llm_cache import LlmCache, LlmCall
from .stage2_candidates import Stage2Result
from .stage4_merging import EVIDENCE_STRENGTH, MODEL_STRENGTH, Decision, MergePair, contexts

MERGE_PROMPT_VERSION = "1"
# Pairs per call: small enough that a cut or malformed answer loses little.
MAX_PAIRS_PER_CALL = 25
# The decider runs this many times and the majority decides (owner decision 2026-11-16). Sending
# the pairs the runs disagree on to Sonnet 5.5 (`decide_with_escalation`) gave the same dev gate
# at about 20 times the price per token, so it is kept only for comparison.
FAST_RUNS = 3
# The committee labelled the dev pairs with these same instructions (2026-11-16); changing them
# changes every cache key, so the decider and label caches start over. A stricter version 2
# (word added or missing = different) was dropped before sending: the owner's anchor marks such
# fragments as the same concept.
SYSTEM_PROMPT = """You decide whether two terms from one university course summary name the same
concept. Each pair comes with short passages where each term appears, and the heading above.
- "same": the terms are interchangeable names of one concept: a translation ("eigenvalue" and
  "ערך עצמי"), an acronym and its expansion, a spelling or inflection variant.
- "different": anything else, including a broader and a narrower concept ("מטריצה" and "מטריצה
  הפיכה"), two related concepts ("binary" and "linear"), or a term and its property.
When unsure, answer "different".
Answer with JSON only: {"decisions": {"<pair id>": "same" | "different", ...}} for every pair."""


def build_prompt(items: list[tuple[str, MergePair, list[str], list[str]]]) -> str:
    """Return the user prompt for one batch of ``(id, pair, first contexts, second contexts)``."""
    blocks = []
    for pair_id, pair, first, second in items:
        lines = [f"Pair {pair_id}: A = {pair.first} | B = {pair.second}"]
        if pair.has_direct_evidence:
            lines.append("The author wrote one of them in parentheses after the other.")
        lines += [f"A: {text}" for text in first] + [f"B: {text}" for text in second]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_decisions(text: str, pair_ids: list[str]) -> dict[str, bool]:
    """Return ``{pair id: is_same}`` for the pairs answered with "same" or "different".

    A missing, unknown or malformed entry is left out; an answer that is not JSON gives nothing.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            data = json.loads(match.group()) if match else {}
        except json.JSONDecodeError:
            data = {}
    answers = data.get("decisions") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        return {}
    verdicts = {}
    for pair_id in pair_ids:
        answer = answers.get(pair_id)
        if isinstance(answer, str) and answer.strip().lower() in ("same", "different"):
            verdicts[pair_id] = answer.strip().lower() == "same"
    return verdicts


def decide_pairs(
    pairs: list[MergePair], stage2: Stage2Result, result, cache: LlmCache, model: str
) -> tuple[list[Decision], int]:
    """Ask the model, through the cache, to decide each pair; return decisions and calls made.

    A decision on a pair with the author's own evidence (a gloss) is stronger, so it wins over
    a plain model decision when a chain must break (design stage 4).
    """
    by_key = {c.key: c for c in stage2.candidates}
    decisions: list[Decision] = []
    calls = 0
    for start in range(0, len(pairs), MAX_PAIRS_PER_CALL):
        batch = pairs[start : start + MAX_PAIRS_PER_CALL]
        items = [
            (
                str(i),
                pair,
                contexts(by_key[pair.first], result),
                contexts(by_key[pair.second], result),
            )
            for i, pair in enumerate(batch, start=1)
        ]
        call = LlmCall("merge", MERGE_PROMPT_VERSION, model, SYSTEM_PROMPT, build_prompt(items))
        verdicts = parse_decisions(cache.answer(call).text, [pair_id for pair_id, *_ in items])
        calls += 1
        for pair_id, pair, *_ in items:
            if pair_id in verdicts:
                strength = EVIDENCE_STRENGTH if pair.has_direct_evidence else MODEL_STRENGTH
                decisions.append(
                    Decision(pair.first, pair.second, verdicts[pair_id], strength, f"model:{model}")
                )
    return decisions, calls


def _verdict_of(runs: list[dict[tuple[str, str], bool]], pair: tuple[str, str]) -> bool | None:
    """Return the verdict every run gave the pair, or None when a run differs or is missing."""
    verdicts = {run.get(pair) for run in runs}
    return verdicts.pop() if len(verdicts) == 1 else None


def decide_with_escalation(
    pairs: list[MergePair],
    stage2: Stage2Result,
    result,
    fast_caches: list[LlmCache],
    strong_cache: LlmCache,
    fast_model: str,
    strong_model: str,
) -> tuple[list[Decision], dict[str, int]]:
    """Decide each pair with the fast model's runs; send the pairs they disagree on further.

    A pair every fast run decided alike keeps that verdict. A pair the runs split on, or that a
    run left unanswered, goes to the strong model, whose verdict is final; a pair it does not
    answer stays undecided (apart). This keeps the strong model's cost to the disputed pairs.

    Returns:
        ``(decisions, counts)``: one decision per decided pair, and ``fast_calls``,
        ``strong_calls``, ``disputed`` counts.
    """
    runs, fast_calls = [], 0
    for cache in fast_caches:
        decisions, calls = decide_pairs(pairs, stage2, result, cache, fast_model)
        runs.append({(d.first, d.second): d for d in decisions})
        fast_calls += calls
    verdict_runs = [{pair: d.is_same for pair, d in run.items()} for run in runs]
    agreed, disputed = [], []
    for pair in pairs:
        key = (pair.first, pair.second)
        if _verdict_of(verdict_runs, key) is None:
            disputed.append(pair)
        else:
            agreed.append(runs[0][key])
    strong, strong_calls = (
        decide_pairs(disputed, stage2, result, strong_cache, strong_model) if disputed else ([], 0)
    )
    counts = {"fast_calls": fast_calls, "strong_calls": strong_calls, "disputed": len(disputed)}
    return agreed + strong, counts


def decide_by_majority(
    pairs: list[MergePair], stage2: Stage2Result, result, caches: list[LlmCache], model: str
) -> tuple[list[Decision], dict[str, int]]:
    """Decide each pair by the majority of the model's runs, one run per cache.

    A pair without a majority (a tie, or no run answered) stays undecided, so it stays apart.

    Returns:
        ``(decisions, counts)``: the majority's decision per decided pair (as the first run that
        gave it), and ``calls`` and ``disputed`` (pairs the runs did not all agree on).
    """
    runs, calls = [], 0
    for cache in caches:
        decisions, made = decide_pairs(pairs, stage2, result, cache, model)
        runs.append({(d.first, d.second): d for d in decisions})
        calls += made
    chosen, disputed = [], 0
    for pair in pairs:
        given = [run[(pair.first, pair.second)] for run in runs if (pair.first, pair.second) in run]
        same = sum(1 for decision in given if decision.is_same)
        disputed += len({decision.is_same for decision in given}) != 1 or len(given) < len(runs)
        if 2 * same == len(given):
            continue
        verdict = 2 * same > len(given)
        chosen.append(next(d for d in given if d.is_same == verdict))
    return chosen, {"calls": calls, "disputed": disputed}
