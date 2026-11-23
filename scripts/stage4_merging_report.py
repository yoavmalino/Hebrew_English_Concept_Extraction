"""Report pipeline stage 4 (merge pairs) on course summaries; sends nothing to any model.

Dev tool, not part of the app. Per course it prints the merge pairs per finder, the volume the
decider would send (calls and prompt characters), and the algorithmic gate on the real keys with
stand-in decisions. With ``--gold-dir`` it also lists "name pairs": two different candidate keys
that the judges' names give for one gold concept, and how many the finders found (finder recall,
reported only). ``--pairs-out`` writes every pair with its contexts, for labelling by a committee.
``--translate-cache DIR`` adds the model translation finder (replay; with ``--send`` it sends the
calls not yet cached, with the key from the normalizer's environment variable).

    python scripts/stage4_merging_report.py COURSE_FOLDER ... \
        --cache GOLD/_normalization_cache [--gold-dir GOLD] [--pairs-out FILE]

Uses the model lemmas from the normalization cache (replay only). Dev courses only.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import concept_committee as committee  # noqa: E402
import stage2_normalization_run as run  # noqa: E402
from concept_pipeline.gold import _PAREN_RE, _SPACED_DASH_RE, gold_from_json, normal_form  # noqa: E402
from concept_pipeline.llm_cache import LlmCache, Mode  # noqa: E402
from concept_pipeline.stage4_merge_decider import MAX_PAIRS_PER_CALL, SYSTEM_PROMPT, build_prompt  # noqa: E402
from concept_pipeline.stage2_model_normalizer import extract_with_model  # noqa: E402
from concept_pipeline.stage4_merging import MODEL_STRENGTH, Decision, check_groups, contexts  # noqa: E402
from concept_pipeline.stage4_merging import decide_by_rule, find_pairs  # noqa: E402
from concept_pipeline.stage4_translation_finder import translation_pairs  # noqa: E402


def stand_in_decisions(pairs) -> list[Decision]:
    """Return a fixed mix of same and different by hash, to exercise the grouping gate."""
    decisions = []
    for pair in pairs:
        digit = hashlib.sha256(f"{pair.first}|{pair.second}".encode()).digest()[0]
        decisions.append(Decision(pair.first, pair.second, digit % 3 != 0, MODEL_STRENGTH, "test"))
    return decisions


# Kinds of name pairs, from the most reliable synonym evidence to none.
SYNONYM, DASH, CONTAINED = "synonym", "dash", "contained"


def _name_sources(name: str) -> list[tuple[str, str]]:
    """Return ``(part, how)`` for a judge's name: "whole", "paren" (inside parentheses) or "dash"."""
    outer = _PAREN_RE.sub(" ", name).strip()
    inner = [part.strip() for part in _PAREN_RE.findall(name)]
    sources = [(name, "whole"), (outer, "whole")] + [(part, "paren") for part in inner]
    for text in [outer, *inner]:
        pieces = _SPACED_DASH_RE.split(text)
        if len(pieces) > 1:
            sources += [(piece.strip(), "dash") for piece in pieces]
    return [(part, how) for part, how in sources if part]


def _pair_kind(first: str, second: str, sources: dict, hits: list[str]) -> str:
    """Return how a name pair is evidenced (`SYNONYM`, `DASH`) or `CONTAINED` when it is not.

    `CONTAINED`: one key's words are inside the other's, or both inside a third key of the same
    concept (head plus addition, "rna" / "rna פולימראז"; or pieces of one name). `SYNONYM`: two
    judges' different names, or a name and its parenthesis ("k-NN (KNN)"). `DASH`: only a piece
    of a name split at a spaced dash, which judges use both for synonyms ("Min – single link")
    and for two things ("Recall – precision plot"); reported apart, because only a label can
    tell them apart.
    """
    words_a, words_b = set(first.split()), set(second.split())
    if words_a <= words_b or words_b <= words_a:
        return CONTAINED
    for other in hits:
        if other not in (first, second) and (words_a | words_b) <= set(other.split()):
            return CONTAINED
    for name_a, how_a in sources[first]:
        for name_b, how_b in sources[second]:
            if "dash" not in (how_a, how_b) and (name_a != name_b or "paren" in (how_a, how_b)):
                return SYNONYM
    return DASH


def name_pairs(gold, stage2) -> list[tuple[str, str, str, str]]:
    """Return ``(concept, key, key, kind)`` for gold concepts whose names give two candidate keys.

    Only concepts that at least two judges put in the core layer count. `kind` is from
    `_pair_kind`; finder recall is reported per kind, and the stage-4 recall gate uses `SYNONYM`.
    """
    keys = {c.key for c in stage2.candidates}
    found = []
    for concept in gold:
        if len(concept.judges("core")) < 2:
            continue
        sources: dict[str, set[tuple[str, str]]] = {}
        for index, name in enumerate(sorted(concept.names)):
            for part, how in _name_sources(name):
                key = " ".join(normal_form(part, stage2.normalizer))
                if key in keys:
                    sources.setdefault(key, set()).add((index, how))
        hits = sorted(sources)
        found += [
            (concept.name, a, b, _pair_kind(a, b, sources, hits))
            for i, a in enumerate(hits)
            for b in hits[i + 1 :]
        ]
    return found


def report_course(
    course: Path, cache: LlmCache, gold_dir: Path | None, translator: LlmCache | None
) -> dict:
    """Return one course's stage-4 report and its pairs with contexts."""
    result = committee._summary_result(course, committee.SUMMARY_MARK)
    stage2, _ = extract_with_model(result, cache, run.DEFAULT_MODEL)
    translations, translate_calls = (
        translation_pairs(stage2, translator, run.DEFAULT_MODEL) if translator else (None, 0)
    )
    pairs, counts = find_pairs(stage2.candidates, translations)
    by_key = {c.key: c for c in stage2.candidates}
    items = [
        {
            "first": p.first,
            "second": p.second,
            "finders": list(p.finders),
            "first_contexts": contexts(by_key[p.first], result),
            "second_contexts": contexts(by_key[p.second], result),
        }
        for p in pairs
    ]
    # Pairs decided by rule ("named") never reach the decider, so they cost no call.
    rule_decisions, model_pairs = decide_by_rule(pairs)
    model_set = set(model_pairs)
    to_model = [(p, it) for p, it in zip(pairs, items) if p in model_set]
    prompt_chars = len(SYSTEM_PROMPT) * -(-len(to_model) // MAX_PAIRS_PER_CALL)
    for start in range(0, len(to_model), MAX_PAIRS_PER_CALL):
        batch = to_model[start : start + MAX_PAIRS_PER_CALL]
        prompt_chars += len(
            build_prompt(
                [
                    (str(i), pair, it["first_contexts"], it["second_contexts"])
                    for i, (pair, it) in enumerate(batch, 1)
                ]
            )
        )
    report = {
        "course": course.name.strip(),
        "candidates": len(stage2.candidates),
        "words": stage2.words,
        **counts,
        "calls": -(-len(to_model) // MAX_PAIRS_PER_CALL),
        "rule_decisions": len(rule_decisions),
        "translate_calls": translate_calls,
        "prompt_chars": prompt_chars,
        "gate": check_groups(sorted(by_key), rule_decisions + stand_in_decisions(model_pairs)),
    }
    if gold_dir is not None:
        folder = next(p for p in gold_dir.iterdir() if p.name.strip() == report["course"])
        gold = gold_from_json(
            json.loads((folder / "concepts.gold.json").read_text("utf-8"))["concepts"]
        )
        proposed = {(p.first, p.second) for p in pairs}
        named = name_pairs(gold, stage2)
        for kind in (SYNONYM, DASH, CONTAINED):
            of_kind = [(a, b) for _, a, b, k in named if k == kind]
            report[f"name_pairs_{kind}"] = len(of_kind)
            report[f"name_pairs_{kind}_found"] = sum(1 for pair in of_kind if pair in proposed)
        report["name_pairs_missed"] = [
            f"[{k}] {c}: {a} / {b}" for c, a, b, k in named if (a, b) not in proposed
        ]
    return report, [{"course": report["course"], **item} for item in items]


def main() -> None:
    """Parse the command line, report each course and the totals, and write pairs if asked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("courses", nargs="+", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path)
    parser.add_argument("--pairs-out", type=Path)
    parser.add_argument("--translate-cache", type=Path, help="run the model translation finder")
    parser.add_argument("--send", action="store_true", help="send translation calls not cached")
    args = parser.parse_args()
    cache = LlmCache(args.cache, Mode.REPLAY)
    translator = None
    if args.translate_cache:
        sender = run.anthropic_sender(run.NORMALIZER_KEY_ENV) if args.send else None
        translator = LlmCache(
            args.translate_cache, Mode.RECORD if args.send else Mode.REPLAY, sender
        )
    totals = {"pairs": 0, "calls": 0, "prompt_chars": 0, "words": 0}
    for kind in (SYNONYM, DASH, CONTAINED):
        totals[f"name_pairs_{kind}"] = totals[f"name_pairs_{kind}_found"] = 0
    every_pair, failed = [], False
    for course in args.courses:
        report, items = report_course(course, cache, args.gold_dir, translator)
        every_pair += items
        finders = " ".join(f"{name} {n}" for name, n in report["by_finder"].items())
        line = (
            f"{report['course']:26} cands {report['candidates']:5} pairs {report['pairs']:4} "
            f"({finders}) skipped {report['skipped_blocks']} calls {report['calls']} "
            f"chars {report['prompt_chars']:,} gate {'ok' if not report['gate'] else 'FAIL'}"
        )
        if "name_pairs_missed" in report:
            line += " name pairs " + " ".join(
                f"{kind} {report[f'name_pairs_{kind}_found']}/{report[f'name_pairs_{kind}']}"
                for kind in (SYNONYM, DASH, CONTAINED)
            )
        print(line)
        for missed in report.get("name_pairs_missed", []):
            print(f"    missed: {missed}")
        for problem in report["gate"]:
            print(f"    gate: {problem}")
        failed = failed or bool(report["gate"])
        for name in totals:
            totals[name] += report.get(name, 0)
    per_thousand = 1000 * totals["pairs"] / max(totals["words"], 1)
    print(
        f"total pairs {totals['pairs']} ({per_thousand:.1f} per 1,000 words), calls "
        f"{totals['calls']}, prompt chars {totals['prompt_chars']:,}, name pairs "
        + ", ".join(
            f"{kind} {totals[f'name_pairs_{kind}_found']}/{totals[f'name_pairs_{kind}']}"
            for kind in (SYNONYM, DASH, CONTAINED)
        )
    )
    print("gate:", "FAILED" if failed else "passed")
    if args.pairs_out:
        args.pairs_out.write_text(json.dumps(every_pair, ensure_ascii=False, indent=1), "utf-8")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
