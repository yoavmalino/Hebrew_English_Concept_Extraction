"""Draw the stratified sample for the stage-2 normalization check and size the model calls.

Dev tool, not part of the app; sends nothing. For each course it extracts the summary, runs stage
2 and lists the words model normalization would send (`words_to_normalize`). It prints, per
course, the words, calls and prompt characters (the input of the cost estimate), and writes a
sample of forms drawn evenly from four kinds (plan section יא: prefix, plural, construct, the
definite ה) for someone to label with the right lemma. Accuracy is then measured against it.

    python scripts/stage2_normalization_sample.py COURSE_FOLDER ... --out FILE
        [--size 300] [--seed 7]

Dev courses only: the held-out courses must not be looked at.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import concept_committee as committee  # noqa: E402

from concept_pipeline.stage2_model_normalizer import (  # noqa: E402
    MAX_WORDS_PER_CALL,
    SYSTEM_PROMPT,
    build_prompt,
    words_to_normalize,
)
from concept_pipeline.stage2_candidates import extract_candidates  # noqa: E402

KINDS = ("definite", "prefix", "plural", "construct")
DEFAULT_SIZE = 300
DEFAULT_SEED = 7
# A shorter word is too often a whole stem ("הר", "בת") to say anything about its prefix.
MIN_KIND_LETTERS = 4
PREFIX_LETTERS = "ובכלמש"


def kind_of(word: str) -> str | None:
    """Return the first kind a Hebrew word could belong to by its letters, or None.

    The order (definite, prefix, plural, construct) makes the kinds disjoint: "המטריצות" counts as
    definite. The kind says what to check, not that the word really has that prefix or ending.
    """
    if not ("א" <= word[0] <= "ת") or len(word) < MIN_KIND_LETTERS:
        return None
    if word[0] == "ה":
        return "definite"
    if word[0] in PREFIX_LETTERS:
        return "prefix"
    if word.endswith(("ים", "ות")):
        return "plural"
    if word.endswith(("י", "ת")):
        return "construct"
    return None


def course_words(course: Path, summary_mark: str) -> list[tuple[str, str]]:
    """Return the words model normalization would send for one course summary."""
    result = committee._summary_result(course, summary_mark)
    return words_to_normalize(extract_candidates(result))


def call_volume(items: list[tuple[str, str]]) -> tuple[int, int]:
    """Return how many calls a file needs and the characters they send."""
    batches = [items[i : i + MAX_WORDS_PER_CALL] for i in range(0, len(items), MAX_WORDS_PER_CALL)]
    characters = sum(len(SYSTEM_PROMPT) + len(build_prompt(batch)) for batch in batches)
    return len(batches), characters


def draw_sample(by_course: dict[str, list[tuple[str, str]]], size: int, seed: int) -> list[dict]:
    """Return `size` forms, an equal share per kind, taken from the courses in turn."""
    rng = random.Random(seed)
    pools: dict[str, dict[str, list[tuple[str, str]]]] = {kind: {} for kind in KINDS}
    for course, items in sorted(by_course.items()):
        for word, example in items:
            kind = kind_of(word)
            if kind:
                pools[kind].setdefault(course, []).append((word, example))
    sample = []
    for kind in KINDS:
        courses = {course: rng.sample(items, len(items)) for course, items in pools[kind].items()}
        share = size // len(KINDS)
        while share and any(courses.values()):
            for course in sorted(courses):
                if share and courses[course]:
                    word, example = courses[course].pop()
                    sample.append(
                        {"course": course, "kind": kind, "word": word, "example": example}
                    )
                    share -= 1
    for item in sample:
        item["lemma"] = None  # filled in by the labeller
    return sample


def main() -> None:
    """Print the call volume per course and write the sample file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("courses", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--summary", default=committee.SUMMARY_MARK)
    args = parser.parse_args()
    by_course, total_calls, total_chars = {}, 0, 0
    for course in args.courses:
        items = course_words(course, args.summary)
        calls, characters = call_volume(items)
        total_calls += calls
        total_chars += characters
        by_course[course.name.strip()] = items
        print(
            f"{course.name.strip():28} words {len(items):6}  calls {calls:3}  chars {characters:9,}"
        )
    print(
        f"{'total':28} words {sum(map(len, by_course.values())):6}  calls {total_calls:3}  "
        f"chars {total_chars:9,}"
    )
    sample = draw_sample(by_course, args.size, args.seed)
    data = {"seed": args.seed, "kinds": list(KINDS), "items": sample}
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"sample: {len(sample)} forms -> {args.out}")


if __name__ == "__main__":
    main()
