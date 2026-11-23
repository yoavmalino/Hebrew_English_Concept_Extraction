"""Measure stage-4 merging against committee labels: the plan's gate and its diagnostics.

Pure, no I/O. Gate (plan section א and stage 4): duplicates below `DUPLICATE_MAX` and over-merging
of gold concepts (`gold_over_merge`, the plan's definition) at most `OVER_MERGE_MAX`, both on the
gold corrected by the labels (`corrected_concepts`); the two numbers must not move between fresh
decider runs by more than their confidence interval (`within_interval`); the algorithmic gate
(`stage4_merging.check_groups`) is checked apart. Pair agreement between runs (`stability`) and
`over_merge` (any labelled pair) are diagnostics. A pair is always a sorted key tuple.
"""

from __future__ import annotations

from collections import Counter

# Plan section א: duplicates (target below 2%) and over-merging (initial target at most 1%,
# stricter because a wrong merge deletes a concept), both re-approved after the pilot.
DUPLICATE_MAX = 0.02
OVER_MERGE_MAX = 0.01
# Plan section א: the final measure moves between fresh runs by no more than its confidence
# interval. Owner decision 2026-11-16: this replaces pair agreement >= 0.98 as the gate, because
# Haiku 5.5 takes no temperature and borderline pairs flip while the gate numbers stay put.
CONFIDENCE_Z = 1.96
# Reported, not gated: share of pairs with the same verdict in every fresh run.
PAIR_AGREEMENT_TARGET = 0.98
# Judges that must agree for a label (committee of three, as for the concept gold).
MIN_AGREEING_JUDGES = 2

Pair = tuple[str, str]


def vote(answers: list[bool | None]) -> bool | None:
    """Return the label that at least `MIN_AGREEING_JUDGES` judges gave, else None."""
    counts = Counter(answer for answer in answers if answer is not None)
    for label in (True, False):
        if counts[label] >= MIN_AGREEING_JUDGES:
            return label
    return None


def _group_of(groups: list[list[str]]) -> dict[str, int]:
    """Return each key's group index."""
    return {key: index for index, group in enumerate(groups) for key in group}


def over_merge(groups: list[list[str]], labels: dict[Pair, bool | None]) -> tuple[int, int]:
    """Return ``(bad, merged)``: merged groups holding a pair labelled different, of all merged.

    A merged group has two or more keys. Only labelled pairs can show an error, so a chain
    joined through unlabelled pairs is checked through its labelled ones.
    """
    group_of = _group_of(groups)
    bad = {
        group_of[a]
        for (a, b), label in labels.items()
        if label is False and a in group_of and group_of.get(a) == group_of.get(b)
    }
    return len(bad), sum(1 for group in groups if len(group) > 1)


def _components(keys: list[str], labels: dict[Pair, bool | None] | None) -> list[set[str]]:
    """Return the keys split into concepts by the pairs labelled same; no labels keeps them one."""
    if labels is None:
        return [set(keys)] if keys else []
    parent = {key: key for key in keys}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for index, a in enumerate(keys):
        for b in keys[index + 1 :]:
            if labels.get(tuple(sorted((a, b)))) is True:
                parent[root(a)] = root(b)
    found: dict[str, set[str]] = {}
    for key in keys:
        found.setdefault(root(key), set()).add(key)
    return list(found.values())


def duplicates(
    groups: list[list[str]], concept_keys: list[list[str]], labels: dict[Pair, bool | None] | None
) -> tuple[int, int]:
    """Return ``(extra, covering)`` for the plan's duplicate rate ``extra / covering``.

    Each gold concept's keys in the file are split into true concepts by the committee's same
    labels (a gold concept that joined related names is split back). `covering` counts the final
    groups that hold any gold key; `extra` adds, per true concept, the groups it is spread over
    minus one. This is the plan's (|S| - unique gold groups in S) / |S| with S = final concepts
    that match the gold.
    """
    group_of = _group_of(groups)
    covering: set[int] = set()
    extra = 0
    for keys in concept_keys:
        present = [key for key in keys if key in group_of]
        covering |= {group_of[key] for key in present}
        for component in _components(present, labels):
            extra += len({group_of[key] for key in component}) - 1
    return extra, len(covering)


def gold_over_merge(
    groups: list[list[str]], concept_keys: list[list[str]], labels: dict[Pair, bool | None] | None
) -> tuple[int, int]:
    """Return ``(bad, merged)`` by the plan's definition: two different gold concepts merged.

    Each gold concept's keys are split into true concepts by the same labels (as in
    `duplicates`). `merged` counts the final groups with two or more keys that hold any gold
    key; a group is bad when it holds two gold keys that no true concept holds together. Pairs
    of keys that name no gold concept (ordinary word forms) do not count, which the stricter
    `over_merge` counts.
    """
    concepts_of: dict[str, set[int]] = {}
    number = 0
    for keys in concept_keys:
        for component in _components(keys, labels):
            for key in component:
                concepts_of.setdefault(key, set()).add(number)
            number += 1
    bad = merged = 0
    for group in groups:
        gold_keys = [key for key in group if key in concepts_of]
        if len(group) < 2 or not gold_keys:
            continue
        merged += 1
        if any(
            not concepts_of[a] & concepts_of[b]
            for index, a in enumerate(gold_keys)
            for b in gold_keys[index + 1 :]
        ):
            bad += 1
    return bad, merged


def is_contained(first: str, second: str) -> bool:
    """Return True when one key's words are all inside the other's (a fragment of a name)."""
    words_first, words_second = set(first.split()), set(second.split())
    return words_first <= words_second or words_second <= words_first


def _is_split(first: str, second: str, labels: dict[Pair, bool | None]) -> bool:
    """Return True when the labels say two keys are different and neither is a fragment."""
    pair = tuple(sorted((first, second)))
    return labels.get(pair) is False and not is_contained(first, second)


def corrected_concepts(
    concept_keys: list[list[str]], labels: dict[Pair, bool | None]
) -> list[list[str]]:
    """Return the gold concepts corrected by the committee's labels where the gold contradicts them.

    Owner decision 2026-11-16, fixed before the held-out run: a gold concept is split where the
    committee labelled two of its keys different, unless one is a fragment of the other (the
    owner's anchor calls "supervised" / "supervised learning" one concept); two gold concepts are
    joined where the committee labelled a pair of their keys same (the gold listed one concept
    twice, as recall / החזרה). A join that would hold a split pair is skipped. Keys are placed in
    sorted order, so the result does not depend on the input order.

    Args:
        concept_keys: Each gold concept's candidate keys in the file.
        labels: Committee votes per sorted key pair; None is unlabelled.

    Returns:
        The corrected concepts as sorted key lists, ordered by their first key.
    """
    parts: list[set[str]] = []
    for keys in concept_keys:
        own: list[set[str]] = []
        for key in sorted(set(keys)):
            home = next((p for p in own if not any(_is_split(key, k, labels) for k in p)), None)
            if home is None:
                own.append({key})
            else:
                home.add(key)
        parts += own
    for (first, second), label in sorted(labels.items(), key=lambda item: item[0]):
        if label is not True:
            continue
        a = next((p for p in parts if first in p), None)
        b = next((p for p in parts if second in p and p is not a), None)
        if a is None or b is None or first in b or second in a:
            continue
        if any(_is_split(x, y, labels) for x in a for y in b):
            continue
        a |= b
        parts.remove(b)
    return sorted((sorted(p) for p in parts), key=lambda keys: keys[0])


def within_interval(measures: list[tuple[int, int]]) -> bool:
    """Return True when a rate moves across runs by no more than its confidence interval.

    The interval is the normal approximation around the first run's rate, ``CONFIDENCE_Z`` times
    its standard error (half-width); the spread is the largest minus the smallest rate.

    Args:
        measures: ``(count, total)`` of the same rate in each fresh run, the first one reported.
    """
    rates = [count / total if total else 0.0 for count, total in measures]
    count, total = measures[0]
    if not total:
        return True
    half_width = CONFIDENCE_Z * (rates[0] * (1 - rates[0]) / total) ** 0.5
    return max(rates) - min(rates) <= half_width


def stability(runs: list[dict[Pair, bool]], pairs: list[Pair]) -> float:
    """Return the share of pairs whose verdict (or missing verdict) is the same in every run."""
    if not pairs:
        return 1.0
    same = sum(1 for pair in pairs if len({run.get(pair) for run in runs}) == 1)
    return same / len(pairs)


def pair_scores(decided: dict[Pair, bool], labels: dict[Pair, bool | None]) -> dict[str, float]:
    """Return the decider's precision and recall of "same" over the labelled pairs.

    An undecided pair counts as "different", because it stays apart.
    """
    labelled = {pair: label for pair, label in labels.items() if label is not None}
    said_same = {pair for pair in labelled if decided.get(pair) is True}
    truly_same = {pair for pair, label in labelled.items() if label}
    hits = len(said_same & truly_same)
    return {
        "labelled": len(labelled),
        "same": len(truly_same),
        "precision": hits / len(said_same) if said_same else 1.0,
        "recall": hits / len(truly_same) if truly_same else 1.0,
    }


def merged_share(groups: list[list[str]], pairs: set[Pair]) -> float:
    """Return the share of pairs whose two keys end in one final group."""
    if not pairs:
        return 1.0
    group_of = _group_of(groups)
    joined = sum(1 for a, b in pairs if a in group_of and group_of[a] == group_of.get(b))
    return joined / len(pairs)
