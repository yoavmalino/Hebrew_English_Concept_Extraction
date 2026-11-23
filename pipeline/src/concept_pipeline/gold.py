"""Concept gold for the stage-2 gate: merge committee answers, vote, match candidates, recall.

Two layers (owner decision 2026-11-16, plan section יא): "must" = needed to solve a past-exam
question; "core" = a concept a lecturer could reasonably ask about, or needed to understand one.
A must concept is also core. Pure logic: the judges' answers come in as parsed JSON.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .normalize import PrefixNormalizer, is_name_match
from .stage2_candidates import NAMED_RESULT_WORDS, SOURCES, STOP_WORDS, Candidate, Stage2Result
from .tokens import split_blocks, words_of

LAYERS = ("must", "core")
# Starting k for k of m judges; chosen against the owner's sample check (design section 11).
DEFAULT_MIN_JUDGES = 2
# Frequency buckets for the report on rare concepts; 1-2 is the risk stage 2 must cover. A name
# that never appears word for word (0) is reported apart and is not part of the gate.
RARE_MAX_MENTIONS = 2
# The stage-2 gate counts concepts named at least this often (owner decision 2026-11-16): a
# concept named once or twice in plain prose is left to stage 5, which would drop it anyway,
# and catching it here needs n-grams of count 1 (pool 856 instead of 182 per 1,000 words on dev).
# Rare concepts are still reported in their own bucket.
GATE_MIN_MENTIONS = RARE_MAX_MENTIONS + 1
GATE_RECALL = 0.95
# A judge's name may hold a synonym: "K-Nearest Neighbors (KNN)", "Idf – inverse document
# frequency". Parentheses and a dash with spaces split it; a bare hyphen or slash does not.
_PAREN_RE = re.compile(r"\(([^()]*)\)")
_SPACED_DASH_RE = re.compile(r"\s[-–—]\s")


@dataclass(frozen=True)
class JudgeConcept:
    """One concept as one judge listed it."""

    judge: str
    layer: str  # "must" or "core"
    name: str
    aliases: tuple[str, ...]
    in_material: bool
    questions: tuple[str, ...] = ()


@dataclass
class GoldConcept:
    """Concepts from all judges that share a normalized form, with the votes for each layer."""

    names: Counter = field(default_factory=Counter)
    forms: set[tuple[str, ...]] = field(default_factory=set)
    must_judges: set[str] = field(default_factory=set)
    core_judges: set[str] = field(default_factory=set)
    in_material: dict[str, bool] = field(default_factory=dict)
    questions: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        """Return the name most judges used (ties broken alphabetically)."""
        return min(self.names, key=lambda name: (-self.names[name], name))

    def judges(self, layer: str) -> set[str]:
        """Return the judges who put the concept in a layer; a must vote also counts for core."""
        return set(self.must_judges) if layer == "must" else self.must_judges | self.core_judges

    def is_in_material(self) -> bool:
        """Return False only when most judges who listed it said the summary lacks it."""
        absent = sum(1 for present in self.in_material.values() if not present)
        return absent * 2 <= len(self.in_material)


def parse_answer(judge: str, answer: dict) -> list[JudgeConcept]:
    """Turn one judge's JSON answer into concepts, skipping malformed entries.

    Args:
        judge: The judge's name (one per provider and model).
        answer: ``{"must": [...], "core": [...]}``; each entry has ``concept`` and optionally
            ``aliases``, ``in_material`` (default true) and ``questions``.

    Returns:
        The well-formed entries in answer order.
    """
    concepts = []
    for layer in LAYERS:
        entries = answer.get(layer) if isinstance(answer, dict) else None
        for entry in entries if isinstance(entries, list) else []:
            name = entry.get("concept") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not name.strip():
                continue
            aliases = tuple(a for a in entry.get("aliases") or [] if isinstance(a, str) and a)
            questions = tuple(str(q) for q in entry.get("questions") or [])
            in_material = entry.get("in_material", True) is not False
            concepts.append(
                JudgeConcept(judge, layer, name.strip(), aliases, in_material, questions)
            )
    return concepts


def normal_form(text: str, normalizer: PrefixNormalizer) -> tuple[str, ...]:
    """Return a name's normal form, split and normalized like the summary's own words."""
    return normalizer.normalize_phrase(tuple(words_of(text)))


def align(concepts: list[JudgeConcept], normalizer: PrefixNormalizer) -> list[GoldConcept]:
    """Group judge concepts by exact normalized name, or by a mutual alias link.

    Two concepts merge when their names have the same normal form, or when each one's aliases
    hold the other's name ("Deadlock" with alias "קיפאון" and "קיפאון" with alias "deadlock").
    A one-way link is not enough: judges list related concepts as aliases, and one-way links
    chained distinct concepts (a semaphore group of ten names, HashSet with queue and stack;
    found 2026-11-16, 405 of 1,105 dev concepts). Two aliases never merge concepts, and near
    names stay apart on purpose: merging them would need a judgement (plan section יא).
    """
    parent = list(range(len(concepts)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    named: dict[tuple[str, ...], int] = {}
    forms_of = []
    for i, concept in enumerate(concepts):
        name_form = normal_form(concept.name, normalizer)
        forms = {name_form} | {normal_form(alias, normalizer) for alias in concept.aliases}
        forms.discard(())
        forms_of.append(forms)
        if not name_form:
            continue
        if name_form in named:
            parent[root(i)] = root(named[name_form])
        else:
            named[name_form] = i
    by_name: dict[tuple[str, ...], list[int]] = {}
    for i, concept in enumerate(concepts):
        by_name.setdefault(normal_form(concept.name, normalizer), []).append(i)
    aliases_of = [{normal_form(a, normalizer) for a in c.aliases} for c in concepts]
    for i, concept in enumerate(concepts):
        own_name = normal_form(concept.name, normalizer)
        for alias in aliases_of[i]:
            for j in by_name.get(alias, []):
                if own_name in aliases_of[j]:
                    parent[root(i)] = root(j)
    groups: dict[int, GoldConcept] = {}
    for i, concept in enumerate(concepts):
        if not forms_of[i]:
            continue
        group = groups.setdefault(root(i), GoldConcept())
        group.names[concept.name] += 1
        group.forms |= forms_of[i]
        (group.must_judges if concept.layer == "must" else group.core_judges).add(concept.judge)
        group.in_material[concept.judge] = group.in_material.get(concept.judge, False) or (
            concept.in_material
        )
        group.questions |= set(concept.questions)
    return sorted(groups.values(), key=lambda group: group.name)


def name_parts(name: str) -> list[str]:
    """Return a judge's name and the synonyms written inside it ("KNN" in "k-NN (KNN)")."""
    parts = [name]
    inner = _PAREN_RE.findall(name)
    outer = _PAREN_RE.sub(" ", name)
    parts += inner + [outer]
    for text in [outer, *inner]:
        parts += _SPACED_DASH_RE.split(text)
    return list(dict.fromkeys(part.strip() for part in parts if part.strip()))


def name_forms(concept: GoldConcept, normalizer: PrefixNormalizer) -> set[tuple[str, ...]]:
    """Return the normal forms of the judges' names, without their alias lists.

    Aliases are left out on purpose: judges list related concepts as aliases ("list" under
    HashSet), and matching them counted wrong finds (plan section יא, 2026-11-16).
    """
    forms = {normal_form(part, normalizer) for name in concept.names for part in name_parts(name)}
    forms.discard(())
    return forms


# Heads that open a name without naming a concept: listed words, the kind of a named result
# ("משפט היינה", "law of large numbers" is English, so its head is "numbers") and "הגדרת X".
UNCREDITED_HEADS = STOP_WORDS | NAMED_RESULT_WORDS | {"הגדרת"}


def head_words(concept: GoldConcept, normalizer: PrefixNormalizer) -> set[str]:
    """Return the head word of each of a concept's name forms (measurement only).

    The head is the first word of a Hebrew name and the last of an English one ("גבול" in
    "גבול חלקי", "sort" in "heap sort"). Owner decision 2026-11-17: a candidate that is only a
    head word of a gold concept's name names the general concept the judges listed by kinds
    ("גבול" for "גבול חלקי" and "גבול של סדרה"), so it is labelled valid, not noise. A head that
    is a listed word ("משפט" in "משפט היינה", "הגדרת") is never credited; a modifier ("מותנה")
    or a word inside a name by chance ("עולם" in "שגיאת העולם") is not a head.
    """
    heads = set()
    for form in name_forms(concept, normalizer):
        if len(form) < 2:
            continue
        head = form[-1] if all(word.isascii() for word in form) else form[0]
        if head not in UNCREDITED_HEADS:
            heads.add(head)
    return heads


def match(
    concept: GoldConcept, candidates: list[Candidate], normalizer: PrefixNormalizer
) -> Candidate | None:
    """Return the candidate that finds a gold concept, or None.

    Found when a name form (`name_forms`) equals a candidate key, or passes `is_name_match`
    against it. An exact match wins; among close ones the most mentioned candidate, then the
    first key alphabetically. Stricter since 2026-11-16: the first rule also used aliases and
    `is_close_normalization`, which inflated recall (plan section יא).
    """
    forms = {" ".join(form) for form in name_forms(concept, normalizer)}
    by_key = {candidate.key: candidate for candidate in candidates}
    exact = sorted(forms & by_key.keys())
    if exact:
        return by_key[exact[0]]
    close = [c for c in candidates if any(is_name_match(form, c.key) for form in forms)]
    return min(close, key=lambda c: (-c.count, c.key)) if close else None


def text_frequency(concept: GoldConcept, result, normalizer: PrefixNormalizer) -> int:
    """Count how often any name form of the concept appears in the summary's prose."""
    forms = name_forms(concept, normalizer)
    if not forms:
        return 0
    longest = max(len(form) for form in forms)
    count = 0
    for block in split_blocks(result):
        for segment in block.segments:
            norms = [normalizer.normalize_word(token.text) for token in segment.tokens]
            for i in range(len(norms)):
                for n in range(1, min(longest, len(norms) - i) + 1):
                    if tuple(norms[i : i + n]) in forms:
                        count += 1
    return count


def _bucket(frequency: int) -> str:
    """Return the frequency bucket name for the rare-concept report."""
    if frequency == 0:
        return "0"
    return (
        f"1-{RARE_MAX_MENTIONS}" if frequency <= RARE_MAX_MENTIONS else f"{RARE_MAX_MENTIONS + 1}+"
    )


def _ratio(found: int, total: int) -> float | None:
    """Return found/total rounded, or None when there is nothing to measure."""
    return round(found / total, 3) if total else None


def recall_report(
    gold: list[GoldConcept], stage2: Stage2Result, result, min_judges: int = DEFAULT_MIN_JUDGES
) -> dict:
    """Return stage-2 recall per layer, per frequency bucket and with each source left out.

    A concept enters a layer when at least `min_judges` judges put it there. Concepts most
    judges marked as missing from the summary leave the denominator and are counted apart;
    concepts below `min_judges` are counted as uncertain, as the design says. "gate" is recall
    on concepts named at least `GATE_MIN_MENTIONS` times, the number the gate compares with
    `GATE_RECALL`. The gate counts
    only concepts whose name (`name_forms`, aliases excluded) appears in the summary at least once (owner decision 2026-11-16:
    an algorithm that reads the summary cannot find a name the summary never writes); those are
    reported apart as "unnamed". "without_source" is the drop-one check: recall if a source
    were removed, counting a concept as lost when its matched candidate came from that source
    alone (another close candidate is not searched).

    Returns:
        ``{layer: {"concepts", "found", "recall", "gate", "by_frequency", "without_source",
        "not_in_material", "unnamed", "missed"}}`` plus ``"uncertain"`` (concepts below k).
    """
    report: dict = {}
    for layer in LAYERS:
        voted = [g for g in gold if len(g.judges(layer)) >= min_judges]
        in_material = [g for g in voted if g.is_in_material()]
        frequency = {id(g): text_frequency(g, result, stage2.normalizer) for g in in_material}
        measured = [g for g in in_material if frequency[id(g)] > 0]
        unnamed = [g for g in in_material if frequency[id(g)] == 0]
        matches = {id(g): match(g, stage2.candidates, stage2.normalizer) for g in in_material}
        found = [g for g in measured if matches[id(g)] is not None]
        buckets: dict[str, list[int]] = {}
        for g in measured:
            counts = buckets.setdefault(_bucket(frequency[id(g)]), [0, 0])
            counts[0] += matches[id(g)] is not None
            counts[1] += 1
        without = {
            source: _ratio(
                sum(1 for g in found if set(matches[id(g)].sources) - {source}), len(measured)
            )
            for source in SOURCES
        }
        unnamed_found = sum(1 for g in unnamed if matches[id(g)] is not None)
        gated = [g for g in measured if frequency[id(g)] >= GATE_MIN_MENTIONS]
        gated_found = sum(1 for g in gated if matches[id(g)] is not None)
        report[layer] = {
            "concepts": len(measured),
            "found": len(found),
            "recall": _ratio(len(found), len(measured)),
            "gate": {
                "concepts": len(gated),
                "found": gated_found,
                "recall": _ratio(gated_found, len(gated)),
            },
            "by_frequency": {
                name: {"found": f, "concepts": t, "recall": _ratio(f, t)}
                for name, (f, t) in sorted(buckets.items())
            },
            "without_source": without,
            "not_in_material": len(voted) - len(in_material),
            "unnamed": {"concepts": len(unnamed), "found": unnamed_found},
            "missed": sorted(g.name for g in measured if matches[id(g)] is None),
        }
    report["uncertain"] = sum(1 for g in gold if 0 < len(g.judges("core")) < min_judges)
    return report


def gold_to_json(gold: list[GoldConcept]) -> list[dict]:
    """Return the gold concepts as JSON-ready data, sorted and with sets as sorted lists."""
    return [
        {
            "name": g.name,
            "names": dict(sorted(g.names.items())),
            "forms": sorted(" ".join(form) for form in g.forms),
            "must_judges": sorted(g.must_judges),
            "core_judges": sorted(g.core_judges),
            "in_material": dict(sorted(g.in_material.items())),
            "questions": sorted(g.questions),
        }
        for g in gold
    ]


def gold_from_json(data: list[dict]) -> list[GoldConcept]:
    """Rebuild gold concepts from `gold_to_json` output."""
    return [
        GoldConcept(
            names=Counter(item["names"]),
            forms={tuple(form.split()) for form in item["forms"]},
            must_judges=set(item["must_judges"]),
            core_judges=set(item["core_judges"]),
            in_material=dict(item["in_material"]),
            questions=set(item["questions"]),
        )
        for item in data
    ]
