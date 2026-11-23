"""Stage 6 of the concept pipeline: choose the learning space's nodes by marginal information.

Design stage 6: a concept enters when it covers knowledge not yet represented or is important
enough on its own, g(c|S) = lambda I(c) + (1 - lambda) DeltaN(c|S). Anchored concepts enter
first, the rest greedily (lazy evaluation; g is monotone and submodular), at most 150 nodes per
learning space, with an epsilon margin to replace the weakest unprotected node, which is hidden,
not deleted. Builds the mention dictionary stage 7 reads. Pure, no I/O and no model. Not used by
the importer yet; lambda, tau and epsilon come from the dev calibration of 2026-11-17.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field

from .stage2_candidates import DEFINED, SUBJECT
from .stage5_importance import FileEvidence, anchored_formulas, canonical_units, is_anchored

# The learning space's only hard cap (owner decision: a course has fewer key concepts than the
# sum of its lectures).
MAX_NODES = 150
# Balance of importance against coverage (design: lambda = 0.3). Calibrated 2026-11-17: 1.0, so
# g = I. On dev every smaller lambda lost recall at the cap (0.3: 0.37 recall of must concepts,
# 1.0: 0.60): coverage favours the many frequent noise phrases of the stage-2 pool. Coverage
# still drives the replacement check and variant B; revisit when the pool is cleaner.
IMPORTANCE_WEIGHT = 1.0
# Entry threshold (design: tau = 0.25, variant A). Calibrated 2026-11-17: with lambda = 1 the cap
# binds first on every dev course, and 0 to 0.7 select alike; 0.35 is the middle of that range,
# so a small file does not take its weakest quarter of concepts.
ENTRY_THRESHOLD = 0.35
# A candidate must beat the weakest node by this margin to replace it (design: epsilon = 0.10).
# Calibrated 2026-11-17: 0.1 to 0.3 alike within 0.02; 0.2 is the middle.
REPLACE_MARGIN = 0.20
# Variant B scales coverage to [0, 1] with the same half point as R (design: u_0 = 3); its
# threshold is searched in 0.1-0.5 at calibration.
COVERAGE_HALF_UNITS = 3.0
VARIANTS = ("A", "B")
# Central mentions kept per node; the design fixes the kinds and their order, not the number.
MAX_DICTIONARY_ENTRIES = 12
# Kinds of central mention, in the design's ranking order; "mention" is the fallback so that
# every node has at least one entry.
DICTIONARY_KINDS = ("definition", "subject", "formula", "co_mention", "mention")
_SENTENCE_END_RE = re.compile(r"[.!?;:\n]")

Unit = tuple[str, int]


@dataclass
class SpaceConcept:
    """A concept of the learning space with what selection reads: importance and coverage."""

    key: str
    importance: float
    coverage: dict[Unit, float]  # canonical unit -> a_cu (strongest mention in that unit)
    omega: dict[Unit, float]  # canonical unit -> its weight
    is_anchored: bool = False
    is_protected: bool = False  # user edits or a learner's mastery data


@dataclass
class Selection:
    """The chosen nodes, the hidden (replaced) ones and the pool, with a short report."""

    nodes: list[str]  # in order of entry
    hidden: list[str]  # replaced nodes: kept with their id, not deleted
    pool: list[str]
    report: dict[str, int] = field(default_factory=dict)


def space_concepts(
    files: list[FileEvidence], importance: dict[str, float], protected: set[str] = frozenset()
) -> list[SpaceConcept]:
    """Return the space's concepts with coverage on canonical units (duplicates counted once).

    A concept is the same across files when its key is the same, as in `stage5_importance.space_importance`.
    """
    canonical = canonical_units(files)
    found: dict[str, SpaceConcept] = {}
    for evidence in files:
        for concept in evidence.concepts:
            entry = found.setdefault(
                concept.key,
                SpaceConcept(concept.key, importance.get(concept.key, 0.0), {}, {}),
            )
            entry.is_anchored = entry.is_anchored or is_anchored(concept)
            entry.is_protected = concept.key in protected
            for m in concept.mentions:
                if m.unit < 0:
                    continue
                unit = canonical[(evidence.file_id, m.unit)]
                entry.coverage[unit] = max(entry.coverage.get(unit, 0.0), m.strength)
                entry.omega[unit] = evidence.unit_omega[m.unit]
    return [found[key] for key in sorted(found)]


def coverage_gain(concept: SpaceConcept, uncovered: dict[Unit, float]) -> float:
    """Return DeltaN(c|S) = sum_u omega_u a_cu prod_{c' in S}(1 - a_c'u).

    `uncovered` holds the product per unit for the current selection; a missing unit is 1.
    """
    return sum(
        concept.omega[unit] * strength * uncovered.get(unit, 1.0)
        for unit, strength in concept.coverage.items()
    )


def marginal_information(
    concept: SpaceConcept,
    uncovered: dict[Unit, float],
    importance_weight: float = IMPORTANCE_WEIGHT,
    variant: str = "A",
) -> float:
    """Return g(c|S) for variant A (coverage in units) or B (coverage scaled to [0, 1])."""
    gain = coverage_gain(concept, uncovered)
    if variant == "B":
        gain = 1.0 - 2.0 ** (-gain / COVERAGE_HALF_UNITS)
    return importance_weight * concept.importance + (1 - importance_weight) * gain


def _uncovered(selected: list[SpaceConcept]) -> dict[Unit, float]:
    """Return prod (1 - a_cu) per unit over the selected concepts."""
    product: dict[Unit, float] = {}
    for concept in selected:
        for unit, strength in concept.coverage.items():
            product[unit] = product.get(unit, 1.0) * (1.0 - strength)
    return product


def _add(uncovered: dict[Unit, float], concept: SpaceConcept) -> None:
    """Update the uncovered products in place after `concept` enters the selection."""
    for unit, strength in concept.coverage.items():
        uncovered[unit] = uncovered.get(unit, 1.0) * (1.0 - strength)


def select_nodes(
    concepts: list[SpaceConcept],
    max_nodes: int = MAX_NODES,
    importance_weight: float = IMPORTANCE_WEIGHT,
    threshold: float = ENTRY_THRESHOLD,
    margin: float = REPLACE_MARGIN,
    variant: str = "A",
) -> Selection:
    """Choose the nodes: anchored first, then greedily by g while g >= tau, then replacements.

    Ties are broken by importance, then key, so the same input always gives the same nodes.
    Anchored concepts beyond the cap stay in the pool and are reported. After the cap, each
    remaining candidate (strongest first) may replace the weakest unprotected node r when
    g(c|S\\{r}) > (1 + epsilon) g(r|S\\{r}); r is hidden. An anchored node is not protected.

    Raises:
        ValueError: An unknown variant.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    by_key = {c.key: c for c in concepts}
    anchored = sorted((c for c in concepts if c.is_anchored), key=lambda c: (-c.importance, c.key))
    nodes = [c.key for c in anchored[:max_nodes]]
    uncovered = _uncovered([by_key[k] for k in nodes])

    def gain(concept: SpaceConcept) -> float:
        return marginal_information(concept, uncovered, importance_weight, variant)

    rest = [c for c in concepts if c.key not in set(nodes)]
    heap = [(-gain(c), -c.importance, c.key) for c in rest]
    heapq.heapify(heap)
    while heap and len(nodes) < max_nodes:
        _, negative_importance, key = heapq.heappop(heap)
        fresh = (-gain(by_key[key]), negative_importance, key)
        # Lazy evaluation: gains only fall as S grows, so a fresh value still ahead of the next
        # stale bound is the true maximum.
        if heap and fresh > heap[0]:
            heapq.heappush(heap, fresh)
            continue
        if -fresh[0] < threshold:
            heapq.heappush(heap, fresh)
            break
        nodes.append(key)
        _add(uncovered, by_key[key])
    hidden = []
    # Replacement is only for a full learning space; below the cap a candidate simply enters.
    if len(nodes) >= max_nodes:
        hidden = _replace_weakest(
            nodes, heap, by_key, importance_weight, threshold, margin, variant
        )
    # A hidden node keeps its id and its dependencies (design stage 6), so it is not pool.
    pool = sorted(set(by_key) - set(nodes) - set(hidden))
    report = {
        "anchored": len(anchored),
        "anchored_over_cap": max(0, len(anchored) - max_nodes),
        "hidden": len(hidden),
    }
    return Selection(nodes, hidden, pool, report)


def _replace_weakest(nodes, heap, by_key, importance_weight, threshold, margin, variant):
    """Replace weak unprotected nodes with stronger candidates once the cap is reached.

    Each candidate above the threshold is tried once, strongest first (by its gain when the
    cap was reached); `nodes` is changed in place. Returns the hidden keys in order.
    """
    hidden: list[str] = []
    candidates = sorted(heap)
    covering = _covering(nodes, by_key)
    for _, _, key in candidates:
        weakest = _weakest_unprotected(nodes, by_key, importance_weight, variant, covering)
        if weakest is None:
            break
        challenger = marginal_information(
            by_key[key],
            _uncovered_without(by_key[key], covering, weakest),
            importance_weight,
            variant,
        )
        incumbent = marginal_information(
            by_key[weakest],
            _uncovered_without(by_key[weakest], covering, weakest),
            importance_weight,
            variant,
        )
        if challenger >= threshold and challenger > (1 + margin) * incumbent:
            nodes[nodes.index(weakest)] = key
            hidden.append(weakest)
            covering = _covering(nodes, by_key)
    return hidden


def _covering(nodes, by_key) -> dict[Unit, list[tuple[str, float]]]:
    """Return, per unit, the selected nodes covering it with a_cu, in the order of `nodes`."""
    covering: dict[Unit, list[tuple[str, float]]] = {}
    for key in nodes:
        for unit, strength in by_key[key].coverage.items():
            covering.setdefault(unit, []).append((key, strength))
    return covering


def _uncovered_without(concept, covering, excluded: str) -> dict[Unit, float]:
    """Return prod (1 - a_c'u) over the selection without `excluded`, on `concept`'s units only.

    The same products `_uncovered` gives for S without `excluded`, multiplied in the same order,
    so the gains are identical; only the units g(concept) reads are computed. This keeps a
    replacement pass linear in the coverage instead of quadratic in the nodes.
    """
    product: dict[Unit, float] = {}
    for unit in concept.coverage:
        for key, strength in covering.get(unit, ()):
            if key != excluded:
                product[unit] = product.get(unit, 1.0) * (1.0 - strength)
    return product


def _weakest_unprotected(nodes, by_key, importance_weight, variant, covering) -> str | None:
    """Return the unprotected node with the lowest g(r|S\\{r}) (ties: importance, key)."""
    best = None
    for key in nodes:
        concept = by_key[key]
        if concept.is_protected:
            continue
        others = _uncovered_without(concept, covering, key)
        value = marginal_information(concept, others, importance_weight, variant)
        rank = (value, concept.importance, key)
        if best is None or rank < best:
            best = rank
    return best[2] if best else None


@dataclass(frozen=True)
class DictionaryEntry:
    """One central mention of a node: where it is, its kind and a_cu, and the passage."""

    file_id: str
    unit: int
    kind: str
    strength: float
    start: int
    text: str


def _sentence_span(text: str, start: int, end: int, block_start: int, block_end: int):
    """Return the sentence around a span, cut at sentence ends and at the block's edges."""
    left = block_start
    for match in _SENTENCE_END_RE.finditer(text, block_start, start):
        left = match.end()
    right_match = _SENTENCE_END_RE.search(text, end, block_end)
    right = right_match.end() if right_match else block_end
    return left, right


def _block_of(info, position: int):
    """Return the block record holding a position, or None."""
    for block in info.blocks:
        if block.start <= position < block.end:
            return block
    return None


def _file_entries(file_id: str, result, concepts, chosen: set[str]) -> dict[str, list]:
    """Return a file's candidate dictionary entries per chosen key, before ranking."""
    info, text = result.structure, result.text
    entries: dict[str, list[DictionaryEntry]] = {}
    sentences: dict[tuple[int, int], set[str]] = {}
    for concept in concepts:
        if concept.key not in chosen:
            continue
        for m in concept.mentions:
            block = _block_of(info, m.start)
            if block is None:
                continue
            span = _sentence_span(text, m.start, m.end, block.start, block.end)
            sentences.setdefault(span, set()).add(concept.key)
            kind = _mention_kind(m, block)
            passage = text[span[0] : span[1]].strip()
            entries.setdefault(concept.key, []).append(
                DictionaryEntry(file_id, m.unit, kind, m.strength, m.start, passage)
            )
    for key, found in entries.items():
        entries[key] = [_with_co_mention(e, key, sentences, text) for e in found]
    for key, formula, record in anchored_formulas(result, concepts):
        if key in chosen:
            entries.setdefault(key, []).append(
                DictionaryEntry(file_id, record.unit, "formula", 1.0, formula.start, formula.latex)
            )
    return entries


def _mention_kind(mention, block) -> str:
    """Return the dictionary kind of a mention before co-mentions are known."""
    if mention.strength >= DEFINED or block.named_unit:
        return "definition"
    if mention.strength >= SUBJECT:
        return "subject"
    return "mention"


def _with_co_mention(entry: DictionaryEntry, key: str, sentences, text) -> DictionaryEntry:
    """Return the entry as a co-mention when its sentence also names another chosen node."""
    if entry.kind != "mention":
        return entry
    for span, keys in sentences.items():
        if span[0] <= entry.start < span[1] and keys - {key}:
            return DictionaryEntry(
                entry.file_id, entry.unit, "co_mention", entry.strength, entry.start, entry.text
            )
    return entry


def mention_dictionary(
    nodes: list[str], files: list[tuple[str, object, list]]
) -> dict[str, list[DictionaryEntry]]:
    """Return each node's central mentions, ranked by kind, then file and position.

    Args:
        nodes: The chosen keys.
        files: ``(file_id, hebmath result, concepts)`` for each file of the learning space.

    Returns:
        ``{key: entries}`` with at most `MAX_DICTIONARY_ENTRIES` each; a node with no central
        mention keeps its plain mentions, so every node has at least one entry. The ranking is
        deterministic (design stage 6).
    """
    chosen = set(nodes)
    gathered: dict[str, list[DictionaryEntry]] = {key: [] for key in nodes}
    for file_id, result, concepts in sorted(files, key=lambda item: item[0]):
        for key, found in _file_entries(file_id, result, concepts, chosen).items():
            gathered[key].extend(found)
    ranked = {}
    for key, found in gathered.items():
        order = sorted(
            set(found), key=lambda e: (DICTIONARY_KINDS.index(e.kind), e.file_id, e.start, e.text)
        )
        central = [e for e in order if e.kind != "mention"]
        ranked[key] = (central or order)[:MAX_DICTIONARY_ENTRIES]
    return ranked
