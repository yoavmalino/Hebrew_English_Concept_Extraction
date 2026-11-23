"""Stage 5 of the concept pipeline: an importance score I(c) in [0, 1] for every merged concept.

Design stage 5: eight signals seen in the document (Q, R, T, M, V_b, V_u, V_s, L), each with its
own weight; a signal that marks too much of a document (rho) passes its weight to the others; a
global and a per-section percentile; then breadth across the learning space's files with
near-duplicate files counted once. Pure, no I/O and no model. Not used by the importer yet.
Weights and T's scale come from the dev calibration of 2026-11-17 (plan, stage 5); the other
parameters keep the design's values, which the calibration found had no measurable effect.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .lexicon import STRUCTURAL_WORDS
from .stage2_candidates import DEFINED, NAMED_RESULT_WORDS, OPERATOR, SUBJECT, Mention
from .tokens import split_blocks

SIGNALS = ("Q", "R", "T", "M", "V_b", "V_u", "V_s", "L")
# Opening weights (design stage 5 table); kept for comparison and the design's worked example.
OPENING_WEIGHTS = {
    "Q": 0.24,
    "R": 0.20,
    "T": 0.15,
    "M": 0.14,
    "L": 0.09,
    "V_b": 0.06,
    "V_u": 0.06,
    "V_s": 0.06,
}
# Calibrated weights (2026-11-17): non-negative logistic regression of "relevant" (gold must or
# core) on rho_i z_i over the 10 dev courses; leave-one-course-out weights differed by at most
# 0.03. R gets 0: with the other signals present, frequency added nothing (frequent phrases are
# mostly noise). With eta = 1 the opening weights rank about as well (nDCG@50 0.41 vs 0.42).
CALIBRATED_WEIGHTS = {
    "Q": 0.2049,
    "R": 0.0,
    "T": 0.2108,
    "M": 0.122,
    "L": 0.084,
    "V_b": 0.1431,
    "V_u": 0.1452,
    "V_s": 0.09,
}
# The formatting and layout signals, with their key in hebmath's ``signal_share``.
STYLE_SIGNALS = {"V_b": "bold", "V_u": "underline", "V_s": "size", "L": "line"}
# A signal marking at most this share of the words keeps full value (design: p_0 = 1/16).
P0_FULL_VALUE_SHARE = 1 / 16
# Units of coverage and sections at which R's two parts reach one half (design: u_0 = 3, s_0 = 2).
R_UNITS_HALF = 3.0
R_SECTIONS_HALF = 2.0
# Anchored formulas at which M reaches one half (design: m_0 = 2), and the anchored formulas a
# file needs before M counts in full (design: n_0 = 10).
M_FORMULAS_HALF = 2.0
M_FILE_FORMULAS_FULL = 10.0
# T: log2 of the document-to-background frequency ratio that maps to 1 (not fixed by the design).
# Calibrated 2026-11-17: 2**4 = 16 times more frequent than the background is "fully specific";
# 3 to 6 rank alike, 8 is worse, and T off costs 0.06 to 0.07 nDCG@50 on dev.
T_MAX_LOG2_RATIO = 4.0
# Local ranks are discounted by eta, more in a section shorter than this many units (design:
# m_0 = 5). Calibrated 2026-11-17: eta = 1 (no discount) beat the design's 0.9 in all 10 dev
# courses (+0.064 nDCG@50); 0.7 to 0.9 rank alike.
LOCAL_DISCOUNT = 1.0
SHORT_SECTION_UNITS = 5
# Weight of breadth across the learning space's files (design: delta = 0.3).
BREADTH_WEIGHT = 0.3
# Near-duplicate files (design stage 5, F*): at least this share of a file's units match a unit
# of the other file, a match being Jaccard at least UNIT_MATCH_JACCARD on shingles of
# SHINGLE_WORDS words. Exact Jaccard: a learning space has few files, so MinHash is not needed.
NEAR_DUPLICATE_UNIT_SHARE = 0.9
UNIT_MATCH_JACCARD = 0.8
SHINGLE_WORDS = 5
COMBINE_MODES = ("max", "soft_or")
_WORD_RE = re.compile(r"[א-תA-Za-z]+")


@dataclass(frozen=True)
class MentionEvidence:
    """One mention with what stage 5 reads from it: unit, section, a_cu and its signals."""

    unit: int
    section: int
    strength: float
    signals: frozenset[str]  # of "bold", "underline", "size", "line", "named", "heading"


@dataclass
class ConceptEvidence:
    """The counts one concept has in one file; every signal is computed from these."""

    key: str
    mentions: list[MentionEvidence]
    formulas_by_section: dict[int, int] = field(default_factory=dict)  # anchored formulas
    word_ratios: list[float] = field(default_factory=list)  # log2 doc/background per word
    t_words: list[str] = field(default_factory=list)  # the words T reads (`specificity_words`)

    def formulas(self) -> int:
        """Return the number of formulas anchored to the concept in the file."""
        return sum(self.formulas_by_section.values())


@dataclass
class FileEvidence:
    """What stage 5 needs from one file: its units, signal shares and concepts."""

    file_id: str  # content hash, so ties and representatives do not depend on file order
    unit_omega: list[float]
    unit_section: list[int]
    unit_shingles: list[frozenset[str]]
    signal_share: dict[str, float]
    anchored_formulas: int
    has_background: bool
    concepts: list[ConceptEvidence]


def _diminishing(count: float, half: float = 1.0) -> float:
    """Return 1 - 2^(-count/half): one half at `half`, each repeat adds less (design z_f)."""
    return 1.0 - 2.0 ** (-count / half)


def information_value(share: float, p0: float = P0_FULL_VALUE_SHARE) -> float:
    """Return rho_f = min(1, log p_f / log p_0); a signal marking everything is worth 0.

    A share of 0 means the signal is absent, so it carries no information in this document.
    """
    if share <= 0.0:
        return 0.0
    if share >= 1.0:
        return 0.0
    return min(1.0, math.log(share) / math.log(p0))


def file_rho(evidence: FileEvidence, p0: float = P0_FULL_VALUE_SHARE) -> dict[str, float]:
    """Return rho per signal for a file; it is fixed per file and reused inside its sections.

    Q's sources (definitions, named theorems) have no share in the document, so rho_Q is 1
    (an interpretation to confirm at calibration). R is always 1. T is 1 when a background
    table exists, else 0, so its weight moves to the others. rho_M = min(1, formulas / n_0).
    """
    rho = {"Q": 1.0, "R": 1.0, "T": 1.0 if evidence.has_background else 0.0}
    rho["M"] = min(1.0, evidence.anchored_formulas / M_FILE_FORMULAS_FULL)
    for signal, share_key in STYLE_SIGNALS.items():
        rho[signal] = information_value(evidence.signal_share.get(share_key, 0.0), p0)
    return rho


def effective_weights(weights: dict[str, float], rho: dict[str, float]) -> dict[str, float]:
    """Return w'_i = w_i rho_i / sum_j w_j rho_j; all zero when no signal carries information."""
    total = sum(weights[s] * rho[s] for s in SIGNALS)
    if total == 0.0:
        return {s: 0.0 for s in SIGNALS}
    return {s: weights[s] * rho[s] / total for s in SIGNALS}


def specificity_words(key: str) -> list[str]:
    """Return the words of a key that T reads: all of them but a named result's result word.

    "משפט" or "חוק" is common in every course, so it would pull "משפט wald" below "wald" once
    stage 4 names a group by its named result (2026-11-20, הסתברות: rank 37 to 105).
    """
    words = key.split()
    specific = [word for word in words if word not in NAMED_RESULT_WORDS]
    return specific or words


def _core_key(concept) -> str:
    """Return the key whose words T reads: a merged concept's most mentioned member, else the key.

    A stage-2 `Candidate` has no `core`; a stage-4 `Concept` named by a longer member keeps the
    words the text uses most (see `stage4_merging.Concept.core`).
    """
    return getattr(concept, "core", "") or concept.key


def _specificity(concept: ConceptEvidence) -> float:
    """Return T: the mean log2 frequency ratio of the concept's words, clipped to [0, 1]."""
    if not concept.word_ratios:
        return 0.0
    mean = sum(concept.word_ratios) / len(concept.word_ratios)
    return min(1.0, max(0.0, mean / T_MAX_LOG2_RATIO))


def signal_values(concept: ConceptEvidence, section: int | None = None) -> dict[str, float]:
    """Return z_i for the eight signals, over the whole file or one main section.

    Args:
        concept: The concept's counts in the file.
        section: None for the file level; a section index for the section level, where R is
            the coverage part only (design gap 18) and only that section's mentions count.

    Returns:
        ``{signal: value in [0, 1]}``.
    """
    mentions = [m for m in concept.mentions if section is None or m.section == section]
    strength_by_unit: dict[int, float] = {}
    for m in mentions:
        if m.unit >= 0:
            strength_by_unit[m.unit] = max(strength_by_unit.get(m.unit, 0.0), m.strength)
    coverage = _diminishing(sum(strength_by_unit.values()), R_UNITS_HALF)
    if section is None:
        spread = _diminishing(len({m.section for m in mentions if m.section >= 0}), R_SECTIONS_HALF)
        frequency = (coverage + spread) / 2
    else:
        frequency = coverage
    defined = sum(1 for m in mentions if m.strength >= DEFINED)
    named = sum(1 for m in mentions if _names_unit(m))
    formulas = (
        concept.formulas() if section is None else concept.formulas_by_section.get(section, 0)
    )
    values = {
        "Q": 1.0 - (1.0 - _diminishing(defined)) * (1.0 - _diminishing(named)),
        "R": frequency,
        "T": _specificity(concept),
        "M": _diminishing(formulas, M_FORMULAS_HALF),
    }
    for signal, share_key in STYLE_SIGNALS.items():
        values[signal] = _diminishing(sum(1 for m in mentions if share_key in m.signals))
    return values


def _names_unit(mention: MentionEvidence) -> bool:
    """Return True for the name of a named theorem: its subject mention, not a piece of it."""
    return "named" in mention.signals and mention.strength >= SUBJECT


def combined_score(values: dict[str, float], weights: dict[str, float]) -> float:
    """Return x(c) = sum_i w'_i z_i for effective weights `weights`."""
    return sum(weights[s] * values[s] for s in SIGNALS)


def percentiles(scores: dict[str, float]) -> dict[str, float]:
    """Return each key's mid-rank percentile in [0, 1]; the single top score gets 1.

    Ties share a value, so the result does not depend on the order of the keys.
    """
    if len(scores) == 1:
        return {key: 1.0 for key in scores}
    ordered = sorted(scores.values())
    n = len(ordered)
    result = {}
    for key, value in scores.items():
        below = _count_below(ordered, value)
        equal = _count_below(ordered, value, inclusive=True) - below
        result[key] = (below + (equal - 1) / 2) / (n - 1)
    return result


def _count_below(ordered: list[float], value: float, inclusive: bool = False) -> int:
    """Return how many sorted values are below `value` (or at most it, when inclusive)."""
    low, high = 0, len(ordered)
    while low < high:
        middle = (low + high) // 2
        if ordered[middle] < value or (inclusive and ordered[middle] == value):
            low = middle + 1
        else:
            high = middle
    return low


def file_importance(
    evidence: FileEvidence,
    weights: dict[str, float] = CALIBRATED_WEIGHTS,
    p0: float = P0_FULL_VALUE_SHARE,
    combine: str = "max",
) -> dict[str, float]:
    """Return I_f(c) for every concept of one file (design stage 5, local and global ranks).

    I_f = max(pct_f(x), eta * min(1, |U_s| / m_0) * max_s pct_s(x_s)); with ``combine`` set to
    "soft_or" the two ranks join as 1 - (1 - a)(1 - b), the design's alternative to calibrate.
    """
    if combine not in COMBINE_MODES:
        raise ValueError(f"combine must be one of {COMBINE_MODES}")
    effective = effective_weights(weights, file_rho(evidence, p0))
    by_key = {c.key: c for c in evidence.concepts}
    global_rank = percentiles(
        {key: combined_score(signal_values(c), effective) for key, c in by_key.items()}
    )
    local_best = {key: 0.0 for key in by_key}
    for section in sorted({m.section for c in by_key.values() for m in c.mentions} - {-1}):
        present = {k: c for k, c in by_key.items() if any(m.section == section for m in c.mentions)}
        section_units = sum(1 for s in evidence.unit_section if s == section)
        discount = LOCAL_DISCOUNT * min(1.0, section_units / SHORT_SECTION_UNITS)
        local = percentiles(
            {k: combined_score(signal_values(c, section), effective) for k, c in present.items()}
        )
        for key, value in local.items():
            local_best[key] = max(local_best[key], discount * value)
    if combine == "max":
        return {key: max(global_rank[key], local_best[key]) for key in by_key}
    return {key: 1 - (1 - global_rank[key]) * (1 - local_best[key]) for key in by_key}


def _jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    """Return the Jaccard similarity of two shingle sets (0 when both are empty)."""
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _matched_units(source: FileEvidence, other: FileEvidence) -> list[bool]:
    """Return, per unit of `source`, whether a unit of `other` matches it."""
    return [
        any(_jaccard(shingles, theirs) >= UNIT_MATCH_JACCARD for theirs in other.unit_shingles)
        for shingles in source.unit_shingles
    ]


def _raw_mass(evidence: FileEvidence) -> float:
    """Return the sum of the file's unit weights before duplicates are removed."""
    return sum(evidence.unit_omega)


def canonical_units(files: list[FileEvidence]) -> dict[tuple[str, int], tuple[str, int]]:
    """Return each unit's canonical unit: itself, or the matching unit of an earlier file.

    Files are ordered by raw mass, then content hash, so the mapping does not depend on the
    order the files are given in. A unit duplicated across files is one unit of knowledge, for
    the file mass (stage 5) and for coverage (stage 6) alike.
    """
    ordered = sorted(files, key=lambda f: (-_raw_mass(f), f.file_id))
    canonical: dict[tuple[str, int], tuple[str, int]] = {}
    for index, evidence in enumerate(ordered):
        for unit, shingles in enumerate(evidence.unit_shingles):
            canonical[(evidence.file_id, unit)] = _first_match(shingles, ordered[:index]) or (
                evidence.file_id,
                unit,
            )
    return canonical


def _first_match(shingles: frozenset[str], earlier: list[FileEvidence]) -> tuple[str, int] | None:
    """Return the first unit of an earlier file that matches the shingles, or None."""
    for other in earlier:
        for unit, theirs in enumerate(other.unit_shingles):
            if _jaccard(shingles, theirs) >= UNIT_MATCH_JACCARD:
                return (other.file_id, unit)
    return None


def file_masses(files: list[FileEvidence]) -> dict[str, float]:
    """Return mu_f: the sum of the file's unit weights, a unit duplicated elsewhere counted once.

    A duplicated unit counts in the file that comes first in `canonical_units`' fixed order.
    """
    canonical = canonical_units(files)
    return {
        f.file_id: sum(
            omega
            for unit, omega in enumerate(f.unit_omega)
            if canonical[(f.file_id, unit)] == (f.file_id, unit)
        )
        for f in files
    }


def near_duplicate_groups(files: list[FileEvidence]) -> list[list[str]]:
    """Return groups of near-duplicate files (transitive), each sorted by representative first.

    A is a near-duplicate of B when at least `NEAR_DUPLICATE_UNIT_SHARE` of A's units match a
    unit of B. The representative is the file with the larger mass, then the smaller hash.
    """
    parent = {f.file_id: f.file_id for f in files}

    def root(file_id: str) -> str:
        while parent[file_id] != file_id:
            parent[file_id] = parent[parent[file_id]]
            file_id = parent[file_id]
        return file_id

    for first in files:
        for second in files:
            if first.file_id >= second.file_id or not first.unit_shingles:
                continue
            share = sum(_matched_units(first, second)) / len(first.unit_shingles)
            reverse = (
                sum(_matched_units(second, first)) / len(second.unit_shingles)
                if second.unit_shingles
                else 0.0
            )
            if max(share, reverse) >= NEAR_DUPLICATE_UNIT_SHARE:
                parent[root(first.file_id)] = root(second.file_id)
    mass = {f.file_id: _raw_mass(f) for f in files}
    groups: dict[str, list[str]] = {}
    for file_id in parent:
        groups.setdefault(root(file_id), []).append(file_id)
    return sorted(
        (sorted(group, key=lambda f: (-mass[f], f)) for group in groups.values()),
        key=lambda group: group[0],
    )


def space_importance(
    files: list[FileEvidence],
    weights: dict[str, float] = CALIBRATED_WEIGHTS,
    p0: float = P0_FULL_VALUE_SHARE,
    combine: str = "max",
    breadth_weight: float = BREADTH_WEIGHT,
) -> dict[str, float]:
    """Return I(c) for every concept of a learning space (design stage 5, learning space level).

    I(c) = (1 - delta) max_f I_f(c) + delta * sum_{f in F*(c)} mu_f / sum_f mu_f. A concept is
    the same across files when its key is the same (cross-file merging is design section 5).
    It is recomputed from the set of files, never accumulated, so file order does not matter.
    """
    masses = file_masses(files)
    total_mass = sum(masses.values())
    representative = {
        member: group[0] for group in near_duplicate_groups(files) for member in group
    }
    best: dict[str, float] = {}
    supporting: dict[str, set[str]] = {}
    for evidence in files:
        for key, value in file_importance(evidence, weights, p0, combine).items():
            best[key] = max(best.get(key, 0.0), value)
            supporting.setdefault(key, set()).add(representative[evidence.file_id])
    result = {}
    for key, value in best.items():
        breadth = sum(masses[f] for f in supporting[key]) / total_mass if total_mass else 0.0
        result[key] = (1 - breadth_weight) * value + breadth_weight * min(1.0, breadth)
    return dict(sorted(result.items()))


def background_ratios(document: dict[str, int], background: dict[str, int]) -> dict[str, float]:
    """Return log2 of each document word's frequency against the background, Laplace smoothed.

    The plan's T (section ג): both frequencies add one and divide by their total plus the
    vocabulary of both, so a word absent from the background still gets a finite ratio. The
    background is the owner's courses as counts only, never the document's own course.

    Args:
        document: Normalized word counts of the file.
        background: Normalized word counts of the background courses.

    Returns:
        ``{word: log2 ratio}`` for every word of the document, the input of `file_evidence`.
    """
    vocabulary = len(set(document) | set(background))
    document_total = sum(document.values()) + vocabulary
    background_total = sum(background.values()) + vocabulary
    return {
        word: math.log2(
            ((count + 1) / document_total) / ((background.get(word, 0) + 1) / background_total)
        )
        for word, count in document.items()
    }


def _shingles(text: str) -> frozenset[str]:
    """Return the unit's word shingles of `SHINGLE_WORDS` words (the whole text when shorter)."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < SHINGLE_WORDS:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(
        " ".join(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)
    )


def _mention_signals(mention: Mention, info, run_at) -> frozenset[str]:
    """Return the formatting and layout signals of a mention, read at its first character."""
    signals = set()
    style = run_at(mention.start)
    if style is not None:
        if style.bold:
            signals.add("bold")
        if style.underline:
            signals.add("underline")
        if style.large:
            signals.add("size")
    block = _block_at(info, mention.start)
    if block is not None:
        if block.standalone:
            signals.add("line")
        if block.named_unit:
            signals.add("named")
        if block.role == "heading":
            signals.add("heading")
    return frozenset(signals)


def _block_at(info, position: int):
    """Return the block record that holds a text position, or None."""
    for block in info.blocks:
        if block.start <= position < block.end:
            return block
    return None


def anchored_formulas(result, concepts) -> list[tuple[str, object, object]]:
    """Return ``(concept key, formula, block record)`` for every formula anchored to a concept.

    Design section 8, deterministic, in order of priority: a formula inside a definition or
    theorem unit anchors to the concept the unit defines (its strongest mention there, DEFINED or
    SUBJECT); a formula after a line that ends with a colon anchors to the strongest concept of
    that line; a formula read through the operator lexicon anchors to that concept.
    Equation-number references are not available yet. An unanchored formula is left out.
    """
    info = result.structure
    found = []
    for block in split_blocks(result):
        record = info.blocks[block.index]
        for formula in block.formulas:
            key = _formula_owner(formula, record, block.index, info, result.text, concepts)
            if key is not None:
                found.append((key, formula, record))
    return found


def _formulas_by_section(anchored) -> dict[str, dict[int, int]]:
    """Return the number of anchored formulas per concept and main section."""
    found: dict[str, dict[int, int]] = {}
    for key, _, record in anchored:
        by_section = found.setdefault(key, {})
        by_section[record.section] = by_section.get(record.section, 0) + 1
    return found


def _formula_owner(formula, record, index, info, text, concepts) -> str | None:
    """Return the key of the concept a formula anchors to, or None (priority as above)."""
    if record.unit >= 0 and info.units[record.unit].type in ("definition", "theorem"):
        owner = _strongest_in(concepts, lambda m: m.unit == record.unit, SUBJECT)
        if owner is not None:
            return owner
    previous = info.blocks[index - 1] if index else None
    if previous is not None and text[previous.start : previous.end].rstrip().endswith(":"):
        owner = _strongest_in(concepts, lambda m: previous.start <= m.start < previous.end, 0.0)
        if owner is not None:
            return owner
    return _strongest_in(
        concepts,
        lambda m: m.strength == OPERATOR and m.start == formula.start and m.end == formula.end,
        0.0,
    )


def _strongest_in(concepts, where, minimum: float) -> str | None:
    """Return the key of the concept with the strongest mention passing `where`, ties by key."""
    best = None
    for concept in concepts:
        strengths = [m.strength for m in concept.mentions if where(m) and m.strength >= minimum]
        if strengths and (best is None or (-max(strengths), concept.key) < best):
            best = (-max(strengths), concept.key)
    return best[1] if best else None


def file_evidence(result, concepts, background: dict[str, float] | None = None) -> FileEvidence:
    """Build a file's stage-5 evidence from a hebmath extraction and its merged concepts.

    Args:
        result: A successful hebmath ``ExtractionResult`` with ``structure``.
        concepts: The file's concepts after stage 4 (``stage4_merging.merge``), or stage-2 candidates.
        background: Log2 frequency ratio of each normalized word against the background table
            (built leave-one-course-out from the owner's courses); None when no table exists,
            which turns T off (rho_T = 0).
    """
    info = result.structure
    runs = sorted(info.runs, key=lambda run: run.start)

    def run_at(position: int):
        for run in runs:
            if run.start <= position < run.end:
                return info.styles[run.style]
        return None

    formulas = _formulas_by_section(anchored_formulas(result, concepts))
    evidence = [
        ConceptEvidence(
            key=c.key,
            mentions=[
                MentionEvidence(m.unit, m.section, m.strength, _mention_signals(m, info, run_at))
                for m in c.mentions
            ],
            formulas_by_section=formulas.get(c.key, {}),
            t_words=specificity_words(_core_key(c)),
        )
        for c in concepts
    ]
    for concept in evidence:
        concept.word_ratios = (
            [background.get(word, 0.0) for word in concept.t_words] if background else []
        )
    return FileEvidence(
        file_id=result.sha256,
        unit_omega=[unit.omega for unit in info.units],
        unit_section=[unit.section for unit in info.units],
        unit_shingles=[_shingles(result.text[unit.start : unit.end]) for unit in info.units],
        signal_share=dict(info.signal_share),
        anchored_formulas=sum(sum(by.values()) for by in formulas.values()),
        has_background=background is not None,
        concepts=evidence,
    )


def is_anchored(concept: ConceptEvidence) -> bool:
    """Return True for a concept stage 6 adds first: defined, a named theorem, or a heading topic.

    Design stage 6: a definition, a named theorem, or the topic of a heading that is not generic.
    A topic is the heading's subject mention (a_cu at least SUBJECT), not a word inside it, and
    a standalone line that is not a heading does not count. A name made only of structural words
    ("סיכום", "מבוא") is a generic heading.
    """
    if all(word in STRUCTURAL_WORDS for word in concept.key.split()):
        return False
    return any(
        m.strength >= DEFINED
        or _names_unit(m)
        or (m.strength >= SUBJECT and "heading" in m.signals)
        for m in concept.mentions
    )
