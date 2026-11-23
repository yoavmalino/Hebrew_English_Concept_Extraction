"""Shared helpers for stage-5/6 calibration on dumped course evidence (pure Python, no numpy).

Loads the per-course dumps written by `dump.py`, builds T's background, computes nDCG@50 and
selection metrics, and fits the non-negative logistic weights. It reads no corpus file itself.
"""

from __future__ import annotations

import math
import pickle
import random
from pathlib import Path

from concept_pipeline import stage5_importance as s5
from concept_pipeline import stage6_selection as s6

# Which label set the metrics read: "label" (2 of 3 judges) or "label_relaxed" (any judge).
LABEL = "label"
# nDCG cut-off: the top of the ranking a student sees first (design stage 5 calibration).
NDCG_AT = 50
ANCHOR_MODES = ("design", "def_named", "def", "none")


def load(folder: Path) -> list[dict]:
    """Return every course dump in `folder`, sorted by file name."""
    courses = []
    for path in sorted(Path(folder).glob("*.pkl")):
        with open(path, "rb") as handle:
            courses.append(pickle.load(handle))
    return courses


# ---------- background (T) ----------
def background_ratios(course: dict, background_courses: list[dict]) -> dict[str, float]:
    """Return log2 doc/background frequency per word, Laplace smoothed; background = others.

    A course is never part of its own background, so T never sees the course it scores.
    """
    document = course["words"]
    background: dict[str, int] = {}
    for other in background_courses:
        for word, count in other["words"].items():
            background[word] = background.get(word, 0) + count
    vocabulary = len(set(document) | set(background))
    document_total, background_total = sum(document.values()), sum(background.values())
    return {
        word: math.log2(
            ((document[word] + 1) / (document_total + vocabulary))
            / ((background.get(word, 0) + 1) / (background_total + vocabulary))
        )
        for word in document
    }


def set_background(course: dict, ratios: dict[str, float] | None) -> None:
    """Fill T's word ratios in the course evidence; None turns T off (rho_T = 0)."""
    evidence = course["evidence"]
    evidence.has_background = ratios is not None
    for concept in evidence.concepts:
        words = getattr(concept, "t_words", None) or s5.specificity_words(concept.key)
        concept.word_ratios = [ratios.get(word, 0.0) for word in words] if ratios else []


# ---------- metrics ----------
def ndcg_at(scores: dict[str, float], labels: dict[str, int], k: int = NDCG_AT) -> float:
    """Return nDCG@k with linear gains (must 2, valid 1, noise 0), ties broken by key."""
    ranked = sorted(scores, key=lambda key: (-scores[key], key))[:k]
    dcg = sum(labels[key] / math.log2(i + 2) for i, key in enumerate(ranked))
    ideal = sorted(labels.values(), reverse=True)[:k]
    best = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    return dcg / best if best else 0.0


def bootstrap_ci(values: list[float], rounds: int = 2000, seed: int = 7) -> tuple[float, ...]:
    """Return (mean, low, high) of the macro mean with a 95% interval, resampling courses."""
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(count)) / count for _ in range(rounds))
    return sum(values) / count, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


# ---------- stage 5 ----------
def features(course: dict, p0: float) -> list[tuple[list[float], int]]:
    """Return per concept the regression features rho_i * z_i (file level) and its label."""
    evidence = course["evidence"]
    rho = s5.file_rho(evidence, p0)
    rows = []
    for concept in evidence.concepts:
        values = s5.signal_values(concept)
        rows.append(([rho[s] * values[s] for s in s5.SIGNALS], course[LABEL][concept.key]))
    return rows


def fit_nonnegative_logistic(
    rows: list[tuple[list[float], int]],
    target,
    iterations: int = 300,
    rate: float = 0.5,
    l2: float = 1e-3,
) -> dict[str, float]:
    """Fit y ~ sigmoid(b + sum w_i x_i) with w_i >= 0 by projected Adam; return weights summing to 1.

    Args:
        rows: Feature rows and labels from `features`.
        target: Maps a label to True for the positive class (relevant, or must only).
        iterations: Full-batch Adam steps.
        rate: Adam step size.
        l2: L2 penalty on the weights (not the bias).
    """
    k = len(s5.SIGNALS)
    weights = [0.1] * k
    bias = 0.0
    first = [0.0] * (k + 1)
    second = [0.0] * (k + 1)
    # Identical rows are merged with a count (most concepts share a few discrete values).
    grouped: dict[tuple, int] = {}
    for x, label in rows:
        key = (tuple(round(v, 9) for v in x), 1.0 if target(label) else 0.0)
        grouped[key] = grouped.get(key, 0) + 1
    data = [(list(x), y, count) for (x, y), count in grouped.items()]
    total_rows = len(rows)
    for step_index in range(1, iterations + 1):
        grad = [0.0] * (k + 1)
        for x, y, count in data:
            z = bias + sum(w * xi for w, xi in zip(weights, x))
            p = 1 / (1 + math.exp(-z)) if z > -30 else 0.0
            delta = (p - y) * count
            grad[k] += delta
            for i in range(k):
                grad[i] += delta * x[i]
        for i in range(k + 1):
            g = grad[i] / total_rows + (l2 * weights[i] if i < k else 0.0)
            first[i] = 0.9 * first[i] + 0.1 * g
            second[i] = 0.999 * second[i] + 0.001 * g * g
            corrected = first[i] / (1 - 0.9**step_index)
            step = rate * corrected / (math.sqrt(second[i] / (1 - 0.999**step_index)) + 1e-8)
            if i < k:
                weights[i] = max(0.0, weights[i] - step)
            else:
                bias -= step
    total = sum(weights)
    return {s: (w / total if total else 1 / k) for s, w in zip(s5.SIGNALS, weights)}


def set_stage5_globals(eta: float, m0: int, t_max: float) -> None:
    """Set the stage-5 module constants a configuration varies (eta, m_0, t_max)."""
    s5.LOCAL_DISCOUNT = eta
    s5.SHORT_SECTION_UNITS = m0
    s5.T_MAX_LOG2_RATIO = t_max


def importance(course: dict, weights: dict[str, float], p0: float, combine: str) -> dict:
    """Return I_f(c) of every concept in a course dump."""
    return s5.file_importance(course["evidence"], weights, p0, combine)


# ---------- stage 6 ----------
def anchor_kinds(concept) -> set[str]:
    """Return which design anchoring reasons a concept has: def, named, heading."""
    kinds: set[str] = set()
    if all(word in s5.STRUCTURAL_WORDS for word in concept.key.split()):
        return kinds
    for mention in concept.mentions:
        if mention.strength >= s5.DEFINED:
            kinds.add("def")
        if s5._names_unit(mention):
            kinds.add("named")
        if mention.strength >= s5.SUBJECT and "heading" in mention.signals:
            kinds.add("heading")
    return kinds


def selection_metrics(course: dict, imp: dict, anchor: str = "design", **params) -> dict:
    """Return recall of must concepts (of those the pool covers), precisions and node count.

    Args:
        course: A course dump.
        imp: Importance per concept key.
        anchor: "design" keeps stage 6's anchoring; the others restrict it to some reasons.
        **params: Passed to `stage6_selection.select_nodes` (lambda, tau, epsilon, variant).
    """
    evidence = course["evidence"]
    concepts = s6.space_concepts([evidence], imp)
    if anchor != "design":
        allowed = {"def_named": {"def", "named"}, "def": {"def"}, "none": set()}[anchor]
        kinds = {c.key: anchor_kinds(c) for c in evidence.concepts}
        for concept in concepts:
            concept.is_anchored = bool(kinds[concept.key] & allowed)
    selection = s6.select_nodes(concepts, **params)
    nodes = selection.nodes
    labels, covers, layer = course[LABEL], course["covers"], course["layer"]
    must = {g for g, layer_name in layer.items() if layer_name == "must"}
    in_pool = {g for key in labels for g in covers[key]} & must
    found = {g for key in nodes for g in covers[key]} & must
    relaxed = course["label_relaxed"]

    def share(predicate) -> float:
        return sum(1 for key in nodes if predicate(key)) / len(nodes) if nodes else 0.0

    return {
        "recall": len(found) / len(in_pool) if in_pool else 1.0,
        "recall_all_must": len(found) / len(must) if must else 1.0,
        "precision": share(lambda key: labels[key] >= 1),
        "precision_relaxed": share(lambda key: relaxed[key] >= 1),
        "strict": share(lambda key: labels[key] == 2),
        "nodes": len(nodes),
        "anchored": selection.report["anchored"],
        "hidden": selection.report["hidden"],
    }
