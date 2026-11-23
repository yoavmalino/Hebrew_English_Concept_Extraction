"""Measure stage-6 selection against concept labels: precision relative to its ceiling.

Pure, no I/O. Owner decision 2026-11-17: plain precision is not the gate, because the selection
fills a fixed number of nodes and a course whose pool holds few relevant concepts cannot reach a
fixed precision even with a perfect ranking. The gate compares precision with the best precision
any selection of the same size could reach (`precision_ceiling`).
"""

from __future__ import annotations

from dataclasses import dataclass

# Plan stage 6: precision relative to the ceiling (initial target, re-approved after the pilot;
# it keeps the design's 0.75, now as a share of what the pool allows).
RELATIVE_PRECISION_MIN = 0.75


@dataclass(frozen=True)
class PrecisionReport:
    """Precision of a selection, its ceiling and their ratio.

    Attributes:
        precision: Relevant selected nodes / selected nodes.
        ceiling: min(1, relevant concepts in the pool / selected nodes).
        relative: precision / ceiling, or None when the pool holds no relevant concept.
    """

    precision: float
    ceiling: float
    relative: float | None


def precision_ceiling(relevant_in_pool: int, selected: int) -> float:
    """Return the best precision a selection of `selected` nodes could reach from this pool.

    Args:
        relevant_in_pool: Pool candidates labelled relevant (must or valid).
        selected: Number of nodes the selection kept.

    Returns:
        min(1, relevant_in_pool / selected); 0.0 for an empty selection.
    """
    if selected <= 0:
        return 0.0
    return min(1.0, relevant_in_pool / selected)


def precision_report(
    selected: list[str], pool: list[str], is_relevant: dict[str, bool]
) -> PrecisionReport:
    """Return precision, ceiling and relative precision of one course's selection.

    Args:
        selected: Keys of the selected nodes.
        pool: Keys of every candidate the selection chose from (the selected ones included).
        is_relevant: Label per key; a key without a label counts as noise.

    Returns:
        The report; `relative` is None when no pool candidate is relevant, so such a course is
        reported apart instead of counting as 0 or 1.
    """
    relevant_selected = sum(1 for key in selected if is_relevant.get(key, False))
    relevant_pool = sum(1 for key in set(pool) if is_relevant.get(key, False))
    precision = relevant_selected / len(selected) if selected else 0.0
    ceiling = precision_ceiling(relevant_pool, len(selected))
    relative = precision / ceiling if ceiling > 0 else None
    return PrecisionReport(precision, ceiling, relative)


def macro_relative(reports: list[PrecisionReport]) -> float | None:
    """Return the mean relative precision over courses that have one, else None.

    A macro mean, as every other stage 5-6 measure: each course counts once whatever its size.
    """
    ratios = [report.relative for report in reports if report.relative is not None]
    return sum(ratios) / len(ratios) if ratios else None


def passes(reports: list[PrecisionReport]) -> bool:
    """Return whether the macro relative precision reaches `RELATIVE_PRECISION_MIN`."""
    mean = macro_relative(reports)
    return mean is not None and mean >= RELATIVE_PRECISION_MIN
