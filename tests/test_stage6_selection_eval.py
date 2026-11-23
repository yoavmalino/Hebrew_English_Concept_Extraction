"""Tests for the stage-6 precision gate relative to its ceiling (pure, no network)."""

import pytest

from concept_pipeline import stage6_selection_eval as se


def test_ceiling_is_relevant_over_selected_and_never_above_one():
    assert se.precision_ceiling(36, 150) == pytest.approx(0.24)
    assert se.precision_ceiling(400, 150) == 1.0
    assert se.precision_ceiling(5, 0) == 0.0


def test_a_perfect_ranking_in_a_poor_pool_gets_full_relative_precision():
    pool = [f"c{i}" for i in range(10)]
    relevant = {"c0": True, "c1": True}
    report = se.precision_report(pool[:4], pool, relevant)
    assert report.precision == pytest.approx(0.5)
    assert report.ceiling == pytest.approx(0.5)
    assert report.relative == pytest.approx(1.0)


def test_a_pool_without_relevant_concepts_is_reported_apart():
    report = se.precision_report(["a"], ["a", "b"], {})
    assert report.relative is None
    assert se.macro_relative([report]) is None and not se.passes([report])


def test_the_gate_uses_the_macro_mean_of_defined_ratios():
    full = se.PrecisionReport(0.5, 0.5, 1.0)
    half = se.PrecisionReport(0.2, 0.4, 0.5)
    empty = se.PrecisionReport(0.0, 0.0, None)
    assert se.macro_relative([full, half, empty]) == pytest.approx(0.75)
    assert se.passes([full, half, empty])
    assert not se.passes([half])
