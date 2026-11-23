"""Tests for pipeline stages 5 (importance) and 6 (selection): the plan's gate items, no network."""

import json
import random
from dataclasses import asdict

import pytest

from test_stage2_candidates import synthetic_result

from concept_pipeline import stage5_importance as s5
from concept_pipeline import stage6_selection as s6
from concept_pipeline.stage2_candidates import DEFINED, MENTION, SUBJECT, extract_candidates
from concept_pipeline.stage4_merging import Concept


def mention(unit=0, section=0, strength=MENTION, *signals):
    return s5.MentionEvidence(unit, section, strength, frozenset(signals))


def file_with(concepts, file_id="f1", units=4, share=None, formulas=0, shingle_seed="a"):
    return s5.FileEvidence(
        file_id=file_id,
        unit_omega=[1.0] * units,
        unit_section=[0] * (units // 2) + [1] * (units - units // 2),
        unit_shingles=[frozenset({f"{shingle_seed}{i} x y z w"}) for i in range(units)],
        signal_share={"bold": 0.03, "underline": 0.01, "size": 0.02, "line": 0.04}
        if share is None
        else share,
        anchored_formulas=formulas,
        has_background=False,
        concepts=concepts,
    )


def test_the_determinant_example_gives_the_design_values():
    determinant = s5.ConceptEvidence(
        "דטרמיננטה",
        [
            mention(0, 0, DEFINED),
            mention(1, 0, SUBJECT, "bold", "size", "line"),
            mention(2, 0, MENTION, "bold"),
            mention(2, 0, MENTION, "bold"),
        ],
    )
    values = s5.signal_values(determinant)
    assert values["Q"] == 0.5 and values["V_b"] == 0.875
    assert (values["V_s"], values["L"], values["V_u"]) == (0.5, 0.5, 0.0)


def test_effective_weights_sum_to_one_and_values_stay_in_range():
    evidence = file_with([s5.ConceptEvidence("a", [mention(0, 0, DEFINED, "bold")])], formulas=3)
    weights = s5.effective_weights(s5.OPENING_WEIGHTS, s5.file_rho(evidence))
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in s5.signal_values(evidence.concepts[0]).values())
    assert all(0.0 <= r <= 1.0 for r in s5.file_rho(evidence).values())


def test_an_all_bold_document_gives_bold_no_weight_and_passes_it_on():
    evidence = file_with([], share={"bold": 1.0, "underline": 0.01, "size": 0.02, "line": 0.04})
    weights = s5.effective_weights(s5.OPENING_WEIGHTS, s5.file_rho(evidence))
    assert weights["V_b"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_a_signal_marking_half_the_words_is_worth_a_quarter():
    assert s5.information_value(0.5) == pytest.approx(0.25)
    assert s5.information_value(1 / 32) == 1.0


def two_files():
    first = file_with(
        [
            s5.ConceptEvidence("a", [mention(0, 0, DEFINED), mention(1, 0)]),
            s5.ConceptEvidence("b", [mention(2, 1)]),
            s5.ConceptEvidence("c", [mention(3, 1, SUBJECT, "line")]),
        ],
        "f1",
    )
    second = file_with(
        [s5.ConceptEvidence("a", [mention(0, 0)]), s5.ConceptEvidence("d", [mention(1, 0)])],
        "f2",
        shingle_seed="b",
    )
    return first, second


def test_importance_is_in_range_and_identical_on_a_second_run():
    files = list(two_files())
    first = json.dumps(s5.space_importance(files), sort_keys=True)
    assert first == json.dumps(s5.space_importance(files), sort_keys=True)
    assert all(0.0 <= value <= 1.0 for value in json.loads(first).values())


def test_importance_does_not_depend_on_the_order_of_files():
    first, second = two_files()
    assert s5.space_importance([first, second]) == s5.space_importance([second, first])


def test_a_near_duplicate_file_does_not_raise_importance():
    first, second = two_files()
    copy = file_with(first.concepts, "f0-copy")
    before = s5.space_importance([first, second])
    after = s5.space_importance([first, second, copy])
    assert after == pytest.approx(before)
    assert s5.near_duplicate_groups([first, copy]) == [["f0-copy", "f1"]]


def test_a_concept_across_files_gets_the_breadth_bonus():
    importance = s5.space_importance(list(two_files()))
    assert importance["a"] > importance["b"]


def test_soft_or_gives_double_strength_a_small_bonus_over_max():
    first, _ = two_files()
    soft = s5.file_importance(first, combine="soft_or")
    hard = s5.file_importance(first, combine="max")
    assert all(soft[key] >= hard[key] for key in hard)
    with pytest.raises(ValueError):
        s5.file_importance(first, combine="sum")


def concept(key, units, importance=0.5, anchored=False, protected=False, strength=0.4):
    coverage = {("f", u): strength for u in units}
    return s6.SpaceConcept(
        key, importance, coverage, {u: 1.0 for u in coverage}, anchored, protected
    )


def test_the_selection_never_passes_the_cap():
    concepts = [concept(f"c{i}", [i], importance=0.9) for i in range(300)]
    selection = s6.select_nodes(concepts)
    assert len(selection.nodes) <= s6.MAX_NODES
    assert set(selection.nodes) | set(selection.pool) == {c.key for c in concepts}


def test_marginal_information_never_grows_when_nodes_are_added():
    rng = random.Random(5)
    concepts = [concept(f"c{i}", rng.sample(range(30), 4), rng.random()) for i in range(40)]
    for _ in range(200):
        small = rng.sample(concepts, 5)
        large = small + rng.sample(concepts, 10)
        target = rng.choice(concepts)
        small_gain = s6.marginal_information(target, s6._uncovered(small))
        large_gain = s6.marginal_information(target, s6._uncovered(large))
        assert large_gain <= small_gain + 1e-12


def test_anchored_concepts_enter_first_and_weak_ones_stay_in_the_pool():
    concepts = [
        concept("anchored", [0], importance=0.0, anchored=True),
        concept("strong", [1, 2, 3], importance=0.9),
        concept("weak", [], importance=0.1),
    ]
    selection = s6.select_nodes(concepts)
    assert selection.nodes[0] == "anchored" and "strong" in selection.nodes
    assert "weak" in selection.pool


def test_a_full_space_hides_the_weakest_node_for_a_much_stronger_one():
    nodes = [concept(f"n{i}", [i], importance=0.5) for i in range(3)]
    newcomer = concept("new", [10, 11, 12, 13], importance=0.9, strength=1.0)
    # Anchored nodes fill the space first, so the newcomer arrives when it is full.
    for node in nodes:
        node.is_anchored = True
    selection = s6.select_nodes(nodes + [newcomer], max_nodes=3)
    assert "new" in selection.nodes and len(selection.hidden) == 1
    assert selection.hidden[0] not in selection.nodes and selection.hidden[0] not in selection.pool


def test_leave_one_out_products_equal_a_full_recomputation():
    rng = random.Random(3)
    nodes = [
        concept(f"n{i}", rng.sample(range(12), 4), strength=rng.choice([0.2, 0.4, 0.7, 1.0]))
        for i in range(8)
    ]
    by_key = {c.key: c for c in nodes}
    keys = [c.key for c in nodes]
    covering = s6._covering(keys, by_key)
    for left_out in keys:
        others = s6._uncovered([by_key[k] for k in keys if k != left_out])
        for target in nodes:
            fast = s6._uncovered_without(target, covering, left_out)
            assert fast == {u: others[u] for u in target.coverage if u in others}


def test_a_protected_node_is_never_replaced():
    nodes = [concept(f"n{i}", [i], importance=0.1, anchored=True, protected=True) for i in range(3)]
    newcomer = concept("new", [10, 11, 12, 13], importance=0.9, strength=1.0)
    selection = s6.select_nodes(nodes + [newcomer], max_nodes=3)
    assert selection.hidden == [] and "new" in selection.pool


def test_an_unknown_variant_is_refused_and_variant_b_scales_coverage():
    with pytest.raises(ValueError):
        s6.select_nodes([], variant="C")
    big = concept("big", list(range(20)), importance=0.0, strength=1.0)
    # The design's lambda, because the calibrated lambda of 1 gives coverage no weight at all.
    gain = s6.marginal_information(big, {}, importance_weight=0.3, variant="B")
    assert 0.0 < gain <= 1.0 - 0.3
    assert s6.marginal_information(big, {}, importance_weight=0.3, variant="A") > 1.0 - 0.3


def test_calibrated_weights_are_non_negative_and_sum_to_one():
    assert set(s5.CALIBRATED_WEIGHTS) == set(s5.SIGNALS)
    assert all(weight >= 0 for weight in s5.CALIBRATED_WEIGHTS.values())
    assert abs(sum(s5.CALIBRATED_WEIGHTS.values()) - 1.0) < 1e-9


def test_background_ratios_favour_the_documents_own_words_and_stay_finite():
    ratios = s5.background_ratios({"מטריצה": 30, "של": 50}, {"של": 5000, "היה": 900})
    assert ratios["מטריצה"] > 0 > ratios["של"]
    assert set(ratios) == {"מטריצה", "של"}


def test_specificity_reads_the_most_mentioned_member_of_a_merged_concept():
    result = synthetic_result()
    candidate = next(c for c in extract_candidates(result).candidates if c.key == "מטריצה הפיכה")
    renamed = Concept("מטריצה הפיכה", [candidate.key], [], [], candidate.mentions, core="הפיכה")
    (concept,) = s5.file_evidence(result, [renamed], {"הפיכה": 3.0, "מטריצה": -2.0}).concepts
    assert concept.t_words == ["הפיכה"]
    assert concept.word_ratios == [3.0]


def test_on_a_real_structure_every_node_has_a_dictionary_entry():
    result = synthetic_result()
    candidates = extract_candidates(result).candidates
    evidence = s5.file_evidence(result, candidates)
    importance = s5.space_importance([evidence])
    concepts = s6.space_concepts([evidence], importance)
    selection = s6.select_nodes(concepts, threshold=0.0)
    dictionary = s6.mention_dictionary(selection.nodes, [(evidence.file_id, result, candidates)])
    assert selection.nodes and all(dictionary[key] for key in selection.nodes)
    defined = next(c for c in concepts if c.key == "מטריצה הפיכה")
    assert defined.is_anchored
    assert dictionary["מטריצה הפיכה"][0].kind == "definition"
    assert json.dumps([asdict(e) for e in dictionary["מטריצה הפיכה"]], ensure_ascii=False)


def test_the_result_word_of_a_named_result_does_not_count_in_specificity():
    assert s5.specificity_words("משפט wald") == ["wald"]
    assert s5.specificity_words("bayes theorem") == ["bayes"]
    assert s5.specificity_words("מטריצה הפיכה") == ["מטריצה", "הפיכה"]
    assert s5.specificity_words("משפט") == ["משפט"]
