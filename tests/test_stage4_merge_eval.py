"""Tests for the stage-4 measures (committee votes, duplicates, over-merging, stability) and the
model synonym finder with a fake model."""

import json

from test_stage2_candidates import synthetic_result

from concept_pipeline import stage4_merge_eval as ev
from concept_pipeline.llm_cache import LlmAnswer, LlmCache, Mode
from concept_pipeline.stage2_candidates import extract_candidates
from concept_pipeline.stage4_merging import find_pairs
from concept_pipeline.stage4_synonym_finder import (
    MAX_GROUP_ENTRIES,
    ordered_entries,
    parse_groups,
    synonym_pairs,
)


def test_two_agreeing_judges_decide_a_label_and_a_split_vote_leaves_none():
    assert ev.vote([True, True, False]) is True
    assert ev.vote([False, None, False]) is False
    assert ev.vote([True, None, False]) is None


def test_a_group_holding_a_pair_labelled_different_is_over_merged():
    groups = [["a", "b", "c"], ["d", "e"], ["f"]]
    labels = {("a", "c"): False, ("d", "e"): True, ("e", "f"): False}
    assert ev.over_merge(groups, labels) == (1, 2)


def test_a_true_concept_spread_over_two_groups_is_one_duplicate():
    groups = [["a"], ["b"], ["c", "d"]]
    # One gold concept named by a, b and c; the committee says a = b, but c is different.
    labels = {("a", "b"): True, ("a", "c"): False, ("b", "c"): False}
    assert ev.duplicates(groups, [["a", "b", "c"]], labels) == (1, 3)


def test_a_gold_concept_that_joined_related_names_is_split_back_and_not_counted():
    groups = [["transcriptase"], ["transcription"]]
    labels = {("transcriptase", "transcription"): False}
    assert ev.duplicates(groups, [["transcriptase", "transcription"]], labels) == (0, 2)


def test_merging_two_gold_concepts_is_over_merging_but_merging_plain_words_is_not():
    groups = [["a", "b"], ["c", "d"], ["x", "y"]]
    concept_keys = [["a"], ["b"], ["c", "d"]]
    labels = {("c", "d"): True, ("x", "y"): False}
    assert ev.gold_over_merge(groups, concept_keys, labels) == (1, 2)
    assert ev.over_merge(groups, labels) == (1, 3)


def test_a_gold_concept_split_by_a_different_label_counts_as_two_concepts():
    groups = [["transcriptase", "transcription"]]
    labels = {("transcriptase", "transcription"): False}
    concept_keys = [["transcriptase", "transcription"]]
    assert ev.gold_over_merge(groups, concept_keys, labels) == (1, 1)


def test_stability_counts_pairs_with_one_verdict_in_every_run():
    pairs = [("a", "b"), ("c", "d")]
    runs = [{("a", "b"): True, ("c", "d"): True}, {("a", "b"): True}, {("a", "b"): True}]
    assert ev.stability(runs, pairs) == 0.5


def test_decider_scores_count_an_undecided_pair_as_different():
    labels = {("a", "b"): True, ("c", "d"): True, ("e", "f"): False, ("g", "h"): None}
    decided = {("a", "b"): True, ("e", "f"): True}
    scores = ev.pair_scores(decided, labels)
    assert (scores["labelled"], scores["precision"], scores["recall"]) == (3, 0.5, 0.5)


class GroupSender:
    """Group the entries of "מטריצה ריבועית" and "square matrix", plus junk the parser drops."""

    def __call__(self, call):
        lines = call.prompt.split("Terms:\n", 1)[1].splitlines()
        number = {line.split(" ", 1)[1]: int(line.split()[0]) for line in lines}
        groups = [[number["מטריצה ריבועית"], number["square matrix"]], [1], "x", [0, 99999]]
        return LlmAnswer(json.dumps({"groups": groups}), 50, 10)


def test_a_synonym_group_becomes_a_pair_for_the_decider(tmp_path):
    stage2 = extract_candidates(synthetic_result())
    pairs, calls = synonym_pairs(stage2, LlmCache(tmp_path, Mode.RECORD, GroupSender()), "m")
    assert calls == 1 and pairs == {("square matrix", "מטריצה ריבועית")}
    found, report = find_pairs(stage2.candidates, synonyms=pairs)
    assert report["by_finder"]["synonym"] == 1
    assert any("synonym" in p.finders for p in found)


def test_entries_are_numbered_in_order_of_first_mention():
    stage2 = extract_candidates(synthetic_result())
    starts = [min(m.start for m in c.mentions) for c in ordered_entries(stage2)]
    assert starts == sorted(starts)


def test_groups_keep_valid_distinct_numbers_and_are_cut_to_the_maximum():
    text = json.dumps({"groups": [[1, 1, 2], [3, True, "4"], list(range(1, 9)), [5, 50]]})
    groups = parse_groups(text, 10)
    assert groups == [[1, 2], list(range(1, MAX_GROUP_ENTRIES + 1))]
    assert parse_groups("not json", 10) == []


def test_without_labels_a_gold_concept_counts_as_one_concept():
    groups = [["supervised", "supervised learning"], ["svm"], ["support vector machine"]]
    concept_keys = [["supervised", "supervised learning"], ["svm", "support vector machine"]]
    labels = {("supervised", "supervised learning"): False}
    assert ev.gold_over_merge(groups, concept_keys, None) == (0, 1)
    assert ev.gold_over_merge(groups, concept_keys, labels) == (1, 1)
    assert ev.duplicates(groups, concept_keys, None) == (1, 3)


def test_a_gold_concept_is_split_where_labelled_different_but_a_fragment_stays():
    concepts = [["precision tradeoff", "recall"], ["supervised", "supervised learning"]]
    labels = {
        ("precision tradeoff", "recall"): False,
        ("supervised", "supervised learning"): False,
    }
    assert ev.corrected_concepts(concepts, labels) == [
        ["precision tradeoff"],
        ["recall"],
        ["supervised", "supervised learning"],
    ]


def test_two_gold_concepts_labelled_same_become_one_and_merging_them_is_not_over_merging():
    concepts = [["recall"], ["החזרה"], ["validation"], ["סט הכשרה"]]
    labels = {("recall", "החזרה"): True, ("validation", "סט הכשרה"): False}
    corrected = ev.corrected_concepts(concepts, labels)
    groups = [["recall", "החזרה"], ["validation", "סט הכשרה"]]
    assert ev.gold_over_merge(groups, corrected, None) == (1, 2)
    assert ev.duplicates([["recall"], ["החזרה"]], corrected, None) == (1, 2)


def test_a_same_label_never_joins_concepts_holding_a_split_pair():
    concepts = [["a b"], ["c"], ["a b", "d"]]
    labels = {("a b", "c"): True, ("c", "d"): True, ("a b", "d"): False}
    corrected = ev.corrected_concepts(concepts, labels)
    assert not any({"a b", "d"} <= set(keys) for keys in corrected)


def test_a_rate_that_moves_within_its_interval_is_stable_and_one_that_jumps_is_not():
    assert ev.within_interval([(5, 422), (5, 425), (4, 416)])
    assert not ev.within_interval([(5, 422), (30, 420), (5, 418)])
