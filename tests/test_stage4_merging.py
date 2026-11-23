"""Tests for stage 4 (merge pairs, evidence, grouping) and its model decider, with a fake model."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from test_stage2_candidates import synthetic_result

from concept_pipeline.llm_cache import LlmAnswer, LlmCache, Mode
from concept_pipeline.stage4_merge_decider import (
    decide_by_majority,
    decide_pairs,
    decide_with_escalation,
    parse_decisions,
)
from concept_pipeline.stage4_translation_finder import parse_choices, translation_pairs
from concept_pipeline.stage2_candidates import SUBJECT, Candidate, Mention, extract_candidates
from concept_pipeline.stage4_merging import (
    EVIDENCE_STRENGTH,
    MAX_KEYS_PER_BLOCK,
    MODEL_STRENGTH,
    USER_STRENGTH,
    Decision,
    MergePair,
    check_groups,
    contexts,
    decide_by_rule,
    find_pairs,
    group_keys,
    merge,
)


def candidates(*keys, aliases=None):
    """Return bare candidates with one mention each; `aliases` maps a key to its aliases."""
    aliases = aliases or {}
    return [
        Candidate(key, ["ngram"], aliases.get(key, []), [Mention(i, i + 1, 0, 0, 0.4, key)])
        for i, key in enumerate(keys)
    ]


def pairs_of(*keys, aliases=None):
    found, _ = find_pairs(candidates(*keys, aliases=aliases))
    return {(p.first, p.second): p.finders for p in found}


def same(a, b, strength=MODEL_STRENGTH):
    return Decision(a, b, True, strength, "test")


def different(a, b, strength=MODEL_STRENGTH):
    return Decision(a, b, False, strength, "test")


def test_full_and_defective_spelling_are_paired():
    assert pairs_of("חיסרון", "חסרון") == {("חיסרון", "חסרון"): ("spelling",)}


def test_an_acronym_is_paired_with_the_phrase_its_initials_spell():
    found = pairs_of("df", "document frequency", 'ט"ל', "טרנספורמציה ליניארית")
    assert ("df", "document frequency") in found
    assert ('ט"ל', "טרנספורמציה ליניארית") in found


def test_one_changed_letter_pairs_long_words_only():
    assert ("הומאוסטאזיס", "הומאוסטזיס") in pairs_of("הומאוסטאזיס", "הומאוסטזיס")
    assert pairs_of("use", "used") == {}


def test_a_confused_glyph_pair_is_found():
    assert pairs_of("מפורט", "מפורש") == {("מפורט", "מפורש"): ("glyph",)}


def test_a_gloss_alias_is_a_pair_with_direct_evidence():
    found, _ = find_pairs(candidates("יציבות", "stability", aliases={"יציבות": ["stability"]}))
    assert [(p.first, p.second, p.has_direct_evidence) for p in found] == [
        ("stability", "יציבות", True)
    ]


def test_a_large_block_of_one_skeleton_is_skipped_and_reported():
    keys = ["אבג" + "י" * i for i in range(MAX_KEYS_PER_BLOCK + 1)]
    found, report = find_pairs(candidates(*keys))
    assert report["skipped_blocks"] == 1
    assert not any("spelling" in p.finders for p in found)


def test_a_chain_breaks_at_its_weakest_link():
    decisions = [
        same("a", "b", USER_STRENGTH),
        same("b", "c", MODEL_STRENGTH),
        different("a", "c", EVIDENCE_STRENGTH),
    ]
    assert group_keys(["a", "b", "c"], decisions) == [["a", "b"], ["c"]]


def test_a_stronger_same_overrides_a_weaker_different_on_the_same_pair():
    decisions = [different("a", "b", MODEL_STRENGTH), same("b", "a", USER_STRENGTH)]
    assert group_keys(["a", "b"], decisions) == [["a", "b"]]


def test_on_equal_strength_different_wins():
    assert group_keys(["a", "b"], [same("a", "b"), different("a", "b")]) == [["a"], ["b"]]


def test_the_gate_passes_on_shuffled_inputs():
    keys = list("abcdefg")
    decisions = [
        same("a", "b"),
        same("c", "d"),
        same("b", "c"),
        different("a", "d"),
        same("e", "f"),
    ]
    assert check_groups(keys, decisions) == []


def test_a_merged_concept_adds_up_mentions_and_takes_the_most_mentioned_name():
    stage2 = extract_candidates(synthetic_result())
    concepts = merge(stage2, [same("מטריצה ריבועית", "square matrix", EVIDENCE_STRENGTH)])
    square = next(c for c in concepts if "square matrix" in c.members)
    counts = {c.key: c.count for c in stage2.candidates}
    assert square.key == "מטריצה ריבועית"
    assert square.count == counts["מטריצה ריבועית"] + counts["square matrix"]
    assert "square matrix" in square.aliases


def test_contexts_hold_the_heading_and_the_passage_of_a_mention():
    result = synthetic_result()
    stage2 = extract_candidates(result)
    square = next(c for c in stage2.candidates if c.key == "square matrix")
    windows = contexts(square, result)
    assert windows and windows[0].startswith("[מרחבים וקטוריים]") and "square matrix" in windows[0]


class MergeSender:
    """Answer "same" for every pair except the last of each batch, which it leaves out."""

    def __init__(self):
        self.calls = 0

    def __call__(self, call):
        self.calls += 1
        ids = [
            line.split()[1].rstrip(":")
            for line in call.prompt.splitlines()
            if line.startswith("Pair ")
        ]
        answer = {pair_id: "same" for pair_id in ids[:-1]}
        return LlmAnswer(json.dumps({"decisions": answer}), 100, 10)


def test_the_decider_keeps_unanswered_pairs_undecided_and_marks_evidence(tmp_path):
    result = synthetic_result()
    stage2 = extract_candidates(result)
    found, _ = find_pairs(stage2.candidates)
    gloss = [p for p in found if p.has_direct_evidence]
    pairs = gloss + [p for p in found if not p.has_direct_evidence][:1]
    sender = MergeSender()
    cache = LlmCache(tmp_path, Mode.RECORD, sender)
    decisions, calls = decide_pairs(pairs, stage2, result, cache, "model-x")
    assert calls == 1 == sender.calls
    assert len(decisions) == len(pairs) - 1
    assert all(d.strength == EVIDENCE_STRENGTH for d in decisions[: len(gloss)])


class VerdictSender:
    """Answer every pair with one fixed verdict, except the pair ids listed as flipped."""

    def __init__(self, verdict, flipped=()):
        self.verdict, self.flipped, self.calls = verdict, set(flipped), 0

    def __call__(self, call):
        self.calls += 1
        ids = [
            line.split()[1].rstrip(":")
            for line in call.prompt.splitlines()
            if line.startswith("Pair ")
        ]
        answer = {
            i: ("same" if (self.verdict != (i in self.flipped)) else "different") for i in ids
        }
        return LlmAnswer(json.dumps({"decisions": answer}), 100, 10)


def test_only_pairs_the_fast_runs_disagree_on_go_to_the_strong_model(tmp_path):
    result = synthetic_result()
    stage2 = extract_candidates(result)
    keys = sorted(c.key for c in stage2.candidates)[:3]
    pairs = [MergePair(keys[0], keys[1], ("synonym",)), MergePair(keys[1], keys[2], ("synonym",))]
    senders = [VerdictSender(True), VerdictSender(True), VerdictSender(True, flipped={"1"})]
    fast = [LlmCache(tmp_path / f"f{n}", Mode.RECORD, s) for n, s in enumerate(senders)]
    strong_sender = VerdictSender(False)
    strong = LlmCache(tmp_path / "strong", Mode.RECORD, strong_sender)
    decisions, counts = decide_with_escalation(
        pairs, stage2, result, fast, strong, "fast", "strong"
    )
    by_pair = {(d.first, d.second): d for d in decisions}
    first = (pairs[0].first, pairs[0].second)
    assert counts["disputed"] == 1 and strong_sender.calls == counts["strong_calls"] == 1
    assert by_pair[first].is_same is False and by_pair[first].by == "model:strong"
    assert all(d.is_same for pair, d in by_pair.items() if pair != first)


def test_the_majority_of_three_runs_decides_and_a_disagreement_is_counted(tmp_path):
    result = synthetic_result()
    stage2 = extract_candidates(result)
    keys = sorted(c.key for c in stage2.candidates)[:3]
    pairs = [MergePair(keys[0], keys[1], ("synonym",)), MergePair(keys[1], keys[2], ("synonym",))]
    senders = [VerdictSender(True), VerdictSender(False, flipped={"2"}), VerdictSender(False)]
    caches = [LlmCache(tmp_path / f"m{n}", Mode.RECORD, s) for n, s in enumerate(senders)]
    decisions, counts = decide_by_majority(pairs, stage2, result, caches, "fast")
    by_pair = {(d.first, d.second): d.is_same for d in decisions}
    assert by_pair == {(keys[0], keys[1]): False, (keys[1], keys[2]): True}
    assert counts == {"calls": 3, "disputed": 2}


def test_a_tie_between_runs_leaves_the_pair_undecided(tmp_path):
    result = synthetic_result()
    stage2 = extract_candidates(result)
    keys = sorted(c.key for c in stage2.candidates)[:2]
    pairs = [MergePair(keys[0], keys[1], ("synonym",))]
    caches = [
        LlmCache(tmp_path / f"t{n}", Mode.RECORD, s)
        for n, s in enumerate([VerdictSender(True), VerdictSender(False)])
    ]
    decisions, _ = decide_by_majority(pairs, stage2, result, caches, "fast")
    assert decisions == []


def test_an_answer_that_is_not_json_decides_nothing():
    assert parse_decisions("I think they are the same", ["1"]) == {}
    assert parse_decisions('{"decisions": {"1": "SAME", "2": "maybe"}}', ["1", "2"]) == {"1": True}


class TranslationSender:
    """Choose the Hebrew entry for "square matrix" by its number, plus an out-of-range number."""

    def __init__(self):
        self.prompts = []

    def __call__(self, call):
        self.prompts.append(call.prompt)
        hebrew = call.prompt.split("Hebrew entries:\n", 1)[1].splitlines()
        number = next(
            int(line.split()[0]) for line in hebrew if line.split(" ", 1)[1] == "מטריצה ריבועית"
        )
        answer = {"terms": {"square matrix": [number, 10_000], "not a term": [1]}}
        return LlmAnswer(json.dumps(answer), 50, 10)


def test_a_chosen_hebrew_entry_becomes_a_translation_pair(tmp_path):
    stage2 = extract_candidates(synthetic_result())
    sender = TranslationSender()
    cache = LlmCache(tmp_path, Mode.RECORD, sender)
    pairs, calls = translation_pairs(stage2, cache, "model-x")
    assert calls == 1 and ("square matrix", "מטריצה ריבועית") in pairs
    assert all(first.isascii() and not second.isascii() for first, second in pairs)
    found, report = find_pairs(stage2.candidates, pairs)
    translated = [p for p in found if "translation" in p.finders]
    assert report["by_finder"]["translation"] == len(translated) >= 1


def test_choices_outside_the_list_or_not_numbers_are_dropped():
    text = '{"terms": {"a": [2, 0, 9, "3", true, 2], "b": "x", "c": [1, 2, 3, 4, 5]}}'
    assert parse_choices(text, ["a", "b", "c"], 5) == {"a": [2], "c": [1, 2, 3]}
    assert parse_choices("no json here", ["a"], 5) == {}


def test_one_word_keys_of_one_length_still_meet_their_typo_variant():
    fillers = [f"מילהארוכה{letter}" for letter in "אבגדהוזחטיכלמנ"]
    found = pairs_of("אנטרופיה", "אנתרופיה", "אנטרופיות", *fillers)
    assert found.get(("אנטרופיה", "אנתרופיה")) == ("typo",)


def test_name_pairs_are_sorted_into_synonym_dash_and_contained():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import stage4_merging_report as report

    whole = {(0, "whole")}
    hits = ["rna", "rna פולימראז", "פולימראז", "support vector", "וקטור תומך", "min", "single link"]
    sources = {key: whole for key in hits}
    sources["וקטור תומך"] = {(1, "whole")}
    sources["min"] = sources["single link"] = {(2, "dash")}
    assert report._pair_kind("rna", "rna פולימראז", sources, hits) == report.CONTAINED
    assert report._pair_kind("rna", "פולימראז", sources, hits) == report.CONTAINED
    assert report._pair_kind("support vector", "וקטור תומך", sources, hits) == report.SYNONYM
    assert report._pair_kind("min", "single link", sources, hits) == report.DASH
    assert report._name_sources("k-NN (KNN)") == [
        ("k-NN (KNN)", "whole"),
        ("k-NN", "whole"),
        ("KNN", "paren"),
    ]


def test_two_swapped_letters_pair_short_english_words():
    assert pairs_of("bais", "bias").get(("bais", "bias")) == ("typo",)
    assert ("רשת", "רתש") not in pairs_of("רשת", "רתש")


def concept_at(key, spans, sources=("ngram",), strength=0.4):
    """Return a concept with one plain mention per (start, end) span."""
    from concept_pipeline.stage4_merging import Concept

    mentions = [Mention(s, e, 0, 0, strength, key) for s, e in spans]
    return Concept(key, [key], list(sources), [], mentions)


def test_a_fragment_inside_one_longer_concept_is_absorbed_without_an_alias():
    from concept_pipeline.stage4_merging import absorb_fragments

    parent = concept_at("חומצה אמינו", [(0, 11), (20, 31), (40, 51)])
    fragment = concept_at("אמינו", [(6, 11), (26, 31), (46, 51)])
    kept, absorbed = absorb_fragments([parent, fragment])
    assert absorbed == {"אמינו": "חומצה אמינו"}
    assert [c.key for c in kept] == ["חומצה אמינו"] and kept[0].aliases == []


def test_a_concept_used_alone_or_in_many_phrases_or_defined_stays():
    from concept_pipeline.stage4_merging import absorb_fragments

    long_one = concept_at("bias variance tradeoff", [(0, 22), (30, 52)])
    alone = concept_at("bias", [(0, 4), (30, 34), (60, 64), (70, 74)])
    phrases = [concept_at(f"גבול {i}", [(100 + 20 * i, 110 + 20 * i)]) for i in range(3)]
    spread = concept_at("גבול", [(100 + 20 * i, 104 + 20 * i) for i in range(3)])
    defined = concept_at("אמינו", [(206, 211)], sources=("definition",))
    holder = concept_at("חומצה אמינו", [(200, 211)])
    _, absorbed = absorb_fragments([long_one, alone, *phrases, spread, defined, holder])
    assert absorbed == {}


def test_a_concept_is_shown_by_a_form_the_text_writes():
    from concept_pipeline.stage4_merging import Concept

    mentions = [Mention(0, 1, 0, 0, 0.4, f) for f in ("השונות המשותפת", "שונות משותפת")]
    assert Concept("שונות משותף", ["שונות משותף"], [], [], mentions).written == "שונות משותפת"
    exact = [Mention(0, 1, 0, 0, 0.4, "מטריצה")]
    assert Concept("מטריצה", ["מטריצה"], [], [], exact).written == "מטריצה"


def test_a_sentence_piece_does_not_absorb_the_term_inside_it():
    from concept_pipeline.stage4_merging import absorb_fragments

    junk = concept_at("גרף אז הוא compatible", [(0, 21), (30, 51)])
    term = concept_at("compatible", [(11, 21), (41, 51)])
    _, absorbed = absorb_fragments([junk, term])
    assert absorbed == {}


def test_a_named_theorem_absorbs_the_name_inside_it():
    from concept_pipeline.stage4_merging import absorb_fragments

    theorem = concept_at("משפט היינה", [(0, 10), (20, 30), (40, 50)])
    name = concept_at("היינה", [(5, 10), (25, 30), (45, 50)])
    _, absorbed = absorb_fragments([theorem, name])
    assert absorbed == {"היינה": "משפט היינה"}


def mentioned(key, times, source="ngram"):
    """Return a candidate from one source, mentioned `times` times at distinct offsets."""
    return Candidate(key, [source], [], [Mention(i, i + 1, 0, 0, 0.4, key) for i in range(times)])


def named_pairs_of(*keys, sources=None):
    """Return the named pairs of candidates written as labels, unless `sources` says otherwise."""
    sources = sources or {}
    found, _ = find_pairs(
        [
            Candidate(key, sources.get(key, ["label"]), [], [Mention(i, i + 1, 0, 0, 0.4, key)])
            for i, key in enumerate(keys)
        ]
    )
    return {(p.first, p.second) for p in found if "named" in p.finders}


def test_named_result_pairs_with_its_bare_name():
    assert named_pairs_of("משפט ויירשטראס", "ויירשטראס") == {("ויירשטראס", "משפט ויירשטראס")}
    assert named_pairs_of("bayes theorem", "bayes") == {("bayes", "bayes theorem")}


def test_named_pair_is_skipped_when_two_results_share_the_name():
    assert named_pairs_of("משפט קושי", "למה קושי", "קושי") == set()


def test_named_tail_after_of_link_pairs():
    assert named_pairs_of("למה של היינה בורל", "היינה בורל") == {
        ("היינה בורל", "למה של היינה בורל")
    }


def test_a_longer_name_that_only_contains_the_bare_name_is_not_paired():
    assert named_pairs_of("משפט בולצאנו ויירשטראס", "ויירשטראס") == set()


def test_a_named_result_seen_only_in_prose_is_not_paired():
    assert named_pairs_of("משפט ערך", "ערך", sources={"משפט ערך": ["ngram"]}) == set()


def test_a_bare_name_with_its_own_definition_or_operator_is_not_paired():
    assert named_pairs_of("משפט מימד", "מימד", sources={"מימד": ["operator", "ngram"]}) == set()


def test_principle_at_the_end_of_an_english_name_is_not_a_result_word():
    assert named_pairs_of("principle component", "component") == set()


def test_merged_concept_is_named_by_the_member_that_contains_the_others():
    stage2 = SimpleNamespace(candidates=[mentioned("פרמה", 13), mentioned("משפט פרמה", 3)])
    (concept,) = merge(stage2, [same("משפט פרמה", "פרמה")])
    assert concept.key == "משפט פרמה"
    assert concept.count == 16


def test_named_pairs_are_decided_by_rule_without_the_model():
    found, _ = find_pairs(
        [
            mentioned("משפט היינה", 2, "label"),
            mentioned("היינה", 3),
            mentioned("חיסרון", 1),
            mentioned("חסרון", 1),
        ]
    )
    rule, model_pairs = decide_by_rule(found)
    assert [(d.first, d.second, d.is_same, d.strength, d.by) for d in rule] == [
        ("היינה", "משפט היינה", True, EVIDENCE_STRENGTH, "rule:named")
    ]
    assert [(p.first, p.second) for p in model_pairs] == [("חיסרון", "חסרון")]


def test_a_user_different_outweighs_a_named_rule():
    found, _ = find_pairs([mentioned("משפט היינה", 2, "label"), mentioned("היינה", 3)])
    rule, _ = decide_by_rule(found)
    user = different("היינה", "משפט היינה", USER_STRENGTH)
    assert group_keys(["משפט היינה", "היינה"], rule + [user]) == [["היינה"], ["משפט היינה"]]


def test_a_longer_member_that_is_not_a_named_result_does_not_rename_the_group():
    stage2 = SimpleNamespace(candidates=[mentioned("פולימראז", 9), mentioned("אנזים פולימראז", 2)])
    (concept,) = merge(stage2, [same("אנזים פולימראז", "פולימראז")])
    assert concept.key == "פולימראז"


def test_a_group_is_named_by_its_fullest_named_result():
    stage2 = SimpleNamespace(
        candidates=[
            mentioned("בולצאנו", 12),
            mentioned("בולצאנו ויירשטראס", 8),
            mentioned("משפט בולצאנו", 3),
            mentioned("משפט בולצאנו ויירשטראס", 2),
        ]
    )
    decisions = [
        same("בולצאנו", "בולצאנו ויירשטראס"),
        same("בולצאנו", "משפט בולצאנו"),
        same("בולצאנו ויירשטראס", "משפט בולצאנו ויירשטראס"),
    ]
    (concept,) = merge(stage2, decisions)
    assert concept.key == "משפט בולצאנו ויירשטראס"


def test_a_group_is_named_by_a_longer_member_the_author_marks_more_strongly():
    label = Candidate("אי רציפות קפיצה", ["label"], [], [Mention(0, 1, 0, 0, SUBJECT, "x")])
    stage2 = SimpleNamespace(candidates=[mentioned("קפיצה", 9), label])
    (concept,) = merge(stage2, [same("אי רציפות קפיצה", "קפיצה")])
    assert concept.key == "אי רציפות קפיצה"
    assert concept.core == "קפיצה"
