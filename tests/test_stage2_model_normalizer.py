"""Tests for model normalization in stage 2: the guard, the lemma map and the two-pass flow."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from concept_pipeline import stage2_model_normalizer
from concept_pipeline.llm_cache import LlmAnswer, LlmCache, Mode
from concept_pipeline.stage2_model_normalizer import (
    build_prompt,
    extract_with_model,
    parse_lemmas,
    words_to_normalize,
)
from concept_pipeline.normalize import LemmaNormalizer
from concept_pipeline.stage2_candidates import check_candidates, extract_candidates
from test_stage2_candidates import synthetic_result

PLURAL_BLOCKS = [
    ("מטריצות", "heading", -1),
    ("מטריצה הפיכה היא מטריצה ריבועית. מטריצה הפיכה שימושית.", "body", 0),
    ("מטריצות הפיכות הן נושא. מטריצות הפיכות חשובות בקורס.", "body", 1),
    ("דוגמה 3 עם vectors ועם vectors נוספים.", "body", 1),
]
LEMMAS = {"מטריצות": "מטריצה", "הפיכות": "הפיכה", "vectors": "vector"}


class LemmaSender:
    """Answer each batch with the known lemmas of its words and count the calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, call):
        self.calls += 1
        words = [line.split("\t")[0] for line in call.prompt.splitlines()[1:]]
        answer = {word: LEMMAS.get(word, word) for word in words}
        return LlmAnswer(json.dumps({"lemmas": answer}, ensure_ascii=False), 100, 20)


def test_a_close_lemma_is_accepted():
    accepted, refused = parse_lemmas('{"lemmas": {"מטריצות": "מטריצה"}}', ["מטריצות"])
    assert accepted == {"מטריצות": "מטריצה"} and refused == {}


def test_a_far_answer_is_refused_and_the_word_stays():
    accepted, refused = parse_lemmas('{"lemmas": {"מטריצות": "וקטור"}}', ["מטריצות"])
    assert accepted == {} and refused == {"מטריצות": "וקטור"}


def test_an_answer_with_more_words_is_refused():
    _, refused = parse_lemmas('{"lemmas": {"מטריצות": "מטריצה הפיכה"}}', ["מטריצות"])
    assert "מטריצות" in refused


def test_an_answer_that_is_not_json_changes_nothing():
    assert parse_lemmas("I cannot help with that", ["מטריצות"]) == ({}, {})


def test_json_wrapped_in_text_is_still_read():
    accepted, _ = parse_lemmas('Here: {"lemmas": {"vectors": "vector"}} done', ["vectors"])
    assert accepted == {"vectors": "vector"}


def test_a_word_the_answer_skips_stays_as_it_is():
    accepted, refused = parse_lemmas('{"lemmas": {}}', ["מטריצות"])
    assert accepted == {} and refused == {}


def test_lemma_chains_are_followed_and_every_lemma_is_its_own_normal_form():
    normalizer = LemmaNormalizer(Counter({"ב1": 1, "ג1": 1}), {"א1": "ב1", "ב1": "ג1"})
    assert normalizer.normalize_word("א1") == "ג1"
    assert normalizer.normalize_word("ג1") == "ג1"


def test_the_prefix_fallback_result_goes_through_the_lemmas():
    counts = Counter({"המטריצות": 1, "מטריצות": 2, "מטריצה": 1})
    normalizer = LemmaNormalizer(counts, {"מטריצות": "מטריצה"})
    assert normalizer.normalize_word("המטריצות") == "מטריצה"


def test_a_lemma_the_document_never_writes_still_builds_the_key():
    counts = Counter({"שונות": 3, "משותפת": 3})
    normalizer = LemmaNormalizer(counts, {"משותפת": "משותף"})
    assert normalizer.normalize_phrase(("שונות", "משותפת")) == ("שונות", "משותף")


def test_words_to_normalize_skips_function_words_and_digits_and_sends_each_word_once():
    stage2 = extract_candidates(synthetic_result(PLURAL_BLOCKS))
    words = [word for word, _ in words_to_normalize(stage2)]
    assert len(words) == len(set(words))
    assert "3" not in words and "היא" not in words
    assert "מטריצות" in words


def test_the_prompt_lists_one_word_and_its_example_per_line():
    prompt = build_prompt([("מטריצות", "מטריצות הפיכות")])
    assert prompt.splitlines()[1] == "מטריצות\tמטריצות הפיכות"


def test_plural_forms_merge_into_one_key_and_the_gate_still_holds(tmp_path):
    result = synthetic_result(PLURAL_BLOCKS)
    before = {c.key for c in extract_candidates(result).candidates}
    cache = LlmCache(tmp_path, Mode.RECORD, LemmaSender())
    stage2, report = extract_with_model(result, cache, "fake-model")
    after = {c.key for c in stage2.candidates}
    assert "מטריצות הפיכות" in before and "מטריצות הפיכות" not in after
    assert "מטריצה הפיכה" in after
    assert report.lemmas["vectors"] == "vector"
    assert check_candidates(stage2, result) == []


def test_a_rerun_replays_from_the_cache_and_sends_nothing(tmp_path):
    result = synthetic_result(PLURAL_BLOCKS)
    sender = LemmaSender()
    extract_with_model(result, LlmCache(tmp_path, Mode.RECORD, sender), "fake-model")
    sent = sender.calls
    again, _ = extract_with_model(result, LlmCache(tmp_path, Mode.REPLAY), "fake-model")
    assert sender.calls == sent and again.candidates


def test_many_words_are_split_into_several_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(stage2_model_normalizer, "MAX_WORDS_PER_CALL", 2)
    result = synthetic_result(PLURAL_BLOCKS)
    words = len(words_to_normalize(extract_candidates(result)))
    _, report = extract_with_model(result, LlmCache(tmp_path, Mode.RECORD, LemmaSender()), "m")
    assert report.calls == -(-words // 2)
