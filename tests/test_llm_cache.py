"""Tests for the pipeline's model-call cache: replay, record and fresh, with a fake sender."""

import json

import pytest


from concept_pipeline.llm_cache import CacheMiss, LlmAnswer, LlmCache, LlmCall, Mode, call_key

CALL = LlmCall("normalize", "v1", "model-a", "system text", "prompt text")


class FakeSender:
    """Answer every call with a numbered reply and remember how many were sent."""

    def __init__(self):
        self.calls = 0

    def __call__(self, call):
        self.calls += 1
        return LlmAnswer(f"answer {self.calls}", input_tokens=10, output_tokens=3)


def test_replay_without_a_recorded_answer_raises_and_sends_nothing(tmp_path):
    sender = FakeSender()
    with pytest.raises(CacheMiss):
        LlmCache(tmp_path, Mode.REPLAY, sender).answer(CALL)
    assert sender.calls == 0


def test_record_sends_once_then_replays_from_disk(tmp_path):
    sender = FakeSender()
    first = LlmCache(tmp_path, Mode.RECORD, sender).answer(CALL)
    again = LlmCache(tmp_path, Mode.RECORD, sender).answer(CALL)
    replayed = LlmCache(tmp_path, Mode.REPLAY).answer(CALL)
    assert first == again == replayed
    assert sender.calls == 1


def test_any_change_in_the_call_gives_a_new_key():
    changed = [
        LlmCall("normalize", "v2", "model-a", "system text", "prompt text"),
        LlmCall("normalize", "v1", "model-b", "system text", "prompt text"),
        LlmCall("normalize", "v1", "model-a", "other system", "prompt text"),
        LlmCall("normalize", "v1", "model-a", "system text", "other prompt"),
    ]
    assert len({call_key(c) for c in changed} | {call_key(CALL)}) == 5


def test_fresh_sends_again_and_keeps_the_recorded_answer(tmp_path):
    sender = FakeSender()
    LlmCache(tmp_path, Mode.RECORD, sender).answer(CALL)
    fresh = LlmCache(tmp_path, Mode.FRESH, sender).answer(CALL)
    assert fresh.text == "answer 2"
    assert LlmCache(tmp_path, Mode.REPLAY).answer(CALL).text == "answer 1"
    lines = (tmp_path / f"normalize-{call_key(CALL)}.fresh.jsonl").read_text().splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["answer 2"]


def test_usage_counts_tokens_of_every_answer_used(tmp_path):
    cache = LlmCache(tmp_path, Mode.RECORD, FakeSender())
    cache.answer(CALL)
    cache.answer(CALL)
    assert (cache.usage.answers, cache.usage.sent) == (2, 1)
    assert (cache.usage.input_tokens, cache.usage.output_tokens) == (20, 6)


def test_record_without_a_sender_is_refused(tmp_path):
    with pytest.raises(ValueError):
        LlmCache(tmp_path, Mode.RECORD).answer(CALL)


def test_no_temporary_file_is_left_after_writing(tmp_path):
    LlmCache(tmp_path, Mode.RECORD, FakeSender()).answer(CALL)
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"]
