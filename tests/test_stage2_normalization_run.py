"""Tests for the normalization run tool: label voting, scoring, gold forms and senders (no network)."""

import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import stage2_normalization_run as run
from concept_pipeline.gold import GoldConcept, name_forms
from concept_pipeline.llm_cache import LlmCall
from concept_pipeline.normalize import LemmaNormalizer

CALL = LlmCall("label", "1", "model-x", "system", "prompt")
JUDGE = {"name": "gemini", "base_url": "https://example.test/v1", "key_env": "TEST_JUDGE_KEY"}


class FakeResponse:
    """A minimal stand-in for an httpx response."""

    def __init__(self, status, payload):
        self.status_code = status
        self.is_error = status >= 400
        self._payload = payload

    def json(self):
        return self._payload


def test_two_agreeing_judges_decide_the_label():
    assert run.decide(["מטריצה", "מטריצה", "מטריצות"]) == "מטריצה"


def test_three_different_answers_leave_the_label_unresolved():
    assert run.decide(["א1", "ב1", "ג1"]) is None


def test_missing_answers_do_not_vote():
    assert run.decide([None, "מרחב", None]) is None
    assert run.decide([None, "מרחב", "מרחב"]) == "מרחב"


def test_the_label_prompt_has_one_word_and_example_per_line():
    prompt = run.label_prompt([{"word": "מטריצות", "example": "מטריצות הפיכות"}])
    assert prompt.splitlines()[1] == "מטריצות\tמטריצות הפיכות"


def test_score_counts_per_kind_and_skips_unresolved_labels(tmp_path, capsys):
    labels = {
        "kinds": ["plural", "prefix"],
        "items": [
            {"course": "c", "kind": "plural", "word": "מטריצות", "lemma": "מטריצה"},
            {"course": "c", "kind": "prefix", "word": "ומרחב", "lemma": "מרחב"},
            {"course": "c", "kind": "prefix", "word": "בסיס", "lemma": None},
        ],
    }
    (tmp_path / "labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    lemmas = tmp_path / "lemmas"
    lemmas.mkdir()
    forms = {"normal_forms": {"מטריצות": "מטריצה", "ומרחב": "ומרחב"}}
    (lemmas / "c.json").write_text(json.dumps(forms, ensure_ascii=False))
    run.command_score(Namespace(labels=tmp_path / "labels.json", lemmas=lemmas))
    out = capsys.readouterr().out
    assert "plural       1/  1" in out and "prefix       0/  1" in out
    assert "1/  2  50.0%  fails" in out


def test_gold_names_are_normalized_with_the_model_lemmas():
    concept = GoldConcept(names=Counter({"מטריצות הפיכות": 2}))
    normalizer = LemmaNormalizer(
        Counter({"מטריצה": 1, "הפיכה": 1}), {"מטריצות": "מטריצה", "הפיכות": "הפיכה"}
    )
    assert name_forms(concept, normalizer) == {("מטריצה", "הפיכה")}


def test_a_busy_provider_is_not_cached_and_names_the_status(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret-value")
    monkeypatch.setattr(run.httpx, "post", lambda *a, **k: FakeResponse(503, {}))
    with pytest.raises(run.SendFailed, match="HTTP 503") as error:
        run.openai_compatible_sender(JUDGE)(CALL)
    assert "secret-value" not in str(error.value)


def test_a_429_names_the_error_code_but_not_the_message(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY", "k")
    body = {"error": {"code": "insufficient_quota", "message": "details for org-secret"}}
    monkeypatch.setattr(run.httpx, "post", lambda *a, **k: FakeResponse(429, body))
    with pytest.raises(run.SendFailed, match="HTTP 429 insufficient_quota") as error:
        run.openai_compatible_sender(JUDGE)(CALL)
    assert "org-secret" not in str(error.value)


def test_an_answer_cut_at_the_output_limit_is_refused(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY", "k")
    payload = {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]}
    monkeypatch.setattr(run.httpx, "post", lambda *a, **k: FakeResponse(200, payload))
    with pytest.raises(run.SendFailed, match="output limit"):
        run.openai_compatible_sender(JUDGE)(CALL)


def test_a_good_answer_keeps_its_text_and_tokens(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY", "k")
    payload = {
        "model": "gemini-x",
        "choices": [{"finish_reason": "stop", "message": {"content": '{"lemmas": {}}'}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }
    monkeypatch.setattr(run.httpx, "post", lambda *a, **k: FakeResponse(200, payload))
    answer = run.openai_compatible_sender(JUDGE)(CALL)
    assert (answer.text, answer.input_tokens, answer.output_tokens) == ('{"lemmas": {}}', 12, 4)


def test_a_missing_key_exits_without_sending(monkeypatch):
    monkeypatch.delenv("TEST_JUDGE_KEY", raising=False)
    with pytest.raises(SystemExit):
        run.openai_compatible_sender(JUDGE)
