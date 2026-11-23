"""Tests of the two-layer concept gold and the committee tool, with fake judges and no network."""

import json
from collections import Counter
import shutil
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from test_stage2_candidates import FIXTURES, synthetic_result

import concept_committee as committee
from concept_pipeline.gold import (
    align,
    gold_from_json,
    gold_to_json,
    head_words,
    is_name_match,
    match,
    name_parts,
    parse_answer,
    recall_report,
)
from concept_pipeline.gold import GoldConcept
from concept_pipeline.normalize import PrefixNormalizer
from concept_pipeline.stage2_candidates import extract_candidates


@pytest.fixture(scope="module")
def document():
    result = synthetic_result()
    return result, extract_candidates(result)


def answers():
    """Three judges: two agree on the must concept under different names, one adds noise."""
    return {
        "a": {
            "must": [
                {
                    "concept": "מטריצה הפיכה",
                    "aliases": ["invertible matrix"],
                    "questions": ["exam1 2"],
                }
            ],
            "core": [
                {"concept": "גרעין"},
                {"concept": "משפט הרנק", "in_material": False},
            ],
        },
        "b": {
            "must": [{"concept": "invertible matrix", "aliases": ["מטריצה הפיכה"]}],
            "core": [{"concept": "גרעין"}, {"concept": "משפט הרנק", "in_material": False}],
        },
        "c": {"must": [], "core": [{"concept": "פרישה ליניארית"}, {"concept": "מספר"}]},
    }


def gold_for(stage2):
    concepts = [c for judge, answer in answers().items() for c in parse_answer(judge, answer)]
    return align(concepts, stage2.normalizer)


def by_name(gold, name):
    return next(g for g in gold if name in g.names)


def test_malformed_entries_are_skipped_and_in_material_defaults_to_true():
    answer = {"must": [{"concept": " "}, "text", {"concept": "גבול"}], "core": None}
    concepts = parse_answer("a", answer)
    assert [c.name for c in concepts] == ["גבול"]
    assert concepts[0].in_material


def test_an_alias_joins_two_judges_names_into_one_concept(document):
    _, stage2 = document
    gold = gold_for(stage2)
    invertible = by_name(gold, "invertible matrix")
    assert invertible.must_judges == {"a", "b"}
    assert "מטריצה הפיכה" in invertible.names


def test_names_that_do_not_match_exactly_stay_apart(document):
    _, stage2 = document
    names = [g.name for g in gold_for(stage2)]
    assert len(names) == len(set(names)) == 5


def test_a_must_vote_also_counts_for_core(document):
    _, stage2 = document
    invertible = by_name(gold_for(stage2), "invertible matrix")
    assert invertible.judges("core") == {"a", "b"}


def test_gold_concept_matches_a_candidate_through_any_judges_name(document):
    _, stage2 = document
    concept = by_name(gold_for(stage2), "invertible matrix")
    found = match(concept, stage2.candidates, stage2.normalizer)
    assert found is not None and found.key == "מטריצה הפיכה"


def test_an_alias_alone_does_not_find_a_concept(document):
    _, stage2 = document
    concepts = parse_answer("a", {"must": [{"concept": "HashSet", "aliases": ["מטריצה הפיכה"]}]})
    gold = align(concepts, stage2.normalizer)
    assert match(gold[0], stage2.candidates, stage2.normalizer) is None


def test_a_name_matches_its_plural_and_a_final_letter_but_not_a_short_word_change():
    assert is_name_match("ערך עצמי", "ערכים עצמיים")
    assert is_name_match("אורתוגונאלי", "אורתוגונלי")
    assert not is_name_match("use", "used")
    assert not is_name_match("comparator", "שאלה")


def test_a_synonym_inside_a_judges_name_counts_as_a_name():
    assert set(name_parts("K-Nearest Neighbors (KNN)")) >= {"KNN", "K-Nearest Neighbors"}
    assert "inverse document frequency" in name_parts("Idf – inverse document frequency")
    assert name_parts("Open/closed") == ["Open/closed"]


def test_recall_counts_only_concepts_with_k_judges_present_in_the_material(document):
    result, stage2 = document
    report = recall_report(gold_for(stage2), stage2, result, min_judges=2)
    assert report["must"]["concepts"] == 1 and report["must"]["found"] == 1
    assert report["core"]["concepts"] == 1  # the must concept; "גרעין" is never named
    assert report["core"]["missed"] == []
    assert report["core"]["not_in_material"] == 1  # "משפט הרנק"
    assert report["uncertain"] == 2  # judge c alone


def test_rare_concepts_are_reported_in_their_own_bucket(document):
    result, stage2 = document
    report = recall_report(gold_for(stage2), stage2, result, min_judges=2)
    assert report["must"]["by_frequency"] == {"1-2": {"found": 1, "concepts": 1, "recall": 1.0}}
    assert "0" not in report["core"]["by_frequency"]


def test_a_concept_the_summary_never_names_is_reported_apart_from_the_gate(document):
    result, stage2 = document
    report = recall_report(gold_for(stage2), stage2, result, min_judges=2)
    assert report["core"]["unnamed"] == {"concepts": 1, "found": 0}  # "גרעין"
    assert report["core"]["recall"] == 1.0


def test_leaving_out_a_source_never_raises_recall(document):
    result, stage2 = document
    layer = recall_report(gold_for(stage2), stage2, result)["core"]
    assert all(value <= layer["recall"] for value in layer["without_source"].values())


def test_gold_survives_a_json_round_trip(document):
    _, stage2 = document
    gold = gold_for(stage2)
    data = json.loads(json.dumps(gold_to_json(gold), ensure_ascii=False))
    assert gold_to_json(gold_from_json(data)) == gold_to_json(gold)


def test_reply_json_is_found_inside_text_or_a_code_fence():
    assert committee._parse_json('```json\n{"must": []}\n```') == {"must": []}


def test_cache_key_changes_with_the_model_and_the_prompt(tmp_path):
    judge = {"name": "j", "kind": "anthropic", "model": "m1", "key_env": "K"}
    first = committee._cache_path(tmp_path, judge, "prompt")
    assert committee._cache_path(tmp_path, judge, "prompt") == first
    assert committee._cache_path(tmp_path, {**judge, "model": "m2"}, "prompt") != first
    assert committee._cache_path(tmp_path, judge, "other") != first


@pytest.fixture
def course(tmp_path):
    folder = tmp_path / "course"
    folder.mkdir()
    shutil.copy(FIXTURES / "struct_he.pdf", folder / "סיכום הקורס.pdf")
    shutil.copy(FIXTURES / "calculus_he.docx", folder / "exam.docx")
    judges = tmp_path / "judges.json"
    entry = {
        "kind": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "model": "m",
        "key_env": "NO_SUCH_KEY",
    }
    judges.write_text(json.dumps({"judges": [{"name": n, **entry} for n in "ab"]}))
    return folder, judges, tmp_path / "cache"


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["concept_committee.py", *map(str, argv)])
    committee.main()


def test_ask_without_send_sends_nothing(monkeypatch, capsys, course):
    folder, judges, cache = course

    def refuse(*args, **kwargs):
        raise AssertionError("nothing may be sent without --send")

    monkeypatch.setattr(committee.httpx, "post", refuse)
    run(monkeypatch, "ask", folder, "--judges", judges, "--cache", cache)
    assert "would send" in capsys.readouterr().out
    assert list(cache.iterdir()) == []


def test_gold_command_merges_cached_answers(monkeypatch, capsys, course, tmp_path):
    folder, judges, cache = course
    prompt, _ = committee.build_prompt(folder, committee.SUMMARY_MARK)
    cache.mkdir()
    for judge in json.loads(judges.read_text())["judges"]:
        answer = {"must": [{"concept": "גבול"}], "core": []}
        path = committee._cache_path(cache, judge, prompt)
        path.write_text(json.dumps({"answer": answer}, ensure_ascii=False))
    out = tmp_path / "gold.json"
    run(monkeypatch, "gold", folder, "--judges", judges, "--cache", cache, "--out", out)
    assert "must: 1 concepts with at least 2 judges" in capsys.readouterr().out
    assert json.loads(out.read_text())["concepts"][0]["must_judges"] == ["a", "b"]


def fake_anthropic(stop_reason, text='{"must": [], "core": []}', model="claude-x"):
    """Return a stand-in for the anthropic package that records the request it gets."""
    calls = {}

    class Stream:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            block = types.SimpleNamespace(type="text", text=text)
            return types.SimpleNamespace(stop_reason=stop_reason, content=[block], model=model)

    class Client:
        def __init__(self, api_key, timeout):
            calls["has_key"] = bool(api_key)
            messages = types.SimpleNamespace(stream=lambda **kwargs: Stream(**kwargs))
            self.beta = types.SimpleNamespace(messages=messages)

    return types.SimpleNamespace(Anthropic=Client), calls


CLAUDE_JUDGE = {"name": "c", "kind": "anthropic", "model": "claude-opus-5-5", "key_env": "TEST_K"}


def test_claude_judge_streams_and_returns_the_model_that_answered(monkeypatch):
    module, calls = fake_anthropic("end_turn", model="claude-opus-5-5")
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("TEST_K", "secret")
    answer, _, served = committee._send_anthropic(CLAUDE_JUDGE, "prompt")
    assert answer == {"must": [], "core": []} and served == "claude-opus-5-5"
    assert calls["has_key"] and calls["output_config"] == {"effort": "high"}


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_claude_judge_answer_that_is_refused_or_cut_is_not_accepted(monkeypatch, stop_reason):
    module, _ = fake_anthropic(stop_reason)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("TEST_K", "secret")
    with pytest.raises(SystemExit):
        committee._send_anthropic(CLAUDE_JUDGE, "prompt")


def test_judge_with_an_unknown_kind_is_rejected(tmp_path):
    path = tmp_path / "judges.json"
    path.write_text(json.dumps({"judges": [{"name": "x", "kind": "other"}]}))
    with pytest.raises(SystemExit):
        committee._load_judges(path)


def test_example_judges_file_is_valid():
    example = Path(committee.__file__).with_name("judges.example.json")
    assert [j["name"] for j in committee._load_judges(example)] == ["claude", "openai", "gemini"]


OPENAI_JUDGE = {
    "name": "o",
    "kind": "openai_compatible",
    "base_url": "https://example.invalid/v1",
    "model": "m",
    "key_env": "TEST_K",
    "timeout_seconds": 1800,
    "extra": {"service_tier": "flex"},
}


def test_openai_judge_sends_extra_fields_and_its_own_timeout(monkeypatch):
    seen = {}

    def post(url, headers, json, timeout):
        seen.update(body=json, timeout=timeout)
        reply = {"model": "m-1", "choices": [{"message": {"content": '{"must": []}'}}]}
        return committee.httpx.Response(200, json=reply)

    monkeypatch.setattr(committee.httpx, "post", post)
    monkeypatch.setenv("TEST_K", "secret")
    answer, _, served = committee._send_openai_compatible(OPENAI_JUDGE, "prompt")
    assert answer == {"must": []} and served == "m-1"
    assert seen["body"]["service_tier"] == "flex" and seen["timeout"] == 1800


def test_openai_judge_error_exits_without_showing_the_key(monkeypatch):
    def post(url, headers, json, timeout):
        return committee.httpx.Response(429, text="resource unavailable")

    monkeypatch.setattr(committee.httpx, "post", post)
    monkeypatch.setenv("TEST_K", "secret")
    with pytest.raises(SystemExit) as exit_info:
        committee._send_openai_compatible(OPENAI_JUDGE, "prompt")
    assert "HTTP 429" in str(exit_info.value) and "secret" not in str(exit_info.value)


def test_judge_without_its_own_timeout_uses_the_default():
    assert committee._timeout(CLAUDE_JUDGE) == committee.REQUEST_TIMEOUT_SECONDS


def fake_batches(status="ended", outcome="succeeded", stop_reason="end_turn"):
    """Return a stand-in anthropic package whose Batches API records what it gets."""
    calls = {"created": []}

    class Batches:
        def create(self, requests):
            calls["created"].append(requests)
            return types.SimpleNamespace(id="batch-1")

        def retrieve(self, batch_id):
            return types.SimpleNamespace(processing_status=status)

        def results(self, batch_id):
            block = types.SimpleNamespace(type="text", text='{"must": [], "core": []}')
            message = types.SimpleNamespace(stop_reason=stop_reason, content=[block], model="cx")
            result = types.SimpleNamespace(type=outcome, message=message)
            return iter([types.SimpleNamespace(custom_id="gold", result=result)])

    class Client:
        def __init__(self, api_key, timeout):
            self.messages = types.SimpleNamespace(batches=Batches())

    return types.SimpleNamespace(Anthropic=Client), calls


BATCH_JUDGE = {**CLAUDE_JUDGE, "batch": True}


@pytest.fixture
def batch_target(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_K", "secret")
    return tmp_path / "c-abc.json", tmp_path / ("c-abc" + committee.BATCH_SUFFIX)


def test_batch_judge_is_submitted_first_without_fallbacks(monkeypatch, batch_target):
    target, pending = batch_target
    module, calls = fake_batches()
    monkeypatch.setitem(sys.modules, "anthropic", module)
    assert committee._ask_judge(BATCH_JUDGE, "prompt", target) is None
    params = calls["created"][0][0]["params"]
    assert "fallbacks" not in params and "stream" not in params
    assert json.loads(pending.read_text())["batch_id"] == "batch-1"


def test_batch_still_running_returns_nothing_and_keeps_waiting(monkeypatch, batch_target):
    target, pending = batch_target
    pending.write_text(json.dumps({"batch_id": "batch-1"}))
    module, calls = fake_batches(status="in_progress")
    monkeypatch.setitem(sys.modules, "anthropic", module)
    assert committee._ask_judge(BATCH_JUDGE, "prompt", target) is None
    assert pending.exists() and calls["created"] == []


def test_finished_batch_returns_the_answer_and_forgets_the_batch(monkeypatch, batch_target):
    target, pending = batch_target
    pending.write_text(json.dumps({"batch_id": "batch-1"}))
    module, _ = fake_batches()
    monkeypatch.setitem(sys.modules, "anthropic", module)
    answer, _, served = committee._ask_judge(BATCH_JUDGE, "prompt", target)
    assert answer == {"must": [], "core": []} and served == "cx"
    assert not pending.exists()


@pytest.mark.parametrize(
    "outcome, stop_reason", [("expired", "end_turn"), ("succeeded", "refusal")]
)
def test_batch_without_a_usable_answer_is_forgotten_so_it_is_asked_again(
    monkeypatch, batch_target, outcome, stop_reason
):
    target, pending = batch_target
    pending.write_text(json.dumps({"batch_id": "batch-1"}))
    module, _ = fake_batches(outcome=outcome, stop_reason=stop_reason)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    with pytest.raises(SystemExit):
        committee._ask_judge(BATCH_JUDGE, "prompt", target)
    assert not pending.exists() and not target.exists()


def test_batch_is_rejected_for_a_judge_that_is_not_claude(tmp_path):
    path = tmp_path / "judges.json"
    judge = {**OPENAI_JUDGE, "batch": True}
    path.write_text(json.dumps({"judges": [judge]}))
    with pytest.raises(SystemExit):
        committee._load_judges(path)


def test_a_key_pasted_into_key_env_is_refused_and_not_printed(tmp_path, capsys):
    path = tmp_path / "judges.json"
    judge = {**OPENAI_JUDGE, "key_env": "sk-not-a-real-key-123"}
    path.write_text(json.dumps({"judges": [judge]}))
    with pytest.raises(SystemExit) as exit_info:
        committee._load_judges(path)
    assert "sk-not-a-real-key" not in str(exit_info.value) + capsys.readouterr().out


def test_concepts_that_share_only_an_alias_stay_apart(document):
    _, stage2 = document
    concepts = parse_answer("a", {"core": [{"concept": "תמונה", "aliases": ["im"]}]})
    concepts += parse_answer("b", {"core": [{"concept": "חלק מדומה", "aliases": ["im"]}]})
    assert len(align(concepts, stage2.normalizer)) == 2


def test_openai_compatible_answer_cut_at_the_limit_is_not_accepted(monkeypatch):
    def post(url, headers, json, timeout):
        choice = {"finish_reason": "length", "message": {"content": '{"must": [{"conc'}}
        return committee.httpx.Response(200, json={"choices": [choice]})

    monkeypatch.setattr(committee.httpx, "post", post)
    monkeypatch.setenv("TEST_K", "secret")
    with pytest.raises(SystemExit) as exit_info:
        committee._send_openai_compatible(OPENAI_JUDGE, "prompt")
    assert "output limit" in str(exit_info.value)


def test_the_gate_counts_only_concepts_named_three_times_or_more(document):
    result, stage2 = document
    report = recall_report(gold_for(stage2), stage2, result, min_judges=2)
    assert report["must"]["gate"] == {"concepts": 0, "found": 0, "recall": None}


def test_a_one_way_alias_does_not_join_two_concepts(document):
    _, stage2 = document
    answers = {
        "a": {"core": [{"concept": "HashSet", "aliases": ["queue"]}]},
        "b": {"core": [{"concept": "queue"}]},
    }
    concepts = [c for judge, answer in answers.items() for c in parse_answer(judge, answer)]
    assert len(align(concepts, stage2.normalizer)) == 2


def test_head_words_are_the_first_hebrew_and_last_english_word_of_a_name():
    concept = GoldConcept(names=Counter({"גבול חלקי": 2, "heap sort": 1, "זיכרון": 1}))
    assert head_words(concept, PrefixNormalizer({})) == {"גבול", "sort"}


def test_a_named_result_or_definition_word_is_never_a_credited_head():
    concept = GoldConcept(names=Counter({"משפט היינה": 2, "הגדרת גבול": 1}))
    assert head_words(concept, PrefixNormalizer({})) == set()
