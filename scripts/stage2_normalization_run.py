"""Run model normalization on the dev courses, label the accuracy sample, score and re-measure.

Dev tool, not part of the app. Every model call goes through `LlmCache`, so a rerun sends nothing;
nothing is sent without `--send`. Keys come from the environment variables the judges file names.

    normalize COURSE ... --cache DIR --out DIR [--model M] [--send]
        Haiku lemmas per course; writes <out>/<course>.json (lemmas, refused, normal forms, usage).
    label --sample FILE --judges FILE --cache DIR --out FILE [--send]
        Judges from providers other than the normalizer's label each sampled word; the first two
        decide when they agree, the third breaks a tie (2 of 3), three answers stay unresolved.
    score --labels FILE --lemmas DIR
        Accuracy of the normalizer's final form against the labels, per kind (gate 95%).
    recall COURSE ... --gold-dir DIR --lemmas DIR [--min-count N] [--k K]
        Stage-2 recall and pool per course with the prefix fallback and with the model lemmas.

Dev courses only: the held-out courses are measured once, by `recall`, never diagnosed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import concept_committee as committee  # noqa: E402

from concept_pipeline.gold import gold_from_json, recall_report  # noqa: E402
from concept_pipeline.llm_cache import CacheMiss, LlmAnswer, LlmCache, LlmCall, Mode  # noqa: E402
from concept_pipeline.stage2_model_normalizer import SYSTEM_PROMPT, extract_with_model  # noqa: E402
from concept_pipeline.stage2_model_normalizer import words_to_normalize  # noqa: E402
from concept_pipeline.stage2_candidates import MIN_PHRASE_COUNT, extract_candidates  # noqa: E402

DEFAULT_MODEL = "claude-haiku-5-5"
NORMALIZER_KEY_ENV = "JUDGE_ANTHROPIC_API_KEY"
# A batch of 400 words answers in roughly 12k tokens; the margin keeps an answer from being cut.
MAX_OUTPUT_TOKENS = 32000
REQUEST_TIMEOUT_SECONDS = 600
LABEL_PROMPT_VERSION = "1"
GATE_ACCURACY = 0.95


class SendFailed(RuntimeError):
    """A call that returned no usable answer; it is not cached, so the next run asks again."""


def _env_key(name: str) -> str:
    """Return the key held by an environment variable; never print it."""
    key = os.environ.get(name)
    if not key:
        sys.exit(f"environment variable {name} is not set")
    return key


def anthropic_sender(key_env: str, max_tokens: int = MAX_OUTPUT_TOKENS):
    """Return a sender for Claude through the official SDK (streamed, long answers)."""
    import anthropic  # requirements-dev.txt; imported only when sending

    client = anthropic.Anthropic(api_key=_env_key(key_env), timeout=REQUEST_TIMEOUT_SECONDS)

    def send(call: LlmCall) -> LlmAnswer:
        with client.messages.stream(
            model=call.model,
            max_tokens=max_tokens,
            system=call.system,
            messages=[{"role": "user", "content": call.prompt}],
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason in ("refusal", "max_tokens"):
            raise SendFailed(f"{call.model}: stop reason {message.stop_reason}; nothing cached")
        text = "".join(block.text for block in message.content if block.type == "text")
        usage = message.usage
        return LlmAnswer(text, usage.input_tokens, usage.output_tokens, message.model)

    return send


def _error_code(response: httpx.Response) -> str:
    """Return the provider's short error code or type, never its message or any header.

    A 429 is either a passing capacity limit (retry helps) or an exhausted quota (it does not);
    only the code tells them apart. The message is left out because it can echo request details.
    """
    try:
        error = response.json().get("error", {})
    except ValueError:
        return "(no JSON body)"
    if isinstance(error, list) and error:
        error = error[0].get("error", {}) if isinstance(error[0], dict) else {}
    if not isinstance(error, dict):
        return ""
    code = error.get("code") or error.get("status") or error.get("type") or ""
    return re.sub(r"[^\w.-]", "", str(code))[:60]


def openai_compatible_sender(judge: dict):
    """Return a sender for an OpenAI-compatible chat endpoint (OpenAI, Gemini)."""
    key = _env_key(judge["key_env"])

    def send(call: LlmCall) -> LlmAnswer:
        body = {
            "model": call.model,
            "messages": [
                {"role": "system", "content": call.system},
                {"role": "user", "content": call.prompt},
            ],
            **judge.get("extra", {}),
        }
        response = httpx.post(
            f"{judge['base_url'].rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
            timeout=float(judge.get("timeout_seconds", REQUEST_TIMEOUT_SECONDS)),
        )
        if response.is_error:
            raise SendFailed(
                f"{judge['name']}: HTTP {response.status_code} {_error_code(response)}; "
                "nothing cached"
            )
        reply = response.json()
        choice = reply["choices"][0]
        if choice.get("finish_reason") == "length":
            raise SendFailed(f"{judge['name']}: answer cut at the output limit; nothing cached")
        usage = reply.get("usage", {})
        return LlmAnswer(
            choice["message"]["content"] or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            reply.get("model", call.model),
        )

    return send


def _course_name(course: Path) -> str:
    """Return the course name as the sample and the gold folders write it (no edge spaces)."""
    return course.name.strip()


def command_normalize(args) -> None:
    """Record the model's lemmas for each course and write the normal form of every word."""
    sender = anthropic_sender(NORMALIZER_KEY_ENV) if args.send else None
    cache = LlmCache(args.cache, Mode.RECORD if args.send else Mode.REPLAY, sender)
    args.out.mkdir(parents=True, exist_ok=True)
    for course in args.courses:
        result = committee._summary_result(course, args.summary)
        stage2, report = extract_with_model(result, cache, args.model)
        words = [word for word, _ in words_to_normalize(extract_candidates(result))]
        data = {
            "model": args.model,
            "calls": report.calls,
            "lemmas": report.lemmas,
            "refused": report.refused,
            "normal_forms": {word: stage2.normalizer.normalize_word(word) for word in words},
        }
        target = args.out / f"{_course_name(course)}.json"
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(
            f"{_course_name(course):28} calls {report.calls:3}  accepted {len(report.lemmas):5}  "
            f"refused {len(report.refused):4}"
        )
    usage = cache.usage
    print(f"sent {usage.sent}  input tokens {usage.input_tokens:,}  output {usage.output_tokens:,}")


def label_prompt(items: list[dict]) -> str:
    """Return the labelling prompt: one ``word<TAB>example`` line per sampled word."""
    lines = [f"{item['word']}\t{item['example']}" for item in items]
    return "Words (word, then an example phrase from the summary):\n" + "\n".join(lines)


def _judge_lemmas(text: str) -> dict[str, str]:
    """Return a judge's ``word -> base form`` map, or exit when the answer is not readable."""
    try:
        data = committee._parse_json(text)
    except json.JSONDecodeError:
        sys.exit("a judge's answer is not valid JSON; delete its cache file and label again")
    lemmas = data.get("lemmas") if isinstance(data, dict) else None
    if not isinstance(lemmas, dict):
        sys.exit("a judge's answer has no lemmas map")
    return {word: str(lemma).strip().lower() for word, lemma in lemmas.items()}


def decide(answers: list[str | None]) -> str | None:
    """Return the label from the judges' answers: two that agree win, otherwise None."""
    votes = Counter(answer for answer in answers if answer)
    if not votes:
        return None
    lemma, count = votes.most_common(1)[0]
    return lemma if count >= 2 else None


def command_label(args) -> None:
    """Ask the labelling judges (same rules as the normalizer) and write the decided labels."""
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    items = sample["items"]
    judges = json.loads(args.judges.read_text(encoding="utf-8"))["judges"]
    prompt = label_prompt(items)
    answers: dict[str, dict[str, str]] = {}
    for judge in judges:
        if judge["kind"] == "anthropic":
            sender = anthropic_sender(judge["key_env"]) if args.send else None
        else:
            sender = openai_compatible_sender(judge) if args.send else None
        cache = LlmCache(args.cache, Mode.RECORD if args.send else Mode.REPLAY, sender)
        call = LlmCall("label", LABEL_PROMPT_VERSION, judge["model"], SYSTEM_PROMPT, prompt)
        try:
            answers[judge["name"]] = _judge_lemmas(cache.answer(call).text)
        except SendFailed as error:
            sys.exit(str(error))
    names = [judge["name"] for judge in judges]
    agreed = unresolved = 0
    for item in items:
        votes = [answers[name].get(item["word"]) for name in names]
        item["votes"] = dict(zip(names, votes))
        item["lemma"] = decide(votes)
        agreed += votes[0] is not None and votes[0] == votes[1]
        unresolved += item["lemma"] is None
    args.out.write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"labelled {len(items)}: first two agree {agreed}, unresolved {unresolved}")


def command_score(args) -> None:
    """Print the normalizer's accuracy on the labelled sample, per kind and in total."""
    sample = json.loads(args.labels.read_text(encoding="utf-8"))
    forms = {}
    for path in args.lemmas.glob("*.json"):
        forms[path.stem] = json.loads(path.read_text(encoding="utf-8"))["normal_forms"]
    right, total, wrong = Counter(), Counter(), []
    for item in sample["items"]:
        if item["lemma"] is None:
            continue
        predicted = forms[item["course"]].get(item["word"], item["word"])
        total[item["kind"]] += 1
        if predicted == item["lemma"]:
            right[item["kind"]] += 1
        else:
            wrong.append((item["course"], item["word"], predicted, item["lemma"]))
    for kind in sample["kinds"]:
        print(f"{kind:10} {right[kind]:3}/{total[kind]:3}  {right[kind] / max(total[kind], 1):.1%}")
    accuracy = sum(right.values()) / max(sum(total.values()), 1)
    verdict = "passes" if accuracy >= GATE_ACCURACY else "fails"
    print(
        f"{'total':10} {sum(right.values()):3}/{sum(total.values()):3}  {accuracy:.1%}  {verdict}"
    )
    for course, word, predicted, lemma in wrong:
        print(f"  {course}: {word} -> {predicted}, label {lemma}")


def _layer_line(report: dict, stage2) -> str:
    """Return one line: recall per layer (all, then the 3+ mention gate) and the pool."""
    parts = [
        f"{layer} {report[layer]['found']}/{report[layer]['concepts']}"
        f" gate {report[layer]['gate']['found']}/{report[layer]['gate']['concepts']}"
        for layer in ("must", "core")
    ]
    pool = 1000 * len(stage2.candidates) / max(stage2.words, 1)
    return "  ".join(parts) + f"  pool {pool:.0f}/1000w"


def command_recall(args) -> None:
    """Print recall and pool per course, without and with the model lemmas, and the totals."""
    cache = LlmCache(args.cache, Mode.REPLAY)
    totals = {"prefix": Counter(), "model": Counter()}
    for course in args.courses:
        name = _course_name(course)
        result = committee._summary_result(course, args.summary)
        folder = next(path for path in args.gold_dir.iterdir() if path.name.strip() == name)
        data = json.loads((folder / "concepts.gold.json").read_text("utf-8"))
        before = extract_candidates(result, args.min_count)
        after, _ = extract_with_model(result, cache, args.model, min_phrase_count=args.min_count)
        for label, stage2 in (("prefix", before), ("model", after)):
            # Gold names are normalized by this run's normalizer, lemmas included (`name_forms`).
            gold = gold_from_json(data["concepts"])
            report = recall_report(gold, stage2, result, args.k)
            print(f"{name:28} {label:6} {_layer_line(report, stage2)}")
            for layer in ("must", "core"):
                totals[label][f"{layer}_found"] += report[layer]["found"]
                totals[label][f"{layer}_all"] += report[layer]["concepts"]
                totals[label][f"{layer}_gate_found"] += report[layer]["gate"]["found"]
                totals[label][f"{layer}_gate_all"] += report[layer]["gate"]["concepts"]
            totals[label]["candidates"] += len(stage2.candidates)
            totals[label]["words"] += stage2.words
    for label, total in totals.items():
        parts = [
            f"{layer} {total[f'{layer}_found']}/{total[f'{layer}_all']} "
            f"{total[f'{layer}_found'] / max(total[f'{layer}_all'], 1):.1%}"
            f" gate {total[f'{layer}_gate_found'] / max(total[f'{layer}_gate_all'], 1):.1%}"
            for layer in ("must", "core")
        ]
        pool = 1000 * total["candidates"] / max(total["words"], 1)
        print(f"{'total':28} {label:6} " + "  ".join(parts) + f"  pool {pool:.0f}/1000w")


def main() -> None:
    """Parse the command line and run one command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["normalize", "label", "score", "recall"])
    parser.add_argument("courses", nargs="*", type=Path)
    parser.add_argument("--summary", default=committee.SUMMARY_MARK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--judges", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--lemmas", type=Path)
    parser.add_argument("--gold-dir", type=Path)
    parser.add_argument("--min-count", type=int, default=MIN_PHRASE_COUNT)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--send", action="store_true", help="really send to the models")
    args = parser.parse_args()
    commands = {
        "normalize": command_normalize,
        "label": command_label,
        "score": command_score,
        "recall": command_recall,
    }
    try:
        commands[args.command](args)
    except SendFailed as error:
        sys.exit(str(error))
    except CacheMiss as error:
        sys.exit(f"{error}; run with --send first")


if __name__ == "__main__":
    main()
