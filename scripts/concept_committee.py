"""Build the two-layer concept gold of one course with a blind committee of models, and score stage 2.

Dev tool, not part of the app (plan section יא). A course folder holds one summary (its name
contains "סיכום", or pass --summary) and past exams. Commands:

    prompt COURSE                          write the prompt and print its size; sends nothing
    ask    COURSE --judges F --cache D     list what would be sent; add --send to send it
    gold   COURSE --judges F --cache D --out GOLD.json [--k 2]   cached answers -> gold
    recall COURSE --gold GOLD.json [--k 2] [--min-count N]       stage-2 recall per layer

Judges never see stage-2 output. Answers are cached by a hash of judge, model and prompt, so a
re-run sends nothing. The judges file lists, per judge, ``name``, ``kind`` ("anthropic" uses
the official SDK; "openai_compatible" needs ``base_url``), ``model``, ``key_env`` (the NAME of
an environment variable that holds its key) and optional ``extra`` (fields added to an
OpenAI-compatible request), ``timeout_seconds``, or ``effort`` and ``batch`` (Claude); see
judges.example.json. A judge with ``"batch": true`` goes through Claude's Batches API at half
price: the first ``ask --send`` submits it, a later one collects the answer. The key is read at
send time and never printed or stored. Sending study material to a provider needs the owner's
approval first (given on 2026-11-16 for Anthropic, OpenAI and Google).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


import httpx  # noqa: E402
from hebmath import Extractor, ExtractorConfig  # noqa: E402

from concept_pipeline.gold import (  # noqa: E402
    DEFAULT_MIN_JUDGES,
    align,
    gold_from_json,
    gold_to_json,
    parse_answer,
    recall_report,
)
from concept_pipeline.stage2_candidates import MIN_PHRASE_COUNT, extract_candidates  # noqa: E402
from concept_pipeline.tokens import clean_text  # noqa: E402

# Bump when the prompt changes: the cache key includes it, so old answers are not reused.
PROMPT_VERSION = "gold-1"
SUFFIXES = (".pdf", ".docx")
SUMMARY_MARK = "סיכום"
# A whole course in one request can take minutes on a large model. A judge may set its own
# "timeout_seconds": OpenAI's flex tier (half price) queues requests and answers more slowly.
REQUEST_TIMEOUT_SECONDS = 600
# Provider error text shown to the owner is cut to this length; it never holds the key.
MAX_ERROR_CHARS = 300
# The core list of a long course can run to thousands of entries; streaming allows a large cap.
ANTHROPIC_MAX_TOKENS = 64000
# key_env names an environment variable; anything else (such as a pasted key) is refused and
# never echoed, because the error would print it to the terminal and to logs.
ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")
# A submitted batch is remembered next to the answer's cache file until its answer is collected.
BATCH_SUFFIX = ".batch.json"
# The one request inside each batch; a batch holds one judge's answer to one course.
BATCH_REQUEST_ID = "gold"

SYSTEM_PROMPT = "You label concepts for an evaluation set of a study tool. Reply with JSON only."
TASK = """Course: {course}

Below are the course summary (the study material) and past exams of the course.

Task 1, "must": for every exam question, list the concepts a student must know to solve it.
Task 2, "core": list the course's core concepts: every concept a lecturer could reasonably ask a
question about, or that is needed to understand such a concept. Include concepts that no exam
here asks about.

Rules:
- A concept is a named idea: a definition, theorem, method, algorithm, structure or process.
  Not a symbol, a number, a whole sentence, or a generic word such as "result" or "example".
- Name each concept the way the summary names it, in the summary's language. Give other names
  you know (Hebrew, English, abbreviations) as aliases.
- in_material: true when the summary explains or uses the concept, false when it is absent.
- List each concept once. A "must" concept is core automatically; do not repeat it in "core".

Reply with this JSON:
{{"must": [{{"concept": "...", "aliases": ["..."], "questions": ["<exam> <question>"],
"in_material": true}}], "core": [{{"concept": "...", "aliases": ["..."], "in_material": true}}]}}
"""


def _extractor() -> Extractor:
    """Return the extractor used for every file: no OCR, as on the server."""
    return Extractor(ExtractorConfig(enable_ocr=False, recovery_ocr_pages=0))


def _course_files(course: Path, summary_mark: str) -> tuple[Path, list[Path]]:
    """Return the course's one summary and its exams; exit when the summary is ambiguous."""
    files = sorted(p for p in course.iterdir() if p.suffix.lower() in SUFFIXES)
    summaries = [p for p in files if summary_mark in p.name]
    if len(summaries) != 1:
        names = ", ".join(p.name for p in summaries) or "none"
        sys.exit(f"need exactly one summary matching {summary_mark!r} in {course}; found {names}")
    return summaries[0], [p for p in files if p != summaries[0]]


def build_prompt(course: Path, summary_mark: str) -> tuple[str, list[str]]:
    """Return the user prompt for a course and the names of exams that could not be read."""
    extractor = _extractor()
    summary, exams = _course_files(course, summary_mark)
    result = extractor.extract_path(summary)
    if not result.ok:
        sys.exit(f"cannot read the summary {summary.name}: {result.error_code}")
    parts = [TASK.format(course=course.name.strip()), "=== SUMMARY ===", clean_text(result.text)]
    skipped = []
    for exam in exams:
        exam_result = extractor.extract_path(exam)
        if not exam_result.ok or not clean_text(exam_result.text).strip():
            skipped.append(exam.name)
            continue
        parts += [f"=== EXAM: {exam.stem} ===", clean_text(exam_result.text)]
    return "\n\n".join(parts), skipped


def _load_judges(path: Path) -> list[dict]:
    """Return the judges from the judges file, checking that each has the needed fields."""
    judges = json.loads(path.read_text(encoding="utf-8"))["judges"]
    for judge in judges:
        if judge.get("kind") not in SENDERS:
            sys.exit(f"judge {judge.get('name', '?')}: kind must be one of {sorted(SENDERS)}")
        needed = {"name", "kind", "model", "key_env"}
        if judge.get("kind") == "openai_compatible":
            needed.add("base_url")
        if judge.get("batch") and judge.get("kind") != "anthropic":
            sys.exit(f"judge {judge['name']}: batch is supported only for kind anthropic")
        missing = needed - judge.keys()
        if missing:
            sys.exit(f"judge {judge.get('name', '?')} lacks {sorted(missing)}")
        if not ENV_NAME_RE.fullmatch(str(judge["key_env"])):
            sys.exit(
                f"judge {judge['name']}: key_env must be the NAME of an environment variable "
                "(such as JUDGE_OPENAI_API_KEY), never the key itself; value not shown"
            )
    return judges


def _cache_path(cache: Path, judge: dict, prompt: str) -> Path:
    """Return the cache file of one judge's answer to one prompt."""
    identity = json.dumps(
        [
            PROMPT_VERSION,
            judge["kind"],
            judge.get("base_url"),
            judge["model"],
            SYSTEM_PROMPT,
            prompt,
        ],
        ensure_ascii=False,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return cache / f"{judge['name']}-{digest}.json"


def _parse_json(content: str) -> dict:
    """Return the JSON object in a reply, also when the model wrapped it in text or a fence."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group())


def _key(judge: dict) -> str:
    """Return the judge's API key from the environment variable its entry names; never print it."""
    key = os.environ.get(judge["key_env"])
    if not key:
        sys.exit(f"environment variable {judge['key_env']} is not set for judge {judge['name']}")
    return key


def _timeout(judge: dict) -> float:
    """Return the judge's request timeout in seconds (its own "timeout_seconds", else default)."""
    return float(judge.get("timeout_seconds", REQUEST_TIMEOUT_SECONDS))


def _send_openai_compatible(judge: dict, prompt: str) -> tuple[dict, str, str]:
    """Ask a judge behind an OpenAI-compatible chat-completions endpoint (OpenAI, Google)."""
    body = {
        "model": judge["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # Provider-specific fields (such as a reasoning effort) come from the judges file.
        **judge.get("extra", {}),
    }
    try:
        response = httpx.post(
            f"{judge['base_url'].rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {_key(judge)}"},
            json=body,
            timeout=_timeout(judge),
        )
    except httpx.TimeoutException:
        sys.exit(f"{judge['name']}: no answer within {_timeout(judge):.0f} s; nothing cached")
    # A failed request is not cached, so running ask again retries only this judge. Flex answers
    # 429 when it has no capacity; that request is not billed.
    if response.is_error:
        detail = response.text[:MAX_ERROR_CHARS]
        sys.exit(f"{judge['name']}: HTTP {response.status_code}, nothing cached: {detail}")
    reply = response.json()
    choice = reply["choices"][0]
    content = choice["message"]["content"] or ""
    # A reply cut at the output limit is not valid JSON; say so instead of a parse traceback.
    if choice.get("finish_reason") == "length":
        sys.exit(
            f"{judge['name']}: answer cut at the output limit; nothing cached (raise max_tokens)"
        )
    try:
        answer = _parse_json(content)
    except json.JSONDecodeError:
        sys.exit(
            f"{judge['name']}: answer is not valid JSON ({len(content):,} chars); nothing cached"
        )
    return answer, content, reply.get("model", judge["model"])


def _anthropic_client(judge: dict):
    """Return an Anthropic SDK client for a judge (the SDK is a dev-only dependency)."""
    import anthropic  # requirements-dev.txt; imported only when sending

    return anthropic.Anthropic(api_key=_key(judge), timeout=_timeout(judge))


def _answer_from_message(judge: dict, message) -> tuple[dict, str, str]:
    """Return a finished Claude message as (answer, raw text, model), or exit if it is unusable.

    A refused or cut answer is never cached, so the next ``ask --send`` asks again.
    """
    if message.stop_reason == "refusal":
        sys.exit(f"judge {judge['name']} declined the request (refusal); nothing was cached")
    if message.stop_reason == "max_tokens":
        sys.exit(f"judge {judge['name']} hit max_tokens; the answer is cut and was not cached")
    content = "".join(block.text for block in message.content if block.type == "text")
    return _parse_json(content), content, message.model


def _send_anthropic(judge: dict, prompt: str) -> tuple[dict, str, str]:
    """Ask a Claude judge through the official SDK, streamed because input and output are long.

    Server-side fallbacks are on: if the model declines, another Claude model answers. The
    model that answered is returned and stored, so a judge replaced this way is visible.
    """
    client = _anthropic_client(judge)
    with client.beta.messages.stream(
        model=judge["model"],
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": judge.get("effort", "high")},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        message = stream.get_final_message()
    return _answer_from_message(judge, message)


def _submit_batch(judge: dict, prompt: str, pending: Path) -> None:
    """Submit one Claude request through the Batches API and remember the batch id.

    The Batches API rejects ``fallbacks`` and ``stream``, so a refusal here is final for the
    batch; running the judge without ``batch`` then retries with fallbacks.
    """
    params = {
        "model": judge["model"],
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": judge.get("effort", "high")},
    }
    batch = _anthropic_client(judge).messages.batches.create(
        requests=[{"custom_id": BATCH_REQUEST_ID, "params": params}]
    )
    pending.write_text(json.dumps({"judge": judge["name"], "batch_id": batch.id}), encoding="utf-8")


def _collect_batch(judge: dict, pending: Path) -> tuple[dict, str, str] | None:
    """Return the answer of a submitted batch, or None while it is still running.

    A batch that ended without an answer (errored, canceled, expired) is forgotten, so the
    next ``ask --send`` submits it again.
    """
    batch_id = json.loads(pending.read_text(encoding="utf-8"))["batch_id"]
    client = _anthropic_client(judge)
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"{judge['name']}: batch {batch.processing_status}; run ask --send again later")
        return None
    outcome = next(iter(client.messages.batches.results(batch_id)), None)
    pending.unlink()
    if outcome is None or outcome.result.type != "succeeded":
        kind = outcome.result.type if outcome else "missing"
        sys.exit(f"judge {judge['name']}: batch result {kind}; nothing cached, ask again")
    return _answer_from_message(judge, outcome.result.message)


def _ask_judge(judge: dict, prompt: str, target: Path) -> tuple[dict, str, str] | None:
    """Return a judge's answer now, or None when it is a batch that is not ready yet."""
    if not judge.get("batch"):
        return SENDERS[judge["kind"]](judge, prompt)
    pending = target.with_name(target.stem + BATCH_SUFFIX)
    if pending.exists():
        return _collect_batch(judge, pending)
    _submit_batch(judge, prompt, pending)
    print(f"{judge['name']}: submitted as a batch (half price); run ask --send again to collect")
    return None


SENDERS = {"openai_compatible": _send_openai_compatible, "anthropic": _send_anthropic}


def command_prompt(args) -> None:
    """Write the prompt to a file (or print it) with its size; sends nothing."""
    prompt, skipped = build_prompt(args.course, args.summary)
    if args.out:
        args.out.write_text(prompt, encoding="utf-8")
    else:
        print(prompt)
    print(f"prompt: {len(prompt):,} characters, {len(prompt.split()):,} words", file=sys.stderr)
    if skipped:
        print(f"exams not readable (skipped): {', '.join(skipped)}", file=sys.stderr)


def command_ask(args) -> None:
    """Send the prompt to every judge without a cached answer, only with --send."""
    prompt, skipped = build_prompt(args.course, args.summary)
    args.cache.mkdir(parents=True, exist_ok=True)
    for judge in _load_judges(args.judges):
        target = _cache_path(args.cache, judge, prompt)
        if target.exists():
            print(f"{judge['name']}: cached, nothing sent")
            continue
        if not args.send:
            if target.with_name(target.stem + BATCH_SUFFIX).exists():
                print(f"{judge['name']}: batch submitted earlier; ask --send collects it")
                continue
            target_host = judge.get("base_url", judge["kind"])
            mode = " as a batch" if judge.get("batch") else ""
            print(f"{judge['name']}: would send {len(prompt):,} characters to {target_host}{mode}")
            continue
        reply = _ask_judge(judge, prompt, target)
        if reply is None:
            continue
        answer, raw, served_model = reply
        record = {
            "judge": judge["name"],
            "model": judge["model"],
            "served_model": served_model,
            "answer": answer,
            "raw": raw,
        }
        target.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{judge['name']}: answered, saved to {target.name}")
    if skipped:
        print(f"exams not readable (skipped): {', '.join(skipped)}")


def _summary_result(course: Path, summary_mark: str):
    """Return the extraction of the course summary."""
    summary, _ = _course_files(course, summary_mark)
    result = _extractor().extract_path(summary)
    if not result.ok:
        sys.exit(f"cannot read the summary {summary.name}: {result.error_code}")
    return result


def command_gold(args) -> None:
    """Merge the cached answers of all judges into the course gold file."""
    prompt, _ = build_prompt(args.course, args.summary)
    concepts = []
    for judge in _load_judges(args.judges):
        target = _cache_path(args.cache, judge, prompt)
        if not target.exists():
            sys.exit(f"no cached answer of {judge['name']} for this prompt; run ask --send first")
        record = json.loads(target.read_text(encoding="utf-8"))
        concepts += parse_answer(judge["name"], record["answer"])
    # Names are normalized with the summary's own word counts, as stage 2 normalizes its keys.
    normalizer = extract_candidates(_summary_result(args.course, args.summary)).normalizer
    gold = align(concepts, normalizer)
    data = {"prompt_version": PROMPT_VERSION, "concepts": gold_to_json(gold)}
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for layer in ("must", "core"):
        kept = sum(1 for g in gold if len(g.judges(layer)) >= args.k)
        print(f"{layer}: {kept} concepts with at least {args.k} judges")


def command_recall(args) -> None:
    """Print stage-2 recall against a course gold file, per layer, as JSON."""
    result = _summary_result(args.course, args.summary)
    stage2 = extract_candidates(result, args.min_count)
    data = json.loads(args.gold.read_text(encoding="utf-8"))
    report = recall_report(gold_from_json(data["concepts"]), stage2, result, args.k)
    report["pool"] = {
        "candidates": len(stage2.candidates),
        "words": stage2.words,
        "per_thousand_words": round(1000 * len(stage2.candidates) / max(stage2.words, 1), 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main() -> None:
    """Parse the command line and run one command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["prompt", "ask", "gold", "recall"])
    parser.add_argument("course", type=Path)
    parser.add_argument("--summary", default=SUMMARY_MARK, help="text in the summary's file name")
    parser.add_argument("--judges", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--send", action="store_true", help="really send to the judges")
    parser.add_argument("--k", type=int, default=DEFAULT_MIN_JUDGES)
    parser.add_argument("--min-count", type=int, default=MIN_PHRASE_COUNT)
    args = parser.parse_args()
    needs = {"ask": ["judges", "cache"], "gold": ["judges", "cache", "out"], "recall": ["gold"]}
    for name in needs.get(args.command, []):
        if getattr(args, name) is None:
            parser.error(f"{args.command} needs --{name}")
    {"prompt": command_prompt, "ask": command_ask, "gold": command_gold, "recall": command_recall}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
