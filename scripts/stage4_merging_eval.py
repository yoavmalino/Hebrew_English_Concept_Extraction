"""Run pipeline stage 4 end to end on course summaries and measure it against a committee.

Dev tool, not part of the app; sends only with ``--send`` (owner approval 2026-11-16, ceiling
$20 across dev and held-out). Per course: stage 2 with model lemmas, the translation and synonym
finders, the decider `DECIDER_RUNS` times (each: the majority of `FAST_RUNS` Haiku runs; separate
caches, for the stability check), then a
committee of judges labels every proposed pair and every gold name pair (same instructions as the
decider), and the plan's gate is computed (`concept_pipeline.stage4_merge_eval`). Every model answer is cached
under the gold folder, so a rerun sends only what is missing. A ledger file keeps the money spent
across runs; a send that would pass ``--budget`` stops the run first.

    python scripts/stage4_merging_eval.py COURSE_FOLDER ... --gold-dir GOLD \
        --judges JUDGES.json --out OUT_DIR --ledger LEDGER.json [--send] [--anchor 40]

Exit code: 0 gate passed, 2 gate failed, 3 a send failed (rerun to retry), 4 budget.
"""

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import concept_committee as committee  # noqa: E402
import stage2_normalization_run as run  # noqa: E402
import stage4_merging_report as report  # noqa: E402
from concept_pipeline import stage4_merge_eval as ev  # noqa: E402
from concept_pipeline.gold import gold_from_json, normal_form  # noqa: E402
from concept_pipeline.llm_cache import LlmCache, LlmCall, Mode  # noqa: E402
from concept_pipeline.stage4_merge_decider import FAST_RUNS, build_prompt, decide_by_majority  # noqa: E402
from concept_pipeline.stage4_merge_decider import SYSTEM_PROMPT as MERGE_SYSTEM_PROMPT  # noqa: E402
from concept_pipeline.stage4_merge_decider import parse_decisions  # noqa: E402
from concept_pipeline.stage2_model_normalizer import extract_with_model  # noqa: E402
from concept_pipeline.normalize import is_name_match  # noqa: E402
from concept_pipeline.stage4_merging import (
    MergePair,
    check_groups,
    contexts,
    find_pairs,
    group_keys,
)  # noqa: E402
from concept_pipeline.stage4_merging import decide_by_rule  # noqa: E402
from concept_pipeline.stage4_synonym_finder import synonym_pairs  # noqa: E402
from concept_pipeline.stage4_translation_finder import translation_pairs  # noqa: E402

# Independent decisions (each the majority of FAST_RUNS Haiku runs) compared for the plan's
# stability check (fresh x3).
DECIDER_RUNS = 3
LABEL_PROMPT_VERSION = "1"
# Pairs per labelling call: large enough for few calls, small enough that a bad answer costs
# little to resend.
LABEL_PAIRS_PER_CALL = 100
# Cost estimate before sending: tokens per prompt character measured on Hebrew-heavy prompts
# (0.70, translation finder 2026-11-16), rounded up; output allows for reasoning tokens.
TOKENS_PER_CHAR_ESTIMATE = 0.8
OUTPUT_TOKENS_PER_PAIR_ESTIMATE = 60
# Haiku 5.5 dollars per million tokens for prompts of 100K tokens or fewer (all of ours).
HAIKU_PRICE = (0.10, 0.50)
ANCHOR_SEED = 7
MIN_CORE_JUDGES = 2


class BudgetExceeded(RuntimeError):
    """Sending more would pass the approved budget."""


class Ledger:
    """Money spent on sent calls across runs, kept in a JSON file; safe across threads."""

    def __init__(self, path: Path, budget: float, prices: dict[str, tuple[float, float]]):
        self.path, self.budget, self.prices = path, budget, prices
        self.spent = json.loads(path.read_text("utf-8"))["spent"] if path.exists() else 0.0
        self._lock = threading.Lock()

    def cost(self, model: str, input_tokens: float, output_tokens: float) -> float:
        """Return the dollars for tokens of a model (unknown models cost as Opus)."""
        price_in, price_out = self.prices.get(model, (4.0, 20.0))
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    def metered(self, send):
        """Return `send` wrapped so every answer is paid for and nothing is sent past budget."""

        def send_paid(call):
            if self.spent >= self.budget:
                raise BudgetExceeded(f"spent ${self.spent:.2f} of ${self.budget:.2f}")
            answer = send(call)
            with self._lock:
                self.spent += self.cost(call.model, answer.input_tokens, answer.output_tokens)
                self.path.write_text(json.dumps({"spent": round(self.spent, 4)}), "utf-8")
            return answer

        return send_paid


def haiku_cache(folder: Path, send: bool, ledger: Ledger, sender: list) -> LlmCache:
    """Return a cache for Haiku calls; one shared sender is built on the first use."""
    if send and not sender:
        sender.append(ledger.metered(run.anthropic_sender(run.NORMALIZER_KEY_ENV)))
    return LlmCache(folder, Mode.RECORD if send else Mode.REPLAY, sender[0] if send else None)


def gold_concept_keys(concept, stage2, by_words: dict) -> dict[str, set]:
    """Return the candidate keys a gold concept's names give, each with where it came from."""
    sources: dict[str, set] = {}
    for index, name in enumerate(sorted(concept.names)):
        for part, how in report._name_sources(name):
            form = " ".join(normal_form(part, stage2.normalizer))
            if not form:
                continue
            same_length = by_words.get(len(form.split()), set())
            hits = {form} & same_length or {k for k in same_length if is_name_match(form, k)}
            for key in hits:
                sources.setdefault(key, set()).add((index, how))
    return sources


def prepare_course(course: Path, gold_dir: Path, caches: dict) -> dict:
    """Return one course's candidates, pairs, decisions per run, groups and items to label."""
    result = committee._summary_result(course, committee.SUMMARY_MARK)
    stage2, _ = extract_with_model(result, caches["normalize"], run.DEFAULT_MODEL)
    translations, _ = translation_pairs(stage2, caches["translate"], run.DEFAULT_MODEL)
    synonyms, _ = synonym_pairs(stage2, caches["synonyms"], run.DEFAULT_MODEL)
    pairs, counts = find_pairs(stage2.candidates, translations, synonyms)
    # "named" pairs are decided by rule and never sent (owner decision 2026-11-20).
    rule_decisions, model_pairs = decide_by_rule(pairs)
    runs, disputed = [], []
    for fast_caches in caches["merge"]:
        decisions, counts_run = decide_by_majority(
            model_pairs, stage2, result, fast_caches, run.DEFAULT_MODEL
        )
        runs.append(rule_decisions + decisions)
        disputed.append(counts_run["disputed"])
    keys = sorted(c.key for c in stage2.candidates)
    by_words: dict[int, set] = {}
    for key in keys:
        by_words.setdefault(len(key.split()), set()).add(key)
    folder = next(p for p in gold_dir.iterdir() if p.name.strip() == course.name.strip())
    gold = gold_from_json(
        json.loads((folder / "concepts.gold.json").read_text("utf-8"))["concepts"]
    )
    concept_keys, name_pairs = [], []
    for concept in gold:
        if len(concept.judges("core")) < MIN_CORE_JUDGES:
            continue
        sources = gold_concept_keys(concept, stage2, by_words)
        hits = sorted(sources)
        concept_keys.append(hits)
        for i, a in enumerate(hits):
            for b in hits[i + 1 :]:
                name_pairs.append((a, b, report._pair_kind(a, b, sources, hits)))
    by_key = {c.key: c for c in stage2.candidates}
    to_label = {(p.first, p.second): p for p in pairs}
    for a, b, kind in name_pairs:
        if kind != report.CONTAINED:
            to_label.setdefault((a, b), MergePair(a, b, ()))
    items = [
        (pair, contexts(by_key[pair.first], result), contexts(by_key[pair.second], result))
        for _, pair in sorted(to_label.items())
    ]
    return {
        "course": course.name.strip(),
        "words": stage2.words,
        "keys": keys,
        "pairs": pairs,
        "counts": counts,
        "runs": [{(d.first, d.second): d.is_same for d in decisions} for decisions in runs],
        "decisions": runs[0],
        "groups": group_keys(keys, runs[0]),
        "groups_by_run": [group_keys(keys, decisions) for decisions in runs],
        "concept_keys": concept_keys,
        "name_pairs": name_pairs,
        "items": items,
        "disputed": disputed,
    }


def label_batches(courses: list[dict], model: str) -> list[tuple[LlmCall, dict]]:
    """Return the labelling calls for one judge: per course, batches with id -> pair maps."""
    batches = []
    for course in courses:
        items = course["items"]
        for start in range(0, len(items), LABEL_PAIRS_PER_CALL):
            chunk = items[start : start + LABEL_PAIRS_PER_CALL]
            numbered = [(str(i), pair, a, b) for i, (pair, a, b) in enumerate(chunk, 1)]
            prompt = build_prompt(numbered)
            call = LlmCall("label_pairs", LABEL_PROMPT_VERSION, model, MERGE_SYSTEM_PROMPT, prompt)
            batches.append((call, {i: (p.first, p.second) for i, p, *_ in numbered}))
    return batches


def judge_sender(judge: dict, ledger: Ledger):
    """Return the metered sender for a judge from the judges file."""
    if judge["kind"] == "anthropic":
        return ledger.metered(run.anthropic_sender(judge["key_env"]))
    return ledger.metered(run.openai_compatible_sender(judge))


def ask_judge(judge: dict, courses: list[dict], cache: LlmCache) -> dict:
    """Return a judge's ``{pair: is_same}``; an unreadable answer is removed so a rerun asks."""
    found = {}
    for call, ids in label_batches(courses, judge["model"]):
        verdicts = parse_decisions(cache.answer(call).text, list(ids))
        if not verdicts:
            cache.path(call).unlink(missing_ok=True)
            raise run.SendFailed(f"{judge['name']}: an answer with no decisions; removed")
        found |= {ids[i]: verdict for i, verdict in verdicts.items()}
    return found


def estimate_labels(judges: list[dict], courses: list[dict], caches: dict, ledger: Ledger) -> float:
    """Return the estimated dollars of the labelling calls not cached yet."""
    total = 0.0
    for judge in judges:
        for call, ids in label_batches(courses, judge["model"]):
            if not caches[judge["name"]].path(call).exists():
                tokens_in = (len(call.system) + len(call.prompt)) * TOKENS_PER_CHAR_ESTIMATE
                total += ledger.cost(
                    judge["model"], tokens_in, len(ids) * OUTPUT_TOKENS_PER_PAIR_ESTIMATE
                )
    return total


def score_course(course: dict, labels: dict) -> dict:
    """Return one course's gate numbers and diagnostics."""
    proposed = {(p.first, p.second) for p in course["pairs"]}
    named = {(a, b) for a, b, kind in course["name_pairs"] if kind != report.CONTAINED}
    positive_names = {pair for pair in named if labels.get(pair) is True}
    positives = {pair for pair, label in labels.items() if label is True}
    # Gate measures use the gold corrected by the labels where it contradicts them, fragments kept
    # as one concept (owner decision 2026-11-16, fixed before held-out; `corrected_concepts`).
    # The raw gold and the gold split by every label are diagnostics.
    groups, concept_keys = course["groups"], course["concept_keys"]
    corrected = ev.corrected_concepts(concept_keys, labels)
    by_run = [
        (ev.gold_over_merge(g, corrected, None), ev.duplicates(g, corrected, None))
        for g in course["groups_by_run"]
    ]
    (bad, merged), (extra, covering) = by_run[0]
    pair_bad, pair_merged = ev.over_merge(groups, labels)
    by_finder = {name: set() for name in course["counts"]["by_finder"]}
    for pair in course["pairs"]:
        for finder in pair.finders:
            by_finder[finder].add((pair.first, pair.second))
    return {
        "course": course["course"],
        "pairs": len(proposed),
        "algorithmic": check_groups(course["keys"], course["decisions"]),
        "over_merge": [bad, merged],
        "over_merge_by_run": [list(om) for om, _ in by_run],
        "duplicates_by_run": [list(dup) for _, dup in by_run],
        "over_merge_raw_gold": list(ev.gold_over_merge(groups, concept_keys, None)),
        "duplicates_raw_gold": list(ev.duplicates(groups, concept_keys, None)),
        "over_merge_any_pair": [pair_bad, pair_merged],
        "over_merge_split_by_labels": list(ev.gold_over_merge(groups, concept_keys, labels)),
        "duplicates_split_by_labels": list(ev.duplicates(groups, concept_keys, labels)),
        "disputed": course["disputed"],
        "duplicates": [extra, covering],
        "stable_pairs": [
            round(ev.stability(course["runs"], sorted(proposed)) * len(proposed)),
            len(proposed),
        ],
        "decider": ev.pair_scores(course["runs"][0], {p: labels.get(p) for p in proposed}),
        "finder_recall": {
            name: [len(found & positives), len(positives)] for name, found in by_finder.items()
        },
        "name_pairs_same": [len(positive_names & proposed), len(positive_names)],
        "name_pairs_merged": round(ev.merged_share(course["groups"], positive_names), 3),
        "unresolved_labels": sum(1 for label in labels.values() if label is None),
    }


def gate(scores: list[dict]) -> dict:
    """Return the pooled gate numbers over all courses and whether each passes."""

    def pooled(field: str) -> float:
        top = sum(s[field][0] for s in scores)
        bottom = sum(s[field][1] for s in scores)
        return top / bottom if bottom else 0.0

    numbers = {
        "algorithmic_ok": not any(s["algorithmic"] for s in scores),
        "duplicates": pooled("duplicates"),
        "over_merge": pooled("over_merge"),
        "over_merge_any_pair": pooled("over_merge_any_pair"),
        "over_merge_split_by_labels": pooled("over_merge_split_by_labels"),
        "duplicates_split_by_labels": pooled("duplicates_split_by_labels"),
        "over_merge_raw_gold": pooled("over_merge_raw_gold"),
        "duplicates_raw_gold": pooled("duplicates_raw_gold"),
        "pair_agreement": pooled("stable_pairs"),
    }
    for field in ("over_merge", "duplicates"):
        runs = [
            tuple(sum(s[f"{field}_by_run"][d][i] for s in scores) for i in (0, 1))
            for d in range(DECIDER_RUNS)
        ]
        numbers[f"{field}_by_run"] = [count / total if total else 0.0 for count, total in runs]
        numbers[f"{field}_stable"] = ev.within_interval(runs)
    numbers["passed"] = (
        numbers["algorithmic_ok"]
        and numbers["duplicates"] < ev.DUPLICATE_MAX
        and numbers["over_merge"] <= ev.OVER_MERGE_MAX
        and numbers["over_merge_stable"]
        and numbers["duplicates_stable"]
    )
    return numbers


def anchor_sample(courses: list[dict], labels: dict, size: int) -> list[dict]:
    """Return `size` labelled pairs for the owner, half same and half different, seeded."""
    pool = {True: [], False: []}
    for course in courses:
        for pair, first, second in course["items"]:
            key = (pair.first, pair.second)
            if labels.get(key) is not None:
                pool[labels[key]].append((course["course"], key, first, second))
    chosen = []
    rng = random.Random(ANCHOR_SEED)
    for label in (True, False):
        chosen += [
            {
                "course": c,
                "first": k[0],
                "second": k[1],
                "first_contexts": a,
                "second_contexts": b,
                "committee": label,
                "answer": None,
            }
            for c, k, a, b in rng.sample(pool[label], min(size // 2, len(pool[label])))
        ]
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    """Prepare every course, label, score, write the report and exit with the gate's code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("courses", nargs="+", type=Path)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--judges", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--anchor", type=int, default=0, help="pairs for the owner's sample")
    parser.add_argument("--send", action="store_true")
    parser.add_argument(
        "--send-only",
        help="send only to this judge; Haiku and the other judges replay from the cache",
    )
    args = parser.parse_args()
    judges = json.loads(args.judges.read_text("utf-8"))["judges"]
    prices = {j["model"]: (j["price_in"], j["price_out"]) for j in judges}
    prices[run.DEFAULT_MODEL] = HAIKU_PRICE
    ledger = Ledger(args.ledger, args.budget, prices)
    # With --send-only, only that judge records; every other cache must already hold its answers
    # (added 2026-11-16 after OpenAI's flex tier answered 429 while the others had finished).
    haiku_sends = args.send and not args.send_only
    haiku: list = []
    gold = args.gold_dir
    try:
        caches = {
            "normalize": haiku_cache(gold / "_normalization_cache", haiku_sends, ledger, haiku),
            "translate": haiku_cache(gold / "_translation_cache", haiku_sends, ledger, haiku),
            "synonyms": haiku_cache(gold / "_synonym_cache", haiku_sends, ledger, haiku),
            # Run folders: decision d uses runs d*FAST_RUNS .. d*FAST_RUNS+FAST_RUNS-1.
            "merge": [
                [
                    haiku_cache(
                        gold / "_merge_cache" / f"run{d * FAST_RUNS + i}",
                        haiku_sends,
                        ledger,
                        haiku,
                    )
                    for i in range(FAST_RUNS)
                ]
                for d in range(DECIDER_RUNS)
            ],
        }
        courses = []
        for course in args.courses:
            courses.append(prepare_course(course, gold, caches))
            print(
                f"prepared {courses[-1]['course']}: {len(courses[-1]['items'])} to label, "
                f"spent ${ledger.spent:.2f}",
                flush=True,
            )
    except BudgetExceeded as error:
        print(f"budget: {error}")
        sys.exit(4)
    except run.SendFailed as error:
        print(f"send failed: {error}")
        sys.exit(3)

    def sends_to(judge: dict) -> bool:
        return args.send and args.send_only in (None, judge["name"])

    label_caches = {
        j["name"]: LlmCache(
            gold / "_pair_label_cache",
            Mode.RECORD if sends_to(j) else Mode.REPLAY,
            judge_sender(j, ledger) if sends_to(j) else None,
        )
        for j in judges
    }
    cost = estimate_labels(judges, courses, label_caches, ledger)
    print(f"labelling estimate ${cost:.2f}; spent so far ${ledger.spent:.2f} of ${args.budget:.2f}")
    if args.send and ledger.spent + cost > args.budget:
        print("budget: the labelling estimate would pass the budget; nothing sent")
        sys.exit(4)
    answers, failures = {}, []
    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        futures = {
            j["name"]: pool.submit(ask_judge, j, courses, label_caches[j["name"]]) for j in judges
        }
        for name, future in futures.items():
            try:
                answers[name] = future.result()
            # Any judge's failure is reported, not swallowed: the run exits 3 and a rerun sends
            # only the calls that are still missing from the cache.
            except Exception as error:  # noqa: BLE001
                failures.append(f"{name}: {type(error).__name__}: {str(error)[:300]}")
    if failures:
        print("\n".join(["send failed:", *failures]))
        sys.exit(3)
    every = {(p.first, p.second) for course in courses for p, *_ in course["items"]}
    labels = {pair: ev.vote([answers[j["name"]].get(pair) for j in judges]) for pair in every}
    scores = [score_course(course, labels) for course in courses]
    for s in scores:
        print(
            f"{s['course']:26} pairs {s['pairs']:5} over-merge {s['over_merge'][0]}/"
            f"{s['over_merge'][1]} (any pair {s['over_merge_any_pair'][0]}/"
            f"{s['over_merge_any_pair'][1]}) disputed {s['disputed'][0]} "
            f"duplicates {s['duplicates'][0]}/{s['duplicates'][1]} stable "
            f"{s['stable_pairs'][0]}/{s['stable_pairs'][1]} decider precision "
            f"{s['decider']['precision']:.3f} recall {s['decider']['recall']:.3f} name pairs "
            f"{s['name_pairs_same'][0]}/{s['name_pairs_same'][1]} merged {s['name_pairs_merged']}"
            f"{' ALGORITHMIC FAIL' if s['algorithmic'] else ''}"
        )
    totals = gate(scores)
    print(
        f"gate: duplicates {totals['duplicates']:.4f} (< {ev.DUPLICATE_MAX}), over-merge "
        f"{totals['over_merge']:.4f} (<= {ev.OVER_MERGE_MAX}; any pair "
        f"{totals['over_merge_any_pair']:.4f}; raw gold {totals['over_merge_raw_gold']:.4f}), "
        f"duplicates raw gold {totals['duplicates_raw_gold']:.4f}, across runs over-merge "
        f"{'stable' if totals['over_merge_stable'] else 'MOVES'} "
        f"{[round(x, 4) for x in totals['over_merge_by_run']]}, duplicates "
        f"{'stable' if totals['duplicates_stable'] else 'MOVES'} "
        f"{[round(x, 4) for x in totals['duplicates_by_run']]}, pair agreement "
        f"{totals['pair_agreement']:.4f} (reported), algorithmic {'ok' if totals['algorithmic_ok'] else 'FAIL'}"
        f" -> {'PASSED' if totals['passed'] else 'FAILED'}; spent ${ledger.spent:.2f}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    name = gold.name
    (args.out / f"{name}.stage4_eval.json").write_text(
        json.dumps({"gate": totals, "courses": scores}, ensure_ascii=False, indent=1), "utf-8"
    )
    labelled = [
        {
            "course": c["course"],
            "first": p.first,
            "second": p.second,
            "finders": list(p.finders),
            "decider": c["runs"][0].get((p.first, p.second)),
            "votes": {j["name"]: answers[j["name"]].get((p.first, p.second)) for j in judges},
            "label": labels.get((p.first, p.second)),
        }
        for c in courses
        for p, *_ in c["items"]
    ]
    (args.out / f"{name}.stage4_labels.json").write_text(
        json.dumps(labelled, ensure_ascii=False, indent=1), "utf-8"
    )
    if args.anchor:
        sample = anchor_sample(courses, labels, args.anchor)
        (args.out / f"{name}.stage4_anchor.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=1), "utf-8"
        )
    sys.exit(0 if totals["passed"] else 2)


if __name__ == "__main__":
    main()
