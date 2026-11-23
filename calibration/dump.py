"""Dump each course's stage-5/6 evidence and gold labels for calibration; sends nothing.

Usage:
    python calibration/dump.py --split dev [--mode algorithmic|replay] [--data-dir DIR] [COURSE...]

Per course: stages 2-4 run on the summary, then `file_evidence` (no background; T is added
later by `calib.set_background`), concept labels from the committee gold (must 2 / core 1 /
noise 0, strict = 2 of 3 judges, relaxed = any judge), the gold concepts each concept covers,
and normalized word counts for T's background. Output: <data>/stage56_results/dumps/<split>/.

Modes:
    algorithmic  no model at all: prefix normalization, rule finders, rule decisions only.
    replay       the model path through the recorded caches in the gold folder. A missing
                 answer stops the run; record the caches first with
                 `scripts/stage4_merging_eval.py ... --send` (costs money).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]

import concept_committee as committee  # noqa: E402
import data_dir  # noqa: E402
import stage2_normalization_run as run  # noqa: E402
import stage4_merging_eval as merging_eval  # noqa: E402

from concept_pipeline.gold import gold_from_json  # noqa: E402
from concept_pipeline.llm_cache import LlmCache, Mode  # noqa: E402
from concept_pipeline.stage2_candidates import extract_candidates  # noqa: E402
from concept_pipeline.stage2_model_normalizer import extract_with_model  # noqa: E402
from concept_pipeline.stage4_merge_decider import FAST_RUNS, decide_by_majority  # noqa: E402
from concept_pipeline.stage4_merging import (  # noqa: E402
    absorb_fragments,
    decide_by_rule,
    find_pairs,
    merge,
)
from concept_pipeline.stage4_synonym_finder import synonym_pairs  # noqa: E402
from concept_pipeline.stage4_translation_finder import translation_pairs  # noqa: E402
from concept_pipeline.stage5_importance import file_evidence  # noqa: E402
from concept_pipeline.tokens import split_blocks  # noqa: E402

MODES = ("algorithmic", "replay")


def _replay(folder: Path) -> LlmCache:
    """Return a replay-only cache: it never sends, and a missing answer is an error."""
    return LlmCache(folder, Mode.REPLAY, None)


def build_concepts(result, gold: Path, mode: str):
    """Return the stage-2 result and the file's concepts after stage 4, in the given mode.

    Args:
        result: The summary's hebmath extraction.
        gold: The split's gold folder, which holds the model caches.
        mode: "algorithmic" or "replay" (see the module docstring).
    """
    if mode == "algorithmic":
        stage2 = extract_candidates(result)
        pairs, _ = find_pairs(stage2.candidates)
        decisions, _ = decide_by_rule(pairs)  # the other pairs stay apart
    else:
        stage2, _ = extract_with_model(
            result, _replay(gold / "_normalization_cache"), run.DEFAULT_MODEL
        )
        translations, _ = translation_pairs(
            stage2, _replay(gold / "_translation_cache"), run.DEFAULT_MODEL
        )
        synonyms, _ = synonym_pairs(stage2, _replay(gold / "_synonym_cache"), run.DEFAULT_MODEL)
        pairs, _ = find_pairs(stage2.candidates, translations, synonyms)
        decisions, model_pairs = decide_by_rule(pairs)
        runs = [_replay(gold / "_merge_cache" / f"run{i}") for i in range(FAST_RUNS)]
        model_decisions, _ = decide_by_majority(
            model_pairs, stage2, result, runs, run.DEFAULT_MODEL
        )
        decisions = decisions + model_decisions
    concepts, _ = absorb_fragments(merge(stage2, decisions))
    return stage2, concepts


def gold_labels(golds, stage2, concepts) -> dict:
    """Return the label sets, the gold concepts each concept covers and the strict layers.

    Strict layers need 2 of 3 judges (the stage-2 gold); relaxed takes any judge's concept,
    because a valid concept one judge listed is not noise.
    """
    by_words: dict[int, set] = {}
    for key in sorted(c.key for c in stage2.candidates):
        by_words.setdefault(len(key.split()), set()).add(key)
    key_gold: dict[str, set[int]] = {}
    layer, layer_relaxed = {}, {}
    for index, concept in enumerate(golds):
        if not concept.judges("core"):
            continue
        is_must = len(concept.judges("must")) >= merging_eval.MIN_CORE_JUDGES
        layer_relaxed[index] = "must" if is_must else "core"
        if len(concept.judges("core")) >= merging_eval.MIN_CORE_JUDGES:
            layer[index] = layer_relaxed[index]
        for key in merging_eval.gold_concept_keys(concept, stage2, by_words):
            key_gold.setdefault(key, set()).add(index)
    covers = {
        c.key: sorted({g for member in c.members for g in key_gold.get(member, ())})
        for c in concepts
    }

    def labels_of(layers_by_id: dict) -> dict[str, int]:
        labels = {}
        for key, ids in covers.items():
            layers = {layers_by_id[g] for g in ids if g in layers_by_id}
            labels[key] = 2 if "must" in layers else 1 if "core" in layers else 0
        return labels

    return {
        "label": labels_of(layer),
        "label_relaxed": labels_of(layer_relaxed),
        "covers": covers,
        "layer": layer,
    }


def dump_course(data: Path, split: str, name: str, mode: str, out: Path) -> None:
    """Write one course's dump and print its label counts."""
    gold = data_dir.gold_dir(data, split)
    course = data_dir.course_dir(data / data_dir.COURSES_FOLDER, name)
    result = committee._summary_result(course, committee.SUMMARY_MARK)
    stage2, concepts = build_concepts(result, gold, mode)
    gold_file = data_dir.course_dir(gold, name) / data_dir.GOLD_FILE
    golds = gold_from_json(json.loads(gold_file.read_text("utf-8"))["concepts"])
    labels = gold_labels(golds, stage2, concepts)
    words = Counter(
        stage2.normalizer.normalize_word(token.text)
        for block in split_blocks(result)
        for segment in block.segments
        for token in segment.tokens
    )
    data_out = {
        "course": name,
        "mode": mode,
        "evidence": file_evidence(result, concepts),
        **labels,
        "gold_names": {i: golds[i].name for i in labels["layer"]},
        "words": words,
    }
    with open(out / f"{name}.pkl", "wb") as handle:
        pickle.dump(data_out, handle)
    counts = Counter(labels["label"].values())
    print(
        f"{name}: concepts {len(concepts)}, must {counts[2]}, core {counts[1]}, noise {counts[0]}"
    )


def main() -> None:
    """Parse the arguments and dump every requested course of the split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_dir.add_data_dir_argument(parser)
    parser.add_argument("--split", choices=data_dir.SPLITS, required=True)
    parser.add_argument("--mode", choices=MODES, default="algorithmic")
    parser.add_argument("--out", type=Path, help="default: <data>/stage56_results/dumps/<split>")
    parser.add_argument("courses", nargs="*", help="course names (default: every gold course)")
    args = parser.parse_args()
    data = data_dir.resolve_data_dir(args.data_dir)
    out = args.out or data / data_dir.RESULTS_FOLDER / "dumps" / args.split
    out.mkdir(parents=True, exist_ok=True)
    for name in args.courses or data_dir.course_names(data, args.split):
        dump_course(data, args.split, name.strip(), args.mode, out)


if __name__ == "__main__":
    main()
