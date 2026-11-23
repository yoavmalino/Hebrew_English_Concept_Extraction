"""Stage-6 calibration grid on dev: anchoring x variant x lambda x tau x epsilon, in parallel.

Usage: python calibration/s6_run.py --config CONFIG_JSON [--data-dir DIR] [--eps 0.1,0.2]
    [--anchors design,def] [--workers N]
CONFIG_JSON: a stage-5 configuration {"label", "scheme", "p0", "t_max", "eta", "m0", "combine"}.
Learned weights are the leave-one-course-out weights of the course's own fold from
runs/s5_<label>.json, so no course is selected with weights fitted on itself. Each finished row
is appended to a checkpoint file, so a stopped run resumes where it was.
Optional narrower grids: S6_LAMBDA, S6_TAU, S6_VARIANTS (comma-separated environment values).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calib  # noqa: E402
import data_dir  # noqa: E402

from concept_pipeline import stage5_importance as s5  # noqa: E402

LAMBDA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
TAU = {
    "A": [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0],
    "B": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7],
}
T_MAX_WHEN_OFF = 6.0  # any value: T has weight 0 when off
NODE_CAP = 150  # stage 6 MAX_NODES; a course at the cap is reported
METRICS = ("recall", "precision", "precision_relaxed", "strict", "recall_all_must", "nodes")
EQUAL = {s: 1 / len(s5.SIGNALS) for s in s5.SIGNALS}

_courses: list[dict] = []
_importance: dict[str, dict] = {}


def _weights_for(config: dict, folds: dict, name: str) -> dict[str, float]:
    """Return the stage-5 weights for one course under a configuration."""
    if config["scheme"] == "opening":
        return s5.OPENING_WEIGHTS
    if config["scheme"] == "equal":
        return EQUAL
    return folds[f"{config['scheme']}|{config['p0']}|{config['t_max']}"][name]


def _init(dump_folder: str, config: dict, folds: dict) -> None:
    """Load the dev dumps and compute each course's importance once per worker."""
    global _courses, _importance
    calib.LABEL = config["label"]
    _courses = calib.load(Path(dump_folder))
    ratios = {
        c["course"]: calib.background_ratios(c, [o for o in _courses if o is not c])
        for c in _courses
    }
    for course in _courses:
        calib.set_background(course, None if config["t_max"] is None else ratios[course["course"]])
    calib.set_stage5_globals(config["eta"], config["m0"], config["t_max"] or T_MAX_WHEN_OFF)
    _importance = {
        c["course"]: calib.importance(
            c, _weights_for(config, folds, c["course"]), config["p0"], config["combine"]
        )
        for c in _courses
    }


def evaluate(args: tuple) -> dict:
    """Return one grid row: per-course selection metrics and their macro means with intervals."""
    anchor, variant, lam, tau, eps = args
    per_course = {
        c["course"]: calib.selection_metrics(
            c, _importance[c["course"]], anchor=anchor, importance_weight=lam, threshold=tau,
            margin=eps, variant=variant,
        )
        for c in _courses
    }  # fmt: skip
    row = {"anchor": anchor, "variant": variant, "lambda": lam, "tau": tau, "eps": eps,
           "by_course": per_course}  # fmt: skip
    for metric in METRICS:
        mean, low, high = calib.bootstrap_ci([m[metric] for m in per_course.values()])
        row[metric] = mean
        row[metric + "_ci"] = [low, high]
    row["at_cap"] = sum(1 for m in per_course.values() if m["nodes"] >= NODE_CAP)
    return row


def _grid(anchors: list[str], eps_values: list[float]) -> list[tuple]:
    """Return every (anchor, variant, lambda, tau, epsilon), honoring the narrowing variables."""
    lambdas = LAMBDA
    taus = TAU
    if os.environ.get("S6_LAMBDA"):
        lambdas = [float(x) for x in os.environ["S6_LAMBDA"].split(",")]
    if os.environ.get("S6_TAU"):
        taus = {v: [float(x) for x in os.environ["S6_TAU"].split(",")] for v in TAU}
    variants = os.environ.get("S6_VARIANTS", "A,B").split(",")
    return [
        (anchor, variant, lam, tau, eps)
        for anchor in anchors
        for variant in variants
        for lam, tau, eps in itertools.product(lambdas, taus[variant], eps_values)
    ]


def main() -> None:
    """Run the stage-6 grid with checkpointing and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_dir.add_data_dir_argument(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eps", default="0.10", help="comma-separated epsilon values")
    parser.add_argument("--anchors", default=",".join(calib.ANCHOR_MODES))
    parser.add_argument("--workers", type=int, default=2, help="processes (default 2)")
    parser.add_argument("--out", default="s6_grid.json", help="file name under runs/")
    args = parser.parse_args()
    results_dir = data_dir.resolve_data_dir(args.data_dir) / data_dir.RESULTS_FOLDER
    config = json.loads(args.config.read_text(encoding="utf-8"))
    label_name = "strict" if config["label"] == "label" else "relaxed"
    s5_file = results_dir / "runs" / f"s5_{label_name}.json"
    folds = json.loads(s5_file.read_text(encoding="utf-8"))["fold_weights"]
    out = results_dir / "runs" / args.out
    grid = _grid(args.anchors.split(","), [float(e) for e in args.eps.split(",")])
    checkpoint = out.with_name(out.name + ".partial.jsonl")
    results = []
    if checkpoint.exists():
        results = [json.loads(line) for line in checkpoint.read_text().splitlines() if line]
    done = {(r["anchor"], r["variant"], r["lambda"], r["tau"], r["eps"]) for r in results}
    todo = [g for g in grid if g not in done]
    print("resuming with", len(results), "done,", len(todo), "to go", flush=True)
    init_args = (str(results_dir / "dumps" / "dev"), config, folds)
    with (
        Pool(args.workers, initializer=_init, initargs=init_args) as pool,
        open(checkpoint, "a", encoding="utf-8") as handle,
    ):
        for count, row in enumerate(pool.imap_unordered(evaluate, todo, chunksize=1), 1):
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if count % 50 == 0:
                print(len(results), "of", len(grid), flush=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf-8")
    print("written", len(results), "rows to", out)


if __name__ == "__main__":
    main()
