"""Stage-5 calibration grid on dev: weights x p0 x T x eta x m0 x combine, nDCG@50 per course.

Usage: python calibration/s5_run.py [--data-dir DIR] [--label label|label_relaxed] [--workers N]
Reads the dev dumps and writes <data>/stage56_results/runs/s5_<strict|relaxed>.json.
Learned weights are fitted leave-one-course-out (the evaluated course never in training), and
T's background for each course is every other dev course. One process per (T, p0) group.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calib  # noqa: E402
import data_dir  # noqa: E402

from concept_pipeline import stage5_importance as s5  # noqa: E402

P0 = [1 / 32, 1 / 16, 1 / 8]
T_MAX = [None, 3.0, 4.0, 6.0, 8.0]  # None = T off
T_MAX_WHEN_OFF = 6.0  # any value: T has weight 0 when off
ETA = [0.7, 0.8, 0.9, 1.0]
M0 = [3, 5, 8]
COMBINE = ["max", "soft_or"]
TARGETS = {"lr_relevant": lambda y: y >= 1, "lr_must": lambda y: y == 2}
EQUAL = {s: 1 / len(s5.SIGNALS) for s in s5.SIGNALS}

# Set in each worker by `_init` (multiprocessing copies module state at fork on Linux only).
_courses: list[dict] = []
_ratios: dict[str, dict] = {}


def _init(dump_folder: str, label: str) -> None:
    """Load the dev dumps and backgrounds into the worker process."""
    global _courses, _ratios
    calib.LABEL = label
    _courses = calib.load(Path(dump_folder))
    _ratios = {
        c["course"]: calib.background_ratios(c, [o for o in _courses if o is not c])
        for c in _courses
    }


def run_group(args: tuple) -> tuple[list[dict], dict]:
    """Return the grid rows and the fold weights of one (t_max, p0) group."""
    t_max, p0 = args
    names = [c["course"] for c in _courses]
    for course in _courses:
        calib.set_background(course, None if t_max is None else _ratios[course["course"]])
    s5.T_MAX_LOG2_RATIO = t_max or T_MAX_WHEN_OFF
    rows = {c["course"]: calib.features(c, p0) for c in _courses}
    schemes = {
        "opening": {n: s5.OPENING_WEIGHTS for n in names},
        "equal": {n: EQUAL for n in names},
    }
    folds = {}
    for scheme, target in TARGETS.items():
        per_fold = {
            test: calib.fit_nonnegative_logistic(
                [r for n in names if n != test for r in rows[n]], target
            )
            for test in names
        }
        schemes[scheme] = per_fold
        folds[f"{scheme}|{p0}|{t_max}"] = per_fold
    results = []
    for (scheme, per_fold), eta, m0, combine in itertools.product(
        schemes.items(), ETA, M0, COMBINE
    ):
        calib.set_stage5_globals(eta, m0, t_max or T_MAX_WHEN_OFF)
        scores = {
            c["course"]: calib.ndcg_at(
                calib.importance(c, per_fold[c["course"]], p0, combine), c[calib.LABEL]
            )
            for c in _courses
        }
        mean, low, high = calib.bootstrap_ci(list(scores.values()))
        results.append(
            {"scheme": scheme, "p0": p0, "t_max": t_max, "eta": eta, "m0": m0,
             "combine": combine, "ndcg": mean, "low": low, "high": high, "by_course": scores}
        )  # fmt: skip
    print("group", t_max, p0, "best", round(max(r["ndcg"] for r in results), 4), flush=True)
    return results, folds


def main() -> None:
    """Run every (T, p0) group in parallel and write the grid."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_dir.add_data_dir_argument(parser)
    parser.add_argument("--label", choices=["label", "label_relaxed"], default="label")
    parser.add_argument("--workers", type=int, default=2, help="processes (default 2)")
    args = parser.parse_args()
    data = data_dir.resolve_data_dir(args.data_dir)
    results_dir = data / data_dir.RESULTS_FOLDER
    dumps = results_dir / "dumps" / "dev"
    out = results_dir / "runs" / f"s5_{'strict' if args.label == 'label' else 'relaxed'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results, fold_weights = [], {}
    groups = list(itertools.product(T_MAX, P0))
    with Pool(args.workers, initializer=_init, initargs=(str(dumps), args.label)) as pool:
        for rows, folds in pool.imap_unordered(run_group, groups):
            results += rows
            fold_weights.update(folds)
    payload = {"label": args.label, "results": results, "fold_weights": fold_weights}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=0), encoding="utf-8")
    print("written", len(results), "rows to", out)


if __name__ == "__main__":
    main()
