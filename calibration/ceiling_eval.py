"""Precision relative to its ceiling (`stage6_selection_eval`) for one configuration, dev and held-out.

Usage: python calibration/ceiling_eval.py [--config configs/final.json] [--data-dir DIR]
The selection always fills up to 150 nodes, so a course whose pool holds few relevant concepts
cannot reach high plain precision; the gate reads precision divided by min(1, relevant / selected).
Weights are fitted on all dev courses; T's background is every dev course except the one scored.
Writes runs/ceiling_eval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calib  # noqa: E402
import data_dir  # noqa: E402

from concept_pipeline import stage6_selection as s6  # noqa: E402
from concept_pipeline import stage6_selection_eval as selection_eval  # noqa: E402

DEFAULT_CONFIG = HERE / "configs" / "final.json"
# The initial per-course mark reported beside the macro gate (stage6_selection_eval).
COURSE_MARK = 0.75


def course_row(course: dict, dev: list[dict], weights: dict, cfg: dict) -> tuple[dict, dict]:
    """Return one course's precision row and its precision reports (strict, relaxed)."""
    others = [o for o in dev if o["course"] != course["course"]]
    calib.set_background(course, calib.background_ratios(course, others))
    imp = calib.importance(course, weights, cfg["p0"], cfg["combine"])
    concepts = s6.space_concepts([course["evidence"]], imp)
    selection = s6.select_nodes(
        concepts, importance_weight=cfg["lambda"], threshold=cfg["tau"], margin=cfg["eps"],
        variant=cfg["variant"],
    )  # fmt: skip
    pool = [concept.key for concept in concepts]
    row = {"course": course["course"], "pool": len(pool), "nodes": len(selection.nodes)}
    reports = {}
    for name, labels in (("strict", course["label"]), ("relaxed", course["label_relaxed"])):
        relevant = {key: labels.get(key, 0) >= 1 for key in pool}
        report = selection_eval.precision_report(selection.nodes, pool, relevant)
        row[name] = {"precision": report.precision, "ceiling": report.ceiling,
                     "relative": report.relative}  # fmt: skip
        reports[name] = report
    must = {g for g, layer in course["layer"].items() if layer == "must"}
    row["must_in_pool"] = len({g for key in pool for g in course["covers"].get(key, ())} & must)
    return row, reports


def main() -> None:
    """Print and write relative precision per course and the macro gate, dev and held-out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_dir.add_data_dir_argument(parser)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    results_dir = data_dir.resolve_data_dir(args.data_dir) / data_dir.RESULTS_FOLDER
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    dev = calib.load(results_dir / "dumps" / "dev")
    held = calib.load(results_dir / "dumps" / "heldout")
    calib.set_stage5_globals(cfg["eta"], cfg["m0"], cfg["t_max"])
    for course in dev:
        calib.set_background(
            course, calib.background_ratios(course, [o for o in dev if o is not course])
        )
    rows_fit = [r for c in dev for r in calib.features(c, cfg["p0"])]
    weights = calib.fit_nonnegative_logistic(rows_fit, lambda y: y >= 1)
    out = {}
    for split, courses in (("dev", dev), ("heldout", held)):
        rows, reports = [], {"strict": [], "relaxed": []}
        for course in courses:
            row, course_reports = course_row(course, dev, weights, cfg)
            rows.append(row)
            for name, report in course_reports.items():
                reports[name].append(report)
            print(split, row["course"], row["pool"], row["nodes"], "must", row["must_in_pool"],
                  *(f"{n} p {row[n]['precision']:.2f} ceil {row[n]['ceiling']:.2f} "
                    f"rel {row[n]['relative']:.2f}" for n in ("strict", "relaxed")))  # fmt: skip
        for name, group in reports.items():
            marked = sum(1 for r in group if r.relative and r.relative >= COURSE_MARK)
            print(split, name, "macro relative", round(selection_eval.macro_relative(group), 3),
                  "passes", selection_eval.passes(group),
                  f"courses >= {COURSE_MARK}:", marked, "/", len(group))  # fmt: skip
        out[split] = rows
    target = results_dir / "runs" / "ceiling_eval.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
