"""Measure the chosen stage-5/6 settings once on a split (dev weights, dev background).

Usage: python calibration/final_eval.py --split heldout [--config configs/final.json]
    [--weights fit|calibrated|opening] [--t-off] [--data-dir DIR]
Weights: "fit" refits them on all dev courses (as at calibration), "calibrated" uses
`CALIBRATED_WEIGHTS` from the code, "opening" the design's opening weights. Each evaluated
course's T background is every dev course except itself (--t-off scores without T). Prints
nDCG@50, recall and precision per course and the macro means, and writes
runs/final_<split>_<config>_<weights>[_t_off].json.
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

from concept_pipeline import stage5_importance as s5  # noqa: E402

DEFAULT_CONFIG = HERE / "configs" / "final.json"
MACRO_METRICS = ("ndcg", "ndcg_relaxed", "recall", "recall_all_must", "precision",
                 "precision_relaxed", "strict", "nodes")  # fmt: skip


def stage5_weights(choice: str, dev: list[dict], p0: float) -> dict[str, float]:
    """Return the stage-5 weights for a --weights choice (see the module docstring)."""
    if choice == "opening":
        return dict(s5.OPENING_WEIGHTS)
    if choice == "calibrated":
        return dict(s5.CALIBRATED_WEIGHTS)
    rows = [r for c in dev for r in calib.features(c, p0)]
    return calib.fit_nonnegative_logistic(rows, lambda y: y >= 1)


def main() -> None:
    """Evaluate one configuration on one split and print and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_dir.add_data_dir_argument(parser)
    parser.add_argument("--split", choices=data_dir.SPLITS, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", choices=["fit", "calibrated", "opening"], default="fit")
    parser.add_argument(
        "--t-off", action="store_true", help="score without T (no background), as the demo does"
    )
    args = parser.parse_args()
    results_dir = data_dir.resolve_data_dir(args.data_dir) / data_dir.RESULTS_FOLDER
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    dev = calib.load(results_dir / "dumps" / "dev")
    targets = dev if args.split == "dev" else calib.load(results_dir / "dumps" / args.split)
    calib.set_stage5_globals(cfg["eta"], cfg["m0"], cfg["t_max"])
    for course in dev:
        calib.set_background(
            course, calib.background_ratios(course, [o for o in dev if o is not course])
        )
    weights = stage5_weights(args.weights, dev, cfg["p0"])
    print("weights:", {k: round(v, 4) for k, v in weights.items()})
    rows = []
    for course in targets:
        others = [o for o in dev if o["course"] != course["course"]]
        ratios = None if args.t_off else calib.background_ratios(course, others)
        calib.set_background(course, ratios)
        imp = calib.importance(course, weights, cfg["p0"], cfg["combine"])
        metrics = calib.selection_metrics(
            course, imp, anchor=cfg["anchor"], importance_weight=cfg["lambda"],
            threshold=cfg["tau"], margin=cfg["eps"], variant=cfg["variant"],
        )  # fmt: skip
        row = {
            "course": course["course"],
            "ndcg": calib.ndcg_at(imp, course["label"]),
            "ndcg_relaxed": calib.ndcg_at(imp, course["label_relaxed"]),
            **metrics,
        }
        rows.append(row)
        shown = {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}
        print(course["course"], f"nDCG {row['ndcg']:.3f} / {row['ndcg_relaxed']:.3f}", shown)
    for metric in MACRO_METRICS:
        mean, low, high = calib.bootstrap_ci([r[metric] for r in rows])
        print(f"macro {metric}: {mean:.3f} [{low:.3f}, {high:.3f}]")
    suffix = "_t_off" if args.t_off else ""
    name = f"final_{args.split}_{args.config.stem}_{args.weights}{suffix}.json"
    out = results_dir / "runs" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"weights": weights, "config": cfg, "rows": rows}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
