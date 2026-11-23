"""Paired per-course comparisons of stage-5 settings against the design's opening values.

Usage: python calibration/paired.py [--data-dir DIR]
Reads runs/s5_strict.json and runs/s5_relaxed.json (from `s5_run.py`). For each change, the
per-course nDCG@50 differences are bootstrapped; "+" marks an interval above zero, "-" below.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_dir  # noqa: E402

# (scheme, p0, t_max, eta, m0, combine): the design's opening values.
BASELINE = ("opening", 0.0625, 6.0, 0.9, 5, "max")
# Each change replaces fields of the baseline, by position.
CHANGES = {
    "T off": {2: None},
    "t_max 3": {2: 3.0},
    "t_max 4": {2: 4.0},
    "t_max 8": {2: 8.0},
    "eta 1.0": {3: 1.0},
    "eta 0.8": {3: 0.8},
    "m0 3": {4: 3},
    "m0 8": {4: 8},
    "soft_or": {5: "soft_or"},
    "p0 1/32": {1: 1 / 32},
    "p0 1/8": {1: 0.125},
    "lr_relevant": {0: "lr_relevant"},
    "lr_must": {0: "lr_must"},
    "equal": {0: "equal"},
    "lr_relevant+t4+eta1": {0: "lr_relevant", 2: 4.0, 3: 1.0},
    "opening+t4+eta1": {2: 4.0, 3: 1.0},
    "lr_relevant+t4+eta1+m0 3": {0: "lr_relevant", 2: 4.0, 3: 1.0, 4: 3},
}


def bootstrap(diffs: list[float], rounds: int = 4000, seed: int = 11) -> tuple[float, ...]:
    """Return (mean, low, high) of the mean difference with a 95% interval."""
    rng = random.Random(seed)
    count = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(count)) / count for _ in range(rounds))
    return sum(diffs) / count, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def _key(row: dict) -> tuple:
    return (row["scheme"], row["p0"], row["t_max"], row["eta"], row["m0"], row["combine"])


def main() -> None:
    """Print the paired comparison table for the strict and relaxed labels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_dir.add_data_dir_argument(parser)
    runs = (
        data_dir.resolve_data_dir(parser.parse_args().data_dir) / data_dir.RESULTS_FOLDER / "runs"
    )
    for label in ("strict", "relaxed"):
        grid = json.loads((runs / f"s5_{label}.json").read_text(encoding="utf-8"))["results"]
        rows = {_key(row): row for row in grid}
        base = rows[BASELINE]
        print("==", label, "baseline", round(base["ndcg"], 4))
        for name, change in CHANGES.items():
            fields = list(BASELINE)
            for position, value in change.items():
                fields[position] = value
            row = rows[tuple(fields)]
            diffs = [row["by_course"][c] - base["by_course"][c] for c in base["by_course"]]
            mean, low, high = bootstrap(diffs)
            flag = "+" if low > 0 else "-" if high < 0 else " "
            wins = sum(d > 0 for d in diffs)
            print(
                f"  {flag} {name:26} {row['ndcg']:.4f}  diff {mean:+.4f} "
                f"[{low:+.4f}, {high:+.4f}]  wins {wins}/{len(diffs)}"
            )


if __name__ == "__main__":
    main()
