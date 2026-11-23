"""Locate the private calibration data: the --data-dir option, else TACTIMIND_DATA_DIR.

The corpus, the committee gold and the model caches are not part of this repository. This module
only resolves where they are on the machine running calibration, and stops with a clear message
when neither the option nor the environment variable says.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DATA_DIR_ENV = "TACTIMIND_DATA_DIR"
# Folder layout inside the data folder (the owner's corpus; any corpus laid out alike works).
COURSES_FOLDER = "training_set"
GOLD_FOLDERS = {"dev": "training_set_gold", "heldout": "training_set_heldout_gold"}
GOLD_FILE = "concepts.gold.json"
RESULTS_FOLDER = "stage56_results"
SPLITS = tuple(GOLD_FOLDERS)


def add_data_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Add the --data-dir option to a calibration script's parser."""
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=f"the folder with the corpus and gold (default: ${DATA_DIR_ENV})",
    )


def resolve_data_dir(value: Path | None) -> Path:
    """Return the data folder from the option or the environment; exit when neither is set.

    Args:
        value: The --data-dir value, or None when the option was not given.

    Returns:
        The data folder, which exists.
    """
    raw = value or (Path(os.environ[DATA_DIR_ENV]) if os.environ.get(DATA_DIR_ENV) else None)
    if raw is None:
        sys.exit(f"no data folder: pass --data-dir or set {DATA_DIR_ENV}")
    folder = raw.expanduser()
    if not folder.is_dir():
        sys.exit(f"the data folder {folder} does not exist")
    return folder


def gold_dir(data: Path, split: str) -> Path:
    """Return the gold folder of a split ("dev" or "heldout")."""
    return data / GOLD_FOLDERS[split]


def course_names(data: Path, split: str) -> list[str]:
    """Return the courses of a split: gold folders that hold a gold file, sorted by name.

    Folder names may end with a space (as the corpus was saved), so names are returned stripped
    and matched stripped by `course_dir`.
    """
    folders = [
        p for p in gold_dir(data, split).iterdir() if p.is_dir() and not p.name.startswith("_")
    ]
    return sorted(p.name.strip() for p in folders if (p / GOLD_FILE).is_file())


def course_dir(parent: Path, name: str) -> Path:
    """Return the folder under `parent` whose stripped name is `name`; exit when there is none."""
    for folder in parent.iterdir():
        if folder.is_dir() and folder.name.strip() == name:
            return folder
    sys.exit(f"no folder named {name!r} in {parent}")
