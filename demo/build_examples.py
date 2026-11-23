"""Precompute the static explorer's examples: one JSON per document, its original, and an index.

Usage: python demo/build_examples.py --out demo/web/data [--generated] FILE[=TITLE] ...
Each FILE is analyzed exactly as the demo API would (`demo/api/analysis.py`: algorithmic mode,
T off). The per-stage summary goes to `<out>/<id>.json` and a copy of the document to
`<out>/../originals/<id>.<ext>`, so the page can show the original beside the extracted text.
The published examples are generated documents (course summaries written by a model for
testing), not real course material.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "api"))

from analysis import analyze  # noqa: E402

_SLUG_RE = re.compile(r"[^a-z0-9]+")
ORIGINALS_FOLDER = "originals"


def slug(path: Path) -> str:
    """Return a file-name-safe identifier for an example, from its file name."""
    return _SLUG_RE.sub("-", path.name.lower()).strip("-")


def main() -> None:
    """Analyze every input file, copy it next to the page, and write its JSON and the index."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--generated",
        action="store_true",
        help="mark the examples as model-written documents; the page then says so",
    )
    parser.add_argument("files", nargs="+", help="FILE or FILE=TITLE")
    args = parser.parse_args()
    originals = args.out.parent / ORIGINALS_FOLDER
    args.out.mkdir(parents=True, exist_ok=True)
    originals.mkdir(parents=True, exist_ok=True)
    index = []
    for item in args.files:
        raw_path, _, title = item.partition("=")
        path = Path(raw_path)
        summary = analyze(path.read_bytes(), filename=path.name)
        example_id = slug(path)
        target = args.out / f"{example_id}.json"
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", "utf-8")
        original_name = f"{example_id}{path.suffix.lower()}"
        shutil.copyfile(path, originals / original_name)
        index.append(
            {
                "id": example_id,
                "title": title or path.stem,
                "format": summary["stage1"]["format"],
                "original": f"{ORIGINALS_FOLDER}/{original_name}",
                "generated": args.generated,
            }
        )
        print(example_id, summary["stage6"]["report"], f"{target.stat().st_size:,} bytes")
    (args.out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
