"""Report pipeline stage 2 (candidates) on a set of documents: pool size, sources and gate checks.

Dev tool, not part of the app. It needs no gold: it prints, per file, the words, the number of
candidates and candidates per 1,000 words (the pool-size bound), how many each source proposed,
and the algorithmic gate (counts add up, mentions inside one unit, idempotent keys, and the same
output on a second run). Exit code 1 when a gate check fails.

    python scripts/stage2_candidates_report.py FILE_OR_FOLDER ... [--snapshots DIR]
        [--min-count N]

Folders are searched recursively for .pdf and .docx. Keep held-out courses out of any tuning run.
"""

import argparse
import json
import sys
from pathlib import Path


from hebmath import Extractor, ExtractorConfig  # noqa: E402

from concept_pipeline.stage2_candidates import (  # noqa: E402
    MIN_PHRASE_COUNT,
    SOURCES,
    check_candidates,
    extract_candidates,
    to_snapshot,
)

SUFFIXES = (".pdf", ".docx")


def _files(paths: list[str]) -> list[Path]:
    """Return the documents named by the arguments, folders expanded, in a stable order."""
    found: list[Path] = []
    for name in paths:
        path = Path(name).expanduser()
        if path.is_dir():
            found += sorted(p for p in path.rglob("*") if p.suffix.lower() in SUFFIXES)
        elif path.suffix.lower() in SUFFIXES:
            found.append(path)
    return found


def _report_file(
    extractor: Extractor, path: Path, snapshots: Path | None, min_count: int
) -> tuple[str, bool]:
    """Run stage 2 on one file; return its report line and whether the gate checks passed."""
    result = extractor.extract_path(path)
    if not result.ok or result.structure is None:
        return f"{path.name[:44]:44}  extraction failed: {result.error_code}", True
    first = extract_candidates(result, min_count)
    problems = check_candidates(first, result)
    snapshot = to_snapshot(first)
    if to_snapshot(extract_candidates(result, min_count)) != snapshot:
        problems.append("a second run gave a different snapshot")
    if snapshots is not None:
        snapshots.mkdir(parents=True, exist_ok=True)
        target = snapshots / f"{path.stem}.stage2.json"
        target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
    per_thousand = 1000 * len(first.candidates) / max(first.words, 1)
    by_source = [sum(1 for c in first.candidates if s in c.sources) for s in SOURCES]
    line = (
        f"{path.name[:44]:44} {first.words:7} {len(first.candidates):6} {per_thousand:7.1f} "
        + " ".join(f"{n:6}" for n in by_source)
        + f"  {len(problems)} problems"
    )
    for problem in problems[:5]:
        line += f"\n    {problem}"
    return line, not problems


def main() -> int:
    """Parse the arguments, report every file, and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--snapshots", type=Path, help="write each file's stage-2 JSON here")
    parser.add_argument(
        "--min-count", type=int, default=MIN_PHRASE_COUNT, help="n-gram frequency threshold"
    )
    args = parser.parse_args()
    extractor = Extractor(ExtractorConfig(enable_ocr=False, recovery_ocr_pages=0))
    header = f"{'file':44} {'words':>7} {'cands':>6} {'per1k':>7} " + " ".join(
        f"{s[:6]:>6}" for s in SOURCES
    )
    print(header)
    passed = True
    for path in _files(args.paths):
        line, ok = _report_file(extractor, path, args.snapshots, args.min_count)
        print(line, flush=True)
        passed = passed and ok
    print("gate: passed" if passed else "gate: FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
