"""Measure and tune pipeline stage 1 (heading detection) against a reviewed corpus.

The corpus is a folder of PDF and DOCX files (keep it outside the repository: the files are
private study material), each with a sidecar ``<file name>.gold.json``::

    {"reviewed": true, "headings": [{"text": "...", "level": 1, "page": 3}, ...]}

``page`` is 1-based (0 or missing for Word files). ``level`` 1 is a main heading: what the
document's own table of contents or bookmarks would list. Titles and table-of-contents lines are
not headings.

Commands (run with the package installed):

    python calibrate_stage1.py sheets CORPUS --review-dir DIR   write one review sheet per file
    python calibrate_stage1.py import CORPUS --review-dir DIR --gold-dir DIR   reviewed sheets -> gold
    python calibrate_stage1.py init   CORPUS   write draft gold files from what stage 1 finds now
    python calibrate_stage1.py report CORPUS   recall and precision per file and in total
    python calibrate_stage1.py sweep  CORPUS   one-parameter sweeps with the plateau rule
    python calibrate_stage1.py invariants CORPUS   stage-1 gate invariants on every file (no gold)
    python calibrate_stage1.py units  CORPUS   unit split per file: counts by type (report only)

Options: ``--gold-dir DIR`` keeps the gold files in a separate folder (mirroring the corpus
sub-folders) so the private files stay untouched; ``--only TEXT`` uses only files whose path
contains TEXT. The corpus folder is searched recursively.

A review sheet is the easy way to label: it lists the main headings stage 1 found, one per line as
``page | text``. The reviewer deletes wrong lines, adds missing ones in the same form, and writes
``REVIEWED`` as the first line; ``import`` turns finished sheets into gold files (level 1 for every
line). Sub-headings are not scored, so the sheets ask only about main headings.

``init`` only drafts: a draft has ``"reviewed": false`` and is ignored until a person corrects it
and sets it to true. Calibrating on stage 1's own output would only confirm it.

Success criteria are the design's (docs/concept_graph_pipeline_design.md, stage 1): recall of
main headings at least 0.90, precision at least 0.85. A heading is found when its text matches
without spaces and punctuation and it starts on the same page.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from hebmath import Extractor, ExtractorConfig
from hebmath import structure as structure_module
from hebmath.invariants import check_invariants
from hebmath.model import Document
from hebmath.pipeline import sniff_format
from hebmath.serialize import serialize_with_spans
from hebmath.structure import StructureInfo, build_structure

RECALL_TARGET = 0.90  # design stage 1: main headings found
PRECISION_TARGET = 0.85  # design stage 1 addition: false headings break the sections
BOOTSTRAP_SAMPLES = 300  # resamples of the files when estimating how noisy a score is
BOOTSTRAP_SEED = 20261113  # fixed so the same corpus gives the same plateau
MAIN_LEVEL = 1
MIN_PARTIAL_CHARS = 6  # a shorter heading is too generic to call a partial match
SUFFIXES = (".pdf", ".docx")
REVIEWED_MARK = "REVIEWED"  # first line of a sheet the reviewer has finished
SHEET_HELP = (
    "# כל שורה היא כותרת ראשית (כמו שהייתה מופיעה בתוכן עניינים): עמוד | טקסט",
    "# מחקו שורות שאינן כותרת ראשית. הוסיפו בצורה הזאת כותרות ראשיות שחסרות (עמוד 0 אם לא ידוע).",
    "# כשסיימתם, החליפו את השורה הראשונה במילה REVIEWED. שורות שמתחילות ב-# מתעלמים מהן.",
)
_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)

# Parameters swept one at a time (module constants of hebmath.structure) and their candidates.
SWEEPS: dict[str, list[float]] = {
    "LARGE_SIZE_RATIO": [1.05, 1.10, 1.15, 1.20, 1.30, 1.40],
    "HEADING_FORMAT_SHARE": [0.5, 0.65, 0.8, 0.9, 1.0],
    "MAJORITY": [0.3, 0.4, 0.5, 0.6, 0.7],
    "MAX_HEADING_CHARS": [60, 90, 120, 160, 200],
    "TITLE_SIZE_RATIO": [1.3, 1.45, 1.6, 1.8, 2.0],
    "BOLD_RATIO": [1.2, 1.3, 1.4, 1.5, 1.6],
}


@dataclass(frozen=True)
class Heading:
    """A heading as (normalized text, level, page)."""

    key: str
    level: int
    page: int


@dataclass
class Case:
    """One file of the corpus: its parsed document, the output text and the gold headings."""

    name: str
    document: Document
    text: str
    spans: list[tuple[int, int, int]]
    gold: list[Heading]


def normalize(text: str) -> str:
    """Return text lowercased without spaces and punctuation."""
    return _NORMALIZE_RE.sub("", text).lower()


def predicted_headings(text: str, info: StructureInfo) -> list[Heading]:
    """Return the headings stage 1 found, in document order."""
    return [
        Heading(normalize(text[b.start : b.end]), b.heading_level, b.page)
        for b in info.blocks
        if b.role == "heading"
    ]


def same_heading(a: Heading, b: Heading) -> bool:
    """Return True when two headings match: same text, same page (a page of 0 matches any)."""
    return a.key == b.key and (not a.page or not b.page or a.page == b.page)


def score_main(pred: list[Heading], gold: list[Heading]) -> tuple[int, int, int]:
    """Count (hits, predicted, gold) for main headings: right text, page and level 1."""
    pred_main = [h for h in pred if h.level == MAIN_LEVEL]
    gold_main = [h for h in gold if h.level == MAIN_LEVEL]
    hits = sum(1 for g in gold_main if any(same_heading(p, g) for p in pred_main))
    return hits, len(pred_main), len(gold_main)


def partial_matches(pred: list[Heading], gold: list[Heading]) -> int:
    """Count gold main headings that are not found but overlap a predicted one on the same page.

    These are boundary errors (a wrapped heading cut in two, a heading with an extra word), not
    headings missed outright. They still count as a miss and a false positive in the scores;
    this number only tells which kind of error it was.
    """
    pred_main = [h for h in pred if h.level == MAIN_LEVEL]
    count = 0
    for g in (h for h in gold if h.level == MAIN_LEVEL):
        if any(same_heading(p, g) for p in pred_main):
            continue
        if any(
            (not p.page or not g.page or p.page == g.page)
            and min(len(p.key), len(g.key)) >= MIN_PARTIAL_CHARS
            and (p.key in g.key or g.key in p.key)
            for p in pred_main
        ):
            count += 1
    return count


def ratios(hits: int, predicted: int, gold: int) -> tuple[float, float, float]:
    """Return (recall, precision, f1); an empty denominator counts as 1.0 for that ratio."""
    recall = hits / gold if gold else 1.0
    precision = hits / predicted if predicted else 1.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return recall, precision, f1


def gold_path(path: Path, folder: Path, gold_dir: Path | None) -> Path:
    """Return where the gold file of a corpus file lives (beside it, or mirrored in gold_dir)."""
    if gold_dir is None:
        return path.with_name(path.name + ".gold.json")
    return gold_dir / path.relative_to(folder).with_name(path.name + ".gold.json")


def load_case(path: Path, sidecar: Path) -> Case | None:
    """Read a corpus file and its reviewed gold; None when there is no reviewed gold."""
    if not sidecar.exists():
        return None
    data = json.loads(sidecar.read_text(encoding="utf8"))
    if not data.get("reviewed"):
        return None
    gold = [
        Heading(normalize(h["text"]), int(h.get("level", 1)), int(h.get("page", 0)))
        for h in data.get("headings", [])
    ]
    document, text, spans = parse(path)
    return Case(path.name, document, text, spans, gold)


def parse(path: Path) -> tuple[Document, str, list[tuple[int, int, int]]]:
    """Run the reader once, so thresholds can be changed without reading the file again."""
    data = path.read_bytes()
    extractor = Extractor(ExtractorConfig(enable_ocr=False, recovery_ocr_pages=0))
    document = extractor._reader(sniff_format(data, path.name))(data)
    text, spans = serialize_with_spans(document)
    return document, text, spans


def corpus_files(folder: Path, only: str = "") -> list[Path]:
    """Return the PDF and Word files under the corpus folder, sorted, filtered by ``only``."""
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUFFIXES and only in str(p.relative_to(folder))
    )


def reviewed_cases(folder: Path, gold_dir: Path | None, only: str) -> list[Case]:
    """Load every file that has reviewed gold."""
    cases = []
    for path in corpus_files(folder, only):
        case = load_case(path, gold_path(path, folder, gold_dir))
        if case:
            cases.append(case)
    return cases


def evaluate(cases: list[Case]) -> list[tuple[str, tuple[int, int, int]]]:
    """Score every case with the current thresholds."""
    results = []
    for case in cases:
        info = build_structure(case.document, case.text, case.spans)
        results.append((case.name, score_main(predicted_headings(case.text, info), case.gold)))
    return results


def evaluate_partials(cases: list[Case]) -> dict[str, int]:
    """Count the partial matches of every case with the current thresholds."""
    found = {}
    for case in cases:
        info = build_structure(case.document, case.text, case.spans)
        found[case.name] = partial_matches(predicted_headings(case.text, info), case.gold)
    return found


def total_f1(scored: list[tuple[str, tuple[int, int, int]]]) -> float:
    """Return the micro F1 over all files."""
    hits = sum(s[1][0] for s in scored)
    predicted = sum(s[1][1] for s in scored)
    gold = sum(s[1][2] for s in scored)
    return ratios(hits, predicted, gold)[2]


def bootstrap_halfwidth(scored: list[tuple[str, tuple[int, int, int]]]) -> float:
    """Return half the width of the 90% interval of the total F1 when files are resampled."""
    rng = random.Random(BOOTSTRAP_SEED)
    scores = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [rng.choice(scored) for _ in scored]
        scores.append(total_f1(sample))
    scores.sort()
    low = scores[int(0.05 * len(scores))]
    high = scores[int(0.95 * len(scores)) - 1]
    return (high - low) / 2


def plateau(values: list[float], f1s: list[float], halfwidth: float) -> float:
    """Pick the middle of the contiguous run of values whose F1 is within the noise of the best.

    The design asks for a stable region, not the sharpest point: the argmax of a noisy curve is
    often the edge of a cliff.
    """
    best = max(range(len(f1s)), key=lambda i: (f1s[i], -abs(i - len(f1s) // 2)))
    low = high = best
    while low > 0 and f1s[low - 1] >= f1s[best] - halfwidth:
        low -= 1
    while high < len(f1s) - 1 and f1s[high + 1] >= f1s[best] - halfwidth:
        high += 1
    return values[(low + high) // 2]


def sheet_path(path: Path, folder: Path, review_dir: Path) -> Path:
    """Return where the review sheet of a corpus file lives (mirroring the corpus sub-folders)."""
    return review_dir / path.relative_to(folder).with_name(path.name + ".txt")


def run_sheets(folder: Path, review_dir: Path, only: str) -> None:
    """Write a review sheet of the predicted main headings for each file without one."""
    for path in corpus_files(folder, only):
        sheet = sheet_path(path, folder, review_dir)
        if sheet.exists():
            continue
        sheet.parent.mkdir(parents=True, exist_ok=True)
        document, text, spans = parse(path)
        info = build_structure(document, text, spans)
        lines = ["UNREVIEWED", *SHEET_HELP]
        for block in info.blocks:
            if block.role == "heading" and block.heading_level == MAIN_LEVEL:
                title = " ".join(text[block.start : block.end].split())
                lines.append(f"{block.page} | {title}")
        sheet.write_text("\n".join(lines) + "\n", encoding="utf8")
        print(f"sheet written: {sheet.name} ({len(lines) - 1 - len(SHEET_HELP)} main headings)")


def read_sheet(sheet: Path) -> list[dict] | None:
    """Parse a finished review sheet into gold headings; None while it is not marked REVIEWED."""
    lines = sheet.read_text(encoding="utf8").splitlines()
    if not lines or lines[0].strip() != REVIEWED_MARK:
        return None
    headings = []
    for line in lines[1:]:
        if not line.strip() or line.lstrip().startswith("#") or "|" not in line:
            continue
        page, title = line.split("|", 1)
        headings.append(
            {"text": title.strip(), "level": MAIN_LEVEL, "page": int(page.strip() or 0)}
        )
    return headings


def run_import(folder: Path, review_dir: Path, gold_dir: Path, only: str) -> None:
    """Turn finished review sheets into reviewed gold files (replacing an older gold file)."""
    for path in corpus_files(folder, only):
        sheet = sheet_path(path, folder, review_dir)
        if not sheet.exists():
            continue
        headings = read_sheet(sheet)
        if headings is None:
            print(f"not reviewed yet: {sheet.name}")
            continue
        target = gold_path(path, folder, gold_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"reviewed": True, "headings": headings}, ensure_ascii=False, indent=2),
            encoding="utf8",
        )
        print(f"gold written: {target.name} ({len(headings)} main headings)")


def run_init(folder: Path, gold_dir: Path | None, only: str) -> None:
    """Write draft gold files for files that have none; never overwrite an existing one."""
    for path in corpus_files(folder, only):
        sidecar = gold_path(path, folder, gold_dir)
        if sidecar.exists():
            continue
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        document, text, spans = parse(path)
        info = build_structure(document, text, spans)
        headings = [
            {
                "text": text[b.start : b.end].strip(),
                "level": b.heading_level,
                "page": b.page,
                "source": next((s.source for s in info.sections if s.start == b.start), "format"),
            }
            for b in info.blocks
            if b.role == "heading"
        ]
        draft = {"reviewed": False, "headings": headings}
        sidecar.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"draft written: {sidecar.name} ({len(headings)} headings)")


def run_report(folder: Path, gold_dir: Path | None, only: str) -> int:
    """Print recall and precision of main headings per file and in total; return the exit code."""
    cases = reviewed_cases(folder, gold_dir, only)
    if not cases:
        print('no reviewed gold files (set "reviewed": true after correcting a draft)')
        return 2
    scored = evaluate(cases)
    partials = evaluate_partials(cases)
    print(f"{'file':40} {'gold':>5} {'found':>5} {'recall':>7} {'precision':>9} {'partial':>8}")
    for name, (hits, predicted, gold) in scored:
        recall, precision, _ = ratios(hits, predicted, gold)
        print(
            f"{name[:40]:40} {gold:5d} {predicted:5d} {recall:7.2f} {precision:9.2f}"
            f" {partials[name]:8d}"
        )
    hits = sum(s[1][0] for s in scored)
    predicted = sum(s[1][1] for s in scored)
    gold = sum(s[1][2] for s in scored)
    recall, precision, f1 = ratios(hits, predicted, gold)
    print(
        f"{'TOTAL (' + str(len(cases)) + ' files)':40} {gold:5d} {predicted:5d} {recall:7.2f} {precision:9.2f}"
    )
    partial = sum(partials.values())
    lenient_recall, lenient_precision, _ = ratios(hits + partial, predicted, gold)
    print(f"partial matches (boundary errors, counted as misses above): {partial}")
    print(
        f"lenient (a partial match counts as found): recall {lenient_recall:.2f}, "
        f"precision {lenient_precision:.2f}; the targets below use the strict numbers"
    )
    verdict = "PASS" if recall >= RECALL_TARGET and precision >= PRECISION_TARGET else "FAIL"
    print(
        f"targets: recall >= {RECALL_TARGET}, precision >= {PRECISION_TARGET}: {verdict} (f1 {f1:.3f})"
    )
    return 0 if verdict == "PASS" else 1


def run_sweep(folder: Path, gold_dir: Path | None, only: str) -> int:
    """Sweep each parameter alone and print the F1 curve with the plateau choice."""
    cases = reviewed_cases(folder, gold_dir, only)
    if not cases:
        print("no reviewed gold files")
        return 2
    base = evaluate(cases)
    halfwidth = bootstrap_halfwidth(base)
    print(
        f"current thresholds: f1 {total_f1(base):.3f}, noise (90% interval half-width) {halfwidth:.3f}"
    )
    for name, values in SWEEPS.items():
        original = getattr(structure_module, name)
        f1s = []
        try:
            for value in values:
                setattr(structure_module, name, value)
                f1s.append(total_f1(evaluate(cases)))
        finally:
            setattr(structure_module, name, original)
        chosen = plateau(values, f1s, halfwidth)
        curve = "  ".join(f"{v}:{f:.3f}" for v, f in zip(values, f1s))
        print(f"{name} (now {original}) -> plateau {chosen}\n    {curve}")
    print(
        "These are one-parameter curves. Change a constant only when the plateau is clearly "
        "different from the current value and the corpus has at least ~140 main headings "
        "(design: +-5% at 90%)."
    )
    return 0


def _extract_file(path: Path):
    """Extract one corpus file the way the server does (no OCR)."""
    extractor = Extractor(ExtractorConfig(enable_ocr=False, recovery_ocr_pages=0))
    return extractor.extract_bytes(path.read_bytes(), filename=path.name)


def run_invariants(folder: Path, only: str) -> int:
    """Print every file that breaks a stage-1 invariant; return 1 when any does."""
    checked = failing = 0
    for path in corpus_files(folder, only):
        result = _extract_file(path)
        if not result.ok or result.structure is None:
            print(f"skipped (not extracted: {result.error_code}): {path.relative_to(folder)}")
            continue
        checked += 1
        problems = check_invariants(result)
        if problems:
            failing += 1
            print(f"FAIL {path.relative_to(folder)}")
            for problem in problems:
                print(f"    {problem}")
    print(f"invariants: {checked} files checked, {failing} failing")
    return 1 if failing else 0


def run_units(folder: Path, only: str) -> int:
    """Print the unit split per file: units by type, unit labels and named units."""
    types = list(structure_module.UNIT_WEIGHTS)
    totals = dict.fromkeys(types + ["labels", "named"], 0)
    print(f"{'file':40} " + " ".join(f"{t[:6]:>6}" for t in types) + f" {'labels':>6} {'named':>6}")
    for path in corpus_files(folder, only):
        result = _extract_file(path)
        if not result.ok or result.structure is None:
            continue
        info = result.structure
        row = {t: sum(1 for u in info.units if u.type == t) for t in types}
        row["labels"] = sum(1 for b in info.blocks if b.role == "unit_label")
        row["named"] = sum(1 for b in info.blocks if b.named_unit)
        for key, value in row.items():
            totals[key] += value
        name = str(path.relative_to(folder))[:40]
        print(f"{name:40} " + " ".join(f"{row[k]:6d}" for k in types + ["labels", "named"]))
    print(f"{'TOTAL':40} " + " ".join(f"{totals[k]:6d}" for k in types + ["labels", "named"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "command",
        choices=("sheets", "import", "init", "report", "sweep", "invariants", "units"),
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--only", default="")
    parser.add_argument("--review-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command in ("sheets", "import"):
        if args.review_dir is None or (args.command == "import" and args.gold_dir is None):
            parser.error("sheets needs --review-dir; import needs --review-dir and --gold-dir")
        if args.command == "sheets":
            run_sheets(args.corpus, args.review_dir, args.only)
        else:
            run_import(args.corpus, args.review_dir, args.gold_dir, args.only)
        return 0
    if args.command == "init":
        run_init(args.corpus, args.gold_dir, args.only)
        return 0
    if args.command == "invariants":
        return run_invariants(args.corpus, args.only)
    if args.command == "units":
        return run_units(args.corpus, args.only)
    run = run_report if args.command == "report" else run_sweep
    return run(args.corpus, args.gold_dir, args.only)


if __name__ == "__main__":
    sys.exit(main())
