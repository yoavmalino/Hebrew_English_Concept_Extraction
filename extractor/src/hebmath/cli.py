"""Command line entry point.

    hebmath INPUT [-o OUTPUT.txt] [--report REPORT.json | --out-dir DIR] [--no-ocr] [--page-markers]

Exit status: 0 on success, 2 when the file could not be extracted (the
reason is printed to stderr and written to the report), 1 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .pipeline import Extractor, ExtractorConfig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hebmath", description="Extract Hebrew text + LaTeX math from PDF/DOCX"
    )
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="output .txt (default: stdout)")
    ap.add_argument("--report", type=Path, help="write a JSON report (status, warnings, stats)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        help="write DIR/<input name>/<input name>.txt and DIR/<input name>/report.json "
        "(the folders are created if missing)",
    )
    ap.add_argument("--no-ocr", action="store_true", help="never run OCR on scanned pages")
    ap.add_argument(
        "--no-plain-math",
        action="store_true",
        help="do not convert formulas typed as plain characters",
    )
    ap.add_argument(
        "--page-markers", action="store_true", help="emit [[hebmath:page=N]] between PDF pages"
    )
    ap.add_argument(
        "--math-line-breaks",
        choices=["formulas", "all", "none"],
        default="formulas",
        help="put formulas on their own line: every formula longer than one symbol "
        "(default), every formula, or none (format v1)",
    )
    ap.add_argument("--version", action="version", version=f"hebmath {__version__}")
    args = ap.parse_args(argv)
    if args.out_dir:
        if args.output or args.report:
            ap.error("--out-dir cannot be combined with -o/--report")
        folder = args.out_dir / args.input.stem
        args.output = folder / f"{args.input.stem}.txt"
        args.report = folder / "report.json"
    for target in (args.output, args.report):
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)

    cfg = ExtractorConfig(
        enable_ocr=not args.no_ocr,
        detect_plain_math=not args.no_plain_math,
        page_markers=args.page_markers,
        math_line_breaks=args.math_line_breaks,
    )
    result = Extractor(cfg).extract_path(args.input)
    if args.report:
        args.report.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not result.ok:
        print(f"hebmath: {result.error_code}: {result.error_message}", file=sys.stderr)
        return 2
    for w in result.warnings:
        print(f"hebmath: warning: {w}", file=sys.stderr)
    if args.output:
        args.output.write_text(result.text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(result.text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
