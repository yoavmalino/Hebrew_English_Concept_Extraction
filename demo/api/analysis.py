"""Run stages 1-6 on one document in algorithmic mode and summarize each stage as JSON.

Shared by the stateless demo API (`app.py`) and the script that precomputes the static
explorer's examples (`demo/build_examples.py`). No model is called and T is off (there is no
background corpus in the demo), so the output shows what the algorithm alone does. Nothing is
stored or logged: the input bytes live only for the call.
"""

from __future__ import annotations

import threading
from collections import Counter
from importlib.metadata import PackageNotFoundError, version

from hebmath import Extractor, ExtractorConfig
from hebmath.safety import Limits

from concept_pipeline import stage5_importance as s5
from concept_pipeline import stage6_selection as s6
from concept_pipeline.stage2_candidates import extract_candidates
from concept_pipeline.stage4_merging import absorb_fragments, decide_by_rule, find_pairs, merge

# Demo limits: small enough for a free instance, large enough for a course summary.
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 150
# How much of each stage the response shows (the full lists can hold thousands of entries).
TOP_CANDIDATES = 40
TOP_IMPORTANCE = 40
MAX_SECTIONS = 60
MAX_RULE_MERGES = 30
# The whole extracted text is returned so the page can show it; the cap only keeps a huge upload
# from producing a response the browser struggles with (a 150-page summary is ~400k characters).
MAX_TEXT_CHARS = 500_000
DEFINITION_CHARS = 220
# A passage must add at least this many characters to the name to be worth showing.
MIN_PASSAGE_EXTRA = 15
SIGNAL_DIGITS = 3

_EXTRACTOR = Extractor(
    ExtractorConfig(
        limits=Limits(max_file_bytes=MAX_FILE_BYTES, max_pdf_pages=MAX_PDF_PAGES),
        # No Tesseract on the demo host: scanned pages are rejected, not read.
        enable_ocr=False,
    )
)
# PDFium is not thread-safe, so one extraction runs at a time (as in the private backend).
_EXTRACT_LOCK = threading.Lock()


class AnalysisError(Exception):
    """The document could not be analyzed; `code` is a stable hebmath error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _package_version(name: str) -> str:
    """Return an installed package's version, or "dev" when it runs from a source tree."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "dev"


def _stage1(result) -> dict:
    """Summarize the extraction: the full text, sections, units and formatting signal shares."""
    info = result.structure
    return {
        "format": result.format,
        "characters": len(result.text),
        "text": result.text[:MAX_TEXT_CHARS],
        "text_truncated": len(result.text) > MAX_TEXT_CHARS,
        "sections": [{"title": s.title, "level": s.level} for s in info.sections[:MAX_SECTIONS]],
        "section_count": len(info.sections),
        "units": dict(Counter(unit.type for unit in info.units)),
        "signal_share": {k: round(v, SIGNAL_DIGITS) for k, v in info.signal_share.items()},
        "warnings": list(result.warnings),
    }


def _stage2(stage2) -> dict:
    """Summarize the candidates: counts and the most mentioned ones with their sources."""
    top = sorted(stage2.candidates, key=lambda c: (-c.count, c.key))[:TOP_CANDIDATES]
    return {
        "candidates": len(stage2.candidates),
        "words": stage2.words,
        "language": stage2.language,
        "top": [{"name": c.form, "count": c.count, "sources": c.sources} for c in top],
    }


def _stage4(pairs, rule_decisions, undecided, concepts, absorbed) -> dict:
    """Summarize merging: pairs per finder, merges decided by rule, and what stayed apart."""
    by_finder = Counter(finder for pair in pairs for finder in pair.finders)
    merged = [[d.first, d.second] for d in rule_decisions if d.is_same]
    return {
        "pairs": len(pairs),
        "pairs_by_finder": dict(sorted(by_finder.items())),
        "merged_by_rule": merged[:MAX_RULE_MERGES],
        "merged_by_rule_count": len(merged),
        # Without a model decider these pairs stay two concepts (the safe side).
        "left_apart": len(undecided),
        "fragments_absorbed": len(absorbed),
        "concepts": len(concepts),
    }


def _stage5(evidence, importance: dict, names: dict) -> dict:
    """Summarize importance: the top concepts with their score and their eight signals."""
    by_key = {concept.key: concept for concept in evidence.concepts}
    top = sorted(importance, key=lambda key: (-importance[key], key))[:TOP_IMPORTANCE]
    rows = []
    for key in top:
        values = s5.signal_values(by_key[key])
        rows.append(
            {
                "name": names[key],
                "importance": round(importance[key], SIGNAL_DIGITS),
                "signals": {s: round(values[s], SIGNAL_DIGITS) for s in s5.SIGNALS if s != "T"},
            }
        )
    return {"t_signal": "off", "weights": s5.CALIBRATED_WEIGHTS, "top": rows}


def _stage6(result, concepts, evidence, importance: dict, names: dict) -> dict:
    """Summarize selection: the chosen nodes in order of entry, with a defining passage each."""
    space = s6.space_concepts([evidence], importance)
    anchored = {concept.key for concept in space if concept.is_anchored}
    selection = s6.select_nodes(space)
    dictionary = s6.mention_dictionary(selection.nodes, [(result.sha256, result, concepts)])
    nodes = []
    for key in selection.nodes:
        entries = dictionary.get(key, [])
        # The top entry is often the heading that only repeats the name; show the first one
        # that says more, else the top one.
        telling = [e for e in entries if len(e.text) > len(names[key]) + MIN_PASSAGE_EXTRA]
        first = (telling or entries or [None])[0]
        nodes.append(
            {
                "name": names[key],
                "importance": round(importance[key], SIGNAL_DIGITS),
                "anchored": key in anchored,
                "kind": first.kind if first else None,
                "passage": first.text[:DEFINITION_CHARS] if first else "",
            }
        )
    return {"pool": len(selection.pool), "report": dict(selection.report), "nodes": nodes}


def analyze(data: bytes, filename: str | None = None) -> dict:
    """Return the per-stage summary of one PDF or DOCX document.

    Args:
        data: The file's bytes; at most `MAX_FILE_BYTES`.
        filename: Used only to help detect a damaged PDF; never stored.

    Returns:
        A JSON-serializable dict with one entry per stage (stage 3 is skipped by design).

    Raises:
        AnalysisError: The file is not a readable PDF or DOCX (corrupt, encrypted, scanned,
            too large); `code` says which.
    """
    with _EXTRACT_LOCK:
        result = _EXTRACTOR.extract_bytes(data, filename=filename)
    if not result.ok or result.structure is None:
        raise AnalysisError(result.error_code or "extraction_failed", result.error_message or "")
    stage2 = extract_candidates(result)
    pairs, _ = find_pairs(stage2.candidates)
    rule_decisions, undecided = decide_by_rule(pairs)
    concepts, absorbed = absorb_fragments(merge(stage2, rule_decisions))
    if not concepts:
        raise AnalysisError("no_concepts", "the document has no candidate terms")
    names = {concept.key: concept.written for concept in concepts}
    evidence = s5.file_evidence(result, concepts, background=None)  # T off
    importance = s5.file_importance(evidence)
    return {
        "versions": {
            "hebmath": _package_version("hebmath"),
            "concept_pipeline": _package_version("concept-pipeline"),
        },
        "mode": "algorithmic",
        "stage1": _stage1(result),
        "stage2": _stage2(stage2),
        "stage4": _stage4(pairs, rule_decisions, undecided, concepts, absorbed),
        "stage5": _stage5(evidence, importance, names),
        "stage6": _stage6(result, concepts, evidence, importance, names),
    }
