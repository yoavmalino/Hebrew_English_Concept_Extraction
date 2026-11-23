"""Tests for tools/calibrate_stage1.py: scoring, drafts, the report and the sweep.

The corpus here is the small fixture set with gold known by construction, so the tool is checked
for correctness; real calibration needs the owner's reviewed corpus.
"""

import json
import shutil
import sys

import pytest
from conftest import FIXTURES, ROOT

sys.path.insert(0, str(ROOT / "tools"))

import calibrate_stage1 as cal  # noqa: E402

GOLD = {
    "struct_he.pdf": [
        {"text": "פרק 1: מספרים מרוכבים", "level": 1, "page": 3},
        {"text": "1.1 הגדרות", "level": 2, "page": 3},
        {"text": "תוצאות חשובות:", "level": 3, "page": 3},
        {"text": "פרק 2: מטריצות", "level": 1, "page": 4},
        {"text": "2.1 פעולות על מטריצות", "level": 2, "page": 4},
    ],
    "struct_he_styles.docx": [
        {"text": "פרק ראשון: אלגברה", "level": 1},
        {"text": "תת כותרת ראשונה", "level": 2},
        {"text": "תת כותרת שנייה", "level": 2},
    ],
    "struct_flat_he.pdf": [],
}


@pytest.fixture
def corpus(tmp_path):
    for name, headings in GOLD.items():
        shutil.copy(FIXTURES / name, tmp_path / name)
        sidecar = tmp_path / (name + ".gold.json")
        sidecar.write_text(json.dumps({"reviewed": True, "headings": headings}), encoding="utf8")
    return tmp_path


def test_report_passes_on_a_corpus_with_known_gold(corpus, capsys):
    assert cal.main(["report", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "TOTAL (3 files)" in out


def test_report_fails_when_gold_has_a_heading_that_is_not_found(corpus, capsys):
    path = corpus / "struct_flat_he.pdf.gold.json"
    path.write_text(
        json.dumps({"reviewed": True, "headings": [{"text": "כותרת שאין", "level": 1, "page": 1}]}),
        encoding="utf8",
    )
    assert cal.main(["report", str(corpus)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_drafts_are_written_but_ignored_until_reviewed(tmp_path, capsys):
    shutil.copy(FIXTURES / "struct_he.pdf", tmp_path / "struct_he.pdf")
    assert cal.main(["init", str(tmp_path)]) == 0
    draft = json.loads((tmp_path / "struct_he.pdf.gold.json").read_text(encoding="utf8"))
    assert draft["reviewed"] is False and len(draft["headings"]) == 5
    assert cal.main(["report", str(tmp_path)]) == 2  # nothing reviewed yet


def test_init_never_overwrites_an_existing_gold_file(corpus):
    sidecar = corpus / "struct_he.pdf.gold.json"
    before = sidecar.read_text(encoding="utf8")
    cal.main(["init", str(corpus)])
    assert sidecar.read_text(encoding="utf8") == before


def test_sweep_runs_restores_the_constants_and_prints_a_plateau(corpus, capsys):
    from hebmath import structure

    before = {name: getattr(structure, name) for name in cal.SWEEPS}
    assert cal.main(["sweep", str(corpus)]) == 0
    assert {name: getattr(structure, name) for name in cal.SWEEPS} == before
    out = capsys.readouterr().out
    assert "plateau" in out and "LARGE_SIZE_RATIO" in out


def test_same_heading_ignores_the_page_when_it_is_unknown():
    a = cal.Heading("abc", 1, 0)
    assert cal.same_heading(a, cal.Heading("abc", 1, 7))
    assert not cal.same_heading(cal.Heading("abc", 1, 2), cal.Heading("abc", 1, 7))
    assert not cal.same_heading(a, cal.Heading("abd", 1, 0))


def test_ratios_handle_empty_denominators():
    assert cal.ratios(0, 0, 0) == (1.0, 1.0, 1.0)
    recall, precision, _ = cal.ratios(3, 4, 6)
    assert (recall, precision) == (0.5, 0.75)


def test_plateau_prefers_the_middle_of_the_flat_region_over_the_sharpest_point():
    values = [1, 2, 3, 4, 5, 6]
    f1s = [0.50, 0.90, 0.91, 0.90, 0.89, 0.60]
    assert cal.plateau(values, f1s, halfwidth=0.03) == 3
    assert cal.plateau(values, f1s, halfwidth=0.0) == 3


def test_gold_dir_mirrors_sub_folders_and_only_filters_files(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    (corpus / "course a").mkdir(parents=True)
    (corpus / "course b").mkdir()
    shutil.copy(FIXTURES / "struct_he.pdf", corpus / "course a" / "summary.pdf")
    shutil.copy(FIXTURES / "struct_he.pdf", corpus / "course b" / "exam.pdf")
    gold = tmp_path / "gold"
    assert cal.main(["init", str(corpus), "--gold-dir", str(gold), "--only", "summary"]) == 0
    assert (gold / "course a" / "summary.pdf.gold.json").exists()
    assert not (gold / "course b").exists()
    assert not list(corpus.rglob("*.gold.json"))  # the private folder is left untouched


def test_sheets_are_imported_only_after_the_reviewer_marks_them(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(FIXTURES / "struct_he.pdf", corpus / "s.pdf")
    review, gold = tmp_path / "review", tmp_path / "gold"
    assert cal.main(["sheets", str(corpus), "--review-dir", str(review)]) == 0
    sheet = review / "s.pdf.txt"
    lines = sheet.read_text(encoding="utf8").splitlines()
    assert lines[0] == "UNREVIEWED"
    assert "3 | פרק 1: מספרים מרוכבים" in lines and "4 | פרק 2: מטריצות" in lines
    cal.main(["import", str(corpus), "--review-dir", str(review), "--gold-dir", str(gold)])
    assert not (gold / "s.pdf.gold.json").exists()
    sheet.write_text(
        "\n".join(
            ["REVIEWED", "# note", "3 | פרק 1: מספרים מרוכבים", "0 | פרק 2: מטריצות | עם קו"]
        ),
        encoding="utf8",
    )
    cal.main(["import", str(corpus), "--review-dir", str(review), "--gold-dir", str(gold)])
    data = json.loads((gold / "s.pdf.gold.json").read_text(encoding="utf8"))
    assert data["reviewed"] is True
    assert [(h["page"], h["text"]) for h in data["headings"]] == [
        (3, "פרק 1: מספרים מרוכבים"),
        (0, "פרק 2: מטריצות | עם קו"),
    ]


def test_partial_matches_are_counted_apart_from_clean_misses():
    gold = [
        cal.Heading(cal.normalize("הרצאה 2 – רגרסיה לוגיסטית והפרדה לינארית"), 1, 12),
        cal.Heading(cal.normalize("כותרת שאינה קיימת בכלל"), 1, 30),
        cal.Heading(cal.normalize("הרצאה 3 – נושא"), 1, 40),
    ]
    pred = [
        cal.Heading(cal.normalize("הרצאה 2 – רגרסיה לוגיסטית"), 1, 12),  # cut short: partial
        cal.Heading(cal.normalize("הרצאה 3 – נושא"), 1, 40),  # exact: a hit
        cal.Heading(cal.normalize("הרצאה 2 – רגרסיה לוגיסטית"), 1, 99),  # other page: no partial
    ]
    assert cal.partial_matches(pred, gold) == 1
    assert cal.score_main(pred, gold) == (1, 3, 3)


def test_report_prints_the_partial_match_column(corpus, capsys):
    cal.main(["report", str(corpus)])
    assert "partial" in capsys.readouterr().out


def test_report_prints_a_lenient_score_next_to_the_strict_one(corpus, capsys):
    cal.main(["report", str(corpus)])
    out = capsys.readouterr().out
    assert "lenient" in out and "strict numbers" in out


def test_invariants_command_passes_on_the_fixture_corpus(corpus, capsys):
    assert cal.main(["invariants", str(corpus)]) == 0
    assert "3 files checked, 0 failing" in capsys.readouterr().out


def test_units_command_prints_a_total_row(corpus, capsys):
    assert cal.main(["units", str(corpus)]) == 0
    assert "TOTAL" in capsys.readouterr().out
