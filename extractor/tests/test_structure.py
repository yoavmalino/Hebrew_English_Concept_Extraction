"""Tests for stage-1 structure: headings, sections, units, runs, signal shares, bold per font.

End-to-end cases use small fixtures with known structure (see make_struct_fixtures.py). Rule-level
cases build a ``Document`` by hand so each rule is tested alone. Gold values below are known by
construction, so they test the rules, not real-document accuracy (that needs the labelled set).
"""

import json

import pytest
from conftest import FIXTURES

from hebmath import Extractor, ExtractorConfig
from hebmath.model import Block, Document, StyledWord, TextSpan
from hebmath.serialize import serialize_with_spans
from hebmath.structure import (
    UNIT_WEIGHTS,
    _unit_heading,
    align_words,
    build_structure,
    classify_bold,
)

# Design stage 1 success criteria: recall 0.90 of main headings, precision 0.85.
HEADING_RECALL_TARGET = 0.90
HEADING_PRECISION_TARGET = 0.85

STRUCT_GOLD = [
    ("פרק 1: מספרים מרוכבים", 1, 3),
    ("1.1 הגדרות", 2, 3),
    ("תוצאות חשובות:", 3, 3),
    ("פרק 2: מטריצות", 1, 4),
    ("2.1 פעולות על מטריצות", 2, 4),
]


def _extract(name: str):
    config = ExtractorConfig(enable_ocr=False, recovery_ocr_pages=0)
    return Extractor(config).extract_bytes((FIXTURES / name).read_bytes(), filename=name)


def _headings(result):
    return [
        (result.text[b.start : b.end].strip(), b.heading_level, b.page)
        for b in result.structure.blocks
        if b.role == "heading"
    ]


def _recall_precision(found, gold):
    hits = len(set(found) & set(gold))
    return hits / len(gold), hits / len(found) if found else 0.0


# ---- end to end -----------------------------------------------------------------------------


def test_pdf_headings_match_gold_with_levels_and_pages():
    result = _extract("struct_he.pdf")
    found = _headings(result)
    recall, precision = _recall_precision(found, STRUCT_GOLD)
    assert recall >= HEADING_RECALL_TARGET and precision >= HEADING_PRECISION_TARGET
    assert found == STRUCT_GOLD


def test_pdf_bookmarks_set_the_headings_even_when_stored_in_visual_order():
    result = _extract("struct_he_outline.pdf")
    s = result.structure
    assert [h[0] for h in _headings(result)] == [g[0] for g in STRUCT_GOLD]
    assert {x.source for x in s.sections} == {"bookmark"}
    assert s.stats["bookmarks"] == 6 and s.stats["bookmarks_matched"] == 5


def test_pdf_title_and_table_of_contents_are_not_headings():
    result = _extract("struct_he.pdf")
    roles = {result.text[b.start : b.end].strip(): b.role for b in result.structure.blocks}
    assert roles["סיכום הקורס אלגברה"] == "title"
    assert roles["תוכן עניינים"] == "toc"
    assert sum(1 for b in result.structure.blocks if b.role == "toc") == 2


def test_pdf_sections_nest_by_level():
    result = _extract("struct_he.pdf")
    sections = result.structure.sections
    assert [s.level for s in sections] == [1, 2, 3, 1, 2]
    assert sections[0].end == sections[3].start  # a main section ends where the next one starts
    assert (
        sections[1].end == sections[3].start
    )  # sub-sections end at the next heading of a higher or equal level


def test_pdf_unit_types_and_weights():
    result = _extract("struct_he.pdf")
    units = result.structure.units
    assert [u.type for u in units] == [
        "paragraph",
        "paragraph",
        "definition",
        "theorem",
        "example",
        "paragraph",
        "definition",
        "exercise",
    ]
    assert sum(u.omega for u in units) / len(units) == pytest.approx(1.0)
    assert units[2].omega / units[0].omega == pytest.approx(
        UNIT_WEIGHTS["definition"] / UNIT_WEIGHTS["paragraph"]
    )


def test_pdf_bold_term_is_a_bold_run():
    result = _extract("struct_he.pdf")
    s = result.structure
    start = result.text.index("צמוד מרוכב")
    run = next(r for r in s.runs if r.start <= start < r.end)
    assert s.styles[run.style].bold
    assert result.text[run.start : run.end].strip() == "צמוד מרוכב"


def test_pdf_signal_shares_are_fractions_and_ignore_the_toc():
    shares = _extract("struct_he.pdf").structure.signal_share
    assert set(shares) == {"bold", "underline", "size", "line"}
    assert all(0 <= v <= 1 for v in shares.values())
    assert shares["bold"] > 0 and shares["underline"] > 0 and shares["size"] > 0


def test_pdf_without_headings_falls_back_to_pages():
    s = _extract("struct_flat_he.pdf").structure
    assert [x.source for x in s.sections] == ["pages", "pages"]
    assert not any(b.role == "heading" for b in s.blocks)


def test_bold_cannot_mark_headings_when_everything_is_bold():
    result = _extract("struct_allbold_he.pdf")
    assert [h[0] for h in _headings(result)] == ["פרק ראשון: הכול מודגש"]
    assert result.structure.signal_share["bold"] == pytest.approx(1.0)


def test_docx_heading_styles_give_the_levels():
    result = _extract("struct_he_styles.docx")
    assert [(h[0], h[1]) for h in _headings(result)] == [
        ("פרק ראשון: אלגברה", 1),
        ("תת כותרת ראשונה", 2),
        ("תת כותרת שנייה", 2),
    ]
    assert {s.source for s in result.structure.sections} == {"style"}


def test_docx_runs_carry_bold_and_underline():
    result = _extract("struct_he_styles.docx")
    s = result.structure

    def style_at(fragment):
        start = result.text.index(fragment)
        return s.styles[next(r for r in s.runs if r.start <= start < r.end).style]

    assert style_at("מושג מודגש").bold
    assert style_at("מושג עם קו תחתון").underline
    assert not style_at("טקסט רגיל עם").bold


def test_docx_without_styles_finds_headings_by_format():
    result = _extract("struct_he_manual.docx")
    assert [(h[0], h[1]) for h in _headings(result)] == [
        ("פרק ראשון: אלגברה", 1),
        ("תת כותרת ראשונה", 2),
    ]
    assert {s.source for s in result.structure.sections} == {"format"}


def test_unreadable_pages_are_listed_as_suspicious():
    config = ExtractorConfig(enable_ocr=False)
    data = (FIXTURES / "scanned_he.pdf").read_bytes()
    result = Extractor(config).extract_bytes(data, filename="scanned_he.pdf")
    assert [s["kind"] for s in result.structure.suspicious] == ["unreadable-page"]


def test_a_failure_in_the_structure_step_never_fails_the_extraction(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("hebmath.pipeline.build_structure", boom)
    result = _extract("struct_he.pdf")
    assert result.ok and result.text and result.structure is None
    assert "structure could not be built" in result.warnings


def test_structure_serializes_to_json():
    data = _extract("struct_he.pdf").structure.to_dict()
    again = json.loads(json.dumps(data))
    assert again["sections"][0]["title"] == "פרק 1: מספרים מרוכבים"
    assert len(again["runs"][0]) == 3
    assert again["stats"]["words_aligned"] > 0


# ---- rules in isolation ---------------------------------------------------------------------


def _word(text, face="Body", weight=400, size=12.0, bold=False, underline=False, color=-1):
    return StyledWord(text, face, weight, size, bold, underline, color)


def _paragraph(text, **style):
    block = Block.paragraph([TextSpan(text)])
    block.words = [_word(w, **style) for w in text.split()]
    return block


def _structure(blocks, toc=()):
    document = Document(blocks=list(blocks), toc=list(toc))
    text, spans = serialize_with_spans(document)
    return text, build_structure(document, text, spans)


def test_bookmarks_override_format_and_set_levels():
    blocks = [
        _paragraph("Introduction"),
        _paragraph("Plain body text that is long enough to be a paragraph of the document."),
        _paragraph("Methods"),
        _paragraph("More body text that is long enough to be a paragraph of the document."),
    ]
    _text, s = _structure(
        blocks, toc=[(1, "Introduction", 0), (2, "Methods", 0), (2, "Missing", 0)]
    )
    roles = [(b.role, b.heading_level) for b in s.blocks]
    assert roles == [("heading", 1), ("body", 0), ("heading", 2), ("body", 0)]
    assert s.stats["bookmarks"] == 3 and s.stats["bookmarks_matched"] == 2
    assert {x.source for x in s.sections} == {"bookmark"}


def test_heading_that_names_a_unit_types_the_next_paragraph():
    blocks = [
        _paragraph("Definition: linear span", size=14.0, underline=True),
        _paragraph("The span of a set is the set of all linear combinations of its vectors."),
        _paragraph("A plain paragraph that follows the definition above in the document."),
    ]
    _text, s = _structure(blocks)
    assert [u.type for u in s.units] == ["definition", "paragraph"]


def test_a_sentence_is_not_a_heading_even_when_large():
    blocks = [
        _paragraph("This line is large but ends like a sentence.", size=20.0),
        _paragraph("Body text of the document that is long enough to be a normal paragraph."),
        _paragraph("Another body paragraph of the document that is long enough as well."),
    ]
    _text, s = _structure(blocks)
    assert not any(b.role == "heading" for b in s.blocks)


def test_bold_is_judged_per_font_family_not_against_one_document_weight():
    words = (
        [_word("body", "TTE1D4E2C8t00", 425)] * 50  # the body font, opaque name
        + [_word("roman", "Times-Roman", 555)] * 20  # a heavier regular font of another family
        + [_word("italic", "Times-Italic", 555)] * 5
        + [_word("heavy", "TTE17A33E0t00", 752)] * 5  # opaque-named bold, 1.77x the body
        + [_word("tagged", "Arial-BoldMT", 685, bold=True)]
    )
    result = classify_bold(words)
    assert result[("Times-Roman", 555, False)] is False
    assert result[("Times-Italic", 555, False)] is False
    assert result[("TTE1D4E2C8t00", 425, False)] is False
    assert result[("TTE17A33E0t00", 752, False)] is True
    assert result[("Arial-BoldMT", 685, True)] is True


def test_same_family_weights_are_compared_with_the_lightest_face():
    words = [_word("a", "Arial", 435)] * 30 + [_word("b", "Arial-Heavy", 685)] * 3
    assert classify_bold(words)[("Arial-Heavy", 685, False)] is True


def test_words_align_through_normalized_text():
    text = "abc  def\nghi"
    words = [_word("abc"), _word("def"), _word("missing"), _word("ghi")]
    found = [(s, e, w.text) for s, e, w in align_words(text, 0, len(text), words)]
    assert found == [(0, 3, "abc"), (5, 8, "def"), (9, 12, "ghi")]


def test_a_word_far_ahead_is_not_matched_to_the_wrong_place():
    text = "start " + "x" * 200 + " end"
    found = align_words(text, 0, len(text), [_word("start"), _word("end")])
    assert [w.text for _s, _e, w in found] == ["start"]


def test_table_of_contents_entries_are_not_counted_in_signal_shares():
    entries = "Intro ........ 1 Methods ........ 2"
    toc = Block.paragraph([TextSpan(entries)])
    toc.words = [_word(w, "Body", 700, bold=True) for w in entries.split()]
    body = _paragraph("Plain body text of the document that is long enough to be a paragraph.")
    _text, s = _structure([toc, body])
    assert s.blocks[0].role == "toc"
    assert s.signal_share["bold"] == 0.0


def test_lecture_and_numbered_headings_take_their_level_from_their_words():
    big = {"size": 14.0, "bold": True}
    blocks = [
        _paragraph("Lecture 3 - Sorting", **big),
        _paragraph("2.1 Binary search", **big),
        _paragraph("2.1.4 Edge cases", **big),
        _paragraph("Body text of the document that is long enough to be a normal paragraph." * 2),
        _paragraph("More body text of the document that is long enough to be a paragraph." * 2),
    ]
    _text, s = _structure(blocks)
    assert [b.heading_level for b in s.blocks if b.role == "heading"] == [1, 2, 3]


def test_a_plain_line_that_names_a_lecture_is_a_heading():
    blocks = [
        _paragraph("Body text of the document that is long enough to be a normal paragraph." * 2),
        _paragraph("הרצאה 4 - אלגוריתם BFS"),
        _paragraph("Body text of the document that is long enough to be a normal paragraph." * 2),
    ]
    _text, s = _structure(blocks)
    assert [(b.role, b.heading_level) for b in s.blocks] == [
        ("body", 0),
        ("heading", 1),
        ("body", 0),
    ]
    assert s.sections[0].source == "pattern"


def test_a_heading_with_a_colon_ranks_below_a_plain_heading_of_the_same_look():
    look = {"size": 14.0, "bold": True}
    blocks = [
        _paragraph("Main topic", **look),
        _paragraph("Body text of the document that is long enough to be a normal paragraph." * 2),
        _paragraph("Introduces something:", **look),
        _paragraph("More body text of the document that is long enough to be a paragraph." * 2),
    ]
    _text, s = _structure(blocks)
    assert [b.heading_level for b in s.blocks if b.role == "heading"] == [1, 2]


def test_formula_debris_is_not_a_heading():
    blocks = [
        _paragraph("$x$ = 3 + 4 = 7 + 1", size=14.0, bold=True),
        _paragraph("Body text of the document that is long enough to be a normal paragraph." * 2),
    ]
    _text, s = _structure(blocks)
    assert not any(b.role == "heading" for b in s.blocks)


def test_a_heading_that_wrapped_onto_two_blocks_is_one_heading():
    look = {"size": 14.0, "bold": True}
    blocks = [
        _paragraph("Lecture 2 - Logistic regression, multiclass classification", **look),
        _paragraph("and the perceptron algorithm", **look),
        _paragraph("Body text of the document that is long enough to be a normal paragraph." * 2),
    ]
    text, s = _structure(blocks)
    roles = [b.role for b in s.blocks]
    assert roles == ["heading", "heading_cont", "body"]
    assert "perceptron" in s.sections[0].title and len(s.sections) == 1


def _page(block, number):
    from hebmath.model import LineStyle

    block.line_styles = [LineStyle(number, 12.0, 400, 0.0, 0.0, 10)]
    return block


def test_a_cover_that_is_only_slightly_larger_does_not_take_level_one():
    cover = {"size": 16.0}
    body = {"size": 11.0}
    chapter = {"size": 14.0, "underline": True}
    long_text = "Body text of the document that is long enough to be a normal paragraph." * 2
    blocks = [
        _page(_paragraph("Biology", **cover), 1),
        _page(_paragraph("Authors of the summary", **cover), 1),
        _page(_paragraph(long_text, **body), 2),
        _page(_paragraph("Macromolecules", **chapter), 2),
        _page(_paragraph(long_text, **body), 3),
        _page(_paragraph("The cell", **chapter), 3),
        _page(_paragraph(long_text, **body), 3),
    ]
    _text, s = _structure(blocks)
    roles = [(b.role, b.heading_level) for b in s.blocks]
    assert roles[:2] == [("title", 0), ("title", 0)]
    assert [b.heading_level for b in s.blocks if b.role == "heading"] == [1, 1]


def test_a_hand_written_contents_list_is_not_headings_or_signal():
    blocks = [
        _paragraph("תוכן ענייניים:"),
        _paragraph("∢ הקדמה"),
        _paragraph("o מושגים בסיסיים"),
        _paragraph("∢ מאקרו מולקולות"),
        _paragraph("Body text of the document that is long enough to be a normal paragraph." * 2),
    ]
    _text, s = _structure(blocks)
    assert [b.role for b in s.blocks] == ["toc", "toc", "toc", "toc", "body"]


def test_named_lectures_outrank_headings_that_only_look_alike():
    plain = "Body text of the document that is long enough to be a normal paragraph." * 2
    topic = {"size": 11.0, "bold": True}
    blocks = [
        _paragraph("הרצאה 1 - מבוא", underline=True),
        _paragraph("Static members", **topic),
        _paragraph(plain),
        _paragraph("הרצאה 2 - מחלקות", underline=True),
        _paragraph(plain),
        _paragraph("תרגול 3 - מנעולים", underline=True),
        _paragraph("Locks", **topic),
        _paragraph(plain),
    ]
    _text, s = _structure(blocks)
    levels = {
        s_text: b.heading_level
        for b, s_text in zip(s.blocks, [_text[b.start : b.end] for b in s.blocks])
        if b.role == "heading"
    }
    assert (
        levels["הרצאה 1 - מבוא"] == levels["הרצאה 2 - מחלקות"] == levels["תרגול 3 - מנעולים"] == 1
    )
    assert levels["Static members"] == levels["Locks"] == 2


def test_lecture_keyword_allows_a_filler_word_and_a_leading_course_word():
    blocks = [
        _paragraph("הרצאה מצגת 4 -"),
        _paragraph("למידה הרצאה 10"),
        _paragraph("Body text of the document that is long enough to be a normal paragraph." * 2),
    ]
    _text, s = _structure(blocks)
    assert [(b.role, b.heading_level) for b in s.blocks] == [
        ("heading", 1),
        ("heading", 1),
        ("body", 0),
    ]


def test_a_document_with_almost_no_letters_is_flagged_and_warned():
    junk = Block.paragraph([TextSpan(". : , ; 1. 2. 3. - . : , " * 60)])
    document = Document(blocks=[junk])
    text, spans = serialize_with_spans(document)
    info = build_structure(document, text, spans)
    assert [s["kind"] for s in info.suspicious] == ["few-letters"]


def test_formulas_do_not_count_against_the_letter_share():
    prose = "Real prose with plenty of letters in it. " * 20
    formulas = "$x^{2}+y^{2}=z^{2}$ " * 40
    document = Document(blocks=[Block.paragraph([TextSpan(prose + formulas)])])
    text, spans = serialize_with_spans(document)
    assert not [
        s for s in build_structure(document, text, spans).suspicious if s["kind"] == "few-letters"
    ]


def test_normal_fixture_is_not_flagged():
    assert not [
        s for s in _extract("struct_he.pdf").structure.suspicious if s["kind"] == "few-letters"
    ]


def _figure_page(inside, outside):
    from hebmath.readers.pdf.extract import PageContent
    from hebmath.readers.pdf.geometry import Atom

    def glyph(x, order):
        return Atom(x0=x, y0=10, x1=x + 5, y1=20, base=10, size=10, text="a", kind="g", order=order)

    atoms = [glyph(10 + 6 * i, i) for i in range(inside)]  # inside the figure box
    atoms += [glyph(300 + 6 * i, 100 + i) for i in range(outside)]  # outside it
    content = PageContent(width=600, height=800)
    content.atoms = atoms
    content.figures = [(0, 0, 200, 100)]
    return content


def test_text_inside_figures_is_dropped_only_when_most_of_the_page_is_not_figure():
    from hebmath import model
    from hebmath.readers.pdf.reader import PdfReader

    reader = PdfReader(ExtractorConfig())
    mostly_text = reader._page_atoms(_figure_page(inside=2, outside=10), model.Document(), True)[0]
    assert len(mostly_text) == 10  # the two figure labels are dropped
    mostly_figure = Document()
    kept = reader._page_atoms(_figure_page(inside=10, outside=2), mostly_figure, True)[0]
    assert len(kept) == 12  # a diagram page keeps its text instead of vanishing
    assert mostly_figure.stats["figure_pages_text_kept"] == 1


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("הרצאה 3 - מבוא", True),
        ("הרצאה מצגת 4 -", True),
        ("למידה הרצאה 10", True),
        ("Duality, Kernels, Multiclass – 4 הרצאה", True),
        ("הרצאה Spectral Clustering – 12", True),
        ("Support Vector Machines – 4 תרגול", True),
        ("הרצאה", False),
        ("בהרצאה הזאת נלמד על שלושה נושאים מרכזיים בלמידה חישובית ובהם גם 3 דוגמאות", False),
        ("משפט 3", False),
    ],
)
def test_names_a_lecture(line, expected):
    from hebmath.structure import names_a_lecture

    assert names_a_lecture(line) is expected


LONG_BODY = "Body text of the document that is long enough to be a normal paragraph. " * 4


def _levels(text, s):
    return {text[b.start : b.end].strip(): b.heading_level for b in s.blocks if b.role == "heading"}


def test_pdf_word_takes_the_fill_colour_of_its_glyphs():
    from hebmath.readers.pdf.geometry import Atom
    from hebmath.readers.pdf.style import make_word

    red = [Atom(0, 0, 5, 10, 0, 12.0, text=c, font="F", color=0xC00000) for c in "red"]
    word = make_word(list("red"), red, set())
    assert word.color == 0xC00000


def test_docx_colour_reads_hex_and_treats_auto_as_unknown():
    from xml.etree import ElementTree

    from hebmath.readers.docx_styles import _color

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def element(value):
        return ElementTree.fromstring(f'<w:color xmlns:w="{w}" w:val="{value}"/>')

    assert _color(element("2F5496")) == 0x2F5496
    assert _color(element("auto")) == -1
    assert _color(element("12")) is None
    assert _color(None) is None


def test_a_colour_other_than_the_body_text_marks_a_style_as_coloured():
    blocks = [
        _paragraph("Blue topic", color=0x2F5496),
        _paragraph(LONG_BODY, color=0x000000),
    ]
    _text, s = _structure(blocks)
    colored = [style for style in s.styles if style.colored]
    assert [style.color for style in colored] == [0x2F5496]
    assert s.body_color == 0


def test_near_black_text_is_not_a_colour():
    blocks = [_paragraph("Dark heading", color=0x202020), _paragraph(LONG_BODY, color=0)]
    _text, s = _structure(blocks)
    assert not any(style.colored for style in s.styles)


def _placed(block, page, x0, x1, top, bottom):
    from hebmath.model import LineStyle

    block.line_styles = [LineStyle(page, 12.0, 400, 0.0, 0.0, 10, x0, x1, top, bottom)]
    return block


def test_blocks_record_page_start_gap_above_and_centering():
    blocks = [
        _placed(_paragraph("Centred title"), 1, 250, 350, 700, 688),
        _placed(_paragraph(LONG_BODY), 1, 50, 550, 640, 600),
        _placed(_paragraph(LONG_BODY), 2, 50, 550, 700, 660),
    ]
    _text, s = _structure(blocks)
    first, second, third = s.blocks
    assert (first.starts_page, first.centered, first.gap_before) == (True, True, -1.0)
    assert (second.starts_page, second.centered, second.gap_before) == (False, False, 4.0)
    assert third.starts_page and third.gap_before == -1.0


def test_blocks_without_geometry_have_unknown_position():
    _text, s = _structure([_paragraph(LONG_BODY)])
    assert s.blocks[0].gap_before == -1.0 and not s.blocks[0].centered


def test_an_underlined_look_that_opens_the_text_outranks_a_bold_look_inside_it():
    chapter = {"underline": True}
    topic = {"bold": True}
    blocks = [
        _paragraph("Chapter one", **chapter),
        _paragraph(LONG_BODY),
        _paragraph("Inner topic", **topic),
        _paragraph(LONG_BODY),
        _paragraph("Chapter two", **chapter),
        _paragraph(LONG_BODY),
        _paragraph("Another topic", **topic),
        _paragraph(LONG_BODY),
    ]
    text, s = _structure(blocks)
    levels = _levels(text, s)
    assert levels["Chapter one"] == levels["Chapter two"] == 1
    assert levels["Inner topic"] == levels["Another topic"] == 2


def test_a_large_line_seen_once_does_not_take_level_one():
    chapter = {"size": 14.0, "bold": True}
    blocks = [
        _page(_paragraph("Chapter one", **chapter), 2),
        _page(_paragraph(LONG_BODY), 2),
        _page(_paragraph("Know this for the exam", size=22.0), 3),
        _page(_paragraph(LONG_BODY), 3),
        _page(_paragraph("Chapter two", **chapter), 4),
        _page(_paragraph(LONG_BODY), 4),
    ]
    text, s = _structure(blocks)
    levels = _levels(text, s)
    assert levels["Chapter one"] == levels["Chapter two"] == 1
    assert levels["Know this for the exam"] > 1
    assert s.stats["main_look_found"] == 1


def test_looks_that_cannot_be_told_apart_share_a_level_and_are_counted():
    blocks = [
        _paragraph("Bold topic", bold=True),
        _paragraph("Underlined topic", underline=True),
        _paragraph(LONG_BODY),
        _paragraph(LONG_BODY),
        _paragraph("Bold again", bold=True),
        _paragraph("Underlined again", underline=True),
        _paragraph(LONG_BODY),
    ]
    text, s = _structure(blocks)
    assert set(_levels(text, s).values()) == {1}
    assert s.stats["heading_level_ties"] == 1


def test_a_numbered_series_makes_plain_lines_headings_without_keywords():
    blocks = []
    for number in (1, 2, 3):
        blocks += [_paragraph(f"Kapitel {number} - Thema"), _paragraph(LONG_BODY)]
    _text, s = _structure(blocks)
    assert [b.heading_level for b in s.blocks if b.role == "heading"] == [1, 1, 1]
    assert {section.source for section in s.sections} == {"series"}
    assert s.stats["series_lines"] == 3


def test_numbers_that_restart_are_not_a_series():
    blocks = []
    for number in (1, 2, 1):
        blocks += [_paragraph(f"Schritt {number} - Thema"), _paragraph(LONG_BODY)]
    _text, s = _structure(blocks)
    assert not any(b.role == "heading" for b in s.blocks)


def test_unit_labels_are_never_a_series():
    blocks = []
    for number in (1, 2, 3):
        blocks += [_paragraph(f"Example {number} - sorting"), _paragraph(LONG_BODY)]
    _text, s = _structure(blocks)
    assert not any(b.role == "heading" for b in s.blocks)


def _lectures_after_a_cover(cover_text, cover_body=""):
    lecture = {"bold": True}
    blocks = [_page(_paragraph(cover_text, **lecture), 1)]
    if cover_body:
        blocks.append(_page(_paragraph(cover_body), 1))
    for page in (2, 3, 4):
        blocks += [
            _page(_paragraph(f"Lecture topic {page}", **lecture), page),
            _page(_paragraph(LONG_BODY), page),
        ]
    return _structure(blocks)


def test_a_cover_with_the_lecture_look_and_no_content_is_a_title():
    text, s = _lectures_after_a_cover("Machine learning summary")
    assert s.blocks[0].role == "title"
    assert sorted(_levels(text, s).values()) == [1, 1, 1]
    assert s.stats["front_matter_titles"] == 1


def test_a_first_page_lecture_with_real_content_stays_a_heading():
    text, s = _lectures_after_a_cover("Lecture topic 1", cover_body=LONG_BODY)
    assert s.blocks[0].role == "heading" and s.blocks[0].heading_level == 1
    assert s.stats["front_matter_titles"] == 0


def test_a_short_section_after_page_one_is_never_front_matter():
    lecture = {"bold": True}
    blocks = [
        _page(_paragraph("Lecture topic 1", **lecture), 1),
        _page(_paragraph(LONG_BODY), 1),
        _page(_paragraph("Lecture topic 2", **lecture), 2),
        _page(_paragraph("Lecture topic 3", **lecture), 3),
        _page(_paragraph(LONG_BODY), 3),
    ]
    text, s = _structure(blocks)
    assert _levels(text, s)["Lecture topic 2"] == 1


def _chapters_with_inner_lines(inner_lines):
    chapter = {"size": 16.0, "bold": True}
    inner = {"bold": True}
    blocks = []
    for number in (1, 2):
        blocks.append(_paragraph(f"Chapter {number}", **chapter))
        for line in inner_lines:
            blocks += [_paragraph(line, **inner), _paragraph(LONG_BODY)]
    return _structure(blocks)


def test_a_line_that_only_names_a_unit_is_a_unit_label_not_a_sub_heading():
    text, s = _chapters_with_inner_lines(["משפט:", "Graph search"])
    roles = {text[b.start : b.end].strip(): b.role for b in s.blocks}
    assert roles["משפט:"] == "unit_label" and roles["Graph search"] == "heading"
    assert not any(section.title.strip() == "משפט:" for section in s.sections)
    assert s.stats["unit_labels"] == 2


def test_a_unit_label_still_types_the_paragraph_under_it():
    text, s = _chapters_with_inner_lines(["הגדרה:"])
    assert [unit.type for unit in s.units] == ["definition", "definition"]


def test_a_topic_that_only_starts_like_a_unit_word_stays_a_heading():
    text, s = _chapters_with_inner_lines(["משפטים חשובים"])
    roles = {text[b.start : b.end].strip(): b.role for b in s.blocks}
    assert roles["משפטים חשובים"] == "heading"


def test_proof_and_remark_lines_are_unit_labels():
    text, s = _chapters_with_inner_lines(["הוכחה:", "הערה:"])
    roles = {text[b.start : b.end].strip(): b.role for b in s.blocks}
    assert roles["הוכחה:"] == "unit_label" and roles["הערה:"] == "unit_label"
    assert "proof" in [unit.type for unit in s.units]


def test_a_numbered_unit_without_a_name_is_still_a_label():
    text, s = _chapters_with_inner_lines(["למה 2:"])
    roles = {text[b.start : b.end].strip(): b.role for b in s.blocks}
    assert roles["למה 2:"] == "unit_label"
    assert _unit_heading("למה 2.1:") == ("theorem", False)
    assert _unit_heading("משפט 3 (קושי-שוורץ)") == ("theorem", True)


def test_a_theorem_with_a_name_stays_a_heading_and_is_flagged():
    text, s = _chapters_with_inner_lines(["משפט פיתגורס", "הגדרה: וקטור עצמי"])
    by_text = {text[b.start : b.end].strip(): b for b in s.blocks}
    assert by_text["משפט פיתגורס"].role == "heading"
    assert by_text["משפט פיתגורס"].named_unit == "theorem"
    assert by_text["הגדרה: וקטור עצמי"].named_unit == "definition"
    assert s.stats["named_units"] == 4


def test_a_superscript_number_in_a_heading_does_not_crash_the_series_check():
    blocks = []
    for number in (1, 2, 3):
        blocks += [_paragraph(f"Topic 10⁷ part {number}", bold=True), _paragraph(LONG_BODY)]
    _text, s = _structure(blocks)
    assert s.blocks


def test_a_plain_line_one_point_above_the_body_is_a_sub_heading():
    chapter = {"size": 16.0, "bold": True}
    blocks = []
    for number in (1, 2):
        blocks.append(_paragraph(f"Chapter {number}", **chapter))
        blocks += [_paragraph(f"Plain topic {number}", size=13.0), _paragraph(LONG_BODY)]
    text, s = _structure(blocks)
    levels = _levels(text, s)
    assert levels["Plain topic 1"] == 2 and levels["Chapter 1"] == 1


def test_a_plain_line_barely_above_the_body_stays_body():
    blocks = []
    for number in (1, 2):
        blocks += [_paragraph(f"Plain topic {number}", size=12.5), _paragraph(LONG_BODY)]
    _text, s = _structure(blocks)
    assert not any(b.role == "heading" for b in s.blocks)
