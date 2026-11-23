"""Tests for the stage-1 formatting metadata (block spans, line styles, modal style).

The metadata must describe the text without changing it: the golden tests in
test_extraction.py already pin the text itself.
"""

import pytest
from conftest import FIXTURES

from hebmath import Extractor, ExtractorConfig
from hebmath.model import LineStyle
from hebmath.readers.pdf.layout import _emphasis_changes

PDF_FIXTURES = [
    "style_he.pdf",
    "tex_en.pdf",
    "tex_pages.pdf",
    "plain_he_libreoffice.pdf",
    "calculus_he_libreoffice.pdf",
]


def _extract(name: str):
    config = ExtractorConfig(enable_ocr=False, recovery_ocr_pages=0)
    return Extractor(config).extract_bytes((FIXTURES / name).read_bytes(), filename=name)


@pytest.mark.parametrize("name", PDF_FIXTURES)
def test_block_spans_rebuild_the_text(name):
    result = _extract(name)
    assert result.ok and result.blocks
    pieces = [result.text[b.start : b.end] for b in result.blocks]
    assert all(pieces)
    assert "\n\n".join(pieces) + "\n" == result.text
    assert [b.start for b in result.blocks] == sorted(b.start for b in result.blocks)


def test_docx_blocks_have_no_line_styles():
    result = _extract("calculus_he.docx")
    assert result.ok and result.blocks
    assert all(not b.line_styles for b in result.blocks)


def test_heading_is_larger_than_the_modal_body_size():
    result = _extract("style_he.pdf")
    first = result.blocks[0]
    assert first.line_styles[0].size > 1.5 * result.doc_style["size"]


def test_underlined_phrase_is_detected_and_plain_text_is_not():
    result = _extract("style_he.pdf")
    shares = [(b.line_styles[0].underline_share, b) for b in result.blocks if b.line_styles]
    full_line = [s for s, b in shares if "כל השורה" in result.text[b.start : b.end]]
    assert full_line and full_line[0] > 0.9
    heading_and_plain = [s for s, b in shares if "קו תחתון" not in result.text[b.start : b.end]]
    assert all(s == 0 for s in heading_and_plain)
    partial = [
        s
        for s, b in shares
        if "באמצע המשפט" in result.text[b.start : b.end]
        and "קו תחתון" in result.text[b.start : b.end]
    ]
    assert partial and 0 < partial[0] < 0.5


def test_bold_text_has_a_higher_weight_than_the_modal_weight():
    result = _extract("style_he.pdf")
    modal = result.doc_style["weight"]
    assert modal > 0
    heavier = [b for b in result.blocks if b.line_styles and b.line_styles[0].weight > 1.3 * modal]
    assert heavier, "the heading is bold, so at least one line is heavier than the body"


def _glyph(text, size, style_size):
    from hebmath.readers.pdf.geometry import Atom

    return Atom(0, 0, 5, 10, 0, size, text=text, font="F", style_size=style_size)


def test_line_and_word_sizes_use_the_size_as_drawn_not_the_font_size():
    from hebmath.readers.pdf.style import line_style, make_word

    # a writer that sets the font to size 1 and scales it with the text matrix
    glyphs = [_glyph(c, 1.0, 20.0) for c in "abc"]
    assert line_style(glyphs, [], 1).size == 20.0
    assert make_word(list("abc"), glyphs, set()).size == 20.0


def test_size_falls_back_to_the_font_size_when_the_drawn_size_is_unknown():
    from hebmath.readers.pdf.style import line_style

    assert line_style([_glyph("a", 12.0, 0.0)], [], 1).size == 12.0


def test_a_thick_real_world_underline_counts_at_its_drawn_size():
    from hebmath.readers.pdf.geometry import Rule
    from hebmath.readers.pdf.style import underlined_glyph_ids

    # measured in a Google Docs summary: 11 pt text drawn from a 14.7 pt font, a 2 pt stroke
    glyphs = [_glyph(c, 14.7, 11.0) for c in "abc"]
    stroke = Rule(x0=0, x1=5, y=-0.9, thickness=2.0)
    assert len(underlined_glyph_ids(glyphs, [stroke])) == 3


def _line_style(underline=0.0, bold=0.0):
    return LineStyle(1, 11.0, 400, bold, underline, 20)


def test_an_underlined_line_after_a_plain_one_starts_a_new_paragraph():
    assert _emphasis_changes(_line_style(), _line_style(underline=1.0))
    assert _emphasis_changes(_line_style(), _line_style(bold=1.0))
    assert not _emphasis_changes(_line_style(underline=0.2), _line_style(underline=0.4))
    assert not _emphasis_changes(None, _line_style(underline=1.0))
