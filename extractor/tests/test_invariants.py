"""Tests for the stage-1 gate invariants (hebmath.invariants).

Real extractions must pass; a structure broken by hand must be reported, so the check is known to
be able to fail.
"""

import dataclasses

import pytest
from conftest import FIXTURES

from hebmath import Extractor, ExtractorConfig
from hebmath.invariants import check_invariants

STRUCTURED = [
    "struct_he.pdf",
    "struct_he_outline.pdf",
    "struct_he_styles.docx",
    "struct_he_manual.docx",
    "struct_flat_he.pdf",
    "calculus_he_libreoffice.pdf",
    "tex_pages.pdf",
]


def _extract(name: str):
    config = ExtractorConfig(enable_ocr=False, recovery_ocr_pages=0)
    return Extractor(config).extract_bytes((FIXTURES / name).read_bytes(), filename=name)


@pytest.mark.parametrize("name", STRUCTURED)
def test_real_extractions_break_no_invariant(name):
    result = _extract(name)
    assert result.ok and result.structure is not None
    assert check_invariants(result) == []


def test_a_failed_extraction_has_nothing_to_check():
    assert check_invariants(_extract("corrupt.pdf")) == []


def test_a_unit_stretched_into_the_next_main_section_is_reported():
    result = _extract("struct_he.pdf")
    info = result.structure
    main = [i for i, s in enumerate(info.sections) if s.level == 1]
    assert len(main) >= 2
    first, second = info.sections[main[0]], info.sections[main[1]]
    index = next(i for i, u in enumerate(info.units) if u.section == main[0])
    info.units[index] = dataclasses.replace(info.units[index], end=second.start + 1)
    problems = check_invariants(result)
    assert any("crosses a main-section boundary" in p for p in problems)
    assert first.end <= second.start + 1


def test_a_block_pointing_at_a_missing_unit_is_reported():
    result = _extract("struct_he.pdf")
    info = result.structure
    info.blocks[0] = dataclasses.replace(info.blocks[0], unit=len(info.units) + 5)
    assert any("does not exist" in p for p in check_invariants(result))


def test_a_block_span_that_differs_from_the_text_is_reported():
    result = _extract("struct_he.pdf")
    info = result.structure
    body = next(i for i, b in enumerate(info.blocks) if b.role == "body")
    info.blocks[body] = dataclasses.replace(info.blocks[body], end=info.blocks[body].end - 1)
    assert any(f"block {body} ends at" in p for p in check_invariants(result))
