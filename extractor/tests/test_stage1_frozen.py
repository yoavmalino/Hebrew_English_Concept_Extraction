"""Pin the stage-1 parameters frozen on 2026-11-14, when the stage-1 gate was passed.

Every later stage is measured on stage-1 output, so a changed threshold invalidates those numbers
(plan file, stage 1: "frozen first"). To change one on purpose: re-run the calibration report on
the dev and held-out gold, record the result and the new value in the plan file, then update the
value here in the same change.
"""

import pytest

from hebmath import structure
from hebmath.readers.pdf import layout, reader, style
from hebmath.readers.pdf.glyphs import recover

FROZEN = {
    structure: {
        "SIZE_STEP": 0.5,
        "BOLD_RATIO": 1.4,
        "LARGE_SIZE_RATIO": 1.15,
        "HEADING_SIZE_RATIO": 1.08,
        "MAJORITY": 0.5,
        "MAX_HEADING_CHARS": 120,
        "MAX_HEADING_WORDS": 14,
        "HEADING_FORMAT_SHARE": 0.8,
        "MAX_HEADING_LEVELS": 4,
        "MAX_LECTURE_TOKENS": 12,
        "LOW_LETTER_SHARE": 0.3,
        "MIN_CHARS_FOR_LETTER_CHECK": 500,
        "MAX_BULLET_CONTENTS": 80,
        "HEADING_WRAP_MIN_CHARS": 35,
        "MIN_LETTER_SHARE": 0.5,
        "TITLE_SIZE_RATIO": 1.6,
        "MAX_TITLE_BLOCKS": 4,
        "MIN_MAIN_SECTION_CHARS": 200,
        "MIN_TITLE_MATCH_CHARS": 4,
        "BOOKMARK_MIN_ENTRIES": 2,
        "MIN_MAIN_OCCURRENCES": 2,
        "MIN_MAIN_SPREAD": 0.5,
        "LEAD_TIE": 0.02,
        "NEAR_BLACK": 0x30,
        "USE_NUMBERED_SERIES": True,
        "MIN_SERIES_MEMBERS": 3,
        "SERIES_EDGE_TOKENS": 3,
        "CENTER_TOLERANCE": 0.08,
        "CENTER_MAX_WIDTH": 0.8,
        "MAX_STANDALONE_CHARS": 80,
        "ALIGN_WINDOW": 80,
        "UNIT_WEIGHTS": {
            "definition": 2.0,
            "theorem": 2.0,
            "paragraph": 1.0,
            "example": 0.5,
            "exercise": 0.5,
            "proof": 1.0,
            "remark": 0.5,
        },
    },
    style: {
        "UNDERLINE_MAX_DROP": 0.40,
        "UNDERLINE_MAX_THICKNESS": 0.25,
        "UNDERLINE_MIN_OVERLAP": 0.5,
    },
    layout: {"EMPHASIS_BREAK_SHARE": 0.5},
    reader: {"FIGURE_TEXT_MAX_SHARE": 0.5},
    recover: {"TRUSTED_HEBREW_SHARE": 0.5},
}

CASES = [
    (module, name, value) for module, values in FROZEN.items() for name, value in values.items()
]


@pytest.mark.parametrize("module,name,value", CASES, ids=[f"{m.__name__}.{n}" for m, n, _ in CASES])
def test_stage1_parameter_is_unchanged_since_the_freeze(module, name, value):
    assert getattr(module, name) == value
