"""Tests of pipeline stage 2: the seven candidate sources, the stop list, counts, and the gate checks."""

import json
from collections import Counter
from pathlib import Path

import pytest

from hebmath import Extractor, ExtractorConfig
from hebmath.model import BlockInfo
from hebmath.pipeline import ExtractionResult
from hebmath.structure import BlockRecord, Run, Section, StructureInfo, StyleRecord, Unit

from concept_pipeline.normalize import PrefixNormalizer, is_close_normalization
from concept_pipeline.stage2_candidates import (
    DEFINED,
    OPERATOR,
    SUBJECT,
    check_candidates,
    extract_candidates,
    to_snapshot,
)

FIXTURES = Path(__file__).resolve().parents[1] / "extractor" / "tests" / "fixtures"

# (text, role, unit) per block of a small synthetic summary; unit 0 is a definition.
BLOCKS = [
    ("מרחבים וקטוריים", "heading", -1),
    ("הגדרה — מטריצה הפיכה:", "heading", 0),
    ("מטריצה הפיכה היא מטריצה ריבועית שיש לה הופכית.", "body", 0),
    ("אוסף הצירופים נקרא הפרישה הליניארית של הקבוצה.", "body", 1),
    ("המטריצה והמטריצה הריבועית (square matrix) שוב: מטריצה ריבועית.", "body", 1),
    ("$\\det A \\ne 0$", "math", 1),
    ("תוצאות חשובות", "heading", -1),
    ("פרישה ליניארית של צירופים, מטריצה וקבוצה.", "body", 2),
]
BOLD_WORD = "הצירופים"


def _record(start, end, role, unit, section=0):
    return BlockRecord(
        kind="paragraph",
        start=start,
        end=end,
        page=1,
        lines=1,
        role=role,
        heading_level=1 if role == "heading" else 0,
        unit=unit,
        section=section,
        subsection=section,
        standalone=role == "heading",
        starts_page=start == 0,
        gap_before=-1.0,
        centered=False,
        words=0,
        bold_share=0.0,
        underline_share=0.0,
        large_share=0.0,
    )


def synthetic_result(blocks_spec=None):
    """Build an extraction result by hand so each source has one known trigger."""
    blocks_spec = BLOCKS if blocks_spec is None else blocks_spec
    spans, text = [], ""
    for i, (piece, _, _) in enumerate(blocks_spec):
        if i:
            text += "\n\n"
        spans.append((len(text), len(text) + len(piece)))
        text += piece
    records = [_record(s, e, role, unit) for (s, e), (_, role, unit) in zip(spans, blocks_spec)]
    units = []
    for index in sorted({unit for _, _, unit in blocks_spec if unit >= 0}):
        members = [spans[i] for i, (_, _, unit) in enumerate(blocks_spec) if unit == index]
        kind = "definition" if index == 0 else "paragraph"
        units.append(Unit(members[0][0], members[-1][1], kind, 1.0, 0))
    bold_start = text.index(BOLD_WORD) if BOLD_WORD in text else 0
    bold_end = bold_start + len(BOLD_WORD)
    plain = StyleRecord("Body", 400, 12.0, False, False, False)
    bold = StyleRecord("Body", 700, 12.0, True, False, False)
    info = StructureInfo(
        styles=[plain, bold],
        runs=[Run(0, bold_start, 0), Run(bold_start, bold_end, 1), Run(bold_end, len(text), 0)],
        blocks=records,
        sections=[Section(blocks_spec[0][0], 1, 0, len(text), "format")],
        units=units,
        signal_share={"bold": 0.02},
    )
    blocks = [BlockInfo("paragraph", s, e) for s, e in spans]
    return ExtractionResult(
        ok=True,
        text=text,
        format="pdf",
        sha256="0",
        extractor_version="test",
        blocks=blocks,
        structure=info,
    )


@pytest.fixture(scope="module")
def stage2():
    result = synthetic_result()
    return result, extract_candidates(result)


def candidate(stage2_result, key):
    return next(c for c in stage2_result.candidates if c.key == key)


def test_label_pattern_and_heading_both_propose_the_named_unit(stage2):
    _, found = stage2
    named = candidate(found, "מטריצה הפיכה")
    assert {"definition", "heading"} <= set(named.sources)
    assert named.mentions[0].strength == DEFINED


def test_copula_in_a_definition_unit_marks_the_defined_term(stage2):
    result, found = stage2
    body_start = result.text.index("מטריצה הפיכה היא")
    mention = next(m for m in candidate(found, "מטריצה הפיכה").mentions if m.start == body_start)
    assert mention.strength == DEFINED


def test_called_pattern_takes_the_term_up_to_the_function_word(stage2):
    _, found = stage2
    assert "definition" in candidate(found, "פרישה ליניארית").sources


def test_bold_word_in_the_opening_block_of_a_unit_is_its_subject(stage2):
    _, found = stage2
    bold = candidate(found, "צירופים")
    assert "emphasis" in bold.sources
    assert bold.strength_by_unit()[1] == SUBJECT


def test_english_gloss_links_both_sides_as_aliases(stage2):
    _, found = stage2
    assert candidate(found, "square matrix").aliases == ["מטריצה ריבועית"]
    hebrew = candidate(found, "מטריצה ריבועית")
    assert "gloss" in hebrew.sources and hebrew.aliases == ["square matrix"]


def test_latex_operator_is_a_weak_mention_of_its_concept(stage2):
    _, found = stage2
    determinant = candidate(found, "דטרמיננטה")
    assert determinant.sources == ["operator"]
    assert [m.strength for m in determinant.mentions] == [OPERATOR]


def test_prefixed_forms_count_toward_the_bare_form(stage2):
    _, found = stage2
    forms = Counter(m.form for m in candidate(found, "מטריצה").mentions)
    assert forms["המטריצה"] == 1 and forms["והמטריצה"] == 1


def test_stop_list_drops_labels_and_structural_headings(stage2):
    _, found = stage2
    keys = {c.key for c in found.candidates}
    assert not keys & {"הגדרה", "תוצאות", "תוצאות חשובות", "של", "a"}


def test_counts_per_unit_and_section_add_up_on_the_synthetic_summary(stage2):
    result, found = stage2
    assert check_candidates(found, result) == []
    total = sum(c.count for c in found.candidates)
    assert total == sum(sum(c.by_unit().values()) for c in found.candidates)


def test_higher_threshold_never_adds_candidates():
    result = synthetic_result()
    loose = {c.key for c in extract_candidates(result, 1).candidates}
    strict = {c.key for c in extract_candidates(result, 3).candidates}
    assert strict <= loose


def test_failed_extraction_is_rejected():
    failed = ExtractionResult(ok=False, text="", format=None, sha256="", extractor_version="t")
    with pytest.raises(ValueError):
        extract_candidates(failed)


@pytest.mark.parametrize(
    "name", ["struct_he.pdf", "struct_he_outline.pdf", "struct_he_styles.docx", "tex_pages.pdf"]
)
def test_gate_holds_and_output_is_deterministic_on_fixture(name):
    extractor = Extractor(ExtractorConfig(enable_ocr=False, recovery_ocr_pages=0))
    result = extractor.extract_path(FIXTURES / name)
    first = extract_candidates(result)
    assert first.candidates
    assert check_candidates(first, result) == []
    snapshot = to_snapshot(first)
    assert to_snapshot(extract_candidates(result)) == snapshot
    assert json.loads(json.dumps(snapshot, ensure_ascii=False)) == snapshot


def test_prefix_is_removed_only_when_the_bare_word_is_attested():
    normalizer = PrefixNormalizer(Counter({"המטריצה": 3, "מטריצה": 1, "הסתברות": 2}))
    assert normalizer.normalize_word("המטריצה") == "מטריצה"
    assert normalizer.normalize_word("הסתברות") == "הסתברות"


def test_risky_prefix_needs_a_bare_word_at_least_as_frequent():
    normalizer = PrefixNormalizer(Counter({"מרחב": 9, "רחב": 1, "לבסיס": 1, "בסיס": 4}))
    assert normalizer.normalize_word("מרחב") == "מרחב"
    assert normalizer.normalize_word("לבסיס") == "בסיס"


def test_normalization_is_idempotent():
    counts = Counter({"והמטריצה": 1, "המטריצה": 2, "מטריצה": 1, "למה": 3, "מה": 5})
    normalizer = PrefixNormalizer(counts)
    for word in counts:
        once = normalizer.normalize_word(word)
        assert normalizer.normalize_word(once) == once
    assert normalizer.normalize_word("למה") == "למה"  # "מה" is a function word


def test_guard_accepts_prefix_plural_and_construct_forms():
    assert is_close_normalization("המטריצות", "מטריצה")
    assert is_close_normalization("מרחבי השורה", "מרחב שורה")
    assert is_close_normalization("Vectors", "vector")


def test_guard_rejects_a_far_answer_or_a_changed_word_count():
    assert not is_close_normalization("מטריצה", "דטרמיננטה")
    assert not is_close_normalization("מרחב וקטורי", "מרחב")


def test_the_guard_accepts_plurals_with_a_final_letter_or_an_ot_ending():
    assert is_close_normalization("ערכים", "ערך")
    assert is_close_normalization("יתרונות", "יתרון")
    assert is_close_normalization("דוגמאות", "דוגמה")
    assert is_close_normalization("ונוסחאות", "נוסחה")
    assert is_close_normalization("מנתחים", "מנתח")


def test_the_guard_still_refuses_a_different_word_after_a_plural_ending():
    assert not is_close_normalization("דוגמאות", "דגם")
    assert not is_close_normalization("ערכים", "ערב")


# Lines seen once that only the label source and the split-subject pattern find.
RARE_BLOCKS = [
    ("פרק ראשון", "heading", -1),
    ("5. נוסחאות וייטה:", "body", 0),
    ("$x_1+x_2=-b/a$", "math", 0),
    ("12. מטריצה $A_{n}$ סקלרית אם היא אלכסונית.", "body", 1),
    ("מטריצה מדורגת תקרא קנונית אם היא מקיימת שני תנאים.", "body", 2),
    ("כל מטריצה סקלרית היא אלכסונית, וגם מטריצה קנונית היא מדורגת.", "body", 3),
    ("לכן מתקיים: הטענה נכונה.", "body", 3),
]


@pytest.fixture(scope="module")
def rare():
    return extract_candidates(synthetic_result(RARE_BLOCKS))


def test_label_line_names_a_term_seen_only_once(rare):
    vieta = candidate(rare, "נוסחאות וייטה")
    assert "label" in vieta.sources and vieta.mentions[0].strength == SUBJECT


def test_label_line_that_opens_with_a_sentence_word_is_not_a_label(rare):
    assert all("לכן" not in c.key.split() for c in rare.candidates if "label" in c.sources)


def test_formula_between_noun_and_adjective_still_defines_the_term(rare):
    assert "definition" in candidate(rare, "מטריצה סקלרית").sources


def test_term_named_by_its_adjective_is_joined_to_the_subject_noun(rare):
    assert "definition" in candidate(rare, "מטריצה קנונית").sources


def test_label_keeps_a_theorem_name_whole_and_drops_a_leading_definition_word():
    blocks = [
        ("פרק", "heading", -1),
        ("4. משפט המימדים:", "body", 0),
        ("4. הגדרת סכום של תת-מרחבים:", "body", 1),
        ("לפי משפט המימדים, וגם סכום של תת-מרחבים.", "body", 2),
    ]
    keys = {c.key for c in extract_candidates(synthetic_result(blocks)).candidates}
    assert {"משפט המימדים", "סכום של תת מרחבים"} <= keys


# Glossary lines, a mixed heading, a word in math mode and a long defined term (2026-11-16).
GLOSSARY_BLOCKS = [
    ("בעיית Rod Cutting:", "heading", -1),
    ("9.3 רציפות במידה שווה\nהגדרה נוספת שמופיעה בשורה הבאה של הכותרת", "heading", -1),
    ("▪ Boxing – מהטיפוס הפרימיטיבי למחלקה העוטפת.", "body", 0),
    ("Template method o – כמו strategy רק עם מחלקה.", "body", 0),
    ("סיווג רב מחלקתי- המסווג מחשב ציון לכל מחלקה.", "body", 1),
    ("$P(A)$ גדול.\nכלל הכלה והפרדה:\n$P(A\\cup B)$", "body", 1),
    ("מסקנה – No Path: אם אין מסלול אז המרחק אינסוף.", "body", 2),
    ("• העתקה רדודה", "body", 2),
    ("הכללה תחת יציבות $Robustness$.", "body", 3),
    ("הכללה תחת יציבות\n$Stability$.", "body", 3),
    ("משתנה ראדמאכר\n$\\sigma=1$\nוזה משפט ארוך שאינו כותרת של שום דבר", "body", 3),
    ("הגרף הזה נקרא דיאגרמת וורונוי המחלק את המישור.", "body", 3),
    ("ולכן גם זה נכון וגם זה", "body", 3),
    ("ונקשר לגדיל שמכונה template strand בתא.", "body", 4),
    ("האלגוריתם בונה מבנה שנקרא עץ מסלולים קצרים ביותר בגרף.", "body", 4),
    ("$0$ קבוצה סופית ־ קבוצה שיש בה מספר סופי של איברים.", "body", 4),
    ("Random controlled trial vs observational study", "heading", -1),
]


@pytest.fixture(scope="module")
def glossary():
    return {c.key: c for c in extract_candidates(synthetic_result(GLOSSARY_BLOCKS)).candidates}


@pytest.mark.parametrize(
    "key",
    ["boxing", "template method", "סיווג רב מחלקתי", "כלל הכלה והפרדה", "no path", "העתקה רדודה"],
)
def test_glossary_line_with_a_dash_colon_or_bullet_names_a_term(glossary, key):
    assert "label" in glossary[key].sources


def test_mixed_script_heading_also_gives_its_english_name(glossary):
    assert "rod cutting" in glossary


def test_each_line_of_a_joined_heading_is_a_name_of_its_own(glossary):
    assert "heading" in glossary["רציפות במידה שווה"].sources


def test_a_word_written_in_math_mode_is_read_as_a_word(glossary):
    assert "robustness" in glossary


def test_a_short_line_in_a_block_of_short_lines_names_a_term(glossary):
    assert "label" in glossary["stability"].sources


def test_a_short_line_beside_a_long_line_is_not_a_label(glossary):
    assert "משתנה ראדמאכר" not in glossary


def test_a_defined_term_also_proposes_its_shorter_beginnings(glossary):
    assert "definition" in glossary["דיאגרמת וורונוי"].sources


def test_a_short_unmarked_line_inside_a_longer_block_is_not_a_label(glossary):
    assert not any("label" in c.sources for k, c in glossary.items() if k.startswith("ולכן"))


def test_a_trigger_with_an_attached_prefix_still_introduces_a_term(glossary):
    assert "definition" in glossary["template strand"].sources


def test_a_superlative_stays_inside_a_defined_term(glossary):
    assert "definition" in glossary["עץ מסלולים קצרים ביותר"].sources


def test_a_maqaf_after_a_formula_bullet_ends_a_label(glossary):
    assert "label" in glossary["קבוצה סופית"].sources


def test_a_comparison_heading_names_both_sides(glossary):
    assert {"random controlled trial", "observational study"} <= glossary.keys()


def keys_of(blocks):
    return {c.key for c in extract_candidates(synthetic_result(blocks)).candidates}


def test_a_word_written_before_numbers_is_a_unit_label_of_the_file():
    blocks = [
        ("שבוע 1", "heading", -1),
        ("גרדיאנט וגרדיאנט שוב.", "body", 0),
        ("שבוע 2", "heading", -1),
        ("שבוע 3", "heading", -1),
    ]
    keys = keys_of(blocks)
    assert "שבוע" not in keys


def test_a_heading_that_is_a_sentence_gives_no_name_with_a_clause_word():
    keys = keys_of([("בשיעור הנוכחי נתאר את המודל כאשר", "heading", -1), ("טקסט.", "body", 0)])
    assert not any(" את " in f" {key} " or "כאשר" in key for key in keys)


def test_a_structural_word_that_only_heads_phrases_is_not_kept_alone():
    blocks = [
        ("שיעור נוכחי", "heading", -1),
        ("שיעור נוכחי. שיעור נוכחי.", "body", 0),
        ("מרחב תוצאות. מרחב תוצאות.", "body", 1),
    ]
    keys = keys_of(blocks)
    assert "שיעור" not in keys and "תוצאות" in keys


def test_a_glued_math_word_splits_only_into_words_the_document_writes():
    from concept_pipeline.tokens import split_glued

    vocabulary = {"bias", "variance", "tradeoff", "var"}
    assert split_glued("biasvariancetradeoff", vocabulary) == ["bias", "variance", "tradeoff"]
    assert split_glued("biasxyz", vocabulary) is None
