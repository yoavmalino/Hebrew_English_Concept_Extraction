"""End-to-end tests against the fixture corpus (see tests/make_fixtures.py)."""

import json
import os
import subprocess
import sys

import pytest
from conftest import FIXTURES, GOLDEN, ROOT, tesseract_has_hebrew

from hebmath import Extractor, ExtractorConfig, extract

TEXT_FIXTURES = [
    "calculus_he.docx",
    "calculus_he_libreoffice.nounicode.pdf",
    "plain_he.docx",
    "tex_en.pdf",
    "tex_hard.pdf",
    "tex_pages.pdf",
    "mathml_he.pdf",
    "mathml_he2.pdf",
    "plain_he_libreoffice.pdf",
    "calculus_he_libreoffice.pdf",
    "owner_only.pdf",
]

ERROR_FIXTURES = {
    "corrupt.pdf": "corrupt_file",
    "empty.pdf": "empty_file",
    "random.bin": "unsupported_format",
    "not_docx.docx": "corrupt_file",
    "truncated.docx": "corrupt_file",
    "encrypted.pdf": "encrypted_file",
    "xxe.docx": "corrupt_file",
    "zipbomb.docx": "resource_limit",
}


def no_ocr() -> Extractor:
    return Extractor(ExtractorConfig(enable_ocr=False))


@pytest.mark.parametrize("name", TEXT_FIXTURES)
def test_golden(name):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not generated on this machine")
    result = no_ocr().extract_path(path)
    assert result.ok, result.error_message
    expected = (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")
    assert result.text == expected


@pytest.mark.parametrize("name, code", sorted(ERROR_FIXTURES.items()))
def test_bad_inputs_fail_cleanly(name, code):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not generated on this machine")
    result = no_ocr().extract_path(path)
    assert not result.ok
    assert result.error_code == code
    assert result.text == ""


def v1(name):
    """Formula-per-line layout off: easier substring checks of the math itself."""
    return (
        Extractor(ExtractorConfig(enable_ocr=False, math_line_breaks="none"))
        .extract_path(FIXTURES / name)
        .text
    )


def test_docx_structures():
    text = v1("calculus_he.docx")
    assert r"$$x_{1,2}=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}$$" in text
    assert r"$$e^{i\pi}+1=0 \tag{1}$$" in text
    assert r"\text{מספר המקרים הרצויים}" in text
    assert "1. מציבים את הנתונים" in text and "א. סעיף ראשון" in text  # list numbering
    assert "[1] לפי הספר של שטיין" in text  # footnotes appended
    assert "$x^{n}$ | $nx^{n-1}$" in text  # table row with math cells
    assert "$x^{2}+y^{2}=r^{2}$" in text  # superscript runs typed in Word


def test_pdf_hebrew_reading_order_and_math():
    text = v1("mathml_he2.pdf")
    assert text.startswith("סיכום: אלגברה לינארית ואנליזה\n")
    assert r"$$e^{x}=\sum_{n=0}^{\infty}\frac{x^{n}}{n!}$$" in text
    assert r"$\sqrt[3]{27}=3$" in text
    assert r"$\frac{1}{1+\frac{1}{x}}$" in text
    assert "עם מספרים כמו 0.95 ו-1%." in text


def test_pdf_tex_structures():
    text = v1("tex_hard.pdf")
    assert r"$$\lim_{n\to\infty}(1+\frac{1}{n})^{n}=e \tag{1}$$" in text
    assert r"\binom{n}{k}" in text
    assert r"$\vec{v}\cdot\vec{w}" in text


def test_page_numbers_removed():
    text = extract(FIXTURES / "tex_pages.pdf").text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert not any(ln.strip().isdigit() for ln in lines)


def test_document_without_math_has_no_delimiters():
    for name in ("plain_he.docx", "plain_he_libreoffice.pdf"):
        assert "$" not in extract(FIXTURES / name).text


@pytest.mark.skipif(not tesseract_has_hebrew(), reason="tesseract with Hebrew model not installed")
def test_scanned_pdf_goes_through_ocr():
    result = extract(FIXTURES / "scanned_he.pdf")
    assert result.ok
    assert result.stats.get("pages_ocr") == 1
    assert "אינטגרלים" in result.text and "נוסחת השורשים" in result.text
    assert any("OCR" in w for w in result.warnings)


def test_scanned_pdf_without_ocr_is_flagged():
    result = no_ocr().extract_path(FIXTURES / "scanned_he.pdf")
    assert result.ok
    assert "[[hebmath:unreadable-page]]" in result.text
    assert result.warnings


def test_bytes_api_and_report():
    data = (FIXTURES / "tex_en.pdf").read_bytes()
    result = extract(data)
    report = result.to_dict()
    assert report["ok"] and report["format"] == "pdf"
    assert len(report["sha256"]) == 64
    json.dumps(report)  # serialisable


def test_deterministic_across_processes():
    """Same input -> byte-identical output, also with different hash seeds."""
    names = [n for n in TEXT_FIXTURES if (FIXTURES / n).exists()]
    code = (
        "import sys,hashlib; sys.path.insert(0, %r); from hebmath import Extractor, ExtractorConfig;"
        "e=Extractor(ExtractorConfig(enable_ocr=False));"
        "print(','.join(hashlib.sha256(e.extract_path(p).text.encode()).hexdigest() for p in sys.argv[1:]))"
        % str(ROOT / "src")
    )
    outs = set()
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run(
            [sys.executable, "-c", code, *[str(FIXTURES / n) for n in names]],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outs.add(res.stdout.strip())
    assert len(outs) == 1


def test_cli(tmp_path):
    out = tmp_path / "out.txt"
    rep = tmp_path / "rep.json"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "hebmath.cli",
            str(FIXTURES / "calculus_he.docx"),
            "-o",
            str(out),
            "--report",
            str(rep),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    assert res.returncode == 0
    assert out.read_text(encoding="utf-8").startswith("פרק 1")
    assert json.loads(rep.read_text(encoding="utf-8"))["ok"] is True
    bad = subprocess.run(
        [sys.executable, "-m", "hebmath.cli", str(FIXTURES / "corrupt.pdf")],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    assert bad.returncode == 2 and "corrupt_file" in bad.stderr


def test_cli_out_dir_creates_per_document_folder(tmp_path):
    src = tmp_path / "input"
    src.mkdir()
    (src / "algebra.docx").write_bytes((FIXTURES / "calculus_he.docx").read_bytes())
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "hebmath.cli",
            str(src / "algebra.docx"),
            "--out-dir",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    assert res.returncode == 0, res.stderr
    folder = tmp_path / "output" / "algebra"
    assert (folder / "algebra.txt").read_text(encoding="utf-8").startswith("פרק 1")
    assert json.loads((folder / "report.json").read_text(encoding="utf-8"))["ok"] is True


def test_legacy_doc_and_encrypted_docx():
    cfb = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600
    assert extract(cfb + b"WordDocument").error_code == "unsupported_format"
    assert extract(cfb + "EncryptedPackage".encode("utf-16-le")).error_code == "encrypted_file"


def test_engines_are_swappable():
    """Components can be replaced without touching the rest of the pipeline."""
    from hebmath.readers.pdf.mathengine import GeometricMathEngine

    class Tagged(GeometricMathEngine):
        def to_latex(self, atoms):
            return r"\boxed{" + super().to_latex(atoms) + "}"

    class FakeOmml:
        def convert(self, element):
            return "OMML"

    pdf = Extractor(ExtractorConfig(enable_ocr=False, pdf_math_engine=Tagged())).extract_path(
        FIXTURES / "tex_en.pdf"
    )
    assert r"$\boxed{" in pdf.text
    docx = Extractor(ExtractorConfig(omml_converter=FakeOmml())).extract_path(
        FIXTURES / "calculus_he.docx"
    )
    assert "$OMML$" in docx.text


def test_formulas_and_text_on_separate_lines():
    text = extract(FIXTURES / "mathml_he2.pdf").text
    for line in text.splitlines():
        stripped = line.strip()
        if "$" in stripped and not stripped.startswith("$$"):
            # a line holding a (non-trivial) formula holds nothing but it (+ punctuation/prefix)
            from hebmath.format import Math, Text, parse_line
            from hebmath.serialize import is_simple_math

            parts = parse_line(stripped)
            maths = [p for p in parts if isinstance(p, Math) and not is_simple_math(p.latex)]
            if maths:
                words = "".join(p.text for p in parts if isinstance(p, Text))
                assert len(words.strip(" .,:;()-")) <= 3, line


def test_unmapped_fonts_are_recovered_without_ocr():
    """PDF with every ToUnicode CMap removed (like PScript5/Ghostscript output).

    Before font recovery such pages went to OCR and lost their mathematics;
    now the text layer is decoded by glyph shape and the result is identical
    to the extraction of the intact file.
    """
    broken = no_ocr().extract_path(FIXTURES / "calculus_he_libreoffice.nounicode.pdf")
    intact = no_ocr().extract_path(FIXTURES / "calculus_he_libreoffice.pdf")
    assert broken.ok and broken.stats.get("recovered_fonts", 0) >= 1
    assert broken.stats.get("pages_ocr", 0) == 0
    assert broken.text == intact.text


def test_unmapped_fonts_degrade_gracefully():
    import difflib

    broken = no_ocr().extract_path(FIXTURES / "mathml_he2.nounicode.pdf")
    intact = no_ocr().extract_path(FIXTURES / "mathml_he2.pdf")
    assert broken.ok and "[[hebmath:unreadable-page]]" not in broken.text
    assert broken.stats["pages_text_layer"] == broken.stats["pages"]
    assert difflib.SequenceMatcher(None, intact.text, broken.text).ratio() > 0.6
    assert r"\sum_{n=0}^{\infty}" in broken.text  # structure survives


def test_glyph_recognizer_is_swappable():
    from hebmath.readers.pdf.glyphs.recover import AtlasRecognizer

    class Counting(AtlasRecognizer):
        calls = 0

        def candidates(self, desc, k, penalty=None):
            Counting.calls += 1
            return super().candidates(desc, k, penalty)

    rec = Counting()
    out = Extractor(ExtractorConfig(enable_ocr=False, glyph_recognizer=rec)).extract_path(
        FIXTURES / "calculus_he_libreoffice.nounicode.pdf"
    )
    assert out.ok and Counting.calls > 10
