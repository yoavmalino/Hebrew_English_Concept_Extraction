"""Unit tests for the format-independent pieces."""

import xml.etree.ElementTree as ET

import pytest

from hebmath.bidi import visual_to_logical
from hebmath.latex.omml import OmmlToLatex
from hebmath.latex.plaintext import linear_to_latex, tag_plain_text
from hebmath.model import Block, Document, InlineMath, Placeholder, TextSpan
from hebmath.serialize import serialize

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def omml(xml: str) -> str:
    el = ET.fromstring(f'<m:oMath xmlns:m="{M}">{xml}</m:oMath>')
    return OmmlToLatex().convert(el)


def r(t, sty=None):
    pr = f'<m:rPr><m:sty m:val="{sty}"/></m:rPr>' if sty else ""
    return f"<m:r>{pr}<m:t>{t}</m:t></m:r>"


# ----------------------------------------------------------------- serializer
def test_delimiters_and_escaping():
    doc = Document(
        blocks=[
            Block.paragraph([TextSpan("מחיר $5 ו-\\ "), InlineMath("x^{2}"), TextSpan(" סוף")]),
            Block.display_math("\\frac{a}{b}", tag="3"),
            Block.paragraph([Placeholder("image")]),
            Block.paragraph([TextSpan("טקסט "), Placeholder("math-object")]),
        ]
    )
    out = serialize(doc, math_line_breaks="none")
    # images are ignored without a trace; lost math keeps its marker
    assert out == (
        "מחיר \\$5 ו-\\\\ $x^{2}$ סוף\n\n$$\\frac{a}{b} \\tag{3}$$\n\n"
        "טקסט [[hebmath:math-object]]\n"
    )


def test_formula_and_text_are_on_separate_lines():
    doc = Document(
        blocks=[
            Block.paragraph(
                [
                    TextSpan("1. "),
                    InlineMath("z+\\overline{z}=2"),
                    TextSpan(" (מדומה) ולכן "),
                    InlineMath("V"),
                    TextSpan(" הוא מרחב, כאשר ("),
                    InlineMath("x>0"),
                    TextSpan(")."),
                ]
            )
        ]
    )
    out = serialize(doc)
    assert out == ("1. $z+\\overline{z}=2$\n(מדומה) ולכן $V$ הוא מרחב, כאשר\n($x>0$).\n")
    lines = out.strip().split("\n")
    assert all(not ln.startswith((".", ",")) for ln in lines)
    # policy "all": even a single symbol gets its own line; "none": format v1
    assert "\n$V$\n" in serialize(doc, math_line_breaks="all")
    assert serialize(doc, math_line_breaks="none").count("\n") == 1


def test_simple_math_classification():
    from hebmath.serialize import is_simple_math

    for latex in (
        "V",
        "\\lambda",
        "A_{n\\times n}",
        "\\mathbb{R}^{n}",
        "x_{i}",
        "A^{t}",
        "\\overline{z}",
    ):
        assert is_simple_math(latex), latex
    for latex in ("x>0", "a+b", "\\frac{1}{2}", "f(x)", "A=B", "\\sum_{i}x_{i}"):
        assert not is_simple_math(latex), latex


def test_serializer_strips_bidi_controls_and_folds_presentation_forms():
    doc = Document(blocks=[Block.paragraph([TextSpan("\u202bשָׁלוֹם\u202c \ufb2a ﬁ")])])
    out = serialize(doc)
    assert "\u202b" not in out and "\u202c" not in out
    assert "\ufb2a" not in out and "fi" in out


def test_empty_document_serializes_to_empty_string():
    assert serialize(Document()) == ""


# ----------------------------------------------------------------- bidi
@pytest.mark.parametrize(
    "visual, logical",
    [
        ("םולש", "שלום"),
        ("abc םולש", "שלום abc"),
        ("2024 תנשב", "בשנת 2024"),
        # glyph shapes as drawn: the logical "(" of an RTL run is drawn mirrored
        (".(עובש) םוי", "יום (שבוע)."),
        ("Hello world", "Hello world"),
    ],
)
def test_visual_to_logical(visual, logical):
    assert visual_to_logical(visual) == logical


# ----------------------------------------------------------------- OMML
def test_omml_fraction_radical_scripts():
    xml = (
        "<m:f><m:num>" + r("−b±") + '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>'
        "<m:sSup><m:e>"
        + r("b")
        + "</m:e><m:sup>"
        + r("2")
        + "</m:sup></m:sSup>"
        + r("−4ac")
        + "</m:e></m:rad></m:num><m:den>"
        + r("2a")
        + "</m:den></m:f>"
    )
    assert omml(xml) == r"\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}"


def test_omml_nary_and_delimiters():
    xml = (
        '<m:nary><m:naryPr><m:chr m:val="∑"/></m:naryPr><m:sub>'
        + r("k=1")
        + "</m:sub><m:sup>"
        + r("n")
        + "</m:sup><m:e>"
        + r("k")
        + "</m:e></m:nary>"
    )
    assert omml(xml) == r"\sum_{k=1}^{n} k"
    assert omml("<m:d><m:e>" + r("x") + "</m:e></m:d>") == "(x)"
    tall = (
        "<m:d><m:e><m:f><m:num>"
        + r("1")
        + "</m:num><m:den>"
        + r("2")
        + "</m:den></m:f></m:e></m:d>"
    )
    assert omml(tall) == r"\left( \frac{1}{2} \right)"


def test_omml_integral_default_char():
    xml = (
        "<m:nary><m:naryPr/><m:sub>"
        + r("0")
        + "</m:sub><m:sup>"
        + r("1")
        + "</m:sup><m:e>"
        + r("x")
        + "</m:e></m:nary>"
    )
    assert omml(xml) == r"\int_{0}^{1} x"


def test_omml_function_accent_bar_matrix():
    assert (
        omml("<m:func><m:fName>" + r("sin", "p") + "</m:fName><m:e>" + r("θ") + "</m:e></m:func>")
        == r"\sin \theta"
    )
    assert (
        omml('<m:acc><m:accPr><m:chr m:val="⃗"/></m:accPr><m:e>' + r("v") + "</m:e></m:acc>")
        == r"\vec{v}"
    )
    assert (
        omml('<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>' + r("z") + "</m:e></m:bar>")
        == r"\overline{z}"
    )
    m = (
        "<m:m><m:mr><m:e>"
        + r("1")
        + "</m:e><m:e>"
        + r("0")
        + "</m:e></m:mr><m:mr><m:e>"
        + r("0")
        + "</m:e><m:e>"
        + r("1")
        + "</m:e></m:mr></m:m>"
    )
    assert omml(m) == r"\begin{matrix} 1 & 0 \\ 0 & 1 \end{matrix}"


def test_omml_hebrew_text_and_equation_number():
    assert omml("<m:r><m:rPr><m:nor/></m:rPr><m:t>מספר המקרים</m:t></m:r>") == r"\text{מספר המקרים}"
    xml = (
        "<m:eqArr><m:e>"
        + r("E=mc")
        + "<m:sSup><m:e/><m:sup>"
        + r("2")
        + "</m:sup></m:sSup>"
        + r("#")
        + "<m:d><m:e>"
        + r("4")
        + "</m:e></m:d></m:e></m:eqArr>"
    )
    assert omml(xml).endswith(r"\tag{4}")


def test_omml_unknown_elements_degrade_gracefully():
    assert omml("<m:futureThing><m:e>" + r("x") + "</m:e></m:futureThing>") == "x"


# ----------------------------------------------------------------- plain-text math
def test_tag_plain_text_finds_formulas_but_not_prose():
    pieces = tag_plain_text("משפט פיתגורס: a² + b² = c², ובשנת 2024 היו 30 סטודנטים.")
    maths = [p.text for p in pieces if p.is_math]
    assert maths == ["a² + b² = c²"]
    assert all(not p.is_math for p in tag_plain_text("if x is big then we stop"))
    assert all(not p.is_math for p in tag_plain_text("ראו עמ' 45–47 (פרק 3)"))


def test_linear_to_latex():
    assert linear_to_latex("a²+b²=c²") == "a^{2}+b^{2}=c^{2}"
    assert linear_to_latex("√(x+1)") == r"\sqrt{x+1}"
    assert linear_to_latex("sinx≤1") == r"\sin x\le1"
    assert linear_to_latex("x^{n-1}") == "x^{n-1}"


# ----------------------------------------------------------------- reference parser
def test_reference_parser_round_trip():
    from hebmath.format import Marker, Math, Text, parse

    doc = Document(
        blocks=[
            Block.paragraph(
                [
                    TextSpan("עלות $3 \\"),
                    InlineMath(r"\frac{a}{b}"),
                    TextSpan(" ו-"),
                    InlineMath(r"\begin{matrix} 1 \\ 2 \end{matrix}"),
                    Placeholder("image"),
                ]
            ),
            Block.display_math(r"e^{i\pi}+1=0", tag="2"),
        ]
    )
    blocks = parse(serialize(doc, math_line_breaks="none"))
    assert blocks[0].parts == [
        Text("עלות $3 \\"),
        Math(r"\frac{a}{b}"),
        Text(" ו-"),
        Math(r"\begin{matrix} 1 \\ 2 \end{matrix}"),
    ]
    assert blocks[1].kind == "display_math" and blocks[1].parts[0] == Math(
        r"e^{i\pi}+1=0", True, "2"
    )
    # format v2: the same paragraph over several lines parses to the same parts
    v2 = parse(serialize(doc))
    assert len(v2[0].lines) == 3
    assert [p for p in v2[0].parts if isinstance(p, Math)] == [
        p for p in blocks[0].parts if isinstance(p, Math)
    ]
    assert parse("א\n$x+1$\nב\n\n$$y$$\n")[0].parts == [Text("א "), Math("x+1"), Text(" ב")]
    assert Marker  # (markers are still parsed; see test_delimiters_and_escaping)


def test_reference_parser_on_real_output(fixtures):
    from hebmath import extract
    from hebmath.format import Math, parse

    blocks = parse(extract(fixtures / "calculus_he.docx").text)
    maths = [p for b in blocks for p in b.parts if isinstance(p, Math)]
    assert len(maths) >= 15
    assert all("$" not in m.latex.replace(r"\$", "") for m in maths)


# ----------------------------------------------------------------- vector shapes
def test_vector_symbol_shapes():
    from hebmath.readers.pdf.vector import _angle, _radical, figure_regions

    move, line = 2, 0
    assert (
        _angle(
            [(move, 522.8, 768.4), (line, 520.3, 761.4), (line, 522.8, 754.2)],
            519.7,
            753.4,
            523.6,
            769.2,
        )
        == "⟨"
    )
    assert (
        _angle(
            [(move, 541.0, 768.4), (line, 543.6, 761.4), (line, 541.0, 754.2)],
            540.2,
            753.4,
            544.2,
            769.2,
        )
        == "⟩"
    )
    rad = [
        (move, 273.7, 571.2),
        (line, 274.9, 572.1),
        (line, 277.8, 565.4),
        (line, 280.9, 582.9),
        (line, 305.9, 582.9),
    ]
    r = _radical(rad, 273.6, 565.0, 306.0, 583.0)
    assert r is not None and r.x1 < 285 and r.bar_x1 == 306.0  # sign + vinculum over the radicand
    dashed = [
        (move, 66.0 + 2 * k, 623.0) if k % 2 == 0 else (line, 66.0 + 2 * k, 623.0)
        for k in range(120)
    ]
    assert (
        _radical(dashed + [(line, 316, 626), (line, 310, 620)], 66.0, 620.1, 315.8, 626.1) is None
    )
    # small strokes inside text lines never make a figure; a big drawing does
    small = [(100.0 + 20 * k, 500.0, 104.0 + 20 * k, 516.0) for k in range(10)]
    assert figure_regions(small, 12.0) == []
    big = small[:3] + [(90.0, 450.0, 200.0, 560.0)]
    assert len(figure_regions(big, 12.0)) == 1


def test_glyph_descriptor_round_trip_and_atlas():
    from hebmath.readers.pdf.glyphs.atlas import load_atlas
    from hebmath.readers.pdf.glyphs.render import Descriptor

    atlas = load_atlas()
    assert len(atlas.glyphs) > 2000 and atlas.tall
    ref = next(g for g in atlas.glyphs if g.char == "∑" and g.source.startswith("STIX"))
    assert Descriptor.from_json(ref.desc.to_json()) == ref.desc
    assert atlas.match(ref.desc, 1)[0].char in ("∑", "Σ")
    heb = next(g for g in atlas.glyphs if g.char == "ש" and g.source.startswith("David"))
    assert atlas.match(heb.desc, 1)[0].char == "ש"


def _glyph(text, x0, y0, x1, y1, size, order, base=None):
    from hebmath.readers.pdf.geometry import Atom

    return Atom(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        base=y0 if base is None else base,
        size=size,
        text=text,
        order=order,
        italic=text.isalpha(),
        lx0=x0,
        lx1=x1,
    )


@pytest.mark.parametrize(
    "glyphs, rule, expected",
    [
        # \overline{z_0} drawn high above the symbol (coordinates from a Word->PScript5 PDF)
        (
            [("z", 244.8, 350.4, 249.4, 356.5, 12.0), ("0", 249.6, 348.1, 252.8, 352.9, 6.9)],
            (243.7, 254.3, 362.6),
            r"\overline{z_{0}}",
        ),
        # \overline{z^n}: the superscript, not the base symbol, is nearest the bar
        (
            [("z", 474.5, 683.0, 479.0, 689.1, 12.0), ("n", 479.6, 689.3, 483.6, 692.4, 7.0)],
            (473.4, 484.9, 696.7),
            r"\overline{z^{n}}",
        ),
    ],
)
def test_overline_over_symbol_with_script(glyphs, rule, expected):
    from hebmath.readers.pdf.geometry import Rule
    from hebmath.readers.pdf.mathengine import GeometricMathEngine

    engine = GeometricMathEngine()
    atoms = [
        _glyph(t, x0, y0, x1, y1, s, i, base=y0 if s > 10 else None)
        for i, (t, x0, y0, x1, y1, s) in enumerate(glyphs)
    ]
    out = engine.build_structures(atoms, [Rule(rule[0], rule[1], rule[2], 0.5)])
    assert engine.to_latex(out) == expected


def test_hebrew_prefix_hyphen_stays_with_formula():
    def para(*segs):
        return serialize(Document(blocks=[Block.paragraph(list(segs))]))

    # PDFs often set a space before the hyphen, or after it, of a prefix letter
    assert (
        para(TextSpan("המינור ה -"), InlineMath("i,j"), TextSpan(" של מטריצה"))
        == "המינור\nה-$i,j$\nשל מטריצה\n"
    )
    assert para(TextSpan("ול- "), InlineMath("BA"), TextSpan(".")) == "ול-$BA$.\n"
    # a real dash after a word is not a prefix
    assert para(TextSpan("זה -"), InlineMath("x+1")) == "זה -\n$x+1$\n"
