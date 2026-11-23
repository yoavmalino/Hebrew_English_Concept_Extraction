"""Generate the test corpus (DOCX written as raw WordprocessingML/OMML, like Word does).

Run:  python tests/make_fixtures.py
PDF variants are produced from these with LibreOffice / Chromium / pdflatex
when those tools are available (see tests/README in docs).
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "fixtures"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

FIXED_DATE = (2024, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------- OMML helpers
def _x(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mr(text, sty=None):
    text = _x(text)
    rpr = f'<m:rPr><m:sty m:val="{sty}"/></m:rPr>' if sty else ""
    return f'<m:r>{rpr}<m:t xml:space="preserve">{text}</m:t></m:r>'


def mtext(text):  # normal (non-math) text inside an equation
    return f'<m:r><m:rPr><m:nor/></m:rPr><m:t xml:space="preserve">{text}</m:t></m:r>'


def e(*xs):
    return "<m:e>" + "".join(xs) + "</m:e>"


def sup(base, s):
    return f"<m:sSup>{e(base)}<m:sup>{s}</m:sup></m:sSup>"


def sub(base, s):
    return f"<m:sSub>{e(base)}<m:sub>{s}</m:sub></m:sSub>"


def subsup(base, s, p):
    return f"<m:sSubSup>{e(base)}<m:sub>{s}</m:sub><m:sup>{p}</m:sup></m:sSubSup>"


def frac(n, d):
    return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"


def rad(body, deg=None):
    if deg is None:
        return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/>{e(body)}</m:rad>'
    return f"<m:rad><m:deg>{deg}</m:deg>{e(body)}</m:rad>"


def nary(ch, lo, hi, body, limloc=None):
    ll = f'<m:limLoc m:val="{limloc}"/>' if limloc else ""
    return f'<m:nary><m:naryPr><m:chr m:val="{ch}"/>{ll}</m:naryPr><m:sub>{lo}</m:sub><m:sup>{hi}</m:sup>{e(body)}</m:nary>'


def integral(lo, hi, body):
    return f"<m:nary><m:naryPr/><m:sub>{lo}</m:sub><m:sup>{hi}</m:sup>{e(body)}</m:nary>"


def delim(*items, beg=None, end=None):
    pr = ""
    if beg is not None or end is not None:
        pr = (
            "<m:dPr>"
            + (f'<m:begChr m:val="{beg}"/>' if beg is not None else "")
            + (f'<m:endChr m:val="{end}"/>' if end is not None else "")
            + "</m:dPr>"
        )
    return "<m:d>" + pr + "".join(e(i) for i in items) + "</m:d>"


def func(name, arg):
    return f"<m:func><m:fName>{mr(name, 'p')}</m:fName>{e(arg)}</m:func>"


def limlow(base, lim):
    return f"<m:func><m:fName><m:limLow>{e(mr(base, 'p'))}<m:lim>{lim}</m:lim></m:limLow></m:fName>"


def matrix(rows):
    return (
        "<m:m>"
        + "".join("<m:mr>" + "".join(e(c) for c in row) + "</m:mr>" for row in rows)
        + "</m:m>"
    )


def omath(*xs):
    return "<m:oMath>" + "".join(xs) + "</m:oMath>"


def omathpara(*xs):
    return "<m:oMathPara>" + omath(*xs) + "</m:oMathPara>"


# ---------------------------------------------------------------- WML helpers
def run(text, *, rtl=True, vert=None, italic=False):
    rpr = (
        "<w:rPr>"
        + ("<w:rtl/>" if rtl else "")
        + ("<w:i/>" if italic else "")
        + (f'<w:vertAlign w:val="{vert}"/>' if vert else "")
        + "</w:rPr>"
    )
    return f'<w:r>{rpr}<w:t xml:space="preserve">{text}</w:t></w:r>'


def para(*content, style=None, num=None, rtl=True):
    ppr = (
        "<w:pPr>"
        + (f'<w:pStyle w:val="{style}"/>' if style else "")
        + (f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num}"/></w:numPr>' if num else "")
        + ("<w:bidi/>" if rtl else "")
        + "</w:pPr>"
    )
    return "<w:p>" + ppr + "".join(content) + "</w:p>"


def table(rows):
    out = "<w:tbl><w:tblPr><w:bidiVisual/></w:tblPr>"
    for row in rows:
        out += "<w:tr>" + "".join(f"<w:tc>{para(*cell)}</w:tc>" for cell in row) + "</w:tr>"
    return out + "</w:tbl>"


def build_docx(
    body_xml: str, *, footnotes: list[str] | None = None, numbering: bool = False
) -> bytes:
    doc = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W}" xmlns:m="{M}" xmlns:r="{R}"><w:body>{body_xml}'
        f"<w:sectPr><w:bidi/></w:sectPr></w:body></w:document>"
    )
    ct = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        + (
            '<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>'
            if footnotes
            else ""
        )
        + (
            '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
            if numbering
            else ""
        )
        + "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + (
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes" Target="footnotes.xml"/>'
            if footnotes
            else ""
        )
        + (
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
            if numbering
            else ""
        )
        + "</Relationships>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:

        def add(name, data):
            zi = zipfile.ZipInfo(name, FIXED_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, data)

        add("[Content_Types].xml", ct)
        add("_rels/.rels", rels)
        add("word/_rels/document.xml.rels", doc_rels)
        add("word/document.xml", doc)
        if footnotes:
            fn = f'<?xml version="1.0" encoding="UTF-8"?><w:footnotes xmlns:w="{W}" xmlns:m="{M}">'
            fn += '<w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>'
            for i, text in enumerate(footnotes, start=1):
                fn += f'<w:footnote w:id="{i}">{para(run(text))}</w:footnote>'
            add("word/footnotes.xml", fn + "</w:footnotes>")
        if numbering:
            add(
                "word/numbering.xml",
                f'<?xml version="1.0" encoding="UTF-8"?><w:numbering xmlns:w="{W}">'
                '<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/>'
                '<w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl></w:abstractNum>'
                '<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:start w:val="1"/>'
                '<w:numFmt w:val="hebrew1"/><w:lvlText w:val="%1."/></w:lvl></w:abstractNum>'
                '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'
                '<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num></w:numbering>',
            )
    return buf.getvalue()


# ---------------------------------------------------------------- documents
def calculus_docx() -> bytes:
    fx = mr("f") + delim(mr("x")) + mr("=") + sup(mr("x"), mr("2")) + mr("+3x−5")
    deriv = (
        sub(mr("f'"), mr("0"))
        + mr("=")
        + limlow("lim", mr("h→0"))
        + e(
            frac(
                mr("f")
                + delim(sub(mr("x"), mr("0")) + mr("+h"))
                + mr("−f")
                + delim(sub(mr("x"), mr("0"))),
                mr("h"),
            )
        )
        + "</m:func>"
    )
    quad = (
        sub(mr("x"), mr("1,2"))
        + mr("=")
        + frac(mr("−b±") + rad(sup(mr("b"), mr("2")) + mr("−4ac")), mr("2a"))
    )
    geo = (
        nary("∑", mr("n=0"), mr("∞"), mr("a") + sup(mr("q"), mr("n")))
        + mr("=")
        + frac(mr("a"), mr("1−q"))
    )
    integ = (
        integral(mr("0"), mr("1"), sup(mr("x"), mr("2")) + mr("dx"))
        + mr("=")
        + frac(mr("1"), mr("3"))
    )
    mat = mr("A=") + delim(matrix([[mr("1"), mr("2")], [mr("3"), mr("4")]]), beg="[", end="]")
    heb_math = (
        mr("P")
        + delim(mr("A"))
        + mr("=")
        + frac(mtext("מספר המקרים הרצויים"), mtext("מספר המקרים הכולל"))
    )
    trig = func("sin", sup(mr("x"), mr("2"))) + mr("+") + func("cos", mr("θ")) + mr("≤2")
    numbered = (
        "<m:eqArr><m:e>"
        + sup(mr("e"), mr("iπ"))
        + mr("+1=0#")
        + delim(mr("1"))
        + "</m:e></m:eqArr>"
    )
    body = "".join(
        [
            para(run("פרק 1: גבולות ונגזרות"), style="Heading1"),
            para(
                run("נגדיר את הפונקציה "),
                omath(fx),
                run(" ונחשב את הנגזרת שלה בנקודה "),
                omath(sub(mr("x"), mr("0")) + mr("=2")),
                run("."),
            ),
            para(run("לפי ההגדרה, הנגזרת היא:")),
            para(omathpara(deriv)),
            para(
                run("נוסחת השורשים של משוואה ריבועית "),
                omath(mr("a") + sup(mr("x"), mr("2")) + mr("+bx+c=0")),
                run(" היא:"),
            ),
            para(omathpara(quad)),
            para(
                run("סכום של טור הנדסי אינסופי כאשר "),
                omath(delim(mr("q"), beg="|", end="|") + mr("<1")),
                run(":"),
            ),
            para(omathpara(geo)),
            para(run("ובנוסף "), omath(integ), run(" וגם "), omath(trig), run(".")),
            para(omathpara(mat)),
            para(run("הסתברות קלאסית: "), omath(heb_math)),
            para(run("זהות אוילר הממוספרת:")),
            para(omathpara(numbered)),
            para(
                run(
                    'טקסט מעורב: בשנת 2024 למדו 30 סטודנטים את הקורס Calculus 1 (חדו"א), והציון הממוצע היה 85.5%.'
                )
            ),
            para(
                run("משפט פיתגורס בכתב רגיל: a² + b² = c², וגם x"),
                run("2", vert="superscript"),
                run(" + y"),
                run("2", vert="superscript"),
                run(" = r"),
                run("2", vert="superscript"),
                run(" כאשר הכתיב עם כתב עילי."),
            ),
            para(
                run("שלבי הפתרון"),
                '<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="1"/></w:r>',
                run(":"),
            ),
            para(run("מציבים את הנתונים"), num=1),
            para(run("פותרים את המשוואה "), omath(mr("2x=6")), num=1),
            para(run("בודקים את התשובה"), num=1),
            para(run("סעיף ראשון"), num=2),
            para(run("סעיף שני"), num=2),
            table(
                [
                    [[run("פונקציה")], [run("נגזרת")]],
                    [[omath(sup(mr("x"), mr("n")))], [omath(mr("n") + sup(mr("x"), mr("n−1")))]],
                    [[omath(func("ln", mr("x")))], [omath(frac(mr("1"), mr("x")))]],
                ]
            ),
        ]
    )
    return build_docx(body, footnotes=["לפי הספר של שטיין, עמוד 12."], numbering=True)


def plain_docx() -> bytes:
    body = "".join(
        [
            para(run("סיכום קורס היסטוריה של עם ישראל"), style="Heading1"),
            para(run("בתקופת בית שני התרחשו שינויים רבים. המרד הגדול פרץ בשנת 66 לספירה.")),
            para(
                run(
                    "The Second Temple period (516 BCE – 70 CE) is discussed in chapter 3.",
                    rtl=False,
                ),
                rtl=False,
            ),
            para(run("שאלה למבחן: מה היו הגורמים למרד? (ראו עמ' 45–47).")),
        ]
    )
    return build_docx(body)


def write_all() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "calculus_he.docx").write_bytes(calculus_docx())
    (OUT / "plain_he.docx").write_bytes(plain_docx())
    (OUT / "corrupt.pdf").write_bytes(
        b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\ngarbage"
    )
    (OUT / "random.bin").write_bytes(bytes(range(256)) * 8)
    (OUT / "empty.pdf").write_bytes(b"")
    (OUT / "not_docx.docx").write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    good = calculus_docx()
    (OUT / "truncated.docx").write_bytes(good[: len(good) // 2])

    # hostile DOCX variants
    (OUT / "xxe.docx").write_bytes(
        _patch_docx(
            good,
            b"<w:document",
            (b'<!DOCTYPE d [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><w:document'),
        )
    )
    (OUT / "zipbomb.docx").write_bytes(_zip_bomb())

    _build_external_pdfs()

    soffice = shutil.which("soffice")
    if soffice:
        for name in ("calculus_he", "plain_he"):
            subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(OUT),
                    str(OUT / f"{name}.docx"),
                ],
                check=False,
                capture_output=True,
                timeout=180,
            )
            src = OUT / f"{name}.pdf"
            if src.exists():
                src.rename(OUT / f"{name}_libreoffice.pdf")
    print("fixtures written to", OUT, file=sys.stderr)


def _patch_docx(data: bytes, old: bytes, new: bytes) -> bytes:
    src = zipfile.ZipFile(io.BytesIO(data))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename == "word/document.xml":
                payload = payload.replace(
                    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', b""
                )
                payload = payload.replace(old, new, 1)
            z.writestr(info, payload)
    return buf.getvalue()


def _zip_bomb() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("[Content_Types].xml", FIXED_DATE), "<Types/>")
        zi = zipfile.ZipInfo("word/document.xml", FIXED_DATE)
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, b"<" + b" " * (600 * 1024 * 1024) + b"/>")
    return buf.getvalue()


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=300, check=False)


def _build_external_pdfs() -> None:
    """PDFs from real producers: pdfLaTeX (Computer Modern), Chromium (MathML, Hebrew)."""
    src = OUT / "src"
    if shutil.which("pdflatex"):
        for name in ("tex_en", "tex_hard", "tex_pages"):
            if (src / f"{name}.tex").exists():
                _run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"], src)
                if (src / f"{name}.pdf").exists():
                    shutil.move(str(src / f"{name}.pdf"), OUT / f"{name}.pdf")
    chrome = shutil.which("chromium") or next(
        iter(sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))), None
    )
    if chrome:
        for name in ("mathml_he", "mathml_he2"):
            _run(
                [
                    str(chrome),
                    "--headless",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT / (name + '.pdf')}",
                    f"{name}.html",
                ],
                src,
            )
    try:
        import pypdfium2 as pdfium

        if (OUT / "mathml_he.pdf").exists():
            pdf = pdfium.PdfDocument(str(OUT / "mathml_he.pdf"))
            bmp = pdf[0].render(scale=300 / 72, grayscale=True)
            _write_image_pdf(bmp, OUT / "scanned_he.pdf")
            pdf.close()
    except ImportError:
        pass
    if shutil.which("qpdf") and (OUT / "tex_en.pdf").exists():
        _run(
            [
                "qpdf",
                "--encrypt",
                "secret",
                "owner",
                "256",
                "--",
                str(OUT / "tex_en.pdf"),
                str(OUT / "encrypted.pdf"),
            ],
            OUT,
        )
        _run(
            [
                "qpdf",
                "--encrypt",
                "",
                "owner",
                "256",
                "--print=none",
                "--",
                str(OUT / "tex_en.pdf"),
                str(OUT / "owner_only.pdf"),
            ],
            OUT,
        )
    if shutil.which("qpdf"):
        for name in ("calculus_he_libreoffice", "mathml_he2"):
            if (OUT / f"{name}.pdf").exists():
                strip_tounicode(OUT / f"{name}.pdf", OUT / f"{name}.nounicode.pdf")


def strip_tounicode(src: Path, dst: Path) -> None:
    """Remove every /ToUnicode CMap: the font-recovery fixtures.

    The result behaves like PDFs from old print drivers (PScript5 + Ghostscript):
    the glyphs draw correctly but the text layer has no usable Unicode.
    """
    import re
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        qdf = Path(tmp) / "a.qdf.pdf"
        _run(["qpdf", "--qdf", "--object-streams=disable", str(src), str(qdf)], tmp)
        data = re.sub(rb"/ToUnicode \d+ 0 R", b"", qdf.read_bytes())
        stripped = Path(tmp) / "b.pdf"
        stripped.write_bytes(data)
        _run(["qpdf", "--deterministic-id", str(stripped), str(dst)], tmp)


def _write_image_pdf(bitmap, path: Path) -> None:
    """A one-page PDF containing only a grayscale image (a 'scan'), no text layer."""
    import zlib

    w, h, stride = bitmap.width, bitmap.height, bitmap.stride
    raw = bytes(bitmap.buffer)
    pixels = b"".join(raw[y * stride : y * stride + w] for y in range(h))
    data = zlib.compress(pixels, 9)
    pw, ph = w * 72 / 300, h * 72 / 300
    content = f"q {pw:.2f} 0 0 {ph:.2f} 0 0 cm /Im0 Do Q".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode(),
        f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode /Length {len(data)} >>\nstream\n".encode()
        + data
        + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.7\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(out.getvalue())


if __name__ == "__main__":
    write_all()
