"""Rebuild the structure fixtures used by test_structure.py (needs Brave/Chromium and pandoc).

The PDFs are printed from tests/fixtures/src/*.html with a Chromium browser:
    <browser> --headless=new --no-pdf-header-footer --print-to-pdf=OUT.pdf file://SRC.html
``struct_he_outline.pdf`` is struct_he.html printed with the extra flag --generate-pdf-document-outline
(it has bookmarks; Chromium stores their titles in visual order).
``struct_he_styles.docx`` comes from pandoc (real Heading styles). ``struct_he_manual.docx`` is
written here by hand: no styles part, headings only look like headings (large, bold runs), which
is the hard case for Word files.
"""

import subprocess
import zipfile
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

MARKDOWN = """# פרק ראשון: אלגברה

טקסט רגיל עם **מושג מודגש** באמצע ועם [מושג עם קו תחתון]{.underline} בנוסף.

## תת כותרת ראשונה

הגדרה: טקסט רגיל כדי שהגודל השכיח יהיה ברור. עוד טקסט רגיל כדי שהגודל השכיח יהיה ברור.

## תת כותרת שנייה

סוף המסמך עם טקסט רגיל נוסף לבדיקה.
"""

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)
ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)


def paragraph(text: str, half_points: int, bold: bool) -> str:
    props = f'<w:sz w:val="{half_points}"/>' + ("<w:b/>" if bold else "")
    return f"<w:p><w:r><w:rPr>{props}</w:rPr><w:t>{text}</w:t></w:r></w:p>"


def build_manual_docx(path: Path) -> None:
    body = "".join(
        [
            paragraph("פרק ראשון: אלגברה", 40, True),
            paragraph(
                "טקסט רגיל ארוך מספיק כדי שיהיה ברור שזו פסקה ולא כותרת, והוא משמש כגוף המסמך.",
                24,
                False,
            ),
            paragraph("תת כותרת ראשונה", 30, True),
            paragraph(
                "עוד טקסט רגיל ארוך מספיק כדי שיהיה ברור שזו פסקה ולא כותרת, והוא משמש כגוף המסמך.",
                24,
                False,
            ),
        ]
    )
    document = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", ROOT_RELS)
        zf.writestr("word/document.xml", document)


def build_styles_docx(path: Path) -> None:
    subprocess.run(
        ["pandoc", "-f", "markdown", "-o", str(path)], input=MARKDOWN.encode(), check=True
    )


if __name__ == "__main__":
    build_manual_docx(FIXTURES / "struct_he_manual.docx")
    build_styles_docx(FIXTURES / "struct_he_styles.docx")
