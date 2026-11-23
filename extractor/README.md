# hebmath: Hebrew text and LaTeX math from PDF and DOCX

Stage 1 of [Hebrew-English Concept Extraction](../README.md), usable on its own. It takes a PDF
or DOCX and returns plain text in correct reading order, with every formula in place as LaTeX:
`$...$` inline and `$$...$$` as a block. It also returns the document's structure (headings,
units, emphasis). The output contract is [docs/FORMAT.md](docs/FORMAT.md) (version 2: each
formula on its own line, images and drawings dropped).

```python
from hebmath import Extractor, ExtractorConfig

extractor = Extractor(ExtractorConfig())  # one instance per worker
result = extractor.extract_bytes(data, filename="summary.pdf")
if result.ok:
    save_text(result.text)  # the text stands on its own
    save_report(result.to_dict())  # status, warnings, stats, engine versions
else:
    log(result.error_code, result.error_message)  # corrupt_file / encrypted_file / ...
```

```bash
hebmath summary.docx -o summary.txt --report summary.json
hebmath scan.pdf --no-ocr
hebmath input/notes.pdf --out-dir output        # -> output/notes/notes.txt + report.json
hebmath summary.pdf --math-line-breaks none     # format v1: formulas inside the line
```

`extract()` never raises on bad input. A corrupt, encrypted, empty or oversized file returns
`ok=False` with a stable error code.

## Why Python

- **PDF engine.** pypdfium2 (Apache-2.0/BSD-3) wraps Chrome's PDFium and gives, for every glyph,
  its ink box, baseline, size and font, plus vector lines. It is fast and robust on damaged
  files. pdf.js gives only text runs, with no per-glyph ink boxes, which makes fractions and
  roots nearly impossible to detect.
- **DOCX/OMML.** `zipfile` and `xml.etree` from the standard library are enough; the OMML to
  LaTeX converter is written here from the standard.
- **OCR.** Tesseract (Apache-2.0) runs as an external process.
- **Future formula models** (such as PP-FormulaNet) are mostly available in Python.

**The only runtime dependency is pypdfium2.** Everything else is the standard library.

## Architecture

```
bytes ─► sniff_format ─► DocxReader ──┐
                     └─► PdfReader ───┼─► Document (IR) ─► serialize() ─► text v2
                          ├ extract.py      (PDFium: glyphs, font identity, lines, images)
                          ├ glyphs/         (decode fonts with no Unicode mapping by glyph shape)
                          ├ vector.py       (classify lines: fraction bar/overline, |, √, ⟨⟩, drawings)
                          ├ mathengine.py   (fractions/roots/limits/accents + baseline parser -> LaTeX)
                          ├ layout.py       (lines, tokens, math regions, bidi, paragraphs)
                          └ ocr.py          (Tesseract for scanned pages)
```

| Module | Role | How to replace it |
|---|---|---|
| `model.py` | Intermediate representation: paragraphs, inline/display formulas, placeholders | – |
| `serialize.py` | The only component that knows the output format | – |
| `format.py` | Reference parser for the format, for later stages | – |
| `latex/omml.py` | OMML to LaTeX, a deterministic tree walk | `ExtractorConfig(omml_converter=...)` |
| `readers/pdf/mathengine.py` | Geometric PDF math engine, no ML | `ExtractorConfig(pdf_math_engine=...)` |
| `readers/pdf/ocr.py` | OCR for scanned pages | `ExtractorConfig(ocr_engine=...)` |
| `latex/plaintext.py` | Detects math typed as plain characters (`a² + b² = c²`) | `detect_plain_math=False` |
| `readers/pdf/glyphs/` | Decodes fonts with no ToUnicode map by glyph shape | `ExtractorConfig(glyph_recognizer=...)`; `recover_fonts=False` |
| `readers/pdf/vector.py` | Classifies vector paths as math symbols or drawings | – |
| `bidi.py` | Own implementation of UAX #9 (W, N, I and L2 rules), visual to logical order | – |

### How it works

**DOCX.** The file is read directly as ZIP and XML. Word stores text in logical order, so mixed
Hebrew and English come out right without reordering. OMML equations are converted by structure:
`m:f` to `\frac`, `m:rad` to `\sqrt[n]{}`, `m:nary` to `\sum`/`\int` with limits, `m:d` to
delimiters, `m:m` to `matrix`, `m:func` to `\sin`, `m:acc` to `\hat`/`\vec`, Word numbering
`#(1)` to `\tag{1}`, Hebrew inside a formula to `\text{}`. Numbered lists (including Hebrew
א/ב/ג numbering), tables, footnotes, text boxes, tracked changes (deletions dropped),
`mc:AlternateContent` and ordinary superscripts (`x` + superscript "2" to `$x^{2}$`) are handled
too.

**PDF with a text layer.**
1. PDFium returns every glyph's Unicode, ink box, baseline origin, size and font. Thin vector
   lines (fraction bars, root bars) are collected.
2. The math engine builds vertical structures inside out. A fraction is a line with a numerator
   right above and a denominator right below, within its width and with no neighbors on the same
   baseline, so an underline or table border is not taken for a fraction. A root is a √ with a
   bar on top. Σ, Π and lim take limits above and below; accents (`\hat`, `\bar`, `\vec`) are
   found by vertical overlap.
3. Lines are built by baseline; small raised or lowered glyphs attach as powers and indices.
4. Each line is split into tokens, classified as Hebrew, Latin prose, "strong" math (a math font
   such as CMMI or Cambria Math, a math character, a structure) or "weak" math (digits,
   brackets). A run of math tokens with at least one strong token is a formula.
5. The formula becomes one atomic left-to-right unit, then the bidi algorithm turns the line from
   visual to logical order, mirroring brackets. A line that is all formula becomes `$$...$$`,
   and an equation number in the margin becomes `\tag`.
6. Lines are joined into paragraphs; page numbers and repeated running heads are removed.

**PDF fonts with no Unicode mapping.** PDF printers such as PScript5 with Ghostscript embed
TrueType subsets whose codes follow order of use, with no ToUnicode table, so the text layer
looks like garbage. Instead of OCR on the page, which loses all math structure, each distinct
glyph is rendered alone and described by a 16×16 coverage grid, its ink box and its topology
(components and holes). The description is matched against an atlas of about 2,900 reference
glyphs from OFL fonts (math, Hebrew, Latin). The choice is then corrected by context (Hebrew vs
Latin, final vs regular letter forms) and optionally cross-checked by OCR of single words.
Coordinates and structure survive, so fractions, indices and matrices do too. A glyph that is
not decoded with confidence is counted in `stats.unresolved_glyphs` with a warning.

**Scanned PDF.** A page with no text layer is rendered at 300 DPI and sent to Tesseract with
`heb+eng`. Formulas on scanned pages are recognized only when they are linear (see limitations).

## Licensing

### What was chosen

| Component | License | Why | Obligations |
|---|---|---|---|
| **pypdfium2** 5.13.0 | Apache-2.0 or BSD-3-Clause | The only permissive engine with exact glyph ink boxes and vector lines; robust on damaged files | Keep copyright and disclaimer when redistributing |
| PDFium (inside pypdfium2) | BSD-3-Clause | – | Same |
| Libraries compiled into PDFium | FreeType (FTL), libjpeg-turbo (IJG/BSD), libpng, zlib, lcms2 (MIT), OpenJPEG (BSD-2), libtiff, Abseil (Apache-2.0), ICU (Unicode) and others | – | **FreeType and libjpeg require attribution in documentation**; exact wording in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |
| Python standard library (`xml.etree`/expat, `zipfile`, `unicodedata`) | PSF-2.0 / MIT | Hardened XML parsing for DOCX | Keep notice |
| **Tesseract** 5.x + Leptonica (optional, external process) | Apache-2.0 / BSD-2 | The most mature permissive OCR for Hebrew | Keep LICENSE/NOTICE if you ship the binary (for example in a Docker image) |
| `heb.traineddata` (tessdata_best) | Apache-2.0 | The model weights' license was checked separately | Same |
| Glyph atlas `data/glyph_atlas.json.gz` | Derived from **SIL OFL 1.1** fonts (STIX Two Math, Libertinus Math, Noto Sans Math, Tinos, Arimo, David Libre, Frank Ruhl Libre, Miriam Libre, Alef, Noto Serif/Sans Hebrew) | OFL fonts only, which allow commercial use. The atlas holds rendered shape descriptors (16×16 grids), not font programs | OFL does not cover rendered output; attribution is kept anyway. fontTools (MIT) is used only to build the atlas |

hebmath is distributed as source under Apache-2.0 (see [LICENSE](LICENSE) and
[NOTICE](NOTICE)). Anyone who ships a built package or a Docker image should include these
notices.

### What was considered and rejected

| Alternative | Problem |
|---|---|
| PyMuPDF / MuPDF | **AGPL-3.0**: a network service must publish its source, or buy a commercial license |
| **lxml** | The library is BSD, but the official wheels **statically link GNU libiconv under LGPL-2.1** (stated in the package's LICENSES.txt). Replaced by the standard `xml.etree` |
| python-bidi | **LGPL**, so UAX #9 was implemented here |
| Nougat (Meta) | Code is MIT, but the **weights are CC-BY-NC** (non-commercial) |
| texify | Code **GPL-3.0**, weights **CC-BY-SA-4.0** |
| Surya / Marker (Datalab) | Weights under **CC-BY-NC-SA** or a **modified OpenRAIL-M** with revenue and funding limits and a "competitor" clause |
| MinerU | Custom license (Apache-2.0 plus extra terms: MAU and revenue thresholds, attribution in online services) |
| pix2tex (LaTeX-OCR) | Code is MIT, but the weights have no stated license and were trained on im2latex-100k (arXiv sources) with no clear license |
| MathType / Microsoft's OMML2MML.XSL | Proprietary |

**A future ML engine that fits:** PaddleOCR's PP-FormulaNet, whose code and model card are both
Apache-2.0. It can be plugged in as an `OcrEngine` or `PdfMathEngine` without changing the rest;
check the licenses of all its dependencies first and pin a version and seed.

> This review is not legal advice. Have counsel read the full texts in
> `site-packages/pypdfium2-*/licenses/` before distributing binaries.

## Settings worth knowing (`ExtractorConfig`)

| Field | Default | What it does |
|---|---|---|
| `math_line_breaks` | `"formulas"` | `"formulas"`: every formula that is not a single symbol on its own line. `"all"`: every formula. `"none"`: format v1 |
| `recover_fonts` | `True` | Decode fonts with no Unicode mapping by shape. `False` falls back to OCR of the page |
| `recovery_ocr_pages` | `6` | Maximum pages OCR'd to cross-check the decoding. `0` turns it off (on one test file: about 10 s instead of 25) |
| `glyph_recognizer` | `None` (atlas) | An alternative glyph recognizer, such as a model, implementing `candidates/tall/distance_to/version` |
| `pdf_math_engine`, `omml_converter`, `ocr_engine` | built in | Replace the math engine, the OMML converter or the OCR |
| `enable_ocr`, `detect_plain_math`, `page_markers` | `True`, `True`, `False` | As in version 1.0 |

## Stability and reproducibility

- Every method is deterministic: no statistical model, no randomness, no dependence on `dict` or
  `set` order. Every sort uses a full key with a tie-breaker (the glyph's position in the
  content stream).
- `test_deterministic_across_processes` runs every file in two processes with different
  `PYTHONHASHSEED` values and checks the output is byte-identical.
- The versions that determine the output are returned in `result.engines` (hebmath, pypdfium2,
  PDFium, Tesseract). Store them next to the text.
- The glyph atlas is built once by `tools/build_glyph_atlas.py` from fonts at pinned commits and
  ships as package data. There is no training at run time. Its version is in
  `engines.glyph_recognizer`. The OCR cross-check changes a decision only when it is
  unambiguous.
- pypdfium2 is pinned to an exact version in `pyproject.toml`. The golden files are a Linux
  reference (as in CI); two golden tests fail on macOS. Run Tesseract single-threaded
  (`OMP_THREAD_LIMIT=1`), because multi-threaded runs are not fully deterministic.
- Upgrading any of these can change the output, which is why `extractor_version` is stored in
  the report and the golden-file tests catch any change.

## Edge cases

| Case | Behavior |
|---|---|
| Formula-heavy document | Tested on 100 mixed pages: about 1.7 s |
| No text layer (scanned) | Hebrew and English OCR, a warning, `stats.pages_ocr`. Without Tesseract: `[[hebmath:unreadable-page]]` and a warning |
| Garbage text layer (fonts with no Unicode) | Decoded by glyph shape, no page OCR. With `recover_fonts=False`: over 30% unmapped characters sends the page to OCR |
| Images, diagrams, vector drawings | Dropped with no marker; text labels inside drawings are dropped too (`stats.figure_glyphs_dropped`) |
| A glyph PDFium reports as a space (code 0x20 in an encoded font) | Restored from uncovered ink (`stats.restored_glyphs`) |
| Corrupt or truncated file | `corrupt_file` (PDFium repairs light damage itself). A fuzz test of 300 damaged files: zero crashes |
| Encrypted PDF | `encrypted_file`; a PDF with only an owner password is read normally |
| Old `.doc` / encrypted DOCX | `unsupported_format` / `encrypted_file` |
| Zip bomb, XXE/DTD in DOCX | `resource_limit` / `corrupt_file` |
| Document with no math | Clean text with no `$` (tested) |
| One broken page in a good document | The page is marked; the rest is extracted |

Run each extraction in its own worker process with an external timeout: PDFium is not
thread-safe.

## Known limitations

1. **Formulas on scanned pages.** Tesseract does not see 2D structure. Fractions, powers and
   roots in a scan come out garbled or are dropped; only simple linear formulas are recognized.
   The fix is a model such as PP-FormulaNet as an `OcrEngine`.
2. **Old equations (Equation Editor 3.0 / MathType)** in DOCX are binary OLE objects. They are
   marked `[[hebmath:math-object]]` and not converted. A formula saved as an image is dropped
   with the other images, per format v2.
3. **Partial PDF structures.** Matrices, `cases` and tall brackets are recognized (also when a
   bracket is built from several pieces), but a sparse or diagonal matrix gets an approximate
   row and column layout. Column separators (`P=(v_1|\dots|v_n)`), aligned `align` systems and
   `\overbrace`/`\underbrace` are not recognized as structures.
4. **Fonts with no Unicode.** Shape decoding is very accurate for common fonts
   (Times/Arial/David/Symbol/Cambria). For a rare Hebrew font without OCR accuracy drops (in the
   FreeSerif test file: similarity about 0.67 without OCR, 0.88 with it). Type3 fonts with no
   Unicode are decoded per page. The work is pure Python: about 10 s for 17 pages of such a file
   (a normal file does not pay this cost).
5. **Two-column layouts** are not supported: lines from both columns at the same height merge.
   Student summaries are almost always single-column.
6. **Subscript vs footnote** is decided by context: a raised digit next to a Hebrew word stays
   text ("מילה1").
7. **Chrome PDFs with MathML on an RTL page.** Chromium writes mirrored codes for some symbols in
   the text layer (`<`/`>`, `∈`/`∋`) even though the page looks right. Mirrored brackets are fixed
   automatically; relation signs cannot always be. Word, LibreOffice and LaTeX are not affected.
8. **Niqqud** is kept as in the PDF. Diacritics the PDF split from their letter are reattached by
   overlap, so rare cases may come out in a different order.
9. **Font sizes and colors** are not kept in the text; headings and emphasis are reported in the
   structure.
10. **Math typed as plain text in Word** is recognized only when it has an unambiguous math sign
    (`=`, `²`, `√`, a Greek letter). "5 + 3" with no equals sign stays text, on purpose.
11. **An operator touching Hebrew.** When a sign such as `-` or `=` sits between a Hebrew word
    and a formula with no space, its logical order is ambiguous, and in rare cases it attaches to
    the wrong side.

## Development and tests

```bash
pip install -e .[dev]
python tests/make_fixtures.py     # builds the fixtures (some need pdflatex, Chromium or qpdf)
pytest -q
```

The test corpus includes a Hebrew DOCX with OMML (fractions, roots, sums, matrices, numbering,
tables, footnotes), PDFs from pdfLaTeX, from Chromium with Hebrew and MathML, and from
LibreOffice, a scanned PDF, and eight hostile or damaged files. Each file's output is compared
with a golden file. All fixtures are synthetic.
