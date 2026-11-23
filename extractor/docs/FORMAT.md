# Output format: `hebmath text v2`

This is the contract between the extractor and the later pipeline stages. Any change to it bumps
the version (`extractor_version`). Version 2 (hebmath 1.1.0) differs from version 1 in three
ways only:

1. Inside a paragraph, a formula and the text next to it are on separate lines.
2. Images and embedded objects are dropped with no marker.
3. Table-of-contents leader dots ("......") are collapsed.

For version 1 output use `math_line_breaks="none"` (or `--math-line-breaks none`). The parser
`hebmath.format.parse()` reads both versions.

## General

- UTF-8 without a BOM. Line endings are `\n` only.
- Text is in **logical order** (reading and typing order), not visual order. Bidi control
  characters are always removed: U+200E, U+200F, U+202A–U+202E and U+2066–U+2069.
- Unicode normalization is NFC. In addition, Hebrew and Latin presentation forms in
  U+FB00–U+FB4F are decomposed to their ordinary forms (for example the `ﬁ` ligature, or a
  precomposed `שׁ`).
- The file is a sequence of **blocks**:
  - exactly one empty line (`\n\n`) separates two blocks;
  - the file ends with a single `\n`, and an empty document gives an empty string;
  - a paragraph block may span several physical lines (see "Line breaks around formulas"); every
    other block is one line.

## Block types

| Block | Form | Notes |
|---|---|---|
| Paragraph | Free text with inline formulas and placeholders | Lines wrapped in the source are joined. Line breaks inside a paragraph appear only around formulas |
| Display formula | `$$<latex>$$` alone on its line | With an equation number: `$$<latex> \tag{<n>}$$` |
| Table row | Cells separated by ` \| ` (space, bar, space) | DOCX only. Always one line, even when it holds formulas |
| Footnotes (DOCX) | A `———` line followed by paragraphs starting with `[n] ` | The reference in the body is `[n]` |

## Formulas

- **Inline**: `$<latex>$`, inside a paragraph, in its place in the text.
- **Display**: `$$<latex>$$`, always a separate block on its own line.
- `<latex>` never holds an unescaped `$` or a newline. **A formula never spans lines.**
- The LaTeX uses only base LaTeX, `amsmath` and `amssymb` commands, so it works with KaTeX and
  MathJax. For example:
  - `\frac`, `\sqrt[n]{}`, `\overline{}`;
  - `\sum_{}^{}`, `\int_{}^{}`, `\lim_{}`, `\binom`;
  - `\left( … \right)`;
  - `\begin{pmatrix}`, `bmatrix`, `vmatrix`, `Vmatrix`, `Bmatrix`, `cases`, `matrix`;
  - `\text{}`, `\operatorname{}`.
- Hebrew text inside a formula is written as `\text{...}` in logical order.
- Equation numbers appear only in display formulas, as a final `\tag{...}`.

### Line breaks around formulas (new in v2)

The default is `math_line_breaks="formulas"`. Inside a paragraph:

1. **Every formula that is not a single symbol is on its own line**: there is a line break
   between it and the text before and after it. For example:
   ```
   נגדיר את ההעתקה
   $T:V\to W$
   ונניח כי
   $\dim V=n$.
   ```
2. **A single symbol stays inside the sentence.** For example:
   - `$V$`, `$\lambda$`, `$\mathbb{R}$`, `$A_{n\times n}$`, `$x^{2}$`, `$\overline{z}$`;
   - anything `hebmath.serialize.is_simple_math()` accepts.

   **Why:** breaking the line around every variable name tears the sentence apart and makes it
   hard to read, with no benefit to analysis. Such a symbol is part of the sentence, not a
   formula of its own.
3. **Punctuation that closes a formula stays on its line**, so a line never starts with
   punctuation. This covers `.`, `,`, `;`, `:`, `)`, `]`, `?`, `!`, `%` and quotes.
4. **An opening bracket right before a formula moves to the formula's line**, so a line never
   ends with `(`.
5. **List numbering** (`1.`, `א.`, `(a)`) right before a formula stays in front of it on the
   same line.
6. **A Hebrew prefix letter with a hyphen** stays attached to the formula, as in `ו-$x>0$` or
   `כש-$n\to\infty$`. The letter is grammatically part of the expression, and splitting it off
   changes the meaning. In borderline cases the rule favors keeping the formula whole.
7. **Consecutive formulas** with no text between them share one line, separated by a space.
8. **Table rows** are never broken.

Other options:

- `math_line_breaks="all"`: a single symbol gets its own line too.
- `math_line_breaks="none"`: v1 behavior, everything on one line.

### Parsing

Use the reference parser `hebmath.format.parse()`, which is tested against the output.

- It returns `ParsedBlock` objects with `parts`: the whole paragraph, its lines joined by a space.
- It also gives `lines`: a list of parts per physical line.

The algorithm:

- A line that starts with `$$` and ends with `$$` is a display formula.
- Paragraph lines are scanned left to right:
  - `\` followed by `\` or `$` is an escaped character;
  - an unescaped `$` opens a formula that runs to the next unescaped `$`. Inside a formula, `\`
    skips the next character.
- Every formula starts and ends on the same line.

## Escaping in plain text

Only two characters are escaped outside formulas:

| In the source | In the output |
|---|---|
| `\` | `\\` |
| `$` | `\$` |

So a `$` in the text (a price, say) is never read as the start of a formula. To restore the
original text: `\\` to `\`, `\$` to `$`.

A run of six or more dots (table-of-contents leaders) is written as ` … `.

## Placeholders

**Images, diagrams and embedded objects are ignored entirely.** No extraction, no description, no
OCR, and no marker in the output. This covers:

- images in DOCX;
- OLE objects that are not equations;
- images in PDF;
- text labels inside PDF vector drawings. These are detected and dropped, and their count is in
  `stats.figure_glyphs_dropped`.

Equations, on the other hand, **are** extracted, whether they are OMML objects, text or glyphs,
and including vector lines such as fraction bars, `\overline`, `|x|` and `√`.

A marker remains only for mathematical or textual content that exists in the source and cannot be
converted:

| Marker | Meaning |
|---|---|
| `[[hebmath:math-object]]` | An old Equation Editor 3.0 or MathType equation (OLE object) that was not converted |
| `[[hebmath:unreadable-page]]` | A PDF page with no text layer while OCR is off or unavailable, or a damaged page |
| `[[hebmath:page=N]]` | A page boundary (only with `page_markers=True`) |

Pattern to match them: `\[\[hebmath:[a-z0-9=\-]+\]\]`.

## Report (JSON)

`ExtractionResult.to_dict()` (or `--report` on the command line) is saved next to the text:

```json
{
  "ok": true,
  "format": "pdf",
  "sha256": "<hash of the input bytes>",
  "extractor_version": "1.1.0",
  "error_code": null,
  "error_message": null,
  "warnings": ["..."],
  "stats": {"display_math": 3, "inline_math": 8, "pages": 2, "pages_ocr": 0,
            "recovered_fonts": 7, "recovered_glyphs": 175, "unresolved_glyphs": 1},
  "engines": {"hebmath": "1.1.0", "pypdfium2": "5.13.0", "pdfium": "153.0.7999.0",
              "glyph_recognizer": "atlas v1 (2935 refs)"}
}
```

`stats` fields added in 1.1:

| Field | Meaning |
|---|---|
| `recovered_fonts` | Fonts with no Unicode mapping decoded by glyph shape |
| `recovered_glyphs` | Glyphs decoded |
| `unresolved_glyphs` | Glyphs not decoded with confidence. Above 0, a warning is added too |
| `recovery_ocr_pages` | Pages OCR'd only to cross-check the decoding |
| `assembled_delimiters` | Tall brackets assembled from several pieces |
| `restored_glyphs` | Glyphs PDFium reported as spaces and that were restored |
| `figure_glyphs_dropped` | Characters dropped because they are labels inside a drawing |

Stable error codes: `empty_file`, `unsupported_format`, `corrupt_file`, `encrypted_file`,
`resource_limit`, `timeout`, `io_error`, `internal_error`.
