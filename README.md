# Hebrew-English Concept Extraction

Turns a course summary into the list of concepts a student needs to learn from it.

The input is a PDF or DOCX written in Hebrew, usually mixed with English terms and math. The
output is up to 150 concepts, ranked by importance, each with the passage from the document that
best introduces it. Linking the concepts into a map is not built yet (see
[Limitations](#limitations-and-known-issues)).

**[Try the live demo](https://yoavmalino.github.io/Hebrew_English_Concept_Extraction/)**: step
through every stage on sample documents, or upload your own file.

## Why this is hard, and what this project does about it

- **Hebrew PDFs store text in visual order.** Copying text out of one gives reversed words and
  scrambled formulas. The extractor rebuilds the reading order with its own implementation of the
  Unicode bidirectional algorithm.
- **Formulas are scattered glyphs and lines.** Fractions, roots, sums, limits and matrices are
  rebuilt from glyph positions and drawn lines, and come out as LaTeX.
- **Some PDFs have no usable text layer.** Some PDF printers embed fonts with no character
  mapping. Instead of OCR, which would lose all math structure, each glyph is matched by shape
  against an atlas of about 2,900 reference glyphs.
- **Hebrew morphology.** One word appears as "גרף", "הגרף", "בגרף". Attached prefixes are
  stripped by a cautious rule, so these count as one term.
- **Two languages for one idea.** A term written as "אנטרופיה (entropy)", an acronym and its
  expansion, or two spellings of one word are recognized as candidates for the same concept.
- **No model needed.** Extraction is geometric and rule-based, so the same file always gives
  byte-identical output. A language model can be added at three points (see below), but the
  default runs with no API key.

## How it works

Six stages. Each takes the previous stage's output; stages 2-6 are pure functions with no
database and no network.

1. **Extraction** (`extractor/`, package `hebmath`). Reads every glyph's position, size and font
   from the PDF (or the XML of a DOCX), groups glyphs into lines by baseline, builds math
   structures from geometry, turns visual order into logical order, and joins lines into
   paragraphs. It also records the structure later stages use: headings, units such as
   definitions and examples, and which words are bold, underlined, larger or on their own line.
2. **Candidates.** Collects every phrase that might be a concept, from seven sources:
   definitions ("X is called…"), headings, emphasized words, English glosses in parentheses,
   label lines, math operators, and phrases of up to four words repeated in the text. Recall comes
   first: a concept lost here never comes back, so most candidates are noise at this point.
3. **Images.** In development: extracting and tagging concept candidates from images and
   diagrams. For now the pipeline goes straight from stage 2 to stage 4.
4. **Merging.** Finds pairs of candidates that may name one concept: a term and its English
   gloss, an acronym and its expansion, spelling variants, one-letter typos, a named theorem and
   its bare name. Pairs a rule can decide safely are merged; in model mode a model decides the
   rest. Grouping is deterministic.
5. **Importance.** Scores each concept from 0 to 1 using eight signals: is it defined or a named
   result, how widely it is spread, how specific it is to this course compared with others, is it
   tied to formulas, is it bold, underlined, in a larger font or on its own line. The weights were
   fitted on real course summaries. A signal that marks too much of a document (for example,
   everything bold) passes its weight to the others.
6. **Selection.** Picks the final list, up to 150. Defined terms, named results and heading topics
   enter first; the rest enter greedily while they add importance. Each concept keeps the passage
   that best introduces it.

### Algorithmic mode and model mode

The default **algorithmic mode** needs only the code. Stage 4 merges only what its rules can
decide; every other pair stays apart, the safe side. This is what the demo runs.

**Model mode** adds three kinds of LLM calls: base forms of Hebrew words (stage 2), finders for
translations and synonyms no rule can see, and a decider for each proposed pair, by majority of
three runs (stage 4). Every call goes through `llm_cache.py`, which records answers and replays
them, so a rerun sends nothing.

Stage 5's specificity signal compares a term's frequency with other courses. With no other
courses to compare against (as in the demo), it is off.

## Install and use

Python 3.11 or later (the extractor alone runs on 3.10).

```bash
pip install ./extractor ./pipeline
```

```python
from hebmath import Extractor, ExtractorConfig

from concept_pipeline import stage5_importance as s5
from concept_pipeline import stage6_selection as s6
from concept_pipeline.stage2_candidates import extract_candidates
from concept_pipeline.stage4_merging import absorb_fragments, decide_by_rule, find_pairs, merge

result = Extractor(ExtractorConfig()).extract_bytes(data, filename="summary.pdf")
stage2 = extract_candidates(result)
pairs, _ = find_pairs(stage2.candidates)
decisions, _undecided = decide_by_rule(pairs)
concepts, _absorbed = absorb_fragments(merge(stage2, decisions))
evidence = s5.file_evidence(result, concepts, background=None)
importance = s5.file_importance(evidence)
selection = s6.select_nodes(s6.space_concepts([evidence], importance))
```

The extractor also works on its own, from the command line:
`hebmath summary.pdf -o summary.txt --report report.json`. Its output format is specified in
[extractor/docs/FORMAT.md](extractor/docs/FORMAT.md), and its design, licensing choices and
edge cases are in [extractor/README.md](extractor/README.md).

Scanned pages need [Tesseract](https://github.com/tesseract-ocr/tesseract) with the Hebrew
model. Without it, scanned pages are marked unreadable and the rest of the document is still
read.

## Demo

- **`demo/web`**: a static page on GitHub Pages. It shows every stage for four sample documents,
  with the original beside the extracted text, and lets visitors upload their own file.
- **`demo/api`**: a small FastAPI service with one endpoint, `POST /analyze`, hosted on Render's
  free tier (`render.yaml`). The uploaded file is analyzed in memory and discarded; nothing is
  stored and requests are not logged.

The four samples are generated documents: course summaries written by a language model for
testing, not real course material.

Run it locally:

```bash
pip install ./extractor ./pipeline -r demo/api/requirements.txt
cd demo/api && uvicorn app:app --port 8766
```

```bash
python -m http.server 8765 --directory demo/web
```

## Evaluation

The gold concept lists come from a committee of three models (Claude Opus 5.5, GPT-6.1 Sol and
Gemini 4 Carbon) that label each course blindly; a concept counts when two of three agree. The
committee was checked against the author's own judgement on a sample. Courses are split into a
development set, used for all tuning, and a held-out set, measured once at the end.

The course files, labels and results are not in this repository, because they come from real
course material. The tools that produced them are: `scripts/concept_committee.py` (the
committee), `scripts/stage2_*` and `scripts/stage4_*` (stage reports; they call a model only with
`--send`), and `calibration/` (the stage 5 and 6 settings, see
[calibration/README.md](calibration/README.md)).

## Limitations and known issues

**Scope**

- **Linking the concepts to each other is not built yet.** The output is a ranked list of
  concepts, not a concept map: relations between concepts (prerequisite, part of, example of)
  are planned and do not exist in any form, here or elsewhere.
- This extraction is one part of a larger, private project: an adaptive learning platform that
  will build a personal concept map from a student's course material and practice it with them.
- Stage 3 (images and diagrams) is not implemented. Text inside images is ignored.

**Extraction: not every PDF converts correctly**

- Scanned pages are not read on the demo server (no OCR there). Formulas on scanned pages are
  read only when they are simple and linear, even with Tesseract installed.
- In fonts with no character mapping, some Hebrew letters of similar shape are confused, most
  often ש and ט.
- Two-column layouts are not supported: lines from both columns merge.
- Old equation objects in Word files (Equation Editor 3.0, MathType) are marked, not converted.
- PDFs printed from Chrome with math on a right-to-left page can mirror some relation signs
  (`<` and `>`).
- The full list is in [extractor/README.md](extractor/README.md#known-limitations).

**Concept quality**

- The selection always fills its list, so a document with few real concepts still gets many, and
  frequent generic words ("טבלה", "עובד") can rank too high.
- Without a model, synonyms with no shared form (two Hebrew words for one idea, an acronym that
  does not spell its expansion) stay as separate concepts.
- The gold labels come from models checked on a sample, so measured precision is a lower bound:
  a valid concept the committee did not list counts as noise.

**Demo**

- The free server is slow, roughly 2 seconds per page, and sleeps when idle: the first upload
  after a quiet period waits up to a minute longer.
- Files are limited to 10 MB and 150 pages.

**Development**

- The extractor's golden test files are a Linux reference (as in CI), so two of those tests fail
  on macOS.
- The code was developed inside the private project and moved here without its history.
  Docstrings that mention "the design", "the plan" or "owner decisions" refer to design
  documents that are not included.

## Development

```bash
pip install -e extractor -e pipeline -r requirements-dev.txt
pytest
ruff check . && ruff format --check .
```

## Credits

**Libraries in the product**

- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) (Apache-2.0 or BSD-3-Clause, used
  under BSD-3-Clause) and [PDFium](https://pdfium.googlesource.com/pdfium/) (BSD-3-Clause): the
  PDF engine and the extractor's only runtime dependency. Installed by pip, not copied into this
  repository; their authors do not endorse this project. PDFium bundles FreeType, libjpeg-turbo, libpng,
  zlib, OpenJPEG, LittleCMS, libtiff, ICU and others. Portions of this software are copyright ©
  The FreeType Project (www.freetype.org). All rights reserved. This software is based in part on
  the work of the Independent JPEG Group.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (Apache-2.0), optional, for
  scanned pages.
- The glyph atlas is built from fonts under the SIL Open Font License: STIX Two Math, Libertinus
  Math, Noto Sans Math, Tinos, Arimo, David Libre, Frank Ruhl Libre, Miriam Libre, Alef, Noto Serif
  Hebrew and Noto Sans Hebrew. Only shape descriptors are shipped, no font files.

**Demo**

- [FastAPI](https://fastapi.tiangolo.com) (MIT) and [Uvicorn](https://www.uvicorn.org)
  (BSD-3-Clause): the demo server.
- [pdf.js](https://github.com/mozilla/pdf.js) (Apache-2.0),
  [docx-preview](https://github.com/VolodymyrBaydalka/docxjs) (Apache-2.0) and
  [JSZip](https://stuk.github.io/jszip/) (MIT or GPL-3.0, used under MIT): show the original
  document and count its pages in the browser. Unmodified copies with their licenses are in
  [demo/web/vendor](demo/web/vendor/README.md).
- Hosted on [GitHub Pages](https://pages.github.com) and [Render](https://render.com).

**Development and evaluation**

- [pytest](https://pytest.org) (MIT), [Ruff](https://github.com/astral-sh/ruff) (MIT),
  [fontTools](https://github.com/fonttools/fonttools) (MIT, builds the glyph atlas),
  [HTTPX](https://www.python-httpx.org) (BSD-3-Clause) and the
  [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) (MIT).
- The bidirectional reordering follows the Unicode Standard Annex #9 (UAX #9).

Exact versions, licenses and obligations: [extractor/THIRD_PARTY_NOTICES.md](extractor/THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
