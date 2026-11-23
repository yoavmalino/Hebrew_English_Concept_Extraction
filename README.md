# Hebrew-English Concept Extraction

Give it a course summary (PDF or DOCX) written in Hebrew, often mixed with English terms and
math. You get back the concepts a student needs to learn from it: about 150 of them, ranked, each
with a passage from the document that defines or uses it.

**Demo:** [yoavmalino.github.io/Hebrew_English_Concept_Extraction](https://yoavmalino.github.io/Hebrew_English_Concept_Extraction/)
walks through every stage on four sample documents.

## What makes it different

- **Hebrew and math read correctly from PDF.** Hebrew PDFs store text in visual order, and
  formulas are scattered glyphs and lines. The extractor rebuilds reading order with its own
  implementation of the Unicode bidi algorithm, and rebuilds fractions, roots, sums, limits and
  matrices from glyph geometry. Every formula comes out as LaTeX (KaTeX/MathJax compatible).
- **No machine learning for extraction.** Every step is geometric or rule-based, so the output is
  deterministic: the same file gives byte-identical output across runs and processes.
- **Fonts with no Unicode mapping are still read.** Some PDF printers embed fonts whose text
  layer is garbage. Instead of sending the page to OCR, which would lose all math structure,
  each glyph is rendered and matched by shape against an atlas of about 2,900 reference glyphs.
- **Structure, not just text.** Headings, units (definition, theorem, example), bold and
  underline are kept, because they are the strongest evidence of what matters.
- **Built for Hebrew morphology and mixed languages.** Attached prefixes (ה, ו, ב, ל, מ, ש,
  כ) are stripped by a cautious rule, so "הגרף" and "בגרף" count as one term. In model mode,
  plurals and construct forms are reduced to base forms too, and a term written in both
  Hebrew and English ("אנטרופיה (entropy)") becomes one concept.
- **Works without a model or an API key.** A model can be added at three points to improve
  results, but the default mode is purely algorithmic.
- **Permissively licensed stack only.** The only runtime dependency is pypdfium2. PyMuPDF
  (AGPL), lxml wheels (bundle LGPL libiconv) and python-bidi (LGPL) were rejected on purpose.
  See [extractor/README.md](extractor/README.md#licensing).
- **Evaluated honestly.** Gold concept lists come from a committee of three models voting 2 of
  3, checked against human judgement. Courses are split into dev (all tuning) and held-out
  (measured once at the end).

## How it works

| Stage | Module | What it does |
|---|---|---|
| 1. Extraction | `hebmath` (`extractor/`) | PDF/DOCX to text in reading order, formulas as LaTeX, plus headings, units and emphasis |
| 2. Candidates | `stage2_candidates.py` | Every term that may be a concept, from 7 sources: definitions, headings, emphasis, glosses in parentheses, labels, LaTeX operators and repeated phrases |
| 3. Image segments | – | Not implemented |
| 4. Merging | `stage4_*.py` | Finds pairs that may be one concept (spelling variants, acronyms, translations, a term and its gloss) and groups them |
| 5. Importance | `stage5_importance.py` | A score from 0 to 1 from 8 signals: is it defined or named, how it is marked, where it appears, how specific it is |
| 6. Selection | `stage6_selection.py` | Picks the final nodes: a concept enters if it is important or covers something no chosen concept covers yet |

Stages 2 to 6 are the package `concept_pipeline` (`pipeline/`), with no database and no network:
model calls go through a sender function you pass in.

### Algorithmic mode and model mode

**Algorithmic mode** (the default) needs nothing but the code. Stage 4 merges only what its
rules can decide; every other pair stays as two concepts, the safe side.

**Model mode** adds three kinds of calls to an LLM:

- base forms of Hebrew words in stage 2 (`stage2_model_normalizer.py`);
- finders for translations and synonyms no rule can see (`stage4_translation_finder.py`,
  `stage4_synonym_finder.py`);
- a decider that judges each proposed pair, by majority of three runs (`stage4_merge_decider.py`).

Every call goes through `llm_cache.py`, which records answers by content and replays them, so a
rerun sends nothing. The prompts live in those modules.

Stage 5 has one more signal, T, which compares a term's frequency with a background of other
courses. Without a background (`background=None`) it is off.

## Install

Python 3.11 or later (the extractor alone runs on 3.10).

```bash
pip install ./extractor ./pipeline
```

OCR of scanned pages needs [Tesseract](https://github.com/tesseract-ocr/tesseract) with the
Hebrew model. Without it, scanned pages are reported as unreadable and the rest of the document
is still read.

## Usage

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

The extractor also has a command line: `hebmath summary.pdf -o summary.txt --report report.json`.
Its output format is specified in [extractor/docs/FORMAT.md](extractor/docs/FORMAT.md).

## Demo

- **`demo/web`**: a static page, deployed to GitHub Pages by `.github/workflows/pages.yml`. It
  shows each stage for four generated course summaries (written by a model for testing, not real
  course material), with the original document beside the full extracted text.
  `demo/build_examples.py` rebuilds its data and copies the originals to `demo/web/originals/`.
  The originals are drawn in the browser by [pdf.js](https://github.com/mozilla/pdf.js) and
  [docx-preview](https://github.com/VolodymyrBaydalka/docxjs) (both Apache-2.0), loaded from a
  CDN at pinned versions only when a document is shown.
- **`demo/api`**: an optional FastAPI service with one endpoint, `POST /analyze`, that runs the
  same analysis on an uploaded file of up to 10 MB. The file is held in memory for the call
  only; nothing is stored and requests are not logged. `render.yaml` deploys it to Render's free
  tier in one step. Put its address in `demo/web/config.js` to turn on the upload form; with no
  address the page shows only the examples.

Run both locally:

```bash
pip install ./extractor ./pipeline -r demo/api/requirements.txt
cd demo/api && uvicorn app:app --port 8766
```

```bash
python -m http.server 8765 --directory demo/web
```

## Development

```bash
pip install -e extractor -e pipeline -r requirements-dev.txt
pytest
ruff check . && ruff format --check .
```

The extractor's golden files are a Linux reference (as in CI), so two of those tests fail on
macOS.

## Data, calibration and evaluation

The course files, gold labels, recorded model answers and per-course results are not in this
repository, because they come from real course material. The tools that produced them are:

- `scripts/concept_committee.py` builds a course's gold concept list from three models that
  label blindly and vote. `scripts/judges.example.json` names the models and the environment
  variables that hold their keys.
- `scripts/stage2_*` and `scripts/stage4_*` report and measure stages 2 and 4. They call a model
  only with `--send`; the stage-4 evaluation also keeps a spending ledger with a budget limit.
- `calibration/` chose the stage-5 weights and stage-6 settings now in the package. Point
  `--data-dir` or `TACTIMIND_DATA_DIR` at a data folder; see
  [calibration/README.md](calibration/README.md).

This code was developed inside a private monorepo and moved here without its history.
Docstrings that mention "the design", "the plan" or "owner decisions" refer to private design
documents that are not included.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party notices (PDFium, its
bundled libraries and the fonts behind the glyph atlas) are in
[extractor/THIRD_PARTY_NOTICES.md](extractor/THIRD_PARTY_NOTICES.md).
