# hebrew-english-concept-extraction

Find the concepts a student needs from a course summary written in Hebrew and English, with
math. The input is one PDF or DOCX. The output is a ranked list of about 150 concepts, each with
a passage from the document that defines or uses it.

The code is split into stages:

1. **Extraction** (`extractor/`, package `hebmath`). Turns the PDF or DOCX into plain text in
   the right reading order, with every formula written as LaTeX, plus the document's structure:
   headings, units, and which words are bold or underlined. It has its own
   [README](extractor/README.md) (in Hebrew) and output format ([FORMAT.md](extractor/docs/FORMAT.md)).
2. **Candidates** (`stage2_candidates.py`). Every term that might be a concept, from seven
   sources: definitions, headings, emphasized text, glosses in parentheses, labels, LaTeX
   operators and repeated phrases.
3. *(Image segments: not implemented.)*
4. **Merging** (`stage4_*.py`). Finds pairs that may be the same concept (spelling variants,
   acronyms, translations, a term and its gloss) and groups them.
5. **Importance** (`stage5_importance.py`). Gives every concept a score from 0 to 1, built from
   eight signals seen in the document, such as whether it is defined or named, how it is
   marked, and where it appears.
6. **Selection** (`stage6_selection.py`). Chooses the nodes of the learning space: a concept
   enters when it is important or covers something no chosen concept covers yet.

Stages 2 to 6 are the package `concept_pipeline` (`pipeline/`). They are pure functions: no
database, no network, no files. The code was developed inside a private monorepo and moved here
without its history. Docstrings refer to "the design", "the plan" and "owner decisions". Those
are the private design documents, and they are not included.

## Two modes

- **Algorithmic mode** needs no model and no API key. Stage 2 runs without model word forms,
  stage 4 merges only what its rules decide, and every other pair stays as two concepts. The
  demo and the tests use this mode.
- **Model mode** adds three kinds of model calls:
  - a cheap model gives each word's base form in stage 2 (`stage2_model_normalizer.py`);
  - finders propose translations and synonyms that no rule can see
    (`stage4_translation_finder.py`, `stage4_synonym_finder.py`);
  - a decider says whether each proposed pair is one concept (`stage4_merge_decider.py`).

  Every call goes through `llm_cache.py`, which records answers by content and replays them, so
  a rerun sends nothing. The prompts are in those modules.

Stage 5 also has a signal, T, that compares a term's frequency with a background of other
courses. With no background (`background=None`), T is off.

## Install

Python 3.11. The two packages install from this repository:

```bash
pip install ./extractor ./pipeline
```

For development (tests, ruff and the model scripts):

```bash
pip install -e extractor -e pipeline -r requirements-dev.txt
pytest
ruff check . && ruff format --check .
```

The golden files of the extractor tests are a Linux reference, so two of them fail on macOS.
OCR of scanned pages needs Tesseract with the Hebrew model. Without it, scanned pages are
reported as unreadable and the rest of the document is still read.

## Use

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

[`demo/api/analysis.py`](demo/api/analysis.py) runs the same steps and summarizes each stage.

## Demo

- `demo/web` is a static page that shows stages 1 to 6 for four synthetic course summaries.
  They were written by a model for testing, so they are not real course material. The page is
  published with GitHub Pages (`.github/workflows/pages.yml`). `demo/build_examples.py`
  rebuilds its data.
- `demo/api` is a small FastAPI service with one call, `POST /analyze`. It takes a PDF or DOCX
  of at most 10 MB and returns the same per-stage summary. It runs in algorithmic mode with T
  off. The file is held in memory only for the call: nothing is stored, and requests are not
  logged. `render.yaml` deploys it, and `demo/web/config.js` holds its address. With no
  address, the page shows only the examples.

To run both locally:

```bash
pip install ./extractor ./pipeline -r demo/api/requirements.txt
cd demo/api && uvicorn app:app --port 8766
```

```bash
python -m http.server 8765 --directory demo/web
```

## Data, calibration and evaluation

The course files, the gold labels, the recorded model answers and the per-course results are
not in this repository. They come from real course material.

- `scripts/concept_committee.py` builds a concept gold for one course from a committee of
  three models, which label blindly and vote 2 of 3. `scripts/judges.example.json` names the
  models and the environment variables that hold their keys (never the keys).
- `scripts/stage2_*` and `scripts/stage4_*` report and measure stages 2 and 4. They send to a
  model only with `--send`.
- `calibration/` holds the code that chose the stage-5 weights and the stage-6 settings now in
  the package. Point `--data-dir` or `TACTIMIND_DATA_DIR` at a data folder. The layout and the
  order of the runs are in [calibration/README.md](calibration/README.md).

The courses are split into dev, used for every choice, and held-out, measured once at the end.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party notices for PDFium and
the glyph atlas fonts are in
[extractor/THIRD_PARTY_NOTICES.md](extractor/THIRD_PARTY_NOTICES.md).
