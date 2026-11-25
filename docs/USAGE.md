# Usage and development

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

with open("summary.pdf", "rb") as file:
    data = file.read()

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
[extractor/docs/FORMAT.md](../extractor/docs/FORMAT.md), and its design, licensing choices and
edge cases are in [extractor/README.md](../extractor/README.md).

Scanned pages need [Tesseract](https://github.com/tesseract-ocr/tesseract) with the Hebrew
model. Without it, scanned pages are marked unreadable and the rest of the document is still
read.

## Algorithmic mode and model mode

The default **algorithmic mode** needs only the code. Stage 4 merges only what its rules can
decide; every other pair stays apart, the safe side. This is what the demo runs.

**Model mode** adds three kinds of LLM calls: base forms of Hebrew words (stage 2), finders for
translations and synonyms no rule can see, and a decider for each proposed pair, by majority of
three runs (stage 4). Every call goes through `llm_cache.py`, which records answers and replays
them, so a rerun sends nothing.

Stage 5's specificity signal compares a term's frequency with other courses. With no other
courses to compare against (as in the demo), it is off.

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

The free server is slow, roughly 2 seconds per page, and sleeps when idle: the first upload after a
quiet period waits up to a minute longer. Files are limited to 10 MB and 150 pages.

## Development

```bash
pip install -e extractor -e pipeline -r requirements-dev.txt
pytest
ruff check . && ruff format --check .
```

- The extractor's golden test files are a Linux reference (as in CI), so two of those tests fail
  on macOS.
- Docstrings that mention "the design", "the plan" or "owner decisions" refer to design documents
  of the private project this code was developed in. They are not included here.
