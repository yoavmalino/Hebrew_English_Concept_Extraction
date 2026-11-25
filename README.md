# Hebrew-English Concept Extraction

Turns a course summary in Hebrew, English or a mix of the two into the list of concepts a student
needs to learn from it.

I started this project in mid-September 2026 as an intermediate step in a larger project: a
generator of concept maps and learning flows (a directed acyclic graph of the material's
hierarchy). It is in active development, and you can follow my progress in the
[live demo](https://yoavmalino.github.io/Hebrew_English_Concept_Extraction/).

**<ins>This repository is the first part, and it runs end to end: from a PDF or DOCX to a list of up
to 150 concepts, each with the passage that introduces it.</ins>** The two graphs are the next part.

A few concepts from the demo's operating systems document, as the pipeline returned them:

| Concept | The passage it comes with |
|---|---|
| מבוי סתום | מבוי סתום הוא מצב שבו קבוצה של תהליכים נמצאת בהמתנה מעגלית, כשכל תהליך בקבוצה ממתין למשאב שמוחזק על ידי תהליך אחר בקבוצה. |
| תקלת דפים | תקלת דפים (page fault) מתרחשת כשתהליך ניגש לדף שאינו נמצא כרגע בזיכרון הפיזי. |
| aging | הפתרון הנפוץ נקרא aging, שבו העדיפות של תהליך ממתין עולה בהדרגה עם הזמן. |
| tlb | כדי למנוע זאת, המעבד מחזיק מטמון קטן של תרגומים שנקרא TLB (Translation Lookaside Buffer). |

The document was generated for testing; it is not real course material.

## Motivation

During my first and second years at the Technion, I struggled to connect the dots between
lectures. I started reading research and even took a paid online course on effective studying.
Testing different study tips taught me one thing: for a concept to stick, and to see the big
picture, I need to understand the structure of the material and build a bridge between the
concepts I already know and the new ones I'm learning.

I used to build concept maps by hand. I skimmed the lecture and wrote down every key concept:
concepts that kept coming back, concepts that appeared in headings, or any term that seemed to be
a concept. Then I split them into groups by context, and only then built the map.

I saw value in building the map myself. But the more I did it, the more I understood that most of
the value comes from understanding the hierarchy of the material, and from asking myself why
"concept A" is connected to "concept B" by a relationship of type X. With dense academic material,
building a map by hand can take a very long time. At some point the cost outgrows the benefit, and
it stops being practical to do for every course.

So I'm trying to build a system that takes PDF or DOCX files of dense academic material and builds
two graphs:

1. **A directed acyclic graph of the hierarchy of the material**: which concepts you need to
   understand before you can understand others.
2. **A directed graph of the relations between concepts**: a concept can have more than one type
   of connection, with many different concepts.

To benefit from such a system, learners still need to make the effort to understand the
connections between concepts. The goal is to focus the effort they already invest in lectures and
homework, not to add another task. The map lets them follow the flow of a lecture and see how each
concept builds on the previous ones.

It also aims to help them find their gap: **what lies between the last concept they understood and
the point where they lost the lecturer.**

## What a good result looks like

This stage is an intermediate step: its output is the set of building blocks for the concept map.
Right now I'm focusing on one question: **which candidates are really concepts of the course, and
which are noise?** Their order doesn't matter yet.

### What counts as a concept

A concept is an idea that belongs to the course's subject and has a meaning of its own there. In
STEM courses, concepts are also working tools: students use them to solve problems and build proofs.

### What a good filter does

- **Keeps the concepts of the course**, especially the ones a student must know.
- **Removes the noise:** general words, sentence fragments, and phrases that wrap a concept in extra
  words, such as "proof of the Weierstrass theorem" instead of "Weierstrass theorem".
- **Works in both Hebrew and English:** the same definition and the same quality bar apply to a
  Hebrew document, an English document, and a document that mixes the two. So far the focus has been
  on Hebrew documents with English terms.

**A missing concept is worse than an extra one.** An extra concept adds noise, but a missing one may
be the very node that connects the central concepts of the map.

### Working targets

I don't know yet what "good enough" is. These are working targets, and I expect to revise them as I
learn what the map really needs. I want to measure them separately for Hebrew and English:

| Goal | Metric | Working target |
|---|---|---|
| Keep the must-know concepts | Recall | 90% |
| Remove the noise | Precision | 70% |

## How it works

| Stage | What it does |
|---|---|
| 1. Extraction | Reads the PDF or DOCX in the right order, formulas included, and keeps the structure: headings, definitions, emphasis |
| 2. Candidates | Collects every phrase that might be a concept. Most of them are noise at this point |
| 3. Images | Not built yet. Meant to recover concepts that appear only in images and diagrams |
| 4. Merging | Finds different names for the same concept |
| 5. Importance | Scores each concept by signals such as definitions, headings, formulas and emphasis |
| 6. Selection | Picks the final list, up to 150, each with the passage that introduces it |

It runs without an LLM by default, and this is what the demo runs. A model can be added in stages 2
and 4, and every answer it gives is recorded, so a rerun sends nothing.

## Key decisions and tradeoffs

- **Rebuild the text instead of copying it.** Hebrew PDFs store text in visual order, so copying it
  gives reversed words and scrambled formulas. I rebuild the reading order with my own implementation
  of the Unicode bidirectional algorithm, and formulas from the positions of glyphs and lines, as
  LaTeX. The price: unusual layouts, such as two columns, are not supported.
- **Glyph shapes instead of OCR** for PDFs with broken fonts. OCR would lose the structure of the
  math, so each glyph is matched by shape against about 2,900 reference glyphs. The price: similar
  Hebrew letters, such as ש and ט, are sometimes confused.
- **Rules before LLMs.** The same file always gives the same result, with no cost and no API key. The
  price: synonyms that share no form are not merged.
- **Recall first.** A concept lost in stage 2 never comes back, so I collect widely and filter later.
  The price: most candidates are noise.
- **When in doubt, don't merge.** A wrong merge deletes a concept. The price: more duplicates.

## Where the project stands, and what's next

**What works well today**

- Extracting text and formulas from most Hebrew and mixed PDF and DOCX files.
- Finding candidates in seven ways, without an LLM: definitions, headings, emphasized words,
  translations in parentheses, label lines, math operators and repeated phrases.

**Now:** separating concepts from noise. My first approach scored concepts with weights I tuned on a
small set of courses, and it overfit, so I'm moving to a different approach.

**Next:** reading text from images before candidates are collected, and merging more without an LLM.

**Later:** understanding diagrams, and building the two graphs: the hierarchy of the material and the
relations between concepts.

I developed this inside a private project and moved it here in November 2026, so the commit history
doesn't show the earlier work.

## Evaluation

To measure the filter, I need to know which concepts belong to each course. A committee of three
models (Claude Opus 5.5, GPT-6.1 Sol and Gemini 4 Carbon) lists them without seeing what my pipeline
found, and a concept counts when at least two agree. I checked the committee against my own judgement
on a sample. Courses are split into a development set, used for all tuning, and a held-out set.

## Limitations

- **Not every PDF is read correctly.** It depends on how the file was made: broken fonts, unusual
  layouts and complex formulas can come out wrong or go missing.
- Text inside images is ignored, and so are scanned pages in the demo.
- Two-column layouts are not supported.
- Formulas on scanned pages are read only when they are simple and linear.
- Old equation objects in Word files are marked, not converted.

The full list for extraction is in [extractor/README.md](extractor/README.md#known-limitations).

## Challenges and open questions

- **How do I read every PDF?** Each program writes PDFs differently. A page can look right while its
  text is stored in the wrong order, or in a font that doesn't say which letter each shape is. An
  error here carries over to every later stage.
- **Where does the list end?** Today it fills up to 150 entries, and I want a cutoff that depends on
  the material. One direction is to add a concept only if it contributes enough to a graph centrality
  measure, but the graph doesn't exist yet.
- **What is "good enough"?** I won't know until I see what the map really needs.
- **How far can merging go without an LLM?** "Context switch" and "החלפת הקשר" name the same idea
  and share no letters.
- **When should images be read?** Text inside an image should be counted with the rest of the
  document, but OCR errors could become candidates.
- **How much can I trust the labels?** They come from models, and a valid concept
  they missed counts as noise, so they limit what I can measure.

## Run it yourself

Python 3.11 or later:

```bash
pip install ./extractor ./pipeline
```

The full pipeline, the demo and the development setup are in [docs/USAGE.md](docs/USAGE.md).

## Credits

This project relies on the open-source software below. Each component remains under its own
license, and its authors do not endorse this project.

| Component | Used for | License |
|---|---|---|
| **Product** | | |
| [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) | Python interface to PDFium | Apache-2.0 or BSD-3-Clause, used under BSD-3-Clause |
| [PDFium](https://pdfium.googlesource.com/pdfium/) | Reading PDF files | BSD-3-Clause |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Reading scanned pages, only when installed (off in the demo) | Apache-2.0 |
| **Demo** | | |
| [FastAPI](https://fastapi.tiangolo.com) | Demo server | MIT |
| [Uvicorn](https://www.uvicorn.org) | Demo server | BSD-3-Clause |
| [pdf.js](https://github.com/mozilla/pdf.js) | Showing PDF files | Apache-2.0 |
| [docx-preview](https://github.com/VolodymyrBaydalka/docxjs) | Showing DOCX files | Apache-2.0 |
| [JSZip](https://stuk.github.io/jszip/) | Opening DOCX files | MIT or GPL-3.0, used under MIT |
| **Development and evaluation** | | |
| [pytest](https://pytest.org) | Tests | MIT |
| [Ruff](https://github.com/astral-sh/ruff) | Linting and formatting | MIT |
| [fontTools](https://github.com/fonttools/fonttools) | Building the glyph atlas | MIT |
| [HTTPX](https://www.python-httpx.org) | Calling model APIs | BSD-3-Clause |
| [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) | Calling model APIs | MIT |

**Fonts.** The glyph atlas is derived from STIX Two Math, Libertinus Math, Noto Sans Math, Tinos,
Arimo, David Libre, Frank Ruhl Libre, Miriam Libre, Alef, Noto Serif Hebrew and Noto Sans Hebrew, all
under the SIL Open Font License 1.1. Only shape descriptors are shipped, not the fonts themselves.
The test files include Latin Modern Math (GUST Font License).

**Standards.** Bidirectional reordering follows Unicode Standard Annex #9.

**Models and services.** I develop this project with the help of Claude (Anthropic). Claude Opus 5.5,
GPT-6.1 Sol and Gemini 4 Carbon produced the evaluation labels. The demo is hosted on
[GitHub Pages](https://pages.github.com) and [Render](https://render.com).

**Required notices.** PDFium includes other libraries, among them FreeType, libjpeg-turbo, libpng,
zlib, OpenJPEG, LittleCMS, libtiff and ICU. Portions of this software are copyright © The FreeType
Project (www.freetype.org). All rights reserved. This software is based in part on the work of the
Independent JPEG Group.

Versions, copyright notices and obligations are in
[extractor/THIRD_PARTY_NOTICES.md](extractor/THIRD_PARTY_NOTICES.md). The licenses of the browser
libraries are in [demo/web/vendor](demo/web/vendor/README.md).

## License

Copyright 2026 Yoav Malinovsky. Licensed under Apache-2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
