# Third-party components and notices

hebmath ships **one** third-party Python runtime dependency (pypdfium2) and,
optionally, calls the **Tesseract** OCR program as an external process. Everything
else is the Python standard library. This file lists every component, its license,
and the attribution or disclaimer obligations that apply.

hebmath itself is released under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
It is distributed as source, so the attribution clauses below apply to anyone who
redistributes it, and to the pypdfium2 wheel pip installs next to it.

## Runtime – Python

| Component | Version | License | Obligations |
|---|---|---|---|
| CPython + stdlib (incl. `xml.etree`, `zipfile`, `zlib`, `unicodedata`) | ≥3.10 | PSF-2.0 | Keep the PSF license notice when redistributing. |
| expat (XML parser inside CPython) | bundled | MIT | Keep copyright notice. |
| pypdfium2 | 5.13.0 | Apache-2.0 **or** BSD-3-Clause (dual; we use it under BSD-3-Clause) | BSD-3: retain copyright + disclaimer; do not use the authors' names for endorsement. Docs/examples of pypdfium2 are CC-BY-4.0 (not shipped by us). |
| PDFium (binary inside pypdfium2) | 153.0.7999.0 | BSD-3-Clause | Retain copyright + disclaimer. |
| pdfium-binaries (build scripts) | – | MIT | Retain notice. |

### Libraries statically compiled into the PDFium binary

The full texts are installed with the wheel in
`site-packages/pypdfium2-5.13.0.dist-info/licenses/`. Summary:

| Library | License | Notes |
|---|---|---|
| FreeType | FreeType License (FTL) – dual-licensed FTL/GPLv2; **used under FTL** | **Attribution required in documentation:** "Portions of this software are copyright © The FreeType Project (www.freetype.org). All rights reserved." |
| libjpeg-turbo | IJG + BSD-3 + zlib | **Attribution required in documentation:** "This software is based in part on the work of the Independent JPEG Group." |
| libpng | libpng license (permissive) | Retain notice. |
| zlib | zlib license | – |
| LittleCMS (lcms2) | MIT | Retain notice. |
| OpenJPEG | BSD-2-Clause | Retain notice. |
| libtiff | libtiff license (BSD-like) | Retain notice. |
| Anti-Grain Geometry 2.3 | AGG 2.3 permissive license | Retain notice. |
| Abseil | Apache-2.0 | Retain license/NOTICE when redistributing. |
| ICU | Unicode License v3 | Retain notice. (The GPL-3.0 text inside icu.txt applies to ICU's *autoconf build scripts* only, with the Autoconf exception; nothing GPL is in the binary.) |
| fast_float, simdutf | MIT / Apache-2.0 | Retain notice. |
| LLVM libc fragments | Apache-2.0 WITH LLVM-exception | Retain notice. |

## Data shipped with hebmath – glyph atlas (`src/hebmath/data/glyph_atlas.json.gz`)

The atlas holds **shape descriptors** of reference glyphs: a 16×16 coverage grid,
an ink box and topology counts per glyph. It is used to identify glyphs of PDF fonts
that carry no Unicode mapping. The descriptors were produced by rendering the fonts
below with PDFium (`tools/build_glyph_atlas.py`). No font program, outline or
bitmap of any font is distributed.

All source fonts are licensed under the **SIL Open Font License 1.1**. The OFL
allows commercial use, and it does not restrict documents or other output
produced with the fonts. It is arguable whether rendered descriptors count as a
"Derivative Work" of the Font Software. We take the conservative reading and keep
attribution here. The atlas is never sold by itself, and the Reserved Font Names
are not used for anything we ship.

| Font | Source (pinned) | Copyright (from the font's name table / OFL.txt) |
|---|---|---|
| STIX Two Math (`STIXTwoMath.input.ttf`) | github.com/stipub/stixfonts @ bea1dc9a | Copyright 2001-2021 The STIX Fonts Project Authors |
| Libertinus Math | github.com/google/fonts `ofl/libertinusmath` @ 1d0e46c7 | Copyright 2025 The Libertinus Math Project Authors |
| Noto Sans Math | google/fonts `ofl/notosansmath` @ 1d0e46c7 | Copyright 2022 Google LLC |
| Tinos (Regular, Bold, Italic) | google/fonts `ofl/tinos` @ 1d0e46c7 | Copyright The Tinos Project Authors |
| Arimo | google/fonts `ofl/arimo` @ 1d0e46c7 | Copyright 2020 The Arimo Project Authors |
| David Libre (Regular, Bold) | google/fonts `ofl/davidlibre` @ 1d0e46c7 | Copyright 2016 The David Libre Project Authors; Reserved Font Names "Hadash", "Gentium", "SIL" |
| Frank Ruhl Libre | google/fonts `ofl/frankruhllibre` @ 1d0e46c7 | Copyright 2015 The Frank Ruhl Libre Project Authors |
| Miriam Libre | google/fonts `ofl/miriamlibre` @ 1d0e46c7 | Copyright 2023 The Miriam Libre Project Authors |
| Alef (Regular, Bold) | google/fonts `ofl/alef` @ 1d0e46c7 | Copyright (c) 2012 HaGilda & Mushon Zer-Aviv; Reserved Font Name "Alef" |
| Noto Serif Hebrew, Noto Sans Hebrew | google/fonts `ofl/notoserifhebrew`, `ofl/notosanshebrew` @ 1d0e46c7 | Copyright 2022/2024 The Noto Project Authors |

The sha256 prefix of each font file is recorded in the atlas itself (`"sources"`).
The full OFL 1.1 text: https://openfontlicense.org/open-font-license-official-text/

## Optional – OCR (external process, not linked)

| Component | Version tested | License | Obligations |
|---|---|---|---|
| Tesseract OCR | 5.3.4 | Apache-2.0 | Keep license/NOTICE when redistributing the binary (e.g. in a container image). |
| Leptonica (Tesseract's image library) | 1.82.0 | BSD-2-Clause-style (Leptonica license) | Retain notice. |
| `heb.traineddata` from tesseract-ocr/tessdata_best @ e12c65a9 | – | Apache-2.0 | Same as Tesseract. |
| `eng.traineddata` | from distro package | Apache-2.0 | – |

## Development / test only (not deployed)

| Component | License | Use |
|---|---|---|
| pytest | MIT | test runner |
| fontTools | MIT | reads font coverage when building the glyph atlas (`tools/build_glyph_atlas.py`); not imported at run time |
| qpdf | Apache-2.0 | generates encrypted fixtures |
| TeX Live (pdflatex) | LPPL / various free | generates fixtures |
| Chromium | BSD-3 + various | generates MathML fixtures |
| Latin Modern Math font (fixture only) | GUST Font License | embedded in a test PDF |
| FreeSerif font (fixture only) | GPL-3.0 **with font exception** | embedded in test PDFs only; never shipped in the product |

## Disclaimer

All components above are provided "AS IS", without warranty of any kind, as stated
in their respective licenses.
