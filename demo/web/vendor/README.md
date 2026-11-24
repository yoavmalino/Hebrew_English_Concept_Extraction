# Third-party viewers

Unmodified copies of the browser libraries the demo page uses to show the original document and
count its pages. They are served with the page, so the demo does not depend on a CDN being
reachable. Each folder keeps the library's own license.

| Folder | Library | Version | License | Source |
|---|---|---|---|---|
| `pdfjs/` | pdf.js (`build/pdf.mjs`, `build/pdf.worker.mjs` from the release zip) | 4.10.38 | Apache-2.0 | https://github.com/mozilla/pdf.js/releases/tag/v4.10.38 |
| `jszip/` | JSZip (`dist/jszip.min.js`) | 3.10.1 | MIT or GPL-3.0, used under MIT | https://github.com/Stuk/jszip/tree/v3.10.1 |
| `docx-preview/` | docx-preview (`dist/docx-preview.min.js`) | 0.3.5 | Apache-2.0 | https://github.com/VolodymyrBaydalka/docxjs/tree/0.3.5 |

SHA-256, to check a copy against its source:

```
a209a2124baa35cbb9015b809926f1bcec9dd1c247296290e205dd9d76cb9128  pdfjs/pdf.mjs
7c237f83fa56bce645d8af51d183c9c56ba7b2d2928ff42754dc7020bea36323  pdfjs/pdf.worker.mjs
acc7e41455a80765b5fd9c7ee1b8078a6d160bbbca455aeae854de65c947d59e  jszip/jszip.min.js
cf0201bc615152efd395ca61f88780a5d1ad12eddaa583d45ab0c4e405b98fdc  docx-preview/docx-preview.min.js
```

To upgrade, replace the files with the new release's, update this table and the paths in
`../app.js`, and check that PDF and DOCX originals still render.
