/**
 * Concept Extraction Explorer: loads a precomputed (or freshly analyzed) document summary and
 * renders one pipeline stage at a time. Every value from a document is inserted as text, never
 * as HTML, so an uploaded file cannot inject markup.
 */
(function () {
  "use strict";

  const config = window.DEMO_CONFIG || {};
  const examplesBox = document.getElementById("examples");
  const uploadForm = document.getElementById("upload");
  const statusLine = document.getElementById("status");
  const progressBox = document.getElementById("progress");
  const uploadButton = document.querySelector(".upload-button");
  // Matches the API's own limit (demo/api/analysis.py MAX_FILE_BYTES), checked here to fail fast.
  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  // A cold start on the free host plus a long file stays well under this.
  const UPLOAD_TIMEOUT_MS = 180000;
  // After this long the wait is most likely the server waking up, so the hint says so.
  const WAKE_HINT_SECONDS = 10;
  const tabsBox = document.getElementById("tabs");
  const stagePanel = document.getElementById("stage");
  const modeBox = document.getElementById("mode");
  const examplesPane = document.getElementById("examples-pane");

  const SOURCE_LABELS = {
    definition: "definition",
    heading: "heading",
    emphasis: "bold/underline",
    ngram: "repeated phrase",
    gloss: "English in parentheses",
    operator: "math operator",
    label: "label line",
  };
  const SIGNAL_LABELS = {
    Q: "defined or a named result",
    R: "spread over units and sections",
    M: "tied to formulas",
    V_b: "bold",
    V_u: "underlined",
    V_s: "larger font",
    L: "on its own line",
  };
  const FINDER_LABELS = {
    gloss: "English in parentheses",
    acronym: "acronym",
    spelling: "full / defective spelling",
    typo: "one-letter typo",
    glyph: "ש / ט glyph swap",
    named: "named result and its bare name",
  };

  // Viewers for the original document, loaded only when stage 1 first shows one. Versions are
  // pinned; pdf.js runs with eval off (an uploaded PDF must not run code).
  const PDFJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs";
  const PDFJS_WORKER_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";
  const JSZIP_URL = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
  const DOCX_PREVIEW_URL = "https://cdn.jsdelivr.net/npm/docx-preview@0.3.5/dist/docx-preview.min.js";
  // A long upload renders only its first pages, so a 150-page PDF does not freeze the tab.
  const MAX_ORIGINAL_PAGES = 40;
  // Canvas pixels per CSS pixel: sharp on high-density screens without huge canvases.
  const MAX_PIXEL_RATIO = 2;

  let current = null;
  // The original document of the current example or upload: {url, format, isHebrew, node}. `node` caches
  // the rendered pages so switching tabs does not render them again.
  let currentOriginal = null;
  let currentIsGenerated = false;
  let currentStage = "stage1";
  // "examples" (the default) or "upload". Each mode keeps its own last document, so switching
  // back and forth never loses a result: {data, isGenerated, original}.
  let mode = "examples";
  const views = { examples: null, upload: null };
  let pdfjsPromise = null;
  let docxPromise = null;

  /**
   * Create an element with optional text and class.
   *
   * @param {string} tag
   * @param {string} [text]
   * @param {string} [className]
   * @returns {HTMLElement}
   */
  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }

  /**
   * Build a row of headline numbers.
   *
   * @param {Array<[string, (string|number)]>} items label and value pairs
   * @returns {HTMLElement}
   */
  function stats(items) {
    const box = el("div", null, "stats");
    for (const [label, value] of items) {
      const stat = el("div", null, "stat");
      stat.append(el("b", typeof value === "number" ? value.toLocaleString("en") : value));
      stat.append(el("span", label));
      box.append(stat);
    }
    return box;
  }

  /**
   * Build a table; each cell is a string, a number or a ready element.
   *
   * @param {string[]} headers
   * @param {Array<Array<(string|number|HTMLElement)>>} rows
   * @returns {HTMLElement}
   */
  function table(headers, rows) {
    const wrap = el("div", null, "table-wrap");
    const tableNode = el("table");
    const head = el("tr");
    headers.forEach((h) => head.append(el("th", h)));
    tableNode.append(head);
    for (const row of rows) {
      const tr = el("tr");
      for (const cell of row) {
        const td = el("td");
        if (cell instanceof HTMLElement) td.append(cell);
        else {
          td.textContent = String(cell);
          if (typeof cell === "number") td.className = "num";
        }
        td.dir = "auto";
        tr.append(td);
      }
      tableNode.append(tr);
    }
    wrap.append(tableNode);
    return wrap;
  }

  /**
   * Return tags for a list of codes, labelled by a dictionary.
   *
   * @param {string[]} codes
   * @param {Object<string, string>} labels
   * @returns {HTMLElement}
   */
  function tags(codes, labels) {
    const box = el("span");
    codes.forEach((code) => box.append(el("span", labels[code] || code, "tag")));
    return box;
  }

  /**
   * Return a small bar and value for a score in [0, 1].
   *
   * @param {number} value
   * @returns {HTMLElement}
   */
  function score(value) {
    const box = el("span", null, "score");
    const bar = el("span", null, "bar");
    bar.style.width = `${Math.round(value * 60)}px`;
    box.append(bar, el("span", value.toFixed(2)));
    return box;
  }

  /**
   * Load a classic script once and resolve when it has run.
   *
   * @param {string} url
   * @returns {Promise<void>}
   */
  function loadScript(url) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = url;
      script.crossOrigin = "anonymous";
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`could not load ${url}`));
      document.head.append(script);
    });
  }

  /**
   * Return the pdf.js module, loading it on first use.
   *
   * @returns {Promise<object>}
   */
  function loadPdfjs() {
    if (!pdfjsPromise) {
      pdfjsPromise = import(PDFJS_URL).then((pdfjs) => {
        pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_URL;
        return pdfjs;
      });
    }
    return pdfjsPromise;
  }

  /**
   * Return the docx-preview library (it needs JSZip as a global), loading both on first use.
   *
   * @returns {Promise<object>}
   */
  function loadDocxPreview() {
    if (!docxPromise) {
      docxPromise = loadScript(JSZIP_URL)
        .then(() => loadScript(DOCX_PREVIEW_URL))
        .then(() => window.docx);
    }
    return docxPromise;
  }

  /**
   * Draw the pages of a PDF into a box, one canvas per page, fitted to the box's width.
   *
   * @param {string} url
   * @param {HTMLElement} box
   * @returns {Promise<void>}
   */
  async function renderPdf(url, box) {
    const pdfjs = await loadPdfjs();
    const pdf = await pdfjs.getDocument({ url, isEvalSupported: false }).promise;
    const width = Math.max(box.clientWidth - 2, 280);
    const ratio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
    const pages = Math.min(pdf.numPages, MAX_ORIGINAL_PAGES);
    box.replaceChildren();
    for (let number = 1; number <= pages; number += 1) {
      const page = await pdf.getPage(number);
      const viewport = page.getViewport({ scale: width / page.getViewport({ scale: 1 }).width });
      const canvas = el("canvas", null, "page");
      canvas.width = Math.floor(viewport.width * ratio);
      canvas.height = Math.floor(viewport.height * ratio);
      canvas.style.aspectRatio = `${viewport.width} / ${viewport.height}`;
      canvas.setAttribute("aria-label", `Page ${number}`);
      box.append(canvas);
      await page.render({
        canvasContext: canvas.getContext("2d"),
        viewport,
        transform: ratio === 1 ? null : [ratio, 0, 0, ratio, 0, 0],
      }).promise;
    }
    if (pdf.numPages > pages) {
      box.append(el("p", `Showing the first ${pages} of ${pdf.numPages} pages.`, "explain"));
    }
  }

  /**
   * Draw a DOCX into a box as Word would lay it out.
   *
   * @param {string} url
   * @param {HTMLElement} box
   * @returns {Promise<void>}
   */
  async function renderDocx(url, box) {
    const docx = await loadDocxPreview();
    const blob = await (await fetch(url)).blob();
    box.replaceChildren();
    await docx.renderAsync(blob, box, null, {
      inWrapper: false,
      ignoreLastRenderedPageBreak: true,
      experimental: false,
    });
    if (currentOriginal.isHebrew) alignRightToLeft(box);
  }

  /**
   * Give each paragraph of a rendered Hebrew DOCX the direction of its own text.
   *
   * Files written by converters (pandoc, Google Docs exports) often hold Hebrew paragraphs with no
   * right-to-left mark, so they render aligned left. A paragraph the author centered or
   * justified keeps that alignment.
   *
   * @param {HTMLElement} box
   */
  function alignRightToLeft(box) {
    for (const block of box.querySelectorAll("p, li, td, th, h1, h2, h3, h4, h5, h6")) {
      block.dir = "auto";
      const align = block.style.textAlign;
      if (!align || align === "left" || align === "start") block.style.textAlign = "start";
    }
  }

  /**
   * Return the original document's box, rendering it the first time.
   *
   * @returns {HTMLElement}
   */
  function originalView() {
    if (currentOriginal.node) return currentOriginal.node;
    const box = el("div", null, "original");
    box.tabIndex = 0;
    box.setAttribute("aria-label", "Original document");
    box.append(el("p", "Loading the original…", "explain"));
    currentOriginal.node = box;
    const render = currentOriginal.format === "pdf" ? renderPdf : renderDocx;
    // Run after the caller has put the box in the page, so it has a width to fit the pages to.
    // (A timeout, not requestAnimationFrame: frames pause while the page is not painted.)
    setTimeout(() =>
      render(currentOriginal.url, box).catch((error) => {
        console.error(error);
        box.replaceChildren(el("p", "The original could not be displayed here.", "explain"));
      }),
    );
    return box;
  }

  /**
   * Build the original and the extracted text side by side, with a switch on narrow screens.
   *
   * @param {HTMLElement} textBox the extracted text
   * @returns {HTMLElement}
   */
  function compareView(textBox) {
    const wrap = el("div", null, "compare");
    wrap.dataset.show = "original";
    const switcher = el("div", null, "compare-switch");
    switcher.setAttribute("role", "group");
    switcher.setAttribute("aria-label", "Show");
    for (const [value, label] of [
      ["original", "Original"],
      ["text", "Extracted text"],
    ]) {
      const button = el("button", label);
      button.type = "button";
      button.setAttribute("aria-pressed", String(value === wrap.dataset.show));
      button.addEventListener("click", () => {
        wrap.dataset.show = value;
        switcher
          .querySelectorAll("button")
          .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      });
      switcher.append(button);
    }
    const originalPane = el("div", null, "pane pane-original");
    originalPane.append(el("h4", "Original"), originalView());
    const textPane = el("div", null, "pane pane-text");
    textPane.append(el("h4", "Extracted text"), textBox);
    wrap.append(switcher, originalPane, textPane);
    return wrap;
  }

  const STAGES = [
    {
      id: "stage1",
      title: "1 · Extraction",
      explain:
        "Reads the PDF or DOCX in true reading order, with every formula as LaTeX. Keeps the " +
        "structure later stages rely on: headings, definitions, examples, emphasis.",
      render(data) {
        const s = data.stage1;
        const units = Object.entries(s.units).sort((a, b) => b[1] - a[1]);
        // One element per line, each with its own direction: a Hebrew line reads right to left,
        // a line that is only a formula left to right.
        // A Hebrew document aligns every line to the right, formula lines included, so the text
        // keeps one edge as it does in the original.
        const isHebrew = data.stage2.language === "he";
        const fullText = el("div", null, isHebrew ? "fulltext rtl-doc" : "fulltext");
        // `excerpt` is what servers before the full-text change returned.
        for (const line of (s.text ?? s.excerpt ?? "").split("\n")) {
          const lineNode = el("div", line || " ", "line");
          lineNode.dir = "auto";
          fullText.append(lineNode);
        }
        fullText.tabIndex = 0;
        fullText.setAttribute("aria-label", "Full extracted text");
        const out = [
          stats([
            ["format", s.format.toUpperCase()],
            ["characters", s.characters],
            ["sections", s.section_count],
            ["units", units.reduce((sum, [, n]) => sum + n, 0)],
          ]),
        ];
        if (currentOriginal) {
          out.push(
            el("h3", "Original vs. extracted"),
            el("p", "Left: the original. Right: what stage 1 extracts. Each side scrolls.", "explain"),
            compareView(fullText),
          );
        } else {
          out.push(
            el("h3", "Extracted text"),
            el("p", "Everything stage 1 extracts. Scroll inside the box.", "explain"),
            fullText,
          );
        }
        if (s.text_truncated) {
          out.push(el("p", "Cut here: the document is very long.", "explain"));
        }
        out.push(
          el("h3", "Units by type"),
          table(
            ["type", "count"],
            units.map(([type, n]) => [type, n]),
          ),
          el("h3", "Headings"),
        );
        const list = el("ul", null, isHebrew ? "sections rtl-doc" : "sections");
        for (const section of s.sections) {
          const item = el("li", section.title);
          item.dir = "auto";
          item.style.paddingInlineStart = `${(section.level - 1) * 20}px`;
          list.append(item);
        }
        out.push(list);
        return out;
      },
    },
    {
      id: "stage2",
      title: "2 · Candidates",
      explain:
        "Every phrase that might be a concept, from seven sources: definitions, headings, " +
        "emphasis, repeated phrases, English glosses, math operators, labels. Most are noise; " +
        "the next stages filter them.",
      render(data) {
        const s = data.stage2;
        return [
          stats([
            ["candidates", s.candidates],
            ["words", s.words],
            ["main language", s.language === "he" ? "Hebrew" : "English"],
          ]),
          el("h3", "Most mentioned candidates"),
          table(
            ["candidate", "mentions", "sources"],
            s.top.map((c) => [c.name, c.count, tags(c.sources, SOURCE_LABELS)]),
          ),
        ];
      },
    },
    {
      id: "stage3",
      title: "3 · Images",
      soon: true,
      explain:
        "In development: extracting and tagging concept candidates from images and diagrams. " +
        "For now the pipeline goes straight from stage 2 to stage 4.",
      render() {
        return [el("p", "Coming next", "soon-badge")];
      },
    },
    {
      id: "stage4",
      title: "4 · Merging",
      explain:
        "Finds different names for one concept: English in parentheses, acronyms, spelling " +
        "variants, typos. Here only rule-safe pairs merge; the full pipeline has a model decide " +
        "the rest.",
      render(data) {
        const s = data.stage4;
        const finders = Object.entries(s.pairs_by_finder).map(([f, n]) => [
          FINDER_LABELS[f] || f,
          n,
        ]);
        const out = [
          stats([
            ["pairs proposed", s.pairs],
            ["merged by rule", s.merged_by_rule_count],
            ["left apart (no model)", s.left_apart],
            ["fragments absorbed", s.fragments_absorbed],
            ["concepts", s.concepts],
          ]),
          el("h3", "Pairs by finder"),
          table(["finder", "pairs"], finders),
        ];
        if (s.merged_by_rule.length) {
          out.push(
            el("h3", "Merged by rule"),
            table(
              ["one name", "other name"],
              s.merged_by_rule.map(([a, b]) => [a, b]),
            ),
          );
        }
        return out;
      },
    },
    {
      id: "stage5",
      title: "5 · Importance",
      explain:
        "Scores each concept from signals in the document, with weights calibrated on real " +
        "course summaries. One signal, rarity across courses, is off in this demo.",
      render(data) {
        const s = data.stage5;
        const signals = Object.keys(SIGNAL_LABELS);
        const rows = s.top.map((c) => [
          c.name,
          score(c.importance),
          tags(
            signals.filter((k) => c.signals[k] >= 0.5),
            SIGNAL_LABELS,
          ),
        ]);
        return [
          el("h3", "Top concepts by importance"),
          el("p", "Tags: signals at 0.5 or higher.", "explain"),
          table(["concept", "importance", "strong signals"], rows),
        ];
      },
    },
    {
      id: "stage6",
      title: "6 · Selection",
      explain:
        "The final list, up to 150. Defined terms, named theorems and heading topics go first, " +
        "then whatever adds importance or coverage. Each comes with the passage that best " +
        "introduces it.",
      render(data) {
        const s = data.stage6;
        const rows = s.nodes.map((n, i) => {
          const name = el("span");
          name.append(el("span", n.name));
          if (n.anchored) name.append(el("span", "anchored", "tag anchored"));
          return [i + 1, name, score(n.importance), n.passage];
        });
        return [
          stats([
            ["nodes", s.nodes.length],
            ["anchored", s.report.anchored],
            ["pool", s.pool],
          ]),
          table(["#", "node", "importance", "passage"], rows),
        ];
      },
    },
  ];

  /** Render the tab row for the stages. */
  function renderTabs() {
    tabsBox.replaceChildren();
    for (const stage of STAGES) {
      const button = el("button", stage.title);
      if (stage.soon) button.append(el("span", "soon", "tab-soon"));
      button.type = "button";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(stage.id === currentStage));
      button.addEventListener("click", () => {
        currentStage = stage.id;
        renderTabs();
        renderStage();
      });
      tabsBox.append(button);
    }
  }

  /** Render the selected stage of the current document. */
  function renderStage() {
    stagePanel.replaceChildren();
    if (!current) return;
    const stage = STAGES.find((s) => s.id === currentStage);
    // Disclosure repeated inside the panel: the notice above is out of view once a reader
    // scrolls into the results.
    if (currentIsGenerated) {
      stagePanel.append(
        el("p", "Generated sample · not real course material", "generated generated-inline"),
      );
    }
    stagePanel.append(el("h2", stage.title), el("p", stage.explain, "explain"));
    stagePanel.append(...stage.render(current));
  }

  /**
   * Build the view of one document, for `show` and for a mode to keep.
   *
   * @param {object} data the JSON from the API or a precomputed file
   * @param {boolean} isGenerated true for the bundled samples, which a model wrote
   * @param {?{url: string, format: string}} original where to read the original document
   * @returns {{data: object, isGenerated: boolean, original: ?object}}
   */
  function makeView(data, isGenerated, original) {
    return {
      data,
      isGenerated,
      original: original
        ? { url: original.url, format: original.format, isHebrew: data.stage2.language === "he", node: null }
        : null,
    };
  }

  /**
   * Show a document's stages, or hide the results when there is no document yet.
   *
   * @param {?{data: object, isGenerated: boolean, original: ?object}} view
   */
  function show(view) {
    current = view ? view.data : null;
    currentIsGenerated = view ? view.isGenerated : false;
    currentOriginal = view ? view.original : null;
    tabsBox.hidden = !view;
    stagePanel.hidden = !view;
    renderTabs();
    renderStage();
  }

  /**
   * Switch between the samples and the user's own file; each keeps its last result.
   *
   * @param {string} next "examples" or "upload"
   */
  function setMode(next) {
    mode = next;
    modeBox
      .querySelectorAll("button")
      .forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === next)));
    examplesPane.hidden = next !== "examples";
    uploadForm.hidden = next !== "upload";
    setStatus("");
    show(views[next]);
  }

  /**
   * Load a precomputed sample and mark its button.
   *
   * @param {{id: string, title: string}} example
   * @param {HTMLButtonElement} button
   * @returns {Promise<void>}
   */
  async function loadExample(example, button) {
    setStatus("Loading…");
    try {
      const response = await fetch(`data/${encodeURIComponent(example.id)}.json`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const original = example.original ? { url: example.original, format: example.format } : null;
      views.examples = makeView(await response.json(), example.generated !== false, original);
      if (mode === "examples") show(views.examples);
      examplesBox.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
      button.setAttribute("aria-pressed", "true");
      setStatus("");
    } catch (error) {
      console.error(error);
      setStatus("This sample didn't load. Reload the page and try again.", true);
    }
  }

  /**
   * Show a status message, styled as an error when `isError` is true.
   *
   * @param {string} text
   * @param {boolean} [isError]
   */
  function setStatus(text, isError = false) {
    statusLine.textContent = text;
    statusLine.classList.toggle("error", isError);
  }

  /**
   * Show the progress box while an upload is analyzed, with elapsed time and a hint that changes
   * when the wait gets long (a free server sleeps when idle and takes up to a minute to wake).
   *
   * @param {string} fileName
   * @returns {function(): void} stops the timer and hides the box
   */
  function startProgress(fileName) {
    const title = document.getElementById("progress-title");
    const detail = document.getElementById("progress-detail");
    const started = Date.now();
    title.textContent = `Analyzing ${fileName}…`;
    const tick = () => {
      const seconds = Math.round((Date.now() - started) / 1000);
      const hint =
        seconds >= WAKE_HINT_SECONDS
          ? "Waking the free demo server. Up to a minute; keep this page open."
          : "Reading the document and finding its concepts.";
      detail.textContent = `${seconds} s · ${hint}`;
    };
    tick();
    const timer = setInterval(tick, 1000);
    progressBox.hidden = false;
    progressBox.scrollIntoView({ block: "nearest", behavior: "smooth" });
    return () => {
      clearInterval(timer);
      progressBox.hidden = true;
    };
  }

  /**
   * Return a message for a file the demo cannot take, or null when it may be sent.
   *
   * @param {File} file
   * @returns {?string}
   */
  function uploadProblem(file) {
    if (!/\.(pdf|docx)$/i.test(file.name)) return "PDF or DOCX only.";
    if (file.size > MAX_UPLOAD_BYTES) return "Over 10 MB. Try a smaller file.";
    if (file.size === 0) return "This file is empty.";
    return null;
  }

  /**
   * Send the chosen file to the demo API and show the result.
   *
   * @returns {Promise<void>}
   */
  async function analyzeUpload() {
    const input = document.getElementById("file");
    const file = input.files[0];
    if (!file) return;
    const problem = uploadProblem(file);
    if (problem) {
      setStatus(problem, true);
      input.value = "";
      return;
    }
    setStatus("");
    input.disabled = true;
    uploadButton.setAttribute("aria-disabled", "true");
    const stopProgress = startProgress(file.name);
    try {
      const response = await fetch(`${config.apiUrl}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream", "X-Filename": file.name },
        body: file,
        signal: AbortSignal.timeout(UPLOAD_TIMEOUT_MS),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        setStatus(body.detail || "Couldn't analyze this file. Try another one.", true);
        return;
      }
      // An upload's original is a local object URL; free the previous upload's one.
      if (views.upload && views.upload.original) URL.revokeObjectURL(views.upload.original.url);
      // The original is shown from the file on this computer; it is not sent anywhere again.
      views.upload = makeView(body, false, {
        url: URL.createObjectURL(file),
        format: body.stage1.format,
      });
      if (mode !== "upload") return;
      currentStage = "stage1";
      show(views.upload);
      setStatus(`Done: ${file.name}`);
      tabsBox.scrollIntoView({ block: "start", behavior: "smooth" });
    } catch (error) {
      console.error(error);
      const timedOut = error.name === "TimeoutError";
      setStatus(
        timedOut
          ? "This took too long. Try again in a minute, or with a shorter file."
          : "Couldn't reach the demo server. Check your connection and try again.",
        true,
      );
    } finally {
      stopProgress();
      input.disabled = false;
      input.value = "";
      uploadButton.removeAttribute("aria-disabled");
    }
  }

  /** Wire the mode switch and upload, render the sample buttons and open the first sample. */
  async function start() {
    // Without an API there is nothing to upload to, so the switch stays a single "Samples" label.
    if (config.apiUrl) {
      modeBox.querySelector('[data-mode="upload"]').hidden = false;
      modeBox
        .querySelectorAll("button")
        .forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
      // Analysis starts as soon as a file is chosen; there is no separate submit step.
      document.getElementById("file").addEventListener("change", analyzeUpload);
      uploadForm.addEventListener("submit", (event) => event.preventDefault());
    }
    try {
      const response = await fetch("data/index.json");
      const examples = await response.json();
      examples.forEach((example, index) => {
        const button = el("button", example.title);
        button.type = "button";
        button.dir = "auto";
        button.setAttribute("aria-pressed", "false");
        button.addEventListener("click", () => loadExample(example, button));
        examplesBox.append(button);
        if (index === 0) loadExample(example, button);
      });
    } catch (error) {
      console.error(error);
      setStatus("The samples didn't load. Reload the page and try again.", true);
    }
  }

  start();
})();
