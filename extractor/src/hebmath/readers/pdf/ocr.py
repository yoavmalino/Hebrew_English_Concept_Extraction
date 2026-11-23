"""OCR for pages without a usable text layer (scans), via the Tesseract CLI.

Tesseract (Apache-2.0) is invoked as a subprocess - no Python binding needed.
Tesseract emits Hebrew in logical order. Determinism: single thread
(``OMP_THREAD_LIMIT=1``), fixed DPI, fixed page segmentation mode and the
model files pinned by the deployment (see docs). OCR text is then passed
through the plain-text math tagger; full 2-D formula recognition on scans is
*not* performed by this engine (see the ``OcrEngine`` protocol to plug one in).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ...errors import ExtractionTimeoutError
from ...model import Block, TextSpan
from ...text_math import promote_lone_math, tag_segments


class OcrEngine(Protocol):
    name: str

    def available(self) -> bool: ...

    def recognize(self, image_pgm: bytes, *, detect_math: bool) -> list[Block]: ...


@dataclass(frozen=True)
class OcrWord:
    """One recognised word: logical-order text, confidence 0..100, pixel box (top-left origin)."""

    text: str
    conf: float
    left: int
    top: int
    width: int
    height: int


class WordOcrEngine(OcrEngine, Protocol):
    """Optional capability: words with boxes (used to identify glyphs of broken fonts)."""

    def recognize_words(self, image_pgm: bytes) -> list[OcrWord]: ...


@dataclass
class TesseractOcr:
    languages: str = "heb+eng"
    psm: int = 3
    oem: int = 1  # LSTM only
    timeout_s: float = 120.0
    binary: str = "tesseract"
    name: str = "tesseract"

    def _path(self) -> str | None:
        return shutil.which(self.binary)

    def version(self) -> str:
        exe = self._path()
        if not exe:
            return "unavailable"
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
        first = (out.stdout or out.stderr).splitlines()
        return first[0].strip() if first else "unknown"

    def installed_languages(self) -> set[str]:
        exe = self._path()
        if not exe:
            return set()
        out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=30)
        return {ln.strip() for ln in out.stdout.splitlines()[1:] if ln.strip()}

    def available(self) -> bool:
        return self._path() is not None

    def effective_languages(self) -> str:
        have = self.installed_languages()
        wanted = [lang for lang in self.languages.split("+") if lang in have]
        return "+".join(wanted) if wanted else "eng"

    def _run(self, image_pgm: bytes, *extra: str) -> str:
        exe = self._path()
        if not exe:
            return ""
        env = dict(os.environ)
        env["OMP_THREAD_LIMIT"] = "1"
        with tempfile.TemporaryDirectory(prefix="hebmath-ocr-") as tmp:
            img = Path(tmp) / "page.pgm"
            img.write_bytes(image_pgm)
            cmd = [
                exe,
                str(img),
                "stdout",
                "-l",
                self.effective_languages(),
                "--psm",
                str(self.psm),
                "--oem",
                str(self.oem),
                "-c",
                "preserve_interword_spaces=0",
                *extra,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=self.timeout_s, env=env)
            except subprocess.TimeoutExpired as exc:
                raise ExtractionTimeoutError("OCR timed out") from exc
        return proc.stdout.decode("utf-8", "replace")

    def recognize(self, image_pgm: bytes, *, detect_math: bool) -> list[Block]:
        return text_to_blocks(self._run(image_pgm), detect_math=detect_math)

    def recognize_words(self, image_pgm: bytes) -> list[OcrWord]:
        out = []
        for row in self._run(image_pgm, "tsv").splitlines()[1:]:
            f = row.split("\t")
            if len(f) < 12 or f[0] != "5" or not f[11].strip():
                continue
            try:
                out.append(
                    OcrWord(f[11].strip(), float(f[10]), int(f[6]), int(f[7]), int(f[8]), int(f[9]))
                )
            except ValueError:
                continue
        return out


def text_to_blocks(text: str, *, detect_math: bool) -> list[Block]:
    blocks: list[Block] = []
    for para in _paragraphs(text):
        segs = [TextSpan(para)]
        if detect_math:
            segs = tag_segments(segs)
        blocks.append(
            promote_lone_math(Block.paragraph(segs)) if detect_math else Block.paragraph(segs)
        )
    return blocks


def _paragraphs(text: str) -> list[str]:
    out = []
    for chunk in text.replace("\f", "\n\n").split("\n\n"):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if lines:
            out.append(" ".join(lines))
    return out
