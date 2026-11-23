"""Stateless demo API: POST /analyze runs the algorithmic pipeline on one uploaded document.

Nothing is stored and nothing about the document is logged: the body is read into memory,
analyzed and dropped. There is no database, no model call and no authentication; the size
limit and the one-at-a-time extraction lock keep a free instance usable.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from analysis import MAX_FILE_BYTES, AnalysisError, analyze

ALLOWED_ORIGINS_ENV = "DEMO_ALLOWED_ORIGINS"
# Header that carries the original file name (optional); kept out of the URL so that access
# logs, which record paths, never see it.
FILENAME_HEADER = "x-filename"
# Plain-language messages per hebmath error code; anything else gets the generic one.
ERROR_MESSAGES = {
    "unsupported_format": "Only PDF and DOCX files are supported.",
    "corrupt_file": "The file looks damaged. Try saving it again.",
    "encrypted_file": "The file is password protected. Remove the password and try again.",
    "file_too_large": "The file is too large for the demo (10 MB, 150 pages).",
    "resource_limit": "The file is too large for the demo (10 MB, 150 pages).",
    "timeout": "The file took too long to read. Try a shorter file.",
    "empty_file": "The file is empty.",
    # A scanned file has no text layer, and the demo runs no OCR, so it ends here too.
    "no_concepts": "No terms were found. Scanned files are not supported in the demo.",
}
GENERIC_ERROR = "The file could not be analyzed."

# Uvicorn's access log is turned off at start (render.yaml); this keeps library loggers quiet too.
logging.getLogger("hebmath").setLevel(logging.WARNING)

app = FastAPI(title="Hebrew-English concept extraction demo", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get(ALLOWED_ORIGINS_ENV, "*").split(",") if o],
    allow_methods=["POST", "GET"],
    allow_headers=["content-type", FILENAME_HEADER],
)


@app.get("/health")
def health() -> dict:
    """Return a liveness answer for the host's health check."""
    return {"status": "ok"}


async def _read_limited(request: Request) -> bytes:
    """Return the request body, refusing it as soon as it passes `MAX_FILE_BYTES`."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_FILE_BYTES:
        raise HTTPException(413, ERROR_MESSAGES["file_too_large"])
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_FILE_BYTES:
            raise HTTPException(413, ERROR_MESSAGES["file_too_large"])
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/analyze")
async def analyze_document(request: Request) -> dict:
    """Analyze the raw PDF or DOCX bytes in the body and return the per-stage summary.

    The body is the file itself (no multipart form). The optional `X-Filename` header helps
    detect a damaged PDF and is not stored.
    """
    data = await _read_limited(request)
    if not data:
        raise HTTPException(400, "Send a PDF or DOCX file as the request body.")
    filename = request.headers.get(FILENAME_HEADER)
    try:
        return await run_in_threadpool(analyze, data, filename)
    except AnalysisError as error:
        raise HTTPException(422, ERROR_MESSAGES.get(error.code, GENERIC_ERROR)) from None
