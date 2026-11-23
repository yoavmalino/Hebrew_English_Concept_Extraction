"""Tests for the stateless demo API: stage summaries, limits, errors and that nothing is written."""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DEMO_API = Path(__file__).resolve().parents[1] / "demo" / "api"
sys.path.insert(0, str(DEMO_API))

import analysis  # noqa: E402
from app import app  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "extractor" / "tests" / "fixtures"


@pytest.fixture
def client():
    return TestClient(app)


def _post(client, data: bytes, filename: str = "doc.docx"):
    return client.post("/analyze", content=data, headers={"x-filename": filename})


def test_analyze_returns_every_stage_for_a_docx(client):
    response = _post(client, (FIXTURES / "struct_he_styles.docx").read_bytes())
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "algorithmic"
    assert {"stage1", "stage2", "stage4", "stage5", "stage6"} <= set(body)
    assert body["stage5"]["t_signal"] == "off"
    assert body["stage6"]["nodes"], "a summary with headings yields at least one node"
    assert all("name" in node and "passage" in node for node in body["stage6"]["nodes"])


def test_analyze_rejects_a_body_over_the_limit(client, monkeypatch):
    monkeypatch.setattr("app.MAX_FILE_BYTES", 10)
    response = _post(client, b"%PDF-" + b"x" * 100, "big.pdf")
    assert response.status_code == 413


def test_analyze_explains_an_unsupported_file_without_internal_details(client):
    response = _post(client, b"just some text, not a document", "notes.txt")
    assert response.status_code == 422
    assert response.json()["detail"] == "Only PDF and DOCX files are supported."


def test_analyze_rejects_an_empty_body(client):
    assert _post(client, b"").status_code == 400


def test_analyze_writes_no_file(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(DEMO_API))
    assert _post(client, (FIXTURES / "plain_he.docx").read_bytes()).status_code in (200, 422)
    assert list(tmp_path.iterdir()) == []
    assert set(os.listdir(DEMO_API)) == before


def test_demo_turns_ocr_off_and_uses_its_own_size_limit():
    limits = analysis._EXTRACTOR.config.limits
    assert limits.max_file_bytes == analysis.MAX_FILE_BYTES
    assert analysis._EXTRACTOR.config.enable_ocr is False
