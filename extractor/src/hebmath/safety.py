"""Shared limits and hardened parsers for untrusted uploads."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass

from .errors import ResourceLimitError


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 100 * 1024 * 1024
    max_zip_uncompressed_bytes: int = 500 * 1024 * 1024
    max_zip_ratio: float = 200.0
    max_zip_entries: int = 20_000
    max_pdf_pages: int = 2_000
    max_ocr_pages: int = 300
    ocr_dpi: int = 300
    page_timeout_s: float = 120.0


def check_zip(zf: zipfile.ZipFile, limits: Limits, compressed_size: int) -> None:
    infos = zf.infolist()
    if len(infos) > limits.max_zip_entries:
        raise ResourceLimitError(f"archive has too many entries ({len(infos)})")
    total = sum(i.file_size for i in infos)
    if total > limits.max_zip_uncompressed_bytes:
        raise ResourceLimitError(f"archive expands to {total} bytes")
    if (
        compressed_size
        and total / max(compressed_size, 1) > limits.max_zip_ratio
        and total > 50 * 1024 * 1024
    ):
        raise ResourceLimitError("suspicious compression ratio (zip bomb?)")
