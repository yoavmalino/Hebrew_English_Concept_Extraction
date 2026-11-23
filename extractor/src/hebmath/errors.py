"""Error taxonomy. Every failure surfaced to callers carries a stable ``code``."""

from __future__ import annotations


class ExtractionError(Exception):
    """Base class for controlled extraction failures."""

    code = "extraction_failed"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class UnsupportedFormatError(ExtractionError):
    code = "unsupported_format"


class CorruptFileError(ExtractionError):
    code = "corrupt_file"


class EncryptedFileError(ExtractionError):
    code = "encrypted_file"


class ResourceLimitError(ExtractionError):
    code = "resource_limit"


class ExtractionTimeoutError(ExtractionError):
    code = "timeout"
