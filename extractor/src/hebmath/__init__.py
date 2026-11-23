"""hebmath - Hebrew-aware PDF/DOCX text extractor with LaTeX math output."""

__version__ = "1.1.0"

from .pipeline import ExtractionResult, Extractor, ExtractorConfig, extract  # noqa: E402

__all__ = ["extract", "Extractor", "ExtractorConfig", "ExtractionResult", "__version__"]
