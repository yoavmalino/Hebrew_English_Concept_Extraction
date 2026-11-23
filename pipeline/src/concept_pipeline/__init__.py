"""Concept-graph pipeline stages (design doc section 4), pure logic with no database or network.

Stage 1 lives in the hebmath extractor; this package starts at stage 2 (candidates). The importer
does not use it yet: it is connected once enough stages pass their gates.
"""
