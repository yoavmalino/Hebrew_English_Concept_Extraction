"""Stage-1 gate invariants: checks that the structure describes the text without breaking it.

Design stage 1 gate: every unit lies in one section, every id points at something real, and the
blocks keep every character of the flat text. Reports problems; never changes the result.
"""

from __future__ import annotations

from .structure import StructureInfo


def _span_problems(text: str, spans: list[tuple[int, int]], name: str) -> list[str]:
    """Return problems with spans that must be in order, inside the text and not overlap."""
    problems = []
    previous_end = 0
    for i, (start, end) in enumerate(spans):
        if not 0 <= start < end <= len(text):
            problems.append(f"{name} {i} span {start}-{end} is empty or outside the text")
        elif start < previous_end:
            problems.append(f"{name} {i} overlaps the {name} before it")
        previous_end = max(previous_end, end)
    return problems


def _uncovered_characters(text: str, spans: list[tuple[int, int]]) -> int:
    """Count non-space characters of the text that no block contains."""
    covered = bytearray(len(text))
    for start, end in spans:
        covered[max(start, 0) : max(end, 0)] = b"\x01" * (max(end, 0) - max(start, 0))
    return sum(1 for i, ch in enumerate(text) if not covered[i] and not ch.isspace())


def _block_problems(result) -> list[str]:
    """Return structure blocks whose span differs from the extracted block they describe.

    One difference is by design: a heading that wrapped onto the next block spans it too, and
    that block gets the role ``heading_cont`` (``structure._merge_wrapped_headings``), so the
    heading's text is ``text[start:end]``. The heading may later become the cover title.
    Anything else is a problem.
    """
    info = result.structure
    if len(info.blocks) != len(result.blocks):
        return ["the structure has a different number of blocks than the extraction"]
    problems = []
    for i, (record, block) in enumerate(zip(info.blocks, result.blocks)):
        if record.start != block.start:
            problems.append(f"block {i} starts at {record.start}, the text block at {block.start}")
        elif record.end != block.end:
            joined = [j for j in range(i + 1, len(info.blocks)) if info.blocks[j].end <= record.end]
            wraps = (
                record.role in ("heading", "title")
                and joined
                and all(info.blocks[j].role == "heading_cont" for j in joined)
                and info.blocks[joined[-1]].end == record.end
            )
            if not wraps:
                problems.append(f"block {i} ends at {record.end}, the text block at {block.end}")
    return problems


def _reference_problems(info: StructureInfo) -> list[str]:
    """Return block references (unit, section, subsection) that point at nothing."""
    problems = []
    for i, block in enumerate(info.blocks):
        if not -1 <= block.unit < len(info.units):
            problems.append(f"block {i} names unit {block.unit}, which does not exist")
        if not -1 <= block.section < len(info.sections):
            problems.append(f"block {i} names section {block.section}, which does not exist")
        if not -1 <= block.subsection < len(info.sections):
            problems.append(f"block {i} names subsection {block.subsection}, which does not exist")
        if block.section >= 0 and info.sections[block.section].level != 1:
            problems.append(f"block {i} has section {block.section}, which is not a main section")
    return problems


def _unit_problems(info: StructureInfo) -> list[str]:
    """Return units that are not inside exactly one main section, or whose blocks disagree."""
    problems = []
    main = [s for s in info.sections if s.level == 1]
    for i, unit in enumerate(info.units):
        if unit.section >= 0:
            section = info.sections[unit.section]
            if not section.start <= unit.start and unit.end <= section.end:
                problems.append(f"unit {i} runs outside its section {unit.section}")
        elif main and unit.start >= main[0].start:
            problems.append(f"unit {i} has no section but starts after the first main heading")
        if sum(1 for s in main if s.start < unit.end and unit.start < s.end) > 1:
            problems.append(f"unit {i} crosses a main-section boundary")
        sections = {b.section for b in info.blocks if b.unit == i}
        if len(sections) > 1:
            problems.append(f"unit {i} has blocks in sections {sorted(sections)}")
        if not any(b.unit == i for b in info.blocks):
            problems.append(f"unit {i} has no block")
    return problems


def check_invariants(result) -> list[str]:
    """Return every stage-1 invariant the extraction result breaks (empty when all hold).

    Args:
        result: An ``ExtractionResult`` from ``Extractor``; one that failed or has no structure
            has nothing to check.

    Returns:
        One plain-English line per problem, in a stable order.
    """
    info = result.structure
    if not result.ok or info is None:
        return []
    text = result.text
    problems = []
    block_spans = [(b.start, b.end) for b in result.blocks]
    problems += _span_problems(text, block_spans, "block")
    uncovered = _uncovered_characters(text, block_spans)
    if uncovered:
        problems.append(f"{uncovered} characters of the text are in no block")
    problems += _block_problems(result)
    problems += _span_problems(text, [(u.start, u.end) for u in info.units], "unit")
    problems += _reference_problems(info)
    problems += _unit_problems(info)
    return problems
