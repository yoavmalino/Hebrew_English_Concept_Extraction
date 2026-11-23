"""Post-processing step shared by readers: tag plain-text math inside prose."""

from __future__ import annotations

from .latex.plaintext import linear_to_latex, tag_plain_text
from .model import Block, BlockKind, InlineMath, Segment, TextSpan


def merge_text_spans(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segments:
        if isinstance(s, TextSpan) and out and isinstance(out[-1], TextSpan):
            out[-1] = TextSpan(out[-1].text + s.text)
        elif isinstance(s, TextSpan) and not s.text:
            continue
        else:
            out.append(s)
    return out


def tag_segments(segments: list[Segment]) -> list[Segment]:
    """Find formulas typed as ordinary characters and turn them into InlineMath."""
    out: list[Segment] = []
    for seg in merge_text_spans(segments):
        if not isinstance(seg, TextSpan):
            out.append(seg)
            continue
        for piece in tag_plain_text(seg.text):
            if piece.is_math:
                latex = linear_to_latex(piece.text)
                out.append(InlineMath(latex) if latex else TextSpan(piece.text))
            else:
                out.append(TextSpan(piece.text))
    return merge_text_spans(out)


def promote_lone_math(block: Block) -> Block:
    """A paragraph that contains nothing but one formula is a display formula."""
    if block.kind is not BlockKind.PARAGRAPH:
        return block
    real = [s for s in block.segments if not (isinstance(s, TextSpan) and not s.text.strip())]
    if len(real) == 1 and isinstance(real[0], InlineMath):
        return Block.display_math(real[0].latex)
    return block
