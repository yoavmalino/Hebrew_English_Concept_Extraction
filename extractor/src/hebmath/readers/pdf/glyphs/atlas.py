"""Reference glyph atlas and nearest-shape matching.

The atlas (``hebmath/data/glyph_atlas.json.gz``) holds shape descriptors of
reference glyphs rendered from SIL-OFL fonts (STIX Two Math for Latin, Greek,
digits, punctuation and mathematical symbols; Tinos, Arimo, David Libre,
Frank Ruhl Libre, Miriam Libre, Alef and Noto Hebrew for Hebrew and common
text glyphs). It is built by ``tools/build_glyph_atlas.py``; only descriptors
(coarse 16x16 coverage grids plus em metrics) are shipped, not the fonts.

Matching is a plain nearest-neighbour search: a cheap geometric pre-filter
(height, aspect ratio, vertical placement against the baseline) followed by
the cosine distance of the coverage grids. It is deterministic and has no
learned parameters.
"""

from __future__ import annotations

import gzip
import heapq
import json
import math
import operator
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from .render import GRID, Descriptor

_PREFILTER = 160


@dataclass(frozen=True, slots=True)
class RefGlyph:
    char: str
    source: str
    desc: Descriptor
    norm: float
    coarse: tuple[int, ...]
    coarse_norm: float


@dataclass(frozen=True, slots=True)
class Match:
    char: str
    distance: float
    source: str


@dataclass(frozen=True)
class Atlas:
    version: str
    glyphs: tuple[RefGlyph, ...]
    tall: tuple[RefGlyph, ...] = ()

    def _features(self) -> list[tuple]:
        """Per reference: unit coarse vector and the geometry terms of _geometry()."""
        cached = getattr(self, "_feat_cache", None)
        if cached is None:
            cached = []
            for g in self.glyphs:
                b = g.desc
                hb, wb = max(b.h, 0.02), max(b.w, 0.02)
                cached.append(
                    (
                        [v / g.coarse_norm for v in g.coarse],
                        (b.y0 + b.y1) / 2,
                        math.log(hb),
                        math.log(wb / hb),
                        hb < 0.45,
                    )
                )
            object.__setattr__(self, "_feat_cache", cached)
        return cached

    def distance_to(self, d: Descriptor, char: str) -> float:
        """Smallest distance between ``d`` and any reference of ``char`` (inf if none)."""
        nd = _norm(d.grid)
        return min(
            (distance(d, g.desc, nd, g.norm) for g in self.glyphs if g.char == char),
            default=float("inf"),
        )

    def match_tall(self, d: Descriptor, k: int = 3) -> list[Match]:
        """Classify an assembled (multi-piece) delimiter, see render_glyphs()."""
        if d.empty:
            return []
        nd = _norm(d.grid)
        scored = sorted(
            (distance(d, g.desc, nd, g.norm) - 0.4 * _geometry(d, g.desc), g.char, g.source)
            for g in self.tall
        )
        out, seen = [], set()
        for dist, ch, src in scored:
            if ch not in seen:
                seen.add(ch)
                out.append(Match(ch, dist, src))
            if len(out) == k:
                break
        return out

    def match(self, d: Descriptor, k: int = 5, penalty=None) -> list[Match]:
        """Best ``k`` distinct characters for descriptor ``d`` (closest first).

        ``penalty(char) -> float`` lets the caller add a prior (e.g. "this font
        draws mathematical symbols, Latin letters are unlikely").
        """
        if d.empty:
            return []
        cd = coarse(d.grid)
        ncd = _norm(cd)
        q = [v / ncd for v in cd]
        mul = operator.mul
        ca = (d.y0 + d.y1) / 2
        ha, wa = max(d.h, 0.02), max(d.w, 0.02)
        lh, la = math.log(ha), math.log(wa / ha)
        place_small = ha < 0.45
        feats = self._features()
        scored_pre = []
        for k, (unit, cb, lhb, lab, small) in enumerate(feats):
            place = 2.0 if (place_small or small) else 0.8
            geo = place * abs(ca - cb) + 0.2 * abs(lh - lhb) + 0.3 * abs(la - lab)
            scored_pre.append((1.0 - sum(map(mul, q, unit)) + 0.4 * geo, k))
        pre = [self.glyphs[k] for _, k in heapq.nsmallest(_PREFILTER, scored_pre)]
        nd = _norm(d.grid)
        scored = sorted(
            (
                (
                    distance(d, g.desc, nd, g.norm) + (penalty(g.char) if penalty else 0.0),
                    g.char,
                    g.source,
                )
                for g in pre
            )
        )
        out: list[Match] = []
        seen = set()
        for dist, ch, src in scored:
            if ch in seen:
                continue
            seen.add(ch)
            out.append(Match(ch, dist, src))
            if len(out) == k:
                break
        return out


def _norm(grid: tuple[int, ...]) -> float:
    return math.sqrt(sum(v * v for v in grid)) or 1.0


def _geometry(a: Descriptor, b: Descriptor) -> float:
    """Difference in proportions and in placement against the baseline (em units).

    Typefaces disagree on exact sizes (Symbol's operators are smaller than
    STIX's), so this is deliberately soft; it mostly separates glyphs whose
    shapes coincide once stretched: '-' vs '_' vs '¯', '.' vs '·', '|' vs '/'.
    """
    ha, hb = max(a.h, 0.02), max(b.h, 0.02)
    wa, wb = max(a.w, 0.02), max(b.w, 0.02)
    ca, cb = (a.y0 + a.y1) / 2, (b.y0 + b.y1) / 2
    # placement is what tells small marks apart (',' vs "'", '.' vs '·')
    place = 2.0 if min(ha, hb) < 0.45 else 0.8
    return (
        place * abs(ca - cb)
        + 0.2 * abs(math.log(ha / hb))
        + 0.3 * abs(math.log((wa / ha) / (wb / hb)))
    )


def distance(
    a: Descriptor, b: Descriptor, na: float | None = None, nb: float | None = None
) -> float:
    """Shape distance: 1 - cosine similarity of the grids, plus geometry and topology terms."""
    na = na or _norm(a.grid)
    nb = nb or _norm(b.grid)
    dot = sum(map(operator.mul, a.grid, b.grid))
    topo = 0.05 * min(abs(a.components - b.components), 2) + 0.08 * min(abs(a.holes - b.holes), 2)
    return (1.0 - dot / (na * nb)) + 0.4 * _geometry(a, b) + topo


def coarse(grid: tuple[int, ...]) -> tuple[int, ...]:
    """GRID x GRID -> (GRID/2) x (GRID/2) by summing 2x2 cells (pre-filter signature)."""
    half = GRID // 2
    return tuple(
        grid[(2 * y) * GRID + 2 * x]
        + grid[(2 * y) * GRID + 2 * x + 1]
        + grid[(2 * y + 1) * GRID + 2 * x]
        + grid[(2 * y + 1) * GRID + 2 * x + 1]
        for y in range(half)
        for x in range(half)
    )


@lru_cache(maxsize=1)
def load_atlas() -> Atlas:
    data = resources.files("hebmath.data").joinpath("glyph_atlas.json.gz").read_bytes()
    doc = json.loads(gzip.decompress(data).decode("utf-8"))
    sources = doc["sources"]

    def refs(items) -> tuple[RefGlyph, ...]:
        out = []
        for ch, src, item in items:
            desc = Descriptor.from_json(item)
            cg = coarse(desc.grid)
            out.append(RefGlyph(ch, sources[src], desc, _norm(desc.grid), cg, _norm(cg)))
        return tuple(out)

    return Atlas(doc["version"], refs(doc["glyphs"]), refs(doc.get("tall", [])))
