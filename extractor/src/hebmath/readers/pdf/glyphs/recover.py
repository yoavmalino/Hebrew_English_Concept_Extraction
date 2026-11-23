"""Recover the characters of PDF fonts that have no usable Unicode mapping.

Typical source: Word documents printed through PScript5 + Ghostscript (and
similar drivers). Every font is embedded as a subset whose codes are assigned
in order of first use and that has neither a ToUnicode CMap nor meaningful
glyph names, e.g. Hebrew "David" and the Windows "Symbol" font used by
MathType/Equation Editor. PDFium reports codes below 0x20 as unmapped and
silently maps the others through a default encoding, so *all* codes of such a
font are wrong. Previously these pages were sent to OCR wholesale, which
destroyed the mathematics that the text layer did contain (sub/superscripts,
fractions, operators, Greek letters).

Method, per document (deterministic, no learned weights):

1. *Shape*: every distinct (font, code) glyph is rendered alone (render.py)
   and matched against the bundled reference atlas (atlas.py). A per-font
   prior is then applied: a font whose glyphs are mostly Hebrew letters is a
   text font (math symbols unlikely), one whose glyphs are mostly symbols is
   a math font (Hebrew unlikely, '−' rather than '–', '⋅' rather than '·').
2. *OCR cross-check* (optional, only for glyphs the shape match is unsure
   of): a few pages are OCR'd with word boxes; OCR words are aligned with the
   glyphs under them and every alignment votes for the glyph's character.
   Words whose known glyphs disagree with the OCR text are discarded, so a
   misread word cannot cast votes.
3. *Assembled delimiters*: tall brackets that the producer built from
   pieces (⎛ ⎜ ⎝, ⎡ ⎢ ⎣, ⎧ ⎨ ⎩ ...) are found as vertical stacks, rendered
   together and classified as one delimiter, then replaced by a single tall
   atom that the math engine treats like any big delimiter.

Glyphs that stay unresolved keep U+FFFD, so the page-level OCR fallback still
applies when a page cannot be read.
"""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from ....bidi import visual_to_logical
from ..geometry import Atom
from .atlas import Atlas, Match, load_atlas
from .render import EMPTY, TALL_STRETCH, Descriptor, render_glyph, render_glyphs

log = logging.getLogger("hebmath.pdf.glyphs")

GlyphKey = tuple[str, int, bool]  # (font identity, code, code is raw i.e. PDFium had no mapping)

REPLACEMENT = "�"

# confidence thresholds on the atlas distance (0 = identical shape)
_CONFIDENT = 0.12
_MARGIN = 0.03
_USABLE = 0.35
# A suspect font whose mapped glyphs are mostly Hebrew letters has a real ToUnicode table: a
# fallback encoding only yields Latin letters. Then only its unmapped codes are recovered and the
# mapped ones are kept (2026-11-14: one unmapped code out of 13,821 glyphs sent a whole LaTeX
# Arial font to shape recovery, which read every ו as "⎮" and every ן as ו).
TRUSTED_HEBREW_SHARE = 0.5

# characters that print identically and only differ by role
_MATH_EQUIV = {"–": "−", "‒": "−", "‐": "−", "⎯": "−", "-": "−", "·": "⋅", "∙": "⋅", "ℵ": "ℵ"}
_TEXT_EQUIV = {"−": "-", "⎯": "—", "⋅": "·", "∙": "·", "ℵ": "א", "│": "|", "∣": "|"}
_TEXT_PUNCT = set(".,:;!?'\"()[]-–—/…%&*׳״־‘’“”")
_BAR_LIKE = set("|│∣⎜⎟⎢⎥⎪⎮⏐")
_FINALS = set("ךםןףץ")
_NON_FINALS = set("כמנפצ")
_FINAL_OF = dict(zip("כמנפצ", "ךםןףץ"))
_NON_FINAL_OF = dict(zip("ךםןףץ", "כמנפצ"))
_PIECES = set("⎛⎜⎝⎞⎟⎠⎡⎢⎣⎤⎥⎦⎧⎨⎩⎫⎬⎭⎪⎮⌠⌡⌈⌉⌊⌋(){}[]⟦⟧") | _BAR_LIKE


# A prior on symbol frequency: in course notes these are common, while their
# rare look-alikes (⇛ for ⇒, ⊝ for ⊕, ≖ for =) almost never occur.
COMMON_MATH = set(
    "+−=<>≤≥≠≈≡≅∼∝±∓×÷⋅∘∗√∞∂∇∑∏∐∫∮∀∃∄¬∧∨⊕⊗⊙∈∉∋⊂⊃⊆⊇⊄⊊∪∩∖∅→←↔⇒⇐⇔↦↑↓⟶⟹⟺"
    "ℝℕℤℚℂℙ…⋯⋮⋱′″‖|()[]{}⟨⟩⌈⌉⌊⌋∠⊥∥△□°"
    "αβγδεϵζηθϑικλμνξπϖρϱσςτυφϕχψωΓΔΘΛΞΠΣΥΦΨΩ"
)


def category(ch: str) -> str:
    if not ch:
        return "none"
    if "א" <= ch <= "ת" or ch in "׳״־":
        return "heb"
    if ch.isascii() and ch.isalpha():
        return "lat"
    if ch.isdigit():
        return "dig"
    if ch in _TEXT_PUNCT:
        return "punct"
    return "math"


class GlyphRecognizer(Protocol):
    """Pluggable shape recogniser (the atlas matcher is the default)."""

    name: str

    def candidates(self, desc: Descriptor, k: int, penalty=None) -> list[Match]: ...

    def tall(self, desc: Descriptor) -> list[Match]: ...


@dataclass
class AtlasRecognizer:
    atlas: Atlas | None = None
    name: str = "glyph-atlas"

    def __post_init__(self) -> None:
        if self.atlas is None:
            self.atlas = load_atlas()

    @property
    def version(self) -> str:
        return f"atlas v{self.atlas.version} ({len(self.atlas.glyphs)} refs)"

    def candidates(self, desc: Descriptor, k: int, penalty=None) -> list[Match]:
        return self.atlas.match(desc, k, penalty)

    def tall(self, desc: Descriptor) -> list[Match]:
        return self.atlas.match_tall(desc)

    def distance_to(self, desc: Descriptor, char: str) -> float:
        return self.atlas.distance_to(desc, char)


@dataclass
class RecoveryConfig:
    enabled: bool = True
    use_ocr: bool = True  # cross-check unsure glyphs with OCR when available
    max_ocr_pages: int = (
        6  # pages OCR'd for the cross-check (they are chosen to cover unsure glyphs)
    )
    ocr_dpi: int = 300


@dataclass
class _Glyph:
    key: GlyphKey
    count: int = 0
    first: tuple[int, int] | None = None  # (page index, char index) of one instance
    desc: Descriptor | None = None
    matches: list[Match] = field(default_factory=list)
    char: str = REPLACEMENT
    method: str = "unresolved"
    confident: bool = False
    # neighbours of every instance (visual order): ("key", GlyphKey) | ("char", str) | None
    context: list[tuple] = field(default_factory=list)


@dataclass
class RecoveryReport:
    fonts: int = 0
    glyphs: int = 0
    unresolved: int = 0
    ocr_pages: int = 0
    ocr_decided: int = 0
    assembled: int = 0
    restored: int = 0
    font_classes: dict[str, str] = field(default_factory=dict)


class FontRecovery:
    def __init__(
        self,
        recognizer: GlyphRecognizer | None = None,
        ocr=None,
        config: RecoveryConfig | None = None,
    ):
        self.recognizer = recognizer or AtlasRecognizer()
        self.ocr = ocr
        self.config = config or RecoveryConfig()

    # ------------------------------------------------------------------
    def run(self, pdf, contents: list, render_page=None) -> RecoveryReport:
        """Rewrite atoms of unmapped fonts in ``contents`` (list[PageContent | None]) in place.

        ``render_page(index, dpi) -> pgm bytes`` is needed for the OCR cross-check.
        """
        report = RecoveryReport()
        cipher = set()
        for c in contents:
            if c is not None:
                cipher |= c.cipher_fonts
        if not cipher or not self.config.enabled:
            return report
        self._trusted = self._trusted_fonts(contents, cipher)
        glyphs = self._collect(contents, cipher)
        self._describe(pdf, glyphs)
        classes = self._classify_fonts(glyphs)
        report.font_classes = {k.split("#")[0] or k: v for k, v in sorted(classes.items())}
        self._decide_by_shape(glyphs, classes)
        for _ in range(2):
            if not self._rerank_by_context(glyphs, classes):
                break
        # OCR can only arbitrate between letters and digits (its language model
        # knows words, not symbols): ask it only about those
        unsure = [
            g
            for g in glyphs.values()
            if not g.confident
            and g.matches
            and any(category(m.char) in ("heb", "dig", "lat") for m in g.matches[:3])
            and classes.get(g.key[0]) != "math"
        ]
        if (
            unsure
            and self.config.use_ocr
            and self.ocr is not None
            and render_page is not None
            and getattr(self.ocr, "available", lambda: False)()
            and hasattr(self.ocr, "recognize_words")
        ):
            report.ocr_pages, report.ocr_decided = self._ocr_cross_check(
                contents, glyphs, classes, unsure, render_page
            )
        self._apply(contents, glyphs, classes)
        report.restored = self._restore_dropped_glyphs(pdf, contents, glyphs, classes)
        report.assembled = self._assemble_delimiters(pdf, contents, cipher)
        report.fonts = len(cipher)
        report.glyphs = len(glyphs)
        report.unresolved = sum(1 for g in glyphs.values() if g.char == REPLACEMENT)
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _key(a: Atom) -> GlyphKey:
        return (a.fkey, a.code, a.raw_code)

    @staticmethod
    def _trusted_fonts(contents, cipher) -> set[str]:
        """Return the suspect fonts whose mapped glyphs are mostly Hebrew (see TRUSTED_HEBREW_SHARE)."""
        mapped: Counter = Counter()
        hebrew: Counter = Counter()
        for c in contents:
            if c is None:
                continue
            for a in c.atoms:
                if a.fkey in cipher and a.kind == "g" and a.text != REPLACEMENT:
                    mapped[a.fkey] += 1
                    hebrew[a.fkey] += category(a.text) == "heb"
        return {fkey for fkey in mapped if hebrew[fkey] >= TRUSTED_HEBREW_SHARE * mapped[fkey]}

    def _suspect(self, a: Atom, cipher) -> bool:
        """Return True when a glyph must be recovered: its font is suspect and not trusted."""
        if a.fkey not in cipher:
            return False
        return a.fkey not in getattr(self, "_trusted", set()) or a.text == REPLACEMENT

    def _collect(self, contents, cipher) -> dict[GlyphKey, _Glyph]:
        glyphs: dict[GlyphKey, _Glyph] = {}
        for pi, c in enumerate(contents):
            if c is None:
                continue
            for a in c.atoms:
                if not self._suspect(a, cipher):
                    continue
                key = self._key(a)
                g = glyphs.get(key)
                if g is None:
                    g = glyphs[key] = _Glyph(key, first=(pi, a.order))
                g.count += 1
            self._neighbours(c, cipher, glyphs)
        return glyphs

    def _neighbours(self, c, cipher, glyphs) -> None:
        """Record, per instance, the glyph drawn directly left and right of it (same word)."""
        rows: dict[int, list[Atom]] = defaultdict(list)
        for a in c.atoms:
            if a.kind == "g" or self._suspect(a, cipher):
                rows[int(round(a.base / max(a.size * 0.3, 0.5)))].append(a)
        for row in rows.values():
            row.sort(key=lambda a: (a.x0, a.order))

            def ref(b: Atom | None):
                if b is None:
                    return None
                if self._suspect(b, cipher):
                    return ("key", self._key(b))
                return ("char", b.text) if b.kind == "g" else None

            for k, a in enumerate(row):
                if not self._suspect(a, cipher):
                    continue
                left = row[k - 1] if k > 0 and a.x0 - row[k - 1].x1 < 0.2 * a.size else None
                right = (
                    row[k + 1] if k + 1 < len(row) and row[k + 1].x0 - a.x1 < 0.2 * a.size else None
                )
                glyphs[self._key(a)].context.append((ref(left), ref(right)))

    def _describe(self, pdf, glyphs: dict[GlyphKey, _Glyph]) -> None:
        import ctypes

        import pypdfium2.raw as raw

        by_page: dict[int, list[_Glyph]] = defaultdict(list)
        for g in glyphs.values():
            by_page[g.first[0]].append(g)
        notdef: dict[int, Descriptor] = {}
        for pi in sorted(by_page):
            page = pdf[pi]
            textpage = page.get_textpage()
            try:
                for g in sorted(by_page[pi], key=lambda x: x.first[1]):
                    obj = raw.FPDFText_GetTextObject(textpage.raw, g.first[1])
                    font = raw.FPDFTextObj_GetFont(obj) if obj else None
                    if not font:
                        g.desc = None
                        continue
                    _, code, is_raw = g.key
                    if is_raw:
                        g.desc = render_glyph(pdf.raw, font, charcode=code)
                        continue
                    # PDFium inverts its own (wrong) mapping back to the char code; for
                    # CID fonts that fails and draws .notdef - then the code is the CID
                    addr = ctypes.cast(font, ctypes.c_void_p).value or 0
                    if addr not in notdef:
                        notdef[addr] = render_glyph(pdf.raw, font, text="\uffff")
                    d = render_glyph(pdf.raw, font, text=chr(code)) if code < 0x110000 else EMPTY
                    nd = notdef[addr]
                    if d.empty or (not nd.empty and d.grid == nd.grid):
                        alt = render_glyph(pdf.raw, font, charcode=code)
                        if not alt.empty and (nd.empty or alt.grid != nd.grid):
                            d = alt
                    g.desc = d
            finally:
                textpage.close()
                page.close()
        for g in glyphs.values():
            if g.desc is not None and not g.desc.empty:
                g.matches = self.recognizer.candidates(g.desc, 8)

    def _classify_fonts(self, glyphs) -> dict[str, str]:
        votes: dict[str, Counter] = defaultdict(Counter)
        for g in glyphs.values():
            if g.matches and g.matches[0].distance <= 0.2:
                votes[g.key[0]][category(g.matches[0].char)] += 1
        classes = {}
        for fkey in {g.key[0] for g in glyphs.values()}:
            v = votes.get(fkey, Counter())
            total = sum(v.values()) or 1
            heb, math_, lat = v["heb"] / total, v["math"] / total, v["lat"] / total
            if heb >= 0.2:
                classes[fkey] = "hebrew"
            elif math_ >= 0.35:
                classes[fkey] = "math"
            elif lat >= 0.3:
                classes[fkey] = "latin"
            else:
                classes[fkey] = "other"
        return classes

    @staticmethod
    def _penalty(font_class: str):
        def hebrew(ch: str) -> float:
            cat = category(ch)
            return 0.10 if cat == "math" else 0.0

        def math_(ch: str) -> float:
            cat = category(ch)
            if cat == "heb":
                return 0.25
            if ch in "–—‒‐·∙⎯'\"’‘":
                return 0.08
            if cat == "math":
                return -0.04 if ch in COMMON_MATH else 0.04
            return 0.02 if cat == "lat" else 0.0

        def latin(ch: str) -> float:
            cat = category(ch)
            return 0.10 if cat == "heb" else 0.05 if cat == "math" else 0.0

        return {"hebrew": hebrew, "math": math_, "latin": latin}.get(font_class)

    @staticmethod
    def _normalise(ch: str, font_class: str) -> str:
        if font_class == "math":
            return _MATH_EQUIV.get(ch, ch)
        if font_class in ("hebrew", "latin"):
            return _TEXT_EQUIV.get(ch, ch)
        return ch

    def _decide_by_shape(self, glyphs, classes) -> None:
        for g in glyphs.values():
            fclass = classes.get(g.key[0], "other")
            if g.desc is None:
                continue
            if g.desc.empty:
                g.char, g.method, g.confident = " ", "blank", True
                continue
            penalty = self._penalty(fclass)
            if penalty is not None:
                g.matches = sorted(
                    (Match(m.char, m.distance + penalty(m.char), m.source) for m in g.matches),
                    key=lambda m: (m.distance, m.char),
                )
            if not g.matches:
                continue
            self._styled_letters(g)
            best = g.matches[0]
            norm = self._normalise(best.char, fclass)
            rivals = [m for m in g.matches[1:] if self._normalise(m.char, fclass) != norm]
            margin = (rivals[0].distance - best.distance) if rivals else 1.0
            g.confident = best.distance <= _CONFIDENT and margin >= _MARGIN
            usable = _USABLE + (
                0.07 if fclass == "math" else 0.0
            )  # symbols vary more across typefaces
            if best.distance <= usable:
                g.char, g.method = norm, "shape"

    def _rerank_by_context(self, glyphs, classes) -> bool:
        """Use the neighbours' (current) identities to separate look-alikes.

        Typeface-independent facts only: a glyph between Hebrew letters is a
        Hebrew letter (not Latin "o"/"I"), between Latin letters it is Latin;
        the final forms ם ן ץ ף ך occur only at the end of a word and their
        non-final partners not there. Returns whether a decision changed.
        """

        def char_of(r) -> str:
            if r is None:
                return ""
            if r[0] == "key":
                return glyphs[r[1]].char if r[1] in glyphs else ""
            return r[1]

        changed = False
        for g in glyphs.values():
            if not g.matches or not g.context or g.char in (" ", REPLACEMENT):
                continue
            n = len(g.context)
            heb = lat = final = inner = lower = 0
            for left, right in g.context:
                lc, rc = char_of(left), char_of(right)
                cats = {category(x) for x in (lc, rc) if x and not x.isspace()}
                if "heb" in cats and "lat" not in cats:
                    heb += 1
                if "lat" in cats and "heb" not in cats:
                    lat += 1
                    lower += any(x.isascii() and x.islower() for x in (lc, rc) if x)
                # right to left: the logically last letter of a word is drawn leftmost
                if category(rc) == "heb":
                    if category(lc) == "heb":
                        inner += 1
                    elif not lc or lc.isspace():
                        final += 1
            fclass = classes.get(g.key[0], "other")
            penalty = {}
            for m in g.matches:
                cat = category(m.char)
                p = 0.0
                if heb >= 0.6 * n and cat in ("lat", "dig"):
                    p += 0.12
                if lat >= 0.6 * n and cat == "heb":
                    p += 0.12
                if lat >= 0.6 * n and lower >= 0.5 * lat and m.char in "IO":
                    p += 0.05  # "l" / "o" inside lowercase words
                mid_word = inner and inner >= 0.3 * (final + inner)
                word_end = final and final >= 0.7 * (final + inner)
                if m.char in _FINALS and mid_word:
                    p += 0.2
                if m.char in _NON_FINALS and word_end:
                    p += 0.15
                # final and non-final forms of a letter are each other's look-alikes
                if (
                    mid_word
                    and m.char in _NON_FINALS
                    and any(x.char == _FINAL_OF[m.char] for x in g.matches[:2])
                ):
                    p -= 0.06
                if (
                    word_end
                    and m.char in _FINALS
                    and any(x.char == _NON_FINAL_OF[m.char] for x in g.matches[:2])
                ):
                    p -= 0.06
                penalty[m.char] = p
            if not any(penalty.values()):
                continue
            ranked = sorted(
                (Match(m.char, m.distance + penalty[m.char], m.source) for m in g.matches),
                key=lambda m: (m.distance, m.char),
            )
            best = ranked[0]
            if best.distance > _USABLE + (0.07 if fclass == "math" else 0.0):
                continue
            new = self._normalise(best.char, fclass)
            if new != g.char and g.method != "ocr":
                g.char, g.method = new, "shape+context"
                changed = True
            g.matches = ranked
        return changed

    def _styled_letters(self, g: _Glyph) -> None:
        """Re-rank styled letters (ℝ ℤ ℕ, script, fraktur) by their base letter's shape.

        Styled designs differ a lot between typefaces (MT Extra draws ℝ as a
        hollow outline, STIX with a doubled stem), but every design keeps the
        silhouette of the base letter, so R-vs-Z is the reliable comparison.
        """
        styled = [
            m
            for m in g.matches[:4]
            if len(unicodedata.normalize("NFKC", m.char)) == 1
            and unicodedata.normalize("NFKC", m.char) != m.char
            and unicodedata.normalize("NFKC", m.char).isalpha()
        ]
        if (
            len(styled) < 2
            or g.matches[0] not in styled
            or not hasattr(self.recognizer, "distance_to")
        ):
            return
        scored = sorted(
            styled,
            key=lambda m: (
                self.recognizer.distance_to(g.desc, unicodedata.normalize("NFKC", m.char)),
                m.char,
            ),
        )
        winner = scored[0]
        if winner is not g.matches[0]:
            g.matches = [Match(winner.char, g.matches[0].distance, winner.source)] + [
                m for m in g.matches if m is not winner
            ]

    # ------------------------------------------------------------------
    def _ocr_cross_check(self, contents, glyphs, classes, unsure, render_page) -> tuple[int, int]:
        unsure_keys = {g.key for g in unsure}
        # choose pages greedily so that they cover as many unsure glyphs as possible
        page_keys: dict[int, set[GlyphKey]] = defaultdict(set)
        for pi, c in enumerate(contents):
            if c is None or c.angle != 0:
                continue
            for a in c.atoms:
                if a.fkey and self._key(a) in unsure_keys:
                    page_keys[pi].add(self._key(a))
        chosen, covered = [], set()
        while page_keys and len(chosen) < self.config.max_ocr_pages:
            pi = max(sorted(page_keys), key=lambda p: len(page_keys[p] - covered))
            if not page_keys[pi] - covered:
                break
            chosen.append(pi)
            covered |= page_keys.pop(pi)
        votes: dict[GlyphKey, Counter] = defaultdict(Counter)
        for pi in sorted(chosen):
            try:
                image = render_page(pi, self.config.ocr_dpi)
                words = self.ocr.recognize_words(image)
            except Exception as exc:  # noqa: BLE001 - the cross-check is best effort
                log.warning("OCR cross-check failed on page %d: %s", pi + 1, exc)
                continue
            self._align(
                contents[pi], words, glyphs, classes, unsure_keys, votes, self.config.ocr_dpi / 72.0
            )
        decided = 0
        for key in sorted(unsure_keys):
            g = glyphs[key]
            v = votes.get(key)
            if not v:
                continue
            top, weight = sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            share = weight / sum(v.values())
            fclass = classes.get(key[0], "other")
            # OCR reads styled letters (ℝ, 𝒜) as their base letter: accept the
            # candidate whose compatibility form is what OCR saw.
            styled = [
                m.char
                for m in g.matches[:4]
                if unicodedata.normalize("NFKC", m.char) == top and m.char != top
            ]
            if styled and weight >= 1.0 and share >= 0.6:
                g.char, g.method = self._normalise(styled[0], fclass), "ocr"
                decided += 1
                continue
            in_candidates = any(
                self._normalise(m.char, fclass) == self._normalise(top, fclass) for m in g.matches
            )
            # OCR is trusted for letters and digits (its language model is good at
            # them) and, for other characters, only when the shape agrees it is plausible.
            shape_cat = category(g.matches[0].char) if g.matches else "none"
            trusted = in_candidates or (
                category(top) in ("heb", "dig", "lat")
                and shape_cat in ("heb", "dig", "lat", "none")
            )
            if weight >= 1.0 and share >= 0.6 and trusted:
                g.char, g.method = self._normalise(top, fclass), "ocr"
                decided += 1
        return len(chosen), decided

    def _align(self, content, words, glyphs, classes, unsure_keys, votes, scale) -> None:
        H = content.height
        atoms = [a for a in content.atoms if a.kind == "g" or (a.fkey and self._key(a) in glyphs)]

        def provisional(a: Atom) -> str | None:
            if a.fkey and self._key(a) in glyphs:
                g = glyphs[self._key(a)]
                if g.char == " ":
                    return None
                return g.char if g.confident else ""  # "" = to be learnt
            return a.text

        for w in words:
            text = "".join(ch for ch in w.text if unicodedata.category(ch) != "Mn")
            if not text:
                continue
            x0, x1 = w.left / scale - 0.5, (w.left + w.width) / scale + 0.5
            ytop, ybot = H - w.top / scale + 0.5, H - (w.top + w.height) / scale - 0.5
            inside = []
            for a in atoms:
                cx, cy = (a.x0 + a.x1) / 2, (a.y0 + a.y1) / 2
                if x0 <= cx <= x1 and ybot <= cy <= ytop:
                    p = provisional(a)
                    if p is not None:
                        inside.append((cx, a, p))
            visual = visual_to_logical(
                text
            )  # logical -> visual (the reordering is its own inverse here)
            if not inside or len(inside) != len(visual):
                continue
            inside.sort(key=lambda t: t[0])
            if any(
                p and p != ch and _TEXT_EQUIV.get(p, p) != _TEXT_EQUIV.get(ch, ch)
                for (_, _, p), ch in zip(inside, visual)
            ):
                continue  # known glyphs disagree: misaligned or misread word
            weight = max(w.conf, 1.0) / 100.0
            for (_, a, p), ch in zip(inside, visual):
                if not p:
                    key = self._key(a)
                    if key in unsure_keys:
                        votes[key][ch] += weight

    # ------------------------------------------------------------------
    def _apply(self, contents, glyphs, classes) -> None:
        for c in contents:
            if c is None:
                continue
            text_chars = unmapped = 0
            for a in c.atoms:
                if a.fkey and self._key(a) in glyphs:
                    g = glyphs[self._key(a)]
                    a.text = g.char
                    a.kind = "sp" if g.char.isspace() else "g"
                    if (
                        a.kind == "g"
                        and (a.x1 - a.x0 <= 0.01 or a.y1 - a.y0 <= 0.01)
                        and g.desc is not None
                        and not g.desc.empty
                    ):
                        # PDFium reports no ink box for codes it took for a space
                        # (code 0x20 drawing "2"): rebuild it from the glyph shape
                        ox = a.lx0 if a.lx1 > a.lx0 else a.x0
                        a.x0, a.x1 = ox + g.desc.x0 * a.size, ox + g.desc.x1 * a.size
                        a.y0, a.y1 = a.base + g.desc.y0 * a.size, a.base + g.desc.y1 * a.size
                    a.math_font = classes.get(a.fkey) == "math"  # also marks unresolved math glyphs
                    if a.math_font and a.text.isascii() and a.text.isalpha():
                        a.italic = True
                if a.kind == "g":
                    text_chars += 1
                    if a.text == REPLACEMENT:
                        unmapped += 1
            c.text_chars, c.unmapped_chars = text_chars, unmapped

    def _restore_dropped_glyphs(self, pdf, contents, glyphs, classes) -> int:
        """Re-insert glyphs that PDFium's text layer dropped as "spaces".

        PDFium treats char code 0x20 as a space and may drop it (repeated or
        "redundant" spaces). In a subset font whose code 0x20 draws a real
        glyph - "≤" in a PScript Symbol subset, "2" in a Hebrew one - the
        character silently disappears ("1 ≤ i ≤ m" became "1im"). Each text
        object of such a font is rendered alone; ink that no reported glyph
        covers, and whose size matches the code-0x20 glyph, is restored.
        """
        import ctypes

        import pypdfium2.raw as raw

        from .fontid import font_identity

        targets = {
            k[0]: g
            for k, g in glyphs.items()
            if k[1] == 0x20
            and not k[2]
            and g.desc is not None
            and not g.desc.empty
            and not g.char.isspace()
            and g.char != REPLACEMENT
        }
        if not targets:
            return 0
        restored = 0
        scale = 4.0
        for pi, c in enumerate(contents):
            if c is None or c.angle != 0 or not (c.cipher_fonts & set(targets)):
                continue
            page = pdf[pi]
            try:
                by_obj: dict[int, list[Atom]] = defaultdict(list)
                for a in c.atoms:
                    if a.obj:
                        by_obj[a.obj].append(a)
                ident: dict[int, str] = {}
                added: list[Atom] = []
                position = 0  # same numbering as extract.text_object_ids()
                for obj in page.get_objects(max_depth=8):
                    if obj.type != raw.FPDF_PAGEOBJ_TEXT:
                        continue
                    position += 1
                    font = raw.FPDFTextObj_GetFont(obj.raw)
                    addr = ctypes.cast(font, ctypes.c_void_p).value or 0
                    if addr not in ident:
                        ident[addr] = font_identity(font)
                    g = targets.get(ident[addr])
                    if g is None:
                        continue
                    oid = position
                    siblings = by_obj.get(oid, [])
                    x0, y0, x1, y1 = obj.get_bounds()
                    size = median_size(siblings) or float(obj.get_font_size() or 0)
                    if size <= 0:
                        continue
                    # only objects whose extent is not covered by their reported glyphs
                    want = g.desc.w * size
                    if siblings and not _uncovered(x0, x1, siblings, 0.6 * want):
                        continue
                    for bx0, by0, bx1, by1 in _ink_blobs(pdf, page, obj, scale, x0, y1):
                        if any(
                            min(bx1, a.x1) - max(bx0, a.x0) > 0.3 * (bx1 - bx0)
                            for a in c.atoms
                            if a.kind == "g" and a.y0 < by1 and a.y1 > by0
                        ):
                            continue  # already in the text layer
                        want_w, want_h = g.desc.w * size, g.desc.h * size
                        if (
                            abs((bx1 - bx0) - want_w) > 0.3 * want_w + 0.6
                            or abs((by1 - by0) - want_h) > 0.3 * want_h + 0.6
                        ):
                            continue
                        base = by0 - g.desc.y0 * size
                        added.append(
                            Atom(
                                x0=bx0,
                                y0=by0,
                                x1=bx1,
                                y1=by1,
                                base=base,
                                size=size,
                                text=g.char,
                                font=siblings[0].font if siblings else "",
                                kind="g",
                                order=min((a.order for a in siblings), default=0),
                                math_font=classes.get(ident[addr]) == "math",
                                code=0x20,
                                lx0=bx0,
                                lx1=bx1,
                                fkey=ident[addr],
                                obj=oid,
                            )
                        )
                if added:
                    c.atoms.extend(added)
                    c.text_chars += len(added)
                    restored += len(added)
            finally:
                page.close()
        return restored

    def _assemble_delimiters(self, pdf, contents, cipher) -> int:
        """Merge vertical stacks of delimiter pieces into one tall delimiter atom."""
        import pypdfium2.raw as raw

        total = 0
        for pi, c in enumerate(contents):
            if c is None:
                continue
            stacks = _find_stacks(
                [
                    a
                    for a in c.atoms
                    if a.kind == "g"
                    and a.fkey in cipher
                    and (a.text in _PIECES or a.text == REPLACEMENT and a.math_font)
                ]
            )
            if not stacks:
                continue
            page = pdf[pi]
            textpage = page.get_textpage()
            try:
                replaced: dict[int, Atom] = {}
                drop: set[int] = set()
                for stack in stacks:
                    obj = raw.FPDFText_GetTextObject(textpage.raw, stack[0].order)
                    font = raw.FPDFTextObj_GetFont(obj) if obj else None
                    if not font:
                        continue
                    top = stack[0]
                    items = []
                    for a in stack:
                        key = self._key(a)
                        dx = (a.lx0 - top.lx0) / top.size
                        dy = (a.base - top.base) / top.size
                        items.append(
                            (key[1], None, dx, dy) if key[2] else (None, chr(key[1]), dx, dy)
                        )
                    d = render_glyphs(pdf.raw, font, items, max_stretch=TALL_STRETCH)
                    best = self.recognizer.tall(d)
                    if not best or best[0].distance > 0.45:
                        continue
                    char = _brace_side(stack) or best[0].char
                    merged = Atom(
                        x0=min(a.x0 for a in stack),
                        y0=min(a.y0 for a in stack),
                        x1=max(a.x1 for a in stack),
                        y1=max(a.y1 for a in stack),
                        base=min(a.base for a in stack),
                        size=top.size,
                        text=char,
                        font=top.font,
                        order=min(a.order for a in stack),
                        kind="g",
                        math_font=True,
                        code=top.code,
                        lx0=min(a.lx0 for a in stack),
                        lx1=max(a.lx1 for a in stack),
                        fkey=top.fkey,
                    )
                    merged.tall = True
                    replaced[id(stack[0])] = merged
                    drop |= {id(a) for a in stack[1:]}
                    total += 1
                if replaced:
                    c.atoms = [replaced.get(id(a), a) for a in c.atoms if id(a) not in drop]
            finally:
                textpage.close()
                page.close()
        return total


def median_size(atoms: list[Atom]) -> float:
    sizes = sorted(a.size for a in atoms if a.kind == "g")
    return sizes[len(sizes) // 2] if sizes else 0.0


def _uncovered(x0: float, x1: float, members: list[Atom], min_gap: float) -> bool:
    """Whether [x0, x1] has a stretch of at least ``min_gap`` not covered by any member's advance box."""
    spans = sorted((min(a.lx0, a.x0), max(a.lx1, a.x1)) for a in members)
    reach = x0
    for s0, s1 in spans:
        if s0 - reach >= min_gap:
            return True
        reach = max(reach, s1)
    return x1 - reach >= min_gap


def _ink_blobs(
    pdf, page, obj, scale: float, left: float, top: float
) -> list[tuple[float, float, float, float]]:
    """Ink boxes (page space) of one text object rendered alone, split at empty columns."""
    import ctypes

    import pypdfium2.raw as raw

    bm = raw.FPDFTextObj_GetRenderedBitmap(pdf.raw, page.raw, obj.raw, ctypes.c_float(scale))
    if not bm:
        return []
    try:
        w, h = raw.FPDFBitmap_GetWidth(bm), raw.FPDFBitmap_GetHeight(bm)
        stride = raw.FPDFBitmap_GetStride(bm)
        buf = ctypes.string_at(raw.FPDFBitmap_GetBuffer(bm), stride * h)
    finally:
        raw.FPDFBitmap_Destroy(bm)
    if w <= 0 or h <= 0 or w * h > 4_000_000:
        return []
    alpha = [buf[y * stride + 3 : y * stride + w * 4 : 4] for y in range(h)]
    cols = [any(row[x] > 96 for row in alpha) for x in range(w)]
    out = []
    x = 0
    while x < w:
        if not cols[x]:
            x += 1
            continue
        s0 = x
        while x < w and cols[x]:
            x += 1
        rows = [y for y in range(h) if any(alpha[y][k] > 96 for k in range(s0, x))]
        out.append(
            (
                left + s0 / scale,
                top - (rows[-1] + 1) / scale,
                left + x / scale,
                top - rows[0] / scale,
            )
        )
    return out


def _brace_side(stack: list[Atom]) -> str | None:
    """A brace is the only assembled delimiter with a distinct *middle* piece.

    The middle piece points away from the content: to the left for '{'.
    """
    if len(stack) < 3:
        return None
    interior = stack[1:-1]
    bars = [a for a in interior if a.text in _BAR_LIKE]
    middles = [a for a in interior if a.text not in _BAR_LIKE]
    if not bars or len(middles) != 1:
        return None
    mid, bar = middles[0], bars[0]
    left_out = bar.x0 - mid.x0
    right_out = mid.x1 - bar.x1
    if max(left_out, right_out) < 0.05 * mid.size:
        return None
    return "{" if left_out > right_out else "}"


def _find_stacks(atoms: list[Atom]) -> list[list[Atom]]:
    """Columns of >= 2 glyphs of one font that touch vertically and share their x range."""
    stacks = []
    used: set[int] = set()
    by_font: dict[str, list[Atom]] = defaultdict(list)
    for a in atoms:
        by_font[a.fkey].append(a)
    for _, group in sorted(by_font.items()):
        group.sort(
            key=lambda a: (-round(a.y1, 1), round(a.x0, 1))
        )  # columns grow from their top piece
        for a in group:
            if id(a) in used:
                continue
            column = [a]
            cur = a
            while True:
                nxt = None
                for b in group:
                    if id(b) in used or b is cur or any(b is x for x in column):
                        continue
                    overlap = min(cur.x1, b.x1) - max(cur.x0, b.x0)
                    narrow = min(cur.x1 - cur.x0, b.x1 - b.x0)
                    gap = cur.y0 - b.y1
                    size = min(cur.size, b.size)
                    # pieces are designed to touch (an extension piece may stop a little
                    # short); glyphs on consecutive lines leave a clear gap
                    reach = 0.35 if (cur.text in _BAR_LIKE or b.text in _BAR_LIKE) else 0.06
                    if (
                        narrow > 0
                        and overlap >= 0.7 * narrow
                        and -0.65 * size <= gap <= reach * size
                    ):
                        if nxt is None or b.y1 > nxt.y1:
                            nxt = b
                if nxt is None:
                    break
                column.append(nxt)
                cur = nxt
            if len(column) >= 2:
                stacks.append(column)
                used |= {id(x) for x in column}
    return stacks
