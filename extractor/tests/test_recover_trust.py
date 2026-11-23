"""Glyph recovery keeps a font's real Hebrew mapping and recovers only its unmapped codes."""

from types import SimpleNamespace

from hebmath.readers.pdf.glyphs.recover import REPLACEMENT, FontRecovery


def _atom(text, fkey="F", kind="g"):
    return SimpleNamespace(text=text, fkey=fkey, kind=kind)


def _page(*texts, fkey="F"):
    return SimpleNamespace(atoms=[_atom(t, fkey) for t in texts])


def _recovery(contents, cipher):
    recovery = FontRecovery.__new__(FontRecovery)
    recovery._trusted = FontRecovery._trusted_fonts(contents, cipher)
    return recovery


def test_a_font_with_a_hebrew_mapping_recovers_only_its_unmapped_glyphs():
    contents = [_page("ש", "ל", "ו", "ם", REPLACEMENT), None]
    recovery = _recovery(contents, {"F"})
    assert recovery._trusted == {"F"}
    assert not recovery._suspect(_atom("ו"), {"F"})
    assert recovery._suspect(_atom(REPLACEMENT), {"F"})


def test_a_font_mapped_to_latin_stays_fully_suspect():
    contents = [_page("a", "b", "c", "d", REPLACEMENT)]
    recovery = _recovery(contents, {"F"})
    assert recovery._trusted == set()
    assert recovery._suspect(_atom("a"), {"F"})


def test_a_font_outside_the_suspect_set_is_never_recovered():
    recovery = _recovery([_page("a")], {"F"})
    assert not recovery._suspect(_atom("a", fkey="G"), {"F"})
