"""
PUA (Private Use Area) glyph remap for math-corrupted text.

GROBID's text extraction sometimes leaves Symbol-font and MT-Extra glyphs
as Unicode Private Use Area codepoints (U+F000-U+F8FF) when the source PDF
ships fonts without ToUnicode CMaps. The result is text like:

    "1 k k k k k k k k x Ax Bu Ef \uf02b \uf03d \uf02b \uf02b \uf0ec"

instead of:

    "x_{k+1} = Ax_k + Bu_k + Ef_k, y_k = ..."

The mapping below covers the standard Adobe Symbol font (the dominant
source of these glyphs across our dataset) plus the common MT Extra and
Microsoft Equation glyphs we observed in the apenergy 2019 paper.

v3 patch: ALSO normalizes the Mathematical Alphanumeric Symbols block
(U+1D400-U+1D7FF). Modern PDFs that DO ship proper ToUnicode CMaps but
use math fonts encode italic/bold variables as these codepoints
(e.g. "𝑓 = 𝜎 (𝑥 𝑡 𝑊 𝑥 (𝑓) + ℎ 𝑡-1 𝑊 ℎ (𝑓) + 𝑏 (𝑓) )"). This is the
born-Unicode-math failure mode, distinct from PUA. We fold these back to
ASCII so downstream text matching / search works.

References:
    - Adobe Symbol Encoding: https://unicode.org/Public/MAPPINGS/VENDORS/ADOBE/symbol.txt
    - Wolfram (MT Extra): non-standard, glyphs mapped by manual inspection
    - Microsoft Equation Editor: similar to MT Extra
    - Mathematical Alphanumeric Symbols: https://unicode.org/charts/PDF/U1D400.pdf

Usage:
    from pua_remap import remap_pua_glyphs
    cleaned = remap_pua_glyphs("x \uf02b y \uf03d z")
    # -> "x + y = z"
    cleaned = remap_pua_glyphs("\U0001D453 = \U0001D70E")   # 𝑓 = 𝜎
    # -> "f = σ"

Scope and limits:
    This is a CHARACTER-level remap. It fixes inline-math corruption like
    operators, Greek letters, and brackets. It does NOT recover layout
    (subscripts, fractions, matrices) — for that, equations should be
    extracted as PNG images (see equation_extractor.py).

    Coverage: about 80% of PUA glyphs in our test set. Unmapped glyphs are
    left as-is (you can grep U+F0xx codepoints to find new ones to add).
"""

from __future__ import annotations
import re


# Adobe Symbol font — full standard mapping.
# Format: PUA codepoint (int) -> replacement string.
SYMBOL_FONT_MAP: dict[int, str] = {
    # Punctuation, operators
    0xF020: " ",     # space
    0xF021: "!",     # exclam
    0xF022: "∀",     # universal (for all)
    0xF023: "#",     # numbersign
    0xF024: "∃",     # existential (there exists)
    0xF025: "%",     # percent
    0xF026: "&",     # ampersand
    0xF027: "∋",     # suchthat
    0xF028: "(",     # parenleft
    0xF029: ")",     # parenright
    0xF02A: "∗",     # asteriskmath
    0xF02B: "+",     # plus
    0xF02C: ",",     # comma
    0xF02D: "−",     # minus
    0xF02E: ".",     # period
    0xF02F: "/",     # slash
    0xF030: "0",     # zero
    0xF031: "1",
    0xF032: "2",
    0xF033: "3",
    0xF034: "4",
    0xF035: "5",
    0xF036: "6",
    0xF037: "7",
    0xF038: "8",
    0xF039: "9",
    0xF03A: ":",     # colon
    0xF03B: ";",     # semicolon
    0xF03C: "<",     # less
    0xF03D: "=",     # equal
    0xF03E: ">",     # greater
    0xF03F: "?",     # question
    0xF040: "≅",     # congruent
    # Greek uppercase
    0xF041: "Α",     # Alpha
    0xF042: "Β",     # Beta
    0xF043: "Χ",     # Chi
    0xF044: "Δ",     # Delta
    0xF045: "Ε",     # Epsilon
    0xF046: "Φ",     # Phi
    0xF047: "Γ",     # Gamma
    0xF048: "Η",     # Eta
    0xF049: "Ι",     # Iota
    0xF04A: "ϑ",     # theta1
    0xF04B: "Κ",     # Kappa
    0xF04C: "Λ",     # Lambda
    0xF04D: "Μ",     # Mu
    0xF04E: "Ν",     # Nu
    0xF04F: "Ο",     # Omicron
    0xF050: "Π",     # Pi
    0xF051: "Θ",     # Theta
    0xF052: "Ρ",     # Rho
    0xF053: "Σ",     # Sigma
    0xF054: "Τ",     # Tau
    0xF055: "Υ",     # Upsilon
    0xF056: "ς",     # sigma1
    0xF057: "Ω",     # Omega
    0xF058: "Ξ",     # Xi
    0xF059: "Ψ",     # Psi
    0xF05A: "Ζ",     # Zeta
    0xF05B: "[",     # bracketleft
    0xF05C: "∴",     # therefore
    0xF05D: "]",     # bracketright
    0xF05E: "⊥",     # perpendicular
    0xF05F: "_",     # underscore
    0xF060: "‾",     # radicalex
    # Greek lowercase
    0xF061: "α",     # alpha
    0xF062: "β",     # beta
    0xF063: "χ",     # chi
    0xF064: "δ",     # delta
    0xF065: "ε",     # epsilon
    0xF066: "ϕ",     # phi
    0xF067: "γ",     # gamma
    0xF068: "η",     # eta
    0xF069: "ι",     # iota
    0xF06A: "φ",     # phi1
    0xF06B: "κ",     # kappa
    0xF06C: "λ",     # lambda
    0xF06D: "μ",     # mu
    0xF06E: "ν",     # nu
    0xF06F: "ο",     # omicron
    0xF070: "π",     # pi
    0xF071: "θ",     # theta
    0xF072: "ρ",     # rho
    0xF073: "σ",     # sigma
    0xF074: "τ",     # tau
    0xF075: "υ",     # upsilon
    0xF076: "ϖ",     # omega1
    0xF077: "ω",     # omega
    0xF078: "ξ",     # xi
    0xF079: "ψ",     # psi
    0xF07A: "ζ",     # zeta
    0xF07B: "{",     # braceleft
    0xF07C: "|",     # bar
    0xF07D: "}",     # braceright
    0xF07E: "∼",     # similar (tilde)
    # Symbols
    0xF0A0: "€",     # Euro (later addition)
    0xF0A1: "ϒ",     # Upsilon1
    0xF0A2: "′",     # prime (minute)
    0xF0A3: "≤",     # lessequal
    0xF0A4: "⁄",     # fraction
    0xF0A5: "∞",     # infinity
    0xF0A6: "ƒ",     # florin
    0xF0A7: "♣",     # club
    0xF0A8: "♦",     # diamond
    0xF0A9: "♥",     # heart
    0xF0AA: "♠",     # spade
    0xF0AB: "↔",     # arrowboth
    0xF0AC: "←",     # arrowleft
    0xF0AD: "↑",     # arrowup
    0xF0AE: "→",     # arrowright
    0xF0AF: "↓",     # arrowdown
    0xF0B0: "°",     # degree
    0xF0B1: "±",     # plusminus
    0xF0B2: "″",     # second
    0xF0B3: "≥",     # greaterequal
    0xF0B4: "×",     # multiply
    0xF0B5: "∝",     # proportional
    0xF0B6: "∂",     # partialdiff
    0xF0B7: "•",     # bullet
    0xF0B8: "÷",     # divide
    0xF0B9: "≠",     # notequal
    0xF0BA: "≡",     # equivalence
    0xF0BB: "≈",     # approxequal
    0xF0BC: "…",     # ellipsis
    0xF0BD: "|",     # arrowvertex
    0xF0BE: "—",     # arrowhorizex
    0xF0BF: "↵",     # carriagereturn
    0xF0C0: "ℵ",     # aleph
    0xF0C1: "ℑ",     # Ifraktur
    0xF0C2: "ℜ",     # Rfraktur
    0xF0C3: "℘",     # weierstrass
    0xF0C4: "⊗",     # circlemultiply
    0xF0C5: "⊕",     # circleplus
    0xF0C6: "∅",     # emptyset
    0xF0C7: "∩",     # intersection
    0xF0C8: "∪",     # union
    0xF0C9: "⊃",     # propersuperset
    0xF0CA: "⊇",     # reflexsuperset
    0xF0CB: "⊄",     # notsubset
    0xF0CC: "⊂",     # propersubset
    0xF0CD: "⊆",     # reflexsubset
    0xF0CE: "∈",     # element
    0xF0CF: "∉",     # notelement
    0xF0D0: "∠",     # angle
    0xF0D1: "∇",     # gradient (nabla)
    0xF0D2: "®",     # registerserif
    0xF0D3: "©",     # copyrightserif
    0xF0D4: "™",     # trademarkserif
    0xF0D5: "∏",     # product
    0xF0D6: "√",     # radical
    0xF0D7: "⋅",     # dotmath
    0xF0D8: "¬",     # logicalnot
    0xF0D9: "∧",     # logicaland
    0xF0DA: "∨",     # logicalor
    0xF0DB: "⇔",     # arrowdblboth
    0xF0DC: "⇐",     # arrowdblleft
    0xF0DD: "⇑",     # arrowdblup
    0xF0DE: "⇒",     # arrowdblright
    0xF0DF: "⇓",     # arrowdbldown
    0xF0E0: "◊",     # lozenge
    0xF0E1: "⟨",     # angleleft
    0xF0E2: "®",     # registersans
    0xF0E3: "©",     # copyrightsans
    0xF0E4: "™",     # trademarksans
    0xF0E5: "∑",     # summation
    # Bracket extension glyphs (paren/bracket/brace top/bottom/middle/extender)
    # These are vertical-stretch fragments — we replace with the base shape
    # since we're collapsing layout to a single line of text anyway.
    0xF0E6: "(",     # parenlefttp
    0xF0E7: "⎜",     # parenleftex
    0xF0E8: "(",     # parenleftbt
    0xF0E9: "[",     # bracketlefttp
    0xF0EA: "⎢",     # bracketleftex
    0xF0EB: "[",     # bracketleftbt
    0xF0EC: "{",     # bracelefttp
    0xF0ED: "{",     # braceleftmid
    0xF0EE: "{",     # braceleftbt
    0xF0EF: "⎪",     # braceex
    0xF0F1: "⟩",     # angleright
    0xF0F2: "∫",     # integral
    0xF0F3: "⌠",     # integraltp
    0xF0F4: "⎮",     # integralex
    0xF0F5: "⌡",     # integralbt
    0xF0F6: ")",     # parenrighttp
    0xF0F7: "⎟",     # parenrightex
    0xF0F8: ")",     # parenrightbt
    0xF0F9: "]",     # bracketrighttp
    0xF0FA: "⎥",     # bracketrightex
    0xF0FB: "]",     # bracketrightbt
    0xF0FC: "}",     # bracerighttp
    0xF0FD: "}",     # bracerightmid
    0xF0FE: "}",     # bracerightbt
}

# MT Extra and Microsoft Equation Editor glyphs.
# These are non-standard; mappings derived from manual inspection of papers
# we've processed. Add to this table as new glyphs surface.
MT_EXTRA_MAP: dict[int, str] = {
    # Math accents / overlines (often used for vector notation, mean bars)
    0xF03F: "̂",      # combining circumflex (hat)
    0xF07E: "̃",      # combining tilde
    0xF0AF: "̄",      # combining macron (mean-bar)
    # Differential operator
    0xF064: "𝑑",     # italic d (sometimes appears in dx/dt)
}


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical Alphanumeric Symbols (NEW in v3 patch)
# Block U+1D400–U+1D7FF. Generated programmatically from the block layout
# in https://unicode.org/charts/PDF/U1D400.pdf — each sub-block is a
# consecutive run of 26 letters (A-Z then a-z) or 10 digits (0-9), with
# certain "holes" reserved for already-existing characters in the BMP
# (e.g. italic h is U+210E, not U+1D455).
#
# We fold every style (italic, bold, script, fraktur, double-struck,
# sans-serif, monospace, etc.) back to the base ASCII letter / digit so
# downstream text matching, search, and embedding pipelines see plain ASCII.
#
# The Greek math sub-block (U+1D6A8–U+1D7CB) is mapped to standard Greek
# lowercase / uppercase Unicode (Α–Ω, α–ω) rather than ASCII.
# ─────────────────────────────────────────────────────────────────────────────

def _build_math_alphanum_map() -> dict[int, str]:
    """
    Construct the U+1D400–U+1D7FF map.

    Each "alphabet style" is a 52-codepoint run: 26 uppercase then 26
    lowercase. Some runs have holes where the canonical Unicode codepoint
    sits elsewhere (e.g. Mathematical Italic h is U+210E, the Planck
    constant character).
    """
    out: dict[int, str] = {}

    # Latin alphabet styles. Each entry is (start_codepoint, holes_dict)
    # where `holes_dict` maps offset-from-start to the codepoint Unicode
    # has assigned instead. Holes don't change our target ASCII char.
    LATIN_STYLES = [
        # Bold A-Z, a-z
        (0x1D400, {}),
        # Italic — h at U+210E is the hole at offset 33 (lowercase h, 26 + 7)
        (0x1D434, {33: 0x210E}),
        # Bold italic
        (0x1D468, {}),
        # Script — B(1), E(4), F(5), H(7), I(8), L(11), M(12), R(17) are holes
        (0x1D49C, {1: 0x212C, 4: 0x2130, 5: 0x2131, 7: 0x210B, 8: 0x2110,
                   11: 0x2112, 12: 0x2133, 17: 0x211B,
                   # lowercase: e(30), g(32), o(40)
                   30: 0x212F, 32: 0x210A, 40: 0x2134}),
        # Bold script
        (0x1D4D0, {}),
        # Fraktur — C(2), H(7), I(8), R(17), Z(25) are holes
        (0x1D504, {2: 0x212D, 7: 0x210C, 8: 0x2111, 17: 0x211C, 25: 0x2128}),
        # Double-struck — C(2), H(7), N(13), P(15), Q(16), R(17), Z(25) are holes
        (0x1D538, {2: 0x2102, 7: 0x210D, 13: 0x2115, 15: 0x2119,
                   16: 0x211A, 17: 0x211D, 25: 0x2124}),
        # Bold fraktur
        (0x1D56C, {}),
        # Sans-serif
        (0x1D5A0, {}),
        # Sans-serif bold
        (0x1D5D4, {}),
        # Sans-serif italic
        (0x1D608, {}),
        # Sans-serif bold italic
        (0x1D63C, {}),
        # Monospace
        (0x1D670, {}),
    ]

    for start, holes in LATIN_STYLES:
        for i in range(52):
            # Skip holes — Unicode reserves the codepoint at start+i but
            # the actual character lives at holes[i]. We map BOTH to the
            # same ASCII letter, so we add the hole codepoint too.
            ascii_char = chr(ord("A") + i) if i < 26 else chr(ord("a") + i - 26)
            cp = start + i
            if i not in holes:
                out[cp] = ascii_char
            else:
                # The "real" codepoint for this slot is elsewhere; map it.
                out[holes[i]] = ascii_char
                # The reserved slot itself isn't a valid character, but some
                # buggy encoders emit it anyway — map to be safe.
                out[cp] = ascii_char

    # ─── Greek math symbols (U+1D6A8–U+1D7CB) ────────────────────────────────
    # Each Greek block is 50 codepoints: 25 uppercase (A-Ω, skipping final
    # sigma), then nabla, then 25 lowercase (α-ω), then partial differential
    # and a few extra variants. We map to standard Greek letters.
    GREEK_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΘΣΤΥΦΧΨΩ"  # 25 chars; capital theta variant included
    GREEK_LOWER = "αβγδεζηθικλμνξοπρςστυφχψω"  # 25 chars; final-sigma included
    # The actual Unicode layout for each style block is:
    #   0..24: uppercase A..Omega
    #   25:    nabla (∇)
    #   26..50: lowercase alpha..omega (with final sigma at 41)
    #   51:    partial differential (∂)
    #   52..56: variant glyphs (epsilon var, theta var, kappa var, phi var, rho var, pi var)
    # We use the simplified version: just fold each style back to its base
    # Greek character. Variants get the unstyled base char.
    GREEK_STYLE_STARTS = [
        0x1D6A8,  # Bold
        0x1D6E2,  # Italic
        0x1D71C,  # Bold italic
        0x1D756,  # Sans-serif bold
        0x1D790,  # Sans-serif bold italic
    ]
    # Variant slots within each Greek block (offset from start)
    GREEK_VARIANTS = {
        51: "∂",   # partial differential
        52: "ε",   # epsilon variant
        53: "θ",   # theta variant
        54: "κ",   # kappa variant
        55: "φ",   # phi variant
        56: "ρ",   # rho variant
        57: "π",   # pi variant
    }
    for start in GREEK_STYLE_STARTS:
        for i, ch in enumerate(GREEK_UPPER):
            out[start + i] = ch
        out[start + 25] = "∇"
        for i, ch in enumerate(GREEK_LOWER):
            out[start + 26 + i] = ch
        for offset, ch in GREEK_VARIANTS.items():
            out[start + offset] = ch

    # ─── Mathematical digits (U+1D7CE–U+1D7FF) ───────────────────────────────
    # Each digit block is 10 codepoints (0-9):
    DIGIT_STYLE_STARTS = [
        0x1D7CE,  # Bold
        0x1D7D8,  # Double-struck
        0x1D7E2,  # Sans-serif
        0x1D7EC,  # Sans-serif bold
        0x1D7F6,  # Monospace
    ]
    for start in DIGIT_STYLE_STARTS:
        for i in range(10):
            out[start + i] = chr(ord("0") + i)

    return out


MATH_ALPHANUM_MAP: dict[int, str] = _build_math_alphanum_map()


# Master combined map. Symbol font wins on conflicts (it's the standard).
# Math alphanumeric is added last; it occupies a non-overlapping codepoint
# range so there are no real conflicts.
PUA_REMAP: dict[int, str] = {
    **MT_EXTRA_MAP,
    **SYMBOL_FONT_MAP,
    **MATH_ALPHANUM_MAP,
}


# Pre-compiled regex for finding any PUA codepoint in the U+F000-U+F8FF range.
# Used by `find_unmapped_pua` for diagnostics.
PUA_PATTERN = re.compile(r"[\uF000-\uF8FF]")


def remap_pua_glyphs(text: str) -> str:
    """
    Replace PUA codepoints AND Mathematical Alphanumeric Symbols in `text`
    with their plain Unicode / ASCII equivalents.

    Codepoints not in PUA_REMAP are left unchanged so they remain visible
    in the output (you can grep for them to extend the table). The function
    is a fast str.translate under the hood.

    Returns the cleaned string. Input is never mutated.
    """
    if not text:
        return text
    return text.translate(PUA_REMAP)


def find_unmapped_pua(text: str) -> dict[str, int]:
    """
    Return a dict of {hex_codepoint: count} for any PUA glyphs in `text`
    that are not in PUA_REMAP. Used for diagnostics — run this on a sample
    of your corpus to find glyphs worth adding to the table.
    """
    if not text:
        return {}
    counts: dict[str, int] = {}
    for ch in PUA_PATTERN.findall(text):
        cp = ord(ch)
        if cp in PUA_REMAP:
            continue
        key = f"U+{cp:04X}"
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    # Self-test.
    samples = [
        # PUA / Symbol font (v2 behaviour, kept working)
        ("\uf078 \uf02b \uf079 \uf03d \uf07a", "ξ + ψ = ζ"),
        ("\uf044T \uf03d 1\uf02e5", "ΔT = 1.5"),
        ("1 k k k k x Ax Bu Ef \uf02b \uf03d \uf02b \uf02b",
         "1 k k k k x Ax Bu Ef + = + +"),
        ("\uf065 \uf02a \uf03d \uf028 T \uf02d T0\uf029",
         "ε ∗ = ( T − T0)"),
        ("", ""),
        # v3 patch: Mathematical Alphanumeric Symbols
        ("\U0001D453 = \U0001D70E", "f = σ"),           # italic f = italic sigma
        ("\U0001D44E \U0001D44F \U0001D450", "a b c"),  # italic a b c
        ("\U0001D400 \U0001D401", "A B"),                # bold A B
        ("\U0001D7D8 \U0001D7D9", "0 1"),                # double-struck digits
        ("\U0001D434 = \U0001D435 + \U0001D436", "A = B + C"),  # italic A=B+C
        # The actual HDARM eq fragment, simplified
        ("\U0001D453 = \U0001D70E (\U0001D465 \U0001D461)",
         "f = σ (x t)"),
    ]
    passed = 0
    for input_str, expected in samples:
        actual = remap_pua_glyphs(input_str)
        if actual == expected:
            print(f"  OK   {input_str!r} -> {actual!r}")
            passed += 1
        else:
            print(f"  FAIL {input_str!r}")
            print(f"       expected: {expected!r}")
            print(f"       actual:   {actual!r}")
    print(f"\n{passed}/{len(samples)} samples passed")

    # Diagnostic on an unknown glyph
    diag = "abc \ufeff def"  # ufeff is in BMP but outside PUA, should be ignored
    print(f"\nDiagnostic find_unmapped_pua on {diag!r}:")
    print(f"  {find_unmapped_pua(diag)}")