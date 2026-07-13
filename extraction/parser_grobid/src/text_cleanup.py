"""
Lightweight text cleanup helpers for parser v8.

Designed for GROBID/PyMuPDF scholarly-text artifacts. The goal is not to
rewrite scientific prose; only remove obvious PDF extraction noise before JSON
writing and LLM use.
"""

from __future__ import annotations
import re
import unicodedata

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NL_RE = re.compile(r"\n{3,}")

# Broken line-/column-hyphenation: "com- plex" -> "complex".
# Keep it conservative to avoid changing real hyphenated terms.
_DEHYPHEN_RE = re.compile(r"\b([A-Za-z]{3,})-\s+([a-z]{2,})\b")

# Common Elsevier/PDF math-symbol corruptions observed in parser samples.
# Keep replacements conservative: some glyphs such as "Â" can be a corrupted
# multiplication sign in numeric expressions, but also appear inside real names
# (e.g., Ângelo). Those are handled contextually below instead of globally.
_SYMBOL_REPLACEMENTS = {
    "1⁄4": "=",
    "¼": "=",
    "À": "-",
    "þ": "+",
    "∕": "/",
    "": "±",
    "": "°",
    "": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",
}


def normalize_math_symbols(text: str) -> str:
    """Normalize common PDF extraction artifacts without changing prose meaning."""
    if not text:
        return ""
    s = str(text)
    for bad, good in _SYMBOL_REPLACEMENTS.items():
        s = s.replace(bad, good)

    # Context-aware multiplication cleanup. Avoid turning names like
    # "Ângelo" into "×ngelo".
    s = re.sub(r"(?<=[0-9)\]])\s*Â\s*(?=[0-9A-Za-z(])", " × ", s)
    s = re.sub(r"(?<=[A-Za-z])\s+Â\s*(?=[0-9(])", " × ", s)

    # Common PDF corruption of ±. Keep it contextual so the acronym AE is not
    # always changed.
    s = re.sub(r"\bAE\s+SEM\b", "± SEM", s)
    s = re.sub(r"(?<=\d)\s+AE\s+(?=\d)", " ± ", s)
    return s


def clean_text_artifacts(text: str) -> str:
    """Clean common PDF extraction artifacts while preserving paragraph breaks."""
    if not text:
        return ""
    s = normalize_math_symbols(str(text))
    s = unicodedata.normalize("NFKC", s)
    s = normalize_math_symbols(s)
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\ufffd", "")  # replacement char
    s = _CONTROL_RE.sub(" ", s)
    s = _DEHYPHEN_RE.sub(r"\1\2", s)

    # Normalize punctuation spacing without removing math spacing completely.
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([([{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]}])", r"\1", s)
    s = re.sub(r"\bFig\s*\.\s*", "Fig. ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bEq\s*\.\s*", "Eq. ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTab\s*\.\s*", "Table ", s, flags=re.IGNORECASE)

    # Keep paragraph breaks but trim each line.
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in s.splitlines()]
    s = "\n".join(lines)
    s = _MULTI_NL_RE.sub("\n\n", s)
    return s.strip()


def clean_equation_text(text: str) -> str:
    """Clean a formula string for inline [Equation N: ...] blocks."""
    if not text:
        return ""
    s = clean_text_artifacts(text)
    # Basic math-symbol/spacing cleanup.
    s = s.replace("•", " ")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("—", "-")
    s = s.replace("Ã", "*")
    s = s.replace("ð", "(").replace("Þ", ")")
    s = s.replace("×", " × ")
    s = re.sub(r"\s*([=+\-*/^])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_FRAGMENT_EQ_RE = re.compile(
    r"\[Equation\s+(?P<id>\d+)(?:,\s*low-confidence)?:\s*(?P<body>.*?)\]",
    re.DOTALL,
)


def _is_equation_fragment(body: str) -> bool:
    """True for broken equation-only fragments such as ')' or '(4)'."""
    b = (body or "").strip()
    if not b:
        return True
    if re.fullmatch(r"[()\[\]{}\s,.;:]+", b):
        return True
    if re.fullmatch(r"\(?\s*\d+[A-Za-z]?\s*\)?", b):
        return True
    if len(b) <= 3 and not re.search(r"[A-Za-zα-ωΑ-Ωψλσ∂ημνκΩ]", b):
        return True
    return False


def merge_broken_equation_blocks(text: str) -> str:
    """Remove/merge standalone broken equation fragments inside section text.

    This handles GROBID cases where a displayed equation is split as:
    [Equation 4: ... (4]
    [Equation 5: )]
    The second block is noise, so it is removed before JSON writing.
    """
    if not text:
        return ""
    out = []
    last = 0
    kept_any = False
    for m in _FRAGMENT_EQ_RE.finditer(text):
        body = m.group("body")
        out.append(text[last:m.start()])
        if _is_equation_fragment(body) and kept_any:
            # Drop standalone fragment; surrounding prose remains intact.
            out.append(" ")
        else:
            out.append(m.group(0))
            kept_any = True
        last = m.end()
    out.append(text[last:])
    return clean_text_artifacts("".join(out))


def equation_block(eq_id: int | str, formula_or_text: str, *, low_confidence: bool = False) -> str:
    """Return a readable equation block for use inside full text."""
    formula = clean_equation_text(formula_or_text)
    if not formula:
        formula = "formula not recovered"
        low_confidence = True
    prefix = f"Equation {eq_id}, low-confidence" if low_confidence else f"Equation {eq_id}"
    return f"[{prefix}: {formula}]"
