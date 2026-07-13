"""
equation_quality.py (v11)

Strict validation and cleanup helpers for production equation extraction.
The goal is to keep real display equations and reject table/prose fragments
that GROBID sometimes emits as <formula>.
"""

from __future__ import annotations
import re

_MATH_SYMBOL_RE = re.compile(r"[=+\-*/×÷∑∏√∫∂∇≤≥≈≠±∞→←↔⇌ρσμλβθαπΔ%]|\b(?:sin|cos|tan|log|ln|exp|min|max)\b", re.I)
_TABLE_WORD_RE = re.compile(
    r"\b(?:Description|Parameter|Parameters|Mean|Max|Min|SD|CV|Standard|Samples?|Rate|Grade|"
    r"Template|Primer|Amplicon|Sequence|Genes?|Expert|Index|weight|LEVEL|NUMBER|Sample codes|"
    r"Conditions|Groups|Chewing|Smoking|Alcohol|Age/Gender|Patient code)\b",
    re.I,
)
_PROSE_WORD_RE = re.compile(
    r"\b(?:Compared with|respectively|calculated according|shown in|described by|represents|"
    r"indicates|where|therefore|because|however|moreover|furthermore|supplementary|"
    r"Figure|Table|Appendix|References|Acknowledgements|Declaration|CRediT)\b",
    re.I,
)
_SENTENCE_VERB_RE = re.compile(r"\b(?:is|are|was|were|has|have|had|shows?|showed|represents?|indicates?|calculated|determined)\b", re.I)
_EQ_NUM_RE = re.compile(r"\((\d{1,3})\)")
_CHECKMARK_RE = re.compile(r"[✓✔☑]")
_TABLE_GRADE_SEQUENCE_RE = re.compile(r"(?:\b[A-G]\b\s+){4,}\d+(?:\.\d+)?\s*(?:±|\+/-)\s*\d+(?:\.\d+)?", re.I)
_TABLE_ROW_SEQUENCE_RE = re.compile(r"(?:\b(?:complex|sample|compound|control|mesotrione|parameter|index|grade)\s*[-_]?\w*\b.*?){2,}", re.I)


def normalize_equation_space(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,;:])", r"\1", text)
    text = re.sub(r"([=+×÷*/])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_printed_number(text: str) -> int | None:
    nums = _EQ_NUM_RE.findall(text or "")
    if nums:
        try:
            return int(nums[-1])
        except Exception:
            return None
    m = re.search(r"(?:^|\s)(\d{1,3})\s*$", text or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def split_multiple_numbered_equations(raw: str) -> list[str]:
    """Split one GROBID formula containing multiple printed numbers.

    Example: '... (3) w_i = ... (4)' -> ['... (3)', 'w_i = ... (4)'].
    """
    text = normalize_equation_space(raw)
    matches = list(_EQ_NUM_RE.finditer(text))
    if len(matches) <= 1:
        return [text] if text else []
    parts = []
    start = 0
    for m in matches:
        end = m.end()
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end
    tail = text[start:].strip()
    if tail:
        # Tail without printed number is usually prose/noise; keep only if mathy.
        if _MATH_SYMBOL_RE.search(tail):
            parts.append(tail)
    return parts


def looks_like_table_or_prose_formula(text: str) -> tuple[bool, list[str]]:
    t = normalize_equation_space(text)
    reasons: list[str] = []
    if not t:
        return True, ["empty_formula_text"]
    if len(t) <= 3:
        return True, ["too_short"]

    words = re.findall(r"[A-Za-z]{3,}", t)
    math_symbols = len(_MATH_SYMBOL_RE.findall(t))
    nums = len(re.findall(r"\d", t))

    if _CHECKMARK_RE.search(t):
        reasons.append("checkmark_table_cells")
    if _TABLE_GRADE_SEQUENCE_RE.search(t):
        reasons.append("grade_letter_table_sequence")
    if _TABLE_ROW_SEQUENCE_RE.search(t) and math_symbols <= 2:
        reasons.append("repeated_table_row_sequence")
    if _TABLE_WORD_RE.search(t):
        reasons.append("table_like_words")
    if _PROSE_WORD_RE.search(t):
        reasons.append("prose_like_words")
    if len(words) >= 8 and math_symbols == 0:
        reasons.append("long_prose_without_math_symbols")
    if len(words) >= 10 and _SENTENCE_VERB_RE.search(t) and math_symbols <= 1:
        reasons.append("sentence_like_formula")
    if len(t) > 220 and len(words) > 18 and math_symbols <= 2:
        reasons.append("very_long_prose_like_formula")
    if re.search(r"\b(?:et al\.?|journal homepage|ScienceDirect|Published by|creativecommons)\b", t, re.I):
        reasons.append("header_footer_or_publisher_text")

    # Reject table-like records even if they contain simple symbols such as ri or X_i.
    hard_table_reasons = {
        "checkmark_table_cells",
        "grade_letter_table_sequence",
        "repeated_table_row_sequence",
    }
    if any(r in reasons for r in hard_table_reasons):
        return True, reasons
    # Table-like text with many numeric cells and no core equation operator is almost always a table row.
    if "table_like_words" in reasons and (math_symbols <= 2 or len(words) >= 5):
        return True, reasons
    # Long numeric result rows such as SAR tables can contain ± and decimals; reject unless they have a real equation sign/function.
    if len(t) > 140 and nums >= 18 and not re.search(r"=|∑|∏|√|∫|∂|∇|\b(?:sin|cos|tan|log|ln|exp)\b", t, re.I):
        reasons.append("long_numeric_table_row")
        return True, reasons
    if any(r in reasons for r in ("long_prose_without_math_symbols", "sentence_like_formula", "very_long_prose_like_formula", "header_footer_or_publisher_text")):
        return True, reasons
    return False, reasons


def equation_confidence(raw: str, clean: str, *, repaired: bool = False) -> tuple[str, list[str]]:
    reasons: list[str] = []
    rejected, reject_reasons = looks_like_table_or_prose_formula(clean or raw)
    if rejected:
        return "rejected", reject_reasons

    t = normalize_equation_space(clean or raw)
    if t.count("(") != t.count(")"):
        reasons.append("unbalanced_parentheses")
    if t.count("[") != t.count("]"):
        reasons.append("unbalanced_brackets")
    if re.search(r"ffiffi|ð|Þ|Á|Ã|�", raw or t):
        reasons.append("pdf_math_glyph_noise")
    if len(re.findall(r"[A-Za-z0-9]", t)) < 3:
        reasons.append("few_alphanumeric_symbols")

    # If a formula is mathy but its formatting is likely lossy, keep but review.
    if re.search(r"\b(?:rho|sigma|alpha|beta)\b", t, re.I):
        reasons.append("possible_lost_greek_symbol")
    if re.search(r"\b\w+\s+\w+\s+\d+\s*/\s*\d+\b", t) and "/" in t:
        reasons.append("possible_fraction_format_loss")

    if reasons:
        return "needs_review", reasons
    if repaired:
        return "repaired", ["merged_split_formula_or_repaired_numbering"]
    return "clean", []


def latex_like_from_plain(text: str) -> str:
    """A conservative latex-like representation.

    This does not pretend to recover perfect LaTeX, but it preserves obvious
    operators and converts common unicode symbols.
    """
    t = normalize_equation_space(text)
    replacements = {
        "×": r"\\times", "÷": r"\\div", "∑": r"\\sum", "∏": r"\\prod",
        "√": r"\\sqrt{}", "∂": r"\\partial", "∇": r"\\nabla", "ρ": r"\\rho",
        "σ": r"\\sigma", "μ": r"\\mu", "λ": r"\\lambda", "β": r"\\beta",
        "θ": r"\\theta", "α": r"\\alpha", "π": r"\\pi", "Δ": r"\\Delta",
        "≤": r"\\le", "≥": r"\\ge", "≈": r"\\approx", "≠": r"\\ne", "±": r"\\pm",
        "→": r"\\to", "←": r"\\leftarrow", "↔": r"\\leftrightarrow", "⇌": r"\\rightleftharpoons",
    }
    for a, b in replacements.items():
        t = t.replace(a, b)
    # Very conservative subscript conversion: X hkl -> X_{hkl}; P tot -> P_{tot}.
    t = re.sub(r"\b([A-Za-z])\s+(hkl|tot|app|skel|max|min|ij|ji|i|j|n|m)\b", r"\1_{\2}", t)
    return t
