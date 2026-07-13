"""
table_quality.py (v13)

Table cleanup and quality scoring helpers.  This module is intentionally
conservative: it preserves raw table text when possible, but flags weak table
structure instead of silently passing it as publication-ready structured data.
"""

from __future__ import annotations
import re

_TABLE_WORDS = re.compile(r"\b(?:Table|Fig(?:ure)?|References|Acknowledg|Declaration|CRediT|Supplementary)\b", re.I)
_FOOTER_RE = re.compile(
    r"\b(?:ScienceDirect|Elsevier|journal homepage|Contents lists available|doi\.org|creativecommons|"
    r"Published by|Accepted|Received|Available online)\b",
    re.I,
)
_HEADER_FOOTER_LINE_RE = re.compile(
    r"^\s*(?:\d+\s*)?(?:[A-Z]\.?\s*)?[A-Z][A-Za-z .-]+\s+et\s+al\.?\s*$|"
    r"^\s*[A-Z][A-Za-z &-]+\s+\d+\s*\([^)]*\)\s*\d+[-–]\d+\s*$",
    re.I,
)
_TABLE_CAPTION_RE = re.compile(r"^\s*Table\s+((?:S\s*)?\d+[A-Za-z]?|[A-Z]\.\s*\d+|[IVXLC]+)\b\.?\s*(.*)$", re.I)


def _norm(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def is_table_mention_not_caption(line: str) -> bool:
    """Return True for prose references like 'Table 1 shows...'"""
    s = _norm(line)
    m = _TABLE_CAPTION_RE.match(s)
    if not m:
        return False
    rest = (m.group(2) or "").strip()
    if re.match(r"^(?:shows?|presents?|lists?|summari[sz]es?|reports?|provides?|indicates?|illustrates?|demonstrates?|contains?|is|are|was|were|has|have)\b", rest, re.I):
        return True
    if re.match(r"^(?:and|or|for|in|from|with|without|of|to)\b", rest, re.I):
        return True
    return False


def clean_table_caption(caption: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    c = _norm(caption)
    if not c:
        return "", ["empty_caption"]
    lines = [ln.strip() for ln in c.splitlines() if ln.strip()]
    kept = []
    for i, ln in enumerate(lines):
        if i > 0 and re.match(r"^(?:Fig(?:ure)?\.?\s+\d+|Table\s+\d+|\d+(?:\.\d+)*\.?\s+[A-Z])\b", ln, re.I):
            reasons.append("caption_cut_at_next_block")
            break
        if i > 0 and re.match(r"^(?:The|This|These|Those|There|In|For|As|We|Our)\b", ln) and len(" ".join(kept)) > 30:
            # likely body prose after caption
            reasons.append("caption_body_leakage_removed")
            break
        if re.search(r"\b(?:Eqs?\.\s*\(|Calculation indexes|Determination of the total)\b", ln, re.I):
            reasons.append("caption_equation_or_body_leakage_removed")
            break
        kept.append(ln)
        if len(" ".join(kept)) > 500:
            reasons.append("caption_truncated_long")
            break
    c = _norm("\n".join(kept))
    return c, reasons


def clean_table_body(text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    lines = [ln.strip() for ln in _norm(text).splitlines() if ln.strip()]
    out: list[str] = []
    data_seen = 0
    for ln in lines:
        if _FOOTER_RE.search(ln) or _HEADER_FOOTER_LINE_RE.match(ln):
            reasons.append("removed_header_footer_line")
            continue
        if re.match(r"^(?:Fig(?:ure)?\.?\s+\d+|References|Declaration of competing interest|CRediT authorship|Acknowledg)", ln, re.I):
            reasons.append("cut_at_non_table_block")
            break
        # Stop after table data if paragraph prose begins.
        if data_seen >= 2 and _looks_like_prose(ln):
            reasons.append("removed_trailing_prose")
            break
        out.append(ln)
        if _is_data_like(ln):
            data_seen += 1
    return _norm("\n".join(out)), sorted(set(reasons))


def _digit_ratio(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    return sum(1 for c in s if c.isdigit() or c in "%.,+-−–—/()<>±=×:") / max(1, len(s))


def _is_data_like(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _digit_ratio(s) >= 0.10:
        return True
    toks = s.split()
    if len(toks) <= 10 and len(toks) >= 2 and not s.endswith("."):
        return True
    if re.search(r"\b(?:yes|no|linear fit|control|treated|untreated|mean|max|min|sd|cv|rate|grade)\b", s, re.I):
        return True
    return False


def _looks_like_prose(line: str) -> bool:
    s = line.strip()
    if len(s) < 45 or _digit_ratio(s) >= 0.18:
        return False
    words = re.findall(r"[A-Za-z]{3,}", s)
    return len(words) >= 8 and bool(re.search(r"\b(?:is|are|was|were|has|have|had|show|shows|showed|indicates?|suggests?|observed|found)\b", s, re.I))


def _split_rows_heuristic(text: str) -> list[list[str]]:
    lines = [ln.strip() for ln in _norm(text).splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    # Simple reconstruction: keep one cell per line when the source is line-based.
    # This is deliberately modest; raw text remains the authoritative table text.
    return [[ln] for ln in lines]


def table_quality(body: str, caption: str = "") -> tuple[float, bool, list[str], list[list[str]]]:
    body = _norm(body)
    cap = _norm(caption)
    reasons: list[str] = []
    lines = [ln for ln in body.splitlines() if ln.strip()]
    data_lines = [ln for ln in lines if _is_data_like(ln)]
    prose_lines = [ln for ln in lines if _looks_like_prose(ln)]

    score = 0.85
    if not cap:
        score -= 0.10
        reasons.append("missing_caption")
    if len(lines) == 0:
        score = 0.0
        reasons.append("empty_table_body")
    elif len(lines) < 3:
        score -= 0.35
        reasons.append("too_few_table_lines")
    elif len(lines) <= 6 and len(body) < 140:
        score -= 0.30
        reasons.append("very_short_recovered_table")
    if len(data_lines) < 2 and len(lines) > 0:
        score -= 0.25
        reasons.append("few_data_like_lines")
    if prose_lines and len(prose_lines) >= max(2, len(data_lines)):
        score -= 0.25
        reasons.append("possible_prose_leakage")
    if _FOOTER_RE.search(body):
        score -= 0.20
        reasons.append("header_footer_leakage")
    if re.search(r"\b(?:Equation|Eqs?\.\s*\(|Calculation indexes)\b", body, re.I):
        score -= 0.20
        reasons.append("equation_or_paragraph_leakage")
    if re.search(r"\bTable\s+\d+\b", body, re.I):
        score -= 0.10
        reasons.append("nested_table_caption_in_body")

    score = max(0.0, min(0.95, score))
    rows = _split_rows_heuristic(body) if body else []
    needs_review = score < 0.70 or bool(set(reasons) & {"empty_table_body", "too_few_table_lines", "possible_prose_leakage", "equation_or_paragraph_leakage"})
    return score, needs_review, sorted(set(reasons)), rows
