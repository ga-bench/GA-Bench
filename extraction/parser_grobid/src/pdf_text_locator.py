"""
pdf_text_locator.py (v13)

Utilities for locating snippets and expected tables in raw PyMuPDF page text
and parsed sections. v13 strengthens expected-table detection so a paper no
longer reports table_count=0 when the text clearly refers to tables.
"""

from __future__ import annotations
import re
from typing import Iterable

_TABLE_REF_RE = re.compile(
    r"\b(?:Supplementary\s+)?Table\s+(?P<num>(?:S\s*)?\d+[A-Za-z]?|[A-Z]\.\s*\d+|[IVXLC]+)\b",
    re.I,
)
_TABLE_RANGE_RE = re.compile(r"\bTables\s+(?P<a>\d+)\s*[-–—]\s*(?P<b>\d+)\b", re.I)
_TABLE_AND_RE = re.compile(r"\bTables\s+(?P<a>\d+)\s+(?:and|,)\s*(?P<b>\d+)\b", re.I)
_TABLE_CAPTION_LINE_RE = re.compile(
    r"^\s*(?:Supplementary\s+)?Table\s+(?P<num>(?:S\s*)?\d+[A-Za-z]?|[A-Z]\.\s*\d+|[IVXLC]+)\b\.?",
    re.I | re.M,
)


def normalize_for_search(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_table_num(num: str | None) -> str:
    if not num:
        return ""
    s = re.sub(r"\s+", "", str(num).upper().strip())
    s = s.replace(".", "") if re.match(r"^[A-Z]\.\d+", s) else s
    return s


def table_label(num: str | None) -> str:
    n = normalize_table_num(num)
    return f"Table {n}" if n else "Table"


def infer_page_for_text(text: str, raw_pages: list[str] | None, *, min_len: int = 16) -> int | None:
    if not text or not raw_pages:
        return None
    q = normalize_for_search(text)
    if len(q) < min_len:
        return None
    for i, page in enumerate(raw_pages, start=1):
        if q in normalize_for_search(page):
            return i
    tokens = q.split()
    windows = []
    for w in (12, 9, 7, 5):
        if len(tokens) < w:
            continue
        for start in range(0, len(tokens) - w + 1):
            win = " ".join(tokens[start:start+w])
            if len(win) >= min_len and re.search(r"\d|[=+\-*/×÷∑∏√∫∂∇≤≥≈≠±%]", win):
                windows.append(win)
    seen = set()
    windows = [x for x in windows if not (x in seen or seen.add(x))]
    for win in windows[:80]:
        for i, page in enumerate(raw_pages, start=1):
            if win in normalize_for_search(page):
                return i
    return None


def table_numbers_in_text(text: str) -> set[str]:
    nums: set[str] = set()
    text = text or ""
    for m in _TABLE_REF_RE.finditer(text):
        n = normalize_table_num(m.group("num"))
        if n:
            nums.add(n)
    for m in _TABLE_RANGE_RE.finditer(text):
        try:
            a, b = int(m.group("a")), int(m.group("b"))
            if 0 < a <= b <= 50:
                nums.update(str(i) for i in range(a, b + 1))
        except Exception:
            pass
    for m in _TABLE_AND_RE.finditer(text):
        nums.add(str(int(m.group("a"))))
        nums.add(str(int(m.group("b"))))
    return nums


def table_caption_numbers_in_text(text: str) -> set[str]:
    nums: set[str] = set()
    for m in _TABLE_CAPTION_LINE_RE.finditer(text or ""):
        n = normalize_table_num(m.group("num"))
        if n:
            nums.add(n)
    return nums


def expected_table_numbers(raw_pages: list[str] | None = None, sections: list[dict] | None = None, tei_text: str | None = None) -> set[str]:
    """Return expected table numbers from captions and references.

    Mentions are included because downstream selection needs truthfulness: it is
    safer to mark a table as unrecovered/needs_review than to falsely claim no
    tables exist.
    """
    nums: set[str] = set()
    for page in raw_pages or []:
        nums.update(table_numbers_in_text(page))
    for sec in sections or []:
        nums.update(table_numbers_in_text(sec.get("text", "") or ""))
        nums.update(table_numbers_in_text(sec.get("text_no_tables", "") or ""))
    if tei_text:
        nums.update(table_numbers_in_text(tei_text))
    return nums


def find_table_caption_page(raw_pages: list[str] | None, num: str) -> int | None:
    n = normalize_table_num(num)
    if not raw_pages or not n:
        return None
    # caption-like lines first
    cap_pat = re.compile(rf"^\s*(?:Supplementary\s+)?Table\s+{re.escape(n)}\b", re.I | re.M)
    ref_pat = re.compile(rf"\bTable\s+{re.escape(n)}\b", re.I)
    for i, page in enumerate(raw_pages, start=1):
        if cap_pat.search(page or ""):
            return i
    for i, page in enumerate(raw_pages, start=1):
        if ref_pat.search(page or ""):
            return i
    return None


def table_sort_value(num: str | None):
    if num is None:
        return (9999, "")
    s = normalize_table_num(num)
    m = re.match(r"(S?)([A-Z]?)(\d+)([A-Z]?)", s)
    if m:
        supp, prefix, n, suffix = m.groups()
        base = int(n)
        if supp:
            base += 2000
        if prefix:
            base += 1000 + (ord(prefix[0]) - 65) * 100
        if suffix:
            base += (ord(suffix[0]) - 64) / 100
        return (base, s)
    roman = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10}
    return (roman.get(s, 9999), s)
