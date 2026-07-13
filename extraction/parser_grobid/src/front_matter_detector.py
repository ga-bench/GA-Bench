"""
front_matter_detector.py (v9)

Detects repository/license/preprint/front-matter pages as metadata only.
It does not delete pages or change the PDF.
"""
from __future__ import annotations
import re

_FRONT_PATTERNS = {
    "repository_or_cover_page": re.compile(r"\b(?:citation for published version|document license|download date|take down policy|institutional repository|university repository)\b", re.I),
    "license_page": re.compile(r"\b(?:general rights|end user agreement|taverne|creative commons|cc\s+by|copyright and moral rights)\b", re.I),
    "arxiv_or_preprint_page": re.compile(r"\b(?:arxiv:\d|arXiv|preprint submitted to|preprint submitted)\b", re.I),
    "graphical_abstract_page": re.compile(r"^\s*graphical abstract\b|\bGraphical Abstract\b", re.I),
    "highlights_page": re.compile(r"^\s*highlights\b|\bHighlights\b", re.I),
}

_ARTICLE_START_RE = re.compile(r"\b(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|1\.?\s+Introduction|Introduction)\b", re.I)
_TITLE_HINT_RE = re.compile(r"\b(?:article info|keywords|received|accepted|available online|journal homepage)\b", re.I)


def detect_front_matter(raw_pages: list[str]) -> dict:
    pages = raw_pages or []
    detected_pages = []
    reasons_by_page: dict[str, list[str]] = {}

    # Only inspect the first few pages; later occurrences of "Highlights" etc.
    # should not be treated as front matter.
    for idx, text in enumerate(pages[:5], start=1):
        t = text or ""
        reasons = [name for name, rx in _FRONT_PATTERNS.items() if rx.search(t)]
        if reasons:
            detected_pages.append(idx)
            reasons_by_page[str(idx)] = reasons

    possible_article_start = 1
    for idx, text in enumerate(pages[:8], start=1):
        t = text or ""
        if _ARTICLE_START_RE.search(t) and (_TITLE_HINT_RE.search(t) or idx > max(detected_pages or [0])):
            possible_article_start = idx
            break

    return {
        "front_matter_detected": bool(detected_pages),
        "front_matter_pages": detected_pages,
        "front_matter_reasons_by_page": reasons_by_page,
        "possible_article_start_page": possible_article_start,
    }
