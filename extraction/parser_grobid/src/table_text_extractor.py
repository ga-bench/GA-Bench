"""
table_text_extractor.py  (v8 — tables.json + placeholders)

Tables come from the PyMuPDF raw page text (the only place Elsevier tables that
ARE text survive as a contiguous, caption-anchored block). Tables that are
embedded as IMAGES yield only a caption in the raw text; those are detected as
"caption-only" and routed to OCR (via an injected callback) so their content is
recovered as Markdown rather than silently lost.

v7.2 changes vs v7.1:
    1. Capture stops at figure captions ("Fig. N"/"Figure N"), at bare page-
       number lines, and at running header/footer lines, in addition to the
       previous stops (next table caption, numbered section heading, prose).
       Trailing footer/figure lines are also trimmed after capture.
    2. Caption-only detection: a block whose body (after caption + legend) has
       no data-like rows is treated as an image table -> OCR.
    3. inject_tables_into_sections() accepts optional `pdf_path`, `ocr_fn`, and
       `bbox_resolver` to OCR image tables. Faithfulness:
         - text table         -> verbatim PyMuPDF text (table_unrecovered=0)
         - image table + OCR   -> OCR'd markdown        (table_unrecovered=0)
         - image table, no OCR -> caption only, FLAGGED (table_unrecovered=1)
"""

from __future__ import annotations
import re
from table_quality import clean_table_caption, clean_table_body, table_quality, is_table_mention_not_caption
from pdf_text_locator import expected_table_numbers, table_sort_value

TABLE_OPEN = "\u27e6TABLE\u27e7"
TABLE_CLOSE = "\u27e6/TABLE\u27e7"

_CAPTION_LINE = re.compile(
    r"^\s*Table\s+(?P<num>(?:[A-Z]\.?)?\d+[A-Za-z]?|[IVXLC]+)\b\.?(?P<rest>.*)$",
    re.IGNORECASE,
)

_SECTION_HEADING_LINE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z]")
_FIGURE_CAPTION_LINE = re.compile(r"^\s*Fig(?:ure)?\.?\s+\d", re.IGNORECASE)
_BARE_PAGENUM_LINE = re.compile(r"^\s*\d{1,4}\s*$")
_FOOTER_LINE = re.compile(
    r"^\s*(?:\d{1,4}\s+)?(?:"
    r"[A-Z]\.?\s+[A-Z][a-z]+.*\bet al\.?"
    r"|[A-Z](?:\.[A-Z])+.*\s*/\s*[^/]+\s+\d+.*"
    r"|[A-Z][A-Za-z .-]+\s+et\s+al\.?"
    r"|.*\b(?:ScienceDirect|Elsevier|Contents lists available|journal homepage|"
    r"Transportation Research Part|Computer Communications|Ad Hoc Networks|"
    r"Acta Biomaterialia)\b.*"
    r"|.*\b(?:https?://|doi\.org|creativecommons|Published by Elsevier)\b.*"
    r")\s*$"
)

_BACK_MATTER_OR_BODY_START = re.compile(
    r"^\s*(?:CRediT authorship|Declaration of competing interest|Acknowledg(?:e)?ments?|"
    r"Appendix|References|Funding|Data availability|Supplementary|Supporting information)\b",
    re.IGNORECASE,
)

_STRONG_PROSE_AFTER_TABLE = re.compile(
    r"^\s*(?:The |This |These |Those |There |Here |Hence |Thus |Therefore|Moreover|Furthermore|"
    r"In addition|In contrast|Overall|Finally|Next|Also|We |Our |A |An |For |As |From )\b",
    re.IGNORECASE,
)

_POST_TABLE_PROSE_START = re.compile(
    r"^\s*(?:Intruder\b|The mentioned tool\b|Next,?\b|Also,?\b|Then,?\b|Here we\b|"
    r"Table\s+\d+\s+presents\b|For these results\b|Please note\b|We included\b|"
    r"We numbered\b|The first execution\b|In other executions\b|In the rest of\b|"
    r"the adversarial model\b|"
    # Common scientific prose accidentally captured after an Elsevier table.
    r"[A-Z]?[A-Za-z0-9-]+(?:,\s*[A-Z]?[A-Za-z0-9-]+)+\s+(?:is|are|was|were|has|have|show|shows|showed|exhibited)\b)"
    , re.IGNORECASE,
)


def _normalise_ws(text: str) -> str:
    # Keep line breaks because tables are more useful as row-like text.
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _digit_ratio(line: str) -> float:
    s = line.strip()
    if not s:
        return 0.0
    relevant = sum(1 for c in s if c.isdigit() or c in "%\u00b1.,/()=<>|:;\u2013-")
    return relevant / len(s)


def _looks_like_prose(line: str) -> bool:
    s = line.strip()
    if len(s) < 60:
        return False
    if _digit_ratio(s) >= 0.12:
        return False
    words = re.findall(r"[A-Za-z]{4,}", s)
    return len(words) >= 8


_PROSE_START_RE = re.compile(
    r"^\s*(?:As shown|As can be seen|The results|These results|This|These|Therefore|Moreover|Furthermore|In addition|"
    r"A summary|We |Our |It is |municipal |explored for |and the variations)\b",
    re.IGNORECASE,
)


def _looks_like_sentence_fragment(line: str) -> bool:
    """Shorter prose fragment that often follows a table in raw PDF text."""
    st = line.strip()
    if len(st) < 28:
        return False
    if _BACK_MATTER_OR_BODY_START.match(st):
        return True
    if _POST_TABLE_PROSE_START.match(st):
        return True
    # Do not let identifiers like P180 alone make prose look like data.
    digit_ratio = _digit_ratio(st)
    if digit_ratio >= 0.22:
        return False
    words = re.findall(r"[A-Za-z]{3,}", st)
    has_verb = re.search(r"\b(?:is|are|was|were|has|have|had|shows?|showed|measured|observed|exhibited|indicated)\b", st, re.I)
    if len(words) >= 7 and has_verb and digit_ratio < 0.22:
        return True
    # Only treat generic prose as trailing prose if it starts like a paragraph.
    # Long descriptive cells inside tables are common and should not be cut.
    if len(words) >= 8 and (_PROSE_START_RE.match(st) or _STRONG_PROSE_AFTER_TABLE.match(st) or st[:1].islower()):
        return True
    return bool(_PROSE_START_RE.match(st))


def _is_stop_line(line: str) -> bool:
    # Do not stop on bare numeric lines inside tables: many table cells are
    # standalone numbers (e.g., 32, 13, 0.058). Bare page numbers are trimmed
    # only at the very end by _trim_trailing_noise().
    return bool(
        _FIGURE_CAPTION_LINE.match(line)
        or _SECTION_HEADING_LINE.match(line)
        or _FOOTER_LINE.match(line)
        or _BACK_MATTER_OR_BODY_START.match(line)
        or _POST_TABLE_PROSE_START.match(line)
    )


def _is_genuine_caption(line: str, next_line: str = ""):
    m = _CAPTION_LINE.match(line)
    if not m:
        return None
    rest = (m.group("rest") or "").strip()
    if rest and rest[0] in ")].,;:":
        return None
    if is_table_mention_not_caption(line):
        return None
    # Body prose such as "Table 6 presents ..." is a table mention, not a caption.
    if re.match(r"^(?:for more details|presents?|shows?|lists?|reports?|provides?|summari[sz]es?|indicates?|illustrates?|is|are|was|were|has|have)\b", rest, re.IGNORECASE):
        return None
    # Reject lower-case prose snippets such as "Table 1. wi and wi-1 are unknown...".
    if rest and rest[:1].islower():
        return None
    if not rest and _SECTION_HEADING_LINE.match(next_line or ""):
        return None
    return m.group("num")


def _is_data_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _digit_ratio(s) >= 0.10:
        return True
    toks = s.split()
    if len(toks) >= 2 and sum(1 for t in toks if len(t) <= 14) >= 2 and not _looks_like_prose(s):
        return True
    return False


def _trim_trailing_noise(lines: list[str]) -> list[str]:
    out = list(lines)
    # Only remove very strong trailing prose/stop lines. Do not remove generic
    # long text because many valid table cells are explanatory sentences.
    while out and (
        _is_stop_line(out[-1])
        or _BARE_PAGENUM_LINE.match(out[-1] or "")
        or _PROSE_START_RE.match(out[-1] or "")
        or _looks_like_sentence_fragment(out[-1] or "")
        or _POST_TABLE_PROSE_START.match(out[-1] or "")
        or re.search(r"continued on next page", out[-1] or "", re.IGNORECASE)
    ):
        out.pop()
    return out


def _truncate_at_trailing_prose(lines: list[str]) -> list[str]:
    """Cut text that clearly starts the paragraph after the table."""
    if len(lines) < 4:
        return lines
    # Keep the first line (Table N) and caption/header area safe.
    data_seen = 0
    for k in range(1, len(lines)):
        if _is_data_line(lines[k]):
            data_seen += 1
        if k >= 3 and data_seen >= 2:
            if (_PROSE_START_RE.match(lines[k] or "")
                or _looks_like_sentence_fragment(lines[k] or "")
                or _POST_TABLE_PROSE_START.match(lines[k] or "")
                or _FOOTER_LINE.match(lines[k] or "")
                or _BACK_MATTER_OR_BODY_START.match(lines[k] or "")):
                return lines[:k]
    return lines



def _line_is_table_continuation(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    if _is_stop_line(s):
        return False
    if _looks_like_sentence_fragment(s):
        return False
    # Table rows often contain short headers/cells, symbols, numbers, or mixed tokens.
    if _digit_ratio(s) >= 0.08:
        return True
    toks = s.split()
    if len(toks) <= 8 and not s.endswith('.'):
        return True
    return False


def _table_low_confidence(body: str) -> bool:
    lines = [ln.strip() for ln in (body or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return True
    data = [ln for ln in lines if _is_data_line(ln)]
    prose = [ln for ln in lines if _looks_like_sentence_fragment(ln)]
    if len(data) < 2:
        return True
    if len(prose) >= 2 and len(prose) >= len(data):
        return True
    return False

def _capture_table_blocks(page_text: str) -> list[dict]:
    lines = page_text.splitlines()
    blocks: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        num = _is_genuine_caption(lines[i], lines[i + 1] if i + 1 < n else "")
        if not num:
            i += 1
            continue
        start = i
        j = i + 1
        consumed = 0
        data_seen = 0
        while j < n:
            ln = lines[j]
            nxt = lines[j + 1] if j + 1 < n else ""
            if _is_genuine_caption(ln, nxt):
                break
            if _is_stop_line(ln):
                break
            if data_seen >= 2 and (_POST_TABLE_PROSE_START.match(ln or "") or _looks_like_sentence_fragment(ln)):
                break
            if data_seen >= 2 and _STRONG_PROSE_AFTER_TABLE.match(ln or "") and not _line_is_table_continuation(ln):
                break
            if _is_data_line(ln):
                data_seen += 1
            j += 1
            consumed += 1
        block_lines = _trim_trailing_noise(_truncate_at_trailing_prose(lines[start:j]))
        if not block_lines:
            i = j
            continue
        legend_end = 1
        while legend_end < len(block_lines) and not _is_data_line(block_lines[legend_end]):
            legend_end += 1
        body = block_lines[legend_end:]
        data_lines = [l for l in body if _is_data_line(l)]
        block_text = _normalise_ws("\n".join(block_lines))
        if block_text:
            blocks.append({
                "num": num,
                "text": block_text,
                "caption_only": len(data_lines) < 2,
            })
        i = j
    return blocks


def _is_continuation_caption(text: str) -> bool:
    return bool(re.search(r"\bcontinued\b", text or "", re.IGNORECASE))


def extract_tables_from_raw_pages(raw_pages: list[str]) -> list[dict]:
    out: list[dict] = []
    by_num: dict[str, dict] = {}
    seen = set()
    for pi, page in enumerate(raw_pages or []):
        for blk in _capture_table_blocks(page or ""):
            caption, body = _split_caption_and_body(blk["text"])
            num_key = str(blk["num"]).lower()
            normalized_head = _normalise_ws((body or blk["text"])[:120]).lower()
            exact_key = (num_key, normalized_head)
            if exact_key in seen:
                continue
            seen.add(exact_key)

            # Merge multi-page continuations such as "Table 2 (continued)" or
            # repeated table headers on the next page. Previously these were
            # either dropped as duplicates or kept as separate incomplete tables.
            if num_key in by_num:
                existing = by_num[num_key]
                append_body = body or blk["text"]
                if append_body and append_body not in existing.get("body", ""):
                    existing["body"] = (existing.get("body", "").rstrip() + "\n" + append_body.strip()).strip()
                    existing["text"] = (existing.get("caption", "").rstrip() + "\n" + existing["body"]).strip()
                    existing["caption_only"] = False if append_body else existing.get("caption_only", False)
                    existing["low_confidence"] = _table_low_confidence(existing.get("body", ""))
                continue

            score, needs_review, reasons, structured_rows = table_quality(body, caption)
            rec = {"num": blk["num"], "text": blk["text"],
                   "caption": caption, "body": body,
                   "page": pi + 1, "caption_only": blk["caption_only"],
                   "low_confidence": bool(needs_review or blk["caption_only"]),
                   "structure_confidence": score,
                   "quality_reasons": reasons,
                   "structured_rows": structured_rows}
            out.append(rec)
            by_num[num_key] = rec
    return out


def assign_tables_to_sections(tables: list[dict], sections: list[dict]) -> dict:
    assignment: dict[int, list[dict]] = {}
    if not sections:
        return assignment
    # Prefer page/nearest-preceding section heading over mention counts. Mention
    # counts often point to Results paragraphs that discuss an earlier Methods
    # table, which caused wrong IMRaD assignments in samples.
    page_index = []
    for idx, sec in enumerate(sections):
        pg = sec.get("page")
        if pg is not None:
            try:
                page_index.append((int(pg), idx))
            except Exception:
                pass
    page_index.sort()
    for tbl in tables:
        target = None
        try:
            tpage = int(tbl.get("page")) if tbl.get("page") is not None else None
        except Exception:
            tpage = None
        if tpage is not None and page_index:
            # nearest preceding section page, or first following section if none
            preceding = [idx for pg, idx in page_index if pg <= tpage]
            if preceding:
                target = preceding[-1]
            else:
                target = page_index[0][1]
        if target is None:
            # Fallback to first mention only when page information is missing.
            ref = re.compile(r"\bTable\s+" + re.escape(str(tbl.get("num") or "")) + r"\b", re.IGNORECASE)
            for idx, sec in enumerate(sections):
                if ref.search(sec.get("text", "") or ""):
                    target = idx
                    break
        if target is None:
            target = 0
        assignment.setdefault(target, []).append(tbl)
    return assignment


def _looks_like_caption_continuation(line: str) -> bool:
    st = (line or "").strip()
    if not st:
        return False
    if _is_stop_line(st):
        return False
    # Captions are often sentence-like and continue after "Table N".
    if st.endswith((".", ";", ":")):
        return True
    words = re.findall(r"[A-Za-z]{3,}", st)
    if len(words) >= 5 and _digit_ratio(st) < 0.12:
        return True
    if st[:1].islower():
        return True
    return False


def _split_caption_and_body(block_text: str) -> tuple[str, str]:
    """Return cleaned caption/legend and cleaned table body."""
    lines = [ln.strip() for ln in (block_text or "").splitlines() if ln.strip()]
    if not lines:
        return "", ""
    caption_lines = [lines[0]]
    i = 1
    # Caption often continues for one or more sentence-like lines after "Table N".
    while i < len(lines) and _looks_like_caption_continuation(lines[i]):
        if re.search(r"\b(?:Eqs?\.\s*\(|Calculation indexes|Determination of|For more details)\b", lines[i], re.I):
            break
        caption_lines.append(lines[i])
        i += 1
    caption, _cap_reasons = clean_table_caption(_normalise_ws("\n".join(caption_lines)))
    body, _body_reasons = clean_table_body("\n".join(lines[i:]).strip())
    return caption, body


def _first_caption_line(block_text: str) -> str:
    cap, _body = _split_caption_and_body(block_text)
    return cap or block_text.split("\n", 1)[0].strip()


def table_placeholder(num: str | None, fallback_id: str | None = None) -> str:
    label = f"Table {num}" if num else (fallback_id or "TABLE")
    return f"[{label} here]"


def _table_label(num: str | None, fallback_id: str | None = None) -> str:
    return f"Table {num}" if num else (fallback_id or "Table")


def _has_same_table(existing: list[dict], num: str | None, content: str) -> bool:
    norm_content = _normalise_ws(content or "")[:120].lower()
    for item in existing:
        if num and str(item.get("num") or "").lower() == str(num).lower():
            return True
        old = _normalise_ws(item.get("raw_text") or item.get("markdown") or "")[:120].lower()
        if norm_content and old and norm_content == old:
            return True
    return False



def _ocr_result_to_fields(ocr_result):
    """Normalize OCR result from either legacy str or v9 structured dict."""
    if isinstance(ocr_result, dict):
        text = (ocr_result.get("markdown") or ocr_result.get("text") or "").strip()
        return {
            "text": text,
            "markdown": (ocr_result.get("markdown") or text).strip(),
            "ocr_confidence": ocr_result.get("ocr_confidence"),
            "structure_confidence": ocr_result.get("structure_confidence"),
            "quality_reasons": ocr_result.get("quality_reasons") or [],
        }
    text = (ocr_result or "").strip() if isinstance(ocr_result, str) else ""
    return {
        "text": text,
        "markdown": text,
        "ocr_confidence": None,
        "structure_confidence": None,
        "quality_reasons": [],
    }


def _structure_confidence_from_text(text: str, caption: str = "") -> float:
    score, _needs_review, _reasons, _structured = table_quality(text, caption)
    return score



def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    norm = [r + [""] * (max_cols - len(r)) for r in rows]
    header = norm[0]
    sep = ["---"] * max_cols
    body = norm[1:]
    def fmt(row):
        return "| " + " | ".join(str(c).replace("|", r"\|") for c in row) + " |"
    return "\n".join([fmt(header), fmt(sep)] + [fmt(r) for r in body])



def _section_index_for_table_num(sections: list[dict], num: str) -> int:
    pat = re.compile(r"\bTable\s+" + re.escape(str(num)) + r"\b", re.I)
    for idx, sec in enumerate(sections or []):
        if pat.search(sec.get("text", "") or "") or pat.search(sec.get("text_no_tables", "") or ""):
            return idx
    return 0


def _add_unrecovered_expected_tables(sections: list[dict], raw_pages: list[str], existing_tables: list[dict]) -> int:
    """Add explicit unrecovered table records for expected-but-missing tables.

    This is a real recovery step for dataset integrity: papers should not report
    table_count=0 when the PDF/TEI clearly contains Table references. If OCR or
    bounding boxes cannot recover the body, the table is kept as an unrecovered
    record so the quality report and downstream filters are truthful.
    """
    expected = expected_table_numbers(raw_pages, sections)
    present = {str(t.get("num") or "").strip().upper() for t in existing_tables if t.get("num")}
    missing = sorted([n for n in expected if n and n.upper() not in present], key=table_sort_value)
    added = 0
    for num in missing:
        idx = _section_index_for_table_num(sections, num)
        if not sections:
            break
        sec = sections[idx]
        sec_tables = sec.setdefault("tables", [])
        label = f"Table {num}"
        placeholder = f"[{label} here]"
        # Avoid duplicates if this function is called more than once.
        if any(str(t.get("num") or "").upper() == str(num).upper() for t in sec_tables):
            continue
        sec_tables.append({
            "num": num,
            "label": label,
            "caption": label,
            "raw_text": "",
            "markdown": "",
            "source": "expected_table_reference_unrecovered",
            "recovered": False,
            "ocr_confidence": None,
            "structure_confidence": 0.0,
            "low_confidence": True,
            "needs_review": True,
            "quality_reasons": ["table_reference_detected_but_body_not_recovered"],
            "structured_rows": [],
            "structured_markdown": "",
            "page": None,
            "placeholder": placeholder,
        })
        base_text = sec.get("text_no_tables") or sec.get("text") or ""
        if placeholder not in base_text:
            sec["text"] = (base_text.rstrip() + "\n\n" + placeholder).strip() if base_text else placeholder
            sec["text_no_tables"] = sec["text"]
        added += 1
    for s in sections or []:
        s["table_count"] = len(s.get("tables", []) or [])
        s["table_unrecovered_count"] = sum(1 for t in (s.get("tables", []) or []) if not t.get("recovered", True))
    return added

def inject_tables_into_sections(
    sections: list[dict],
    raw_pages: list[str],
    *,
    pdf_path=None,
    ocr_fn=None,
    bbox_resolver=None,
) -> dict:
    """Attach tables to sections without injecting table content into text.

    v8 behavior:
      - full section text contains only a placeholder, e.g. [Table 2 here]
      - actual table content is stored under section["tables"] for tables.json
      - text_no_tables is kept aligned with text so downstream code sees one
        clean IMRaD text file, not separate with/without table variants.
    """
    stats = {"injected": 0, "text_tables": 0, "ocr_tables": 0, "unrecovered": 0}

    tables = extract_tables_from_raw_pages(raw_pages)
    if not tables:
        for s in sections:
            if not (s.get("text_no_tables") or "").strip():
                s["text_no_tables"] = s.get("text", "") or ""
            s.setdefault("tables", s.get("tables", []) or [])
        added_missing = _add_unrecovered_expected_tables(sections, raw_pages, [])
        stats["unrecovered"] += added_missing
        stats["expected_missing_tables"] = added_missing
        return stats

    assignment = assign_tables_to_sections(tables, sections)

    for idx, sec in enumerate(sections):
        base_text = sec.get("text_no_tables") or sec.get("text", "") or ""
        assigned = assignment.get(idx, [])
        sec_tables = sec.get("tables", []) or []
        text_parts = [base_text] if base_text else []

        for tbl in assigned:
            content = tbl.get("body") or tbl["text"]
            caption_text = tbl.get("caption") or _first_caption_line(tbl["text"])
            recovered = True
            source = "pymupdf_text"
            markdown = content or ""
            ocr_confidence = None
            structure_confidence = _structure_confidence_from_text(content, caption_text)
            quality_reasons = []

            needs_ocr = bool(tbl.get("caption_only") or tbl.get("low_confidence"))
            if needs_ocr:
                ocr_text = None
                if ocr_fn is not None and pdf_path is not None:
                    bbox = None
                    if bbox_resolver is not None:
                        bbox = bbox_resolver(tbl["num"], tbl.get("page"))
                    if bbox is not None:
                        page, xywh = bbox
                        ocr_text = ocr_fn(pdf_path, page, xywh, "table")
                ocr_fields = _ocr_result_to_fields(ocr_text)
                if ocr_fields["text"] and len(ocr_fields["text"].strip()) > len((content or "").strip()):
                    content = ocr_fields["text"]
                    markdown = ocr_fields["markdown"]
                    source = "ocr_low_confidence" if tbl.get("low_confidence") else "ocr"
                    stats["ocr_tables"] += 1
                    recovered = True
                    ocr_confidence = ocr_fields["ocr_confidence"]
                    structure_confidence = ocr_fields["structure_confidence"]
                    quality_reasons = ocr_fields["quality_reasons"]
                elif tbl.get("caption_only"):
                    markdown = content or ""
                    ocr_confidence = None
                    structure_confidence = 0.0
                    quality_reasons = ["caption_only_table_no_ocr_text"]
                    recovered = False
                    source = "caption_only"
                    stats["unrecovered"] += 1
                else:
                    markdown = content or ""
                    ocr_confidence = None
                    structure_confidence = _structure_confidence_from_text(content, caption_text)
                    quality_reasons = ["pymupdf_table_low_confidence"]
                    source = "pymupdf_text_low_confidence"
                    stats["text_tables"] += 1
            else:
                stats["text_tables"] += 1

            # Final cleanup/scoring of the table body regardless of source.
            cleaned_content, cleanup_reasons = clean_table_body(content)
            if cleaned_content:
                content = cleaned_content
                markdown = cleaned_content if not markdown or source.startswith("pymupdf") else markdown
            q_score, q_needs_review, q_reasons, structured_rows = table_quality(content, caption_text)
            if structure_confidence is None or q_score < structure_confidence:
                structure_confidence = q_score
            quality_reasons = sorted(set((quality_reasons or []) + cleanup_reasons + q_reasons + (tbl.get("quality_reasons") or [])))

            num = tbl.get("num")
            if _has_same_table(sec_tables, num, content):
                continue

            fallback_id = f"table{len(sec_tables) + 1:02d}"
            label = _table_label(num, fallback_id)
            placeholder = table_placeholder(num, fallback_id)
            if placeholder not in base_text and placeholder not in "\n".join(text_parts):
                text_parts.append(placeholder)

            sec_tables.append({
                "num": num,
                "label": label,
                "caption": caption_text,
                "raw_text": content if recovered else "",
                "markdown": markdown if recovered else "",
                "source": source,
                "recovered": recovered,
                "ocr_confidence": ocr_confidence,
                "structure_confidence": structure_confidence,
                "low_confidence": bool(tbl.get("low_confidence")) or source.endswith("low_confidence") or (structure_confidence is not None and structure_confidence < 0.65),
                "needs_review": (not recovered) or bool(tbl.get("low_confidence")) or bool(q_needs_review) or (structure_confidence is not None and structure_confidence < 0.70),
                "quality_reasons": quality_reasons,
                "structured_rows": structured_rows,
                "structured_markdown": _rows_to_markdown(structured_rows),
                "page": tbl.get("page"),
                "placeholder": placeholder,
            })
            stats["injected"] += 1

        sec["text"] = "\n\n".join(p for p in text_parts if p).strip()
        sec["text_no_tables"] = sec["text"]
        sec["tables"] = sec_tables
        sec["table_count"] = len(sec_tables)
        sec["table_unrecovered_count"] = sum(1 for t in sec_tables if not t.get("recovered", True))

    added_missing = _add_unrecovered_expected_tables(sections, raw_pages, tables)
    stats["unrecovered"] += added_missing
    stats["expected_missing_tables"] = added_missing
    return stats


_SENTINEL_SPAN = re.compile(
    re.escape(TABLE_OPEN) + r".*?" + re.escape(TABLE_CLOSE), re.DOTALL
)


def strip_table_sentinels(text: str, keep_content: bool) -> str:
    if not text:
        return text
    if keep_content:
        out = text.replace(TABLE_OPEN + "\n", "").replace("\n" + TABLE_CLOSE, "")
        out = out.replace(TABLE_OPEN, "").replace(TABLE_CLOSE, "")
    else:
        out = _SENTINEL_SPAN.sub("", text)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
# ---------------------------------------------------------------------------
# v13 final table recovery layer
# ---------------------------------------------------------------------------
# The v12 extractor was good when it saw a clean 'Table N' caption block, but
# some Elsevier PDFs expose tables only as body references or as caption-like
# orphan sections.  The wrapper below preserves the v12 behavior and then adds
# an expected-table audit/recovery pass so missing tables are explicit.

from pdf_text_locator import normalize_table_num, table_label as _expected_table_label, find_table_caption_page

_INJECT_TABLES_INTO_SECTIONS_V12 = inject_tables_into_sections

_TABLE_CAPTURE_STOP_RE = re.compile(
    r"^\s*(?:"
    r"Table\s+(?:S\s*)?\d+"
    r"|Fig(?:ure)?\.?\s+\d+"
    r"|\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z]"
    r"|References\b|Acknowledg(?:e)?ments?\b|Declaration\b|CRediT\b|Funding\b|Conclusion\b"
    r")",
    re.I,
)


def _table_caption_or_reference_regex(num: str) -> re.Pattern:
    n = normalize_table_num(num)
    if n.startswith("S") and n[1:].isdigit():
        num_pat = rf"S\s*{re.escape(n[1:])}"
    else:
        num_pat = re.escape(n)
    return re.compile(rf"\b(?:Supplementary\s+)?Table\s+{num_pat}\b\.?\s*(.*)$", re.I)


def _extract_caption_body_from_page_for_num(page_text: str, num: str) -> tuple[str, str]:
    """Best-effort raw page recovery for a single expected table.

    Returns (caption, body). Body can be empty if only a mention was found.
    """
    lines = [ln.strip() for ln in (page_text or "").splitlines() if ln.strip()]
    if not lines:
        return _expected_table_label(num), ""
    pat = _table_caption_or_reference_regex(num)
    starts = [i for i, ln in enumerate(lines) if pat.search(ln)]
    if not starts:
        return _expected_table_label(num), ""

    # Prefer caption-like line near top of a block, not prose 'Table N shows'.
    start = starts[0]
    for i in starts:
        ln = lines[i]
        after = pat.search(ln).group(1).strip() if pat.search(ln) else ""
        if not re.match(r"^(?:shows?|presents?|lists?|reports?|provides?|summari[sz]es?|indicates?|illustrates?|was|were|is|are|has|have)\b", after, re.I):
            start = i
            break

    block = [lines[start]]
    data_seen = 0
    for j in range(start + 1, min(len(lines), start + 80)):
        ln = lines[j]
        if j > start + 1 and _TABLE_CAPTURE_STOP_RE.match(ln):
            break
        if _FOOTER_LINE.match(ln) or _BARE_PAGENUM_LINE.match(ln):
            continue
        if data_seen >= 2 and (_looks_like_sentence_fragment(ln) or _STRONG_PROSE_AFTER_TABLE.match(ln)):
            break
        block.append(ln)
        if _is_data_line(ln):
            data_seen += 1
    block = _trim_trailing_noise(_truncate_at_trailing_prose(block))
    caption, body = _split_caption_and_body("\n".join(block))
    if not caption:
        caption = _expected_table_label(num)
    body, _ = clean_table_body(body)
    return caption, body


def _find_section_for_expected_table(sections: list[dict], num: str, page: int | None) -> int:
    n = normalize_table_num(num)
    pat = re.compile(rf"\bTable\s+{re.escape(n)}\b", re.I)
    for idx, sec in enumerate(sections or []):
        if pat.search(sec.get("text", "") or "") or pat.search(sec.get("text_no_tables", "") or ""):
            return idx
    if page is not None:
        candidates = []
        for idx, sec in enumerate(sections or []):
            try:
                sp = int(sec.get("page")) if sec.get("page") is not None else None
            except Exception:
                sp = None
            if sp is not None and sp <= page:
                candidates.append((sp, idx))
        if candidates:
            return sorted(candidates)[-1][1]
    return 0


def ensure_expected_tables_in_sections(sections: list[dict], raw_pages: list[str] | None) -> dict:
    """Ensure every expected table is either recovered or explicitly unrecovered.

    Returns stats used by output_writer and quality_report.
    """
    raw_pages = raw_pages or []
    expected = expected_table_numbers(raw_pages, sections)
    expected_sorted = sorted(expected, key=table_sort_value)

    # First, try to upgrade existing caption-only/unrecovered table records
    # using page-text recovery. This handles PDFs where v12 detected the caption
    # but did not capture the body.
    for sec in sections or []:
        for t in sec.get("tables", []) or []:
            n = normalize_table_num(t.get("num"))
            if not n:
                continue
            has_body = bool((t.get("raw_text") or t.get("markdown") or "").strip())
            if t.get("recovered", True) and has_body:
                continue
            page = t.get("page") or find_table_caption_page(raw_pages, n)
            caption = t.get("caption") or _expected_table_label(n)
            body = ""
            if page is not None and 1 <= int(page) <= len(raw_pages):
                cap2, body = _extract_caption_body_from_page_for_num(raw_pages[int(page) - 1], n)
                caption = cap2 or caption
            if body and len([ln for ln in body.splitlines() if ln.strip()]) >= 2:
                q_score, q_needs_review, q_reasons, structured_rows = table_quality(body, caption)
                t.update({
                    "caption": caption,
                    "raw_text": body,
                    "markdown": body,
                    "source": "pdf_page_text_recovery",
                    "recovered": True,
                    "structure_confidence": q_score,
                    "low_confidence": q_score < 0.65,
                    "needs_review": True if q_needs_review or q_score < 0.70 else bool(t.get("needs_review", False)),
                    "quality_reasons": sorted(set((t.get("quality_reasons") or []) + q_reasons + ["caption_only_upgraded_from_page_text"])),
                    "structured_rows": structured_rows,
                    "structured_markdown": _rows_to_markdown(structured_rows),
                    "page": int(page),
                })

    present: set[str] = set()
    for sec in sections or []:
        for t in sec.get("tables", []) or []:
            n = normalize_table_num(t.get("num"))
            if n:
                present.add(n)

    recovered_added = 0
    unrecovered_added = 0
    for num in expected_sorted:
        if num in present:
            continue
        page = find_table_caption_page(raw_pages, num)
        caption = _expected_table_label(num)
        body = ""
        if page is not None and 1 <= page <= len(raw_pages):
            caption, body = _extract_caption_body_from_page_for_num(raw_pages[page - 1], num)

        q_score, q_needs_review, q_reasons, structured_rows = table_quality(body, caption)
        recovered = bool(body and len([ln for ln in body.splitlines() if ln.strip()]) >= 2)
        source = "pdf_page_text_recovery" if recovered else "expected_table_reference_unrecovered"
        idx = _find_section_for_expected_table(sections, num, page)
        if not sections:
            continue
        sec = sections[idx]
        sec_tables = sec.setdefault("tables", [])
        label = _expected_table_label(num)
        placeholder = table_placeholder(num, None)
        if any(normalize_table_num(t.get("num")) == num for t in sec_tables):
            continue
        reasons = list(q_reasons or [])
        if recovered:
            reasons.extend(["expected_table_recovered_from_page_text", "raw_page_table_recovery"])
            recovered_added += 1
        else:
            reasons.append("expected_table_not_recovered")
            unrecovered_added += 1

        sec_tables.append({
            "num": num,
            "label": label,
            "caption": caption,
            "raw_text": body if recovered else "",
            "markdown": body if recovered else "",
            "source": source,
            "recovered": recovered,
            "ocr_confidence": None,
            "structure_confidence": q_score if recovered else 0.0,
            "low_confidence": (not recovered) or q_score < 0.65,
            "needs_review": True,
            "quality_reasons": sorted(set(reasons)),
            "structured_rows": structured_rows if recovered else [],
            "structured_markdown": _rows_to_markdown(structured_rows) if recovered else "",
            "page": page,
            "placeholder": placeholder,
        })
        base_text = sec.get("text_no_tables") or sec.get("text") or ""
        if placeholder not in base_text:
            sec["text"] = (base_text.rstrip() + "\n\n" + placeholder).strip() if base_text else placeholder
            sec["text_no_tables"] = sec["text"]
        present.add(num)

    for s in sections or []:
        s["table_count"] = len(s.get("tables", []) or [])
        s["table_unrecovered_count"] = sum(1 for t in (s.get("tables", []) or []) if not t.get("recovered", True))

    recovered_total = 0
    unrecovered_total = 0
    for sec in sections or []:
        for t in sec.get("tables", []) or []:
            if normalize_table_num(t.get("num")) in expected:
                if t.get("recovered", True):
                    recovered_total += 1
                else:
                    unrecovered_total += 1

    return {
        "expected_table_count": len(expected_sorted),
        "expected_table_labels": [_expected_table_label(n) for n in expected_sorted],
        "expected_table_numbers": expected_sorted,
        "expected_tables_recovered": recovered_total,
        "expected_tables_unrecovered": max(0, len(expected_sorted) - recovered_total),
        "expected_recovery_added": recovered_added,
        "expected_unrecovered_added": unrecovered_added,
    }


def inject_tables_into_sections(
    sections: list[dict],
    raw_pages: list[str],
    *,
    pdf_path=None,
    ocr_fn=None,
    bbox_resolver=None,
) -> dict:
    """v13 wrapper: run v12 extraction, then enforce expected-table accounting."""
    stats = _INJECT_TABLES_INTO_SECTIONS_V12(
        sections, raw_pages,
        pdf_path=pdf_path, ocr_fn=ocr_fn, bbox_resolver=bbox_resolver,
    )
    expected_stats = ensure_expected_tables_in_sections(sections, raw_pages)
    stats.update(expected_stats)
    stats["unrecovered"] = max(int(stats.get("unrecovered", 0) or 0), int(expected_stats.get("expected_tables_unrecovered", 0) or 0))
    return stats
