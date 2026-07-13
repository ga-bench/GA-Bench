"""
Per-paper output writer (v8).

v8 changes:
  1. One clean IMRaD file: fulltext_imrad.json.
  2. Table content is written to tables.json only; body text keeps placeholders
     such as [Table 2 here]. No fulltext_imrad_with_tables.json is produced.
  3. Figure/table captions are removed from body text when detected and replaced
     with placeholders. Captions remain in figures.json/tables.json.
  4. Equation placeholders are resolved into readable in-text blocks:
     [Equation N: ...].
  5. Figure JSON keeps assignment evidence, possible sections, and mentions.
"""

from __future__ import annotations
import json
import shutil
import datetime
import re
from pathlib import Path

from table_text_extractor import inject_tables_into_sections, ensure_expected_tables_in_sections
from text_cleanup import clean_text_artifacts
from equation_extractor import build_equations_dict
from label_utils import parse_label, asset_id
from quality_report import build_quality_report
import ocr_engine


IMRAD_CATEGORIES = ["introduction", "methods", "results", "discussion"]


def _now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_raw_pages(pdf_path) -> tuple[list[str], str | None]:
    if not pdf_path:
        return [], None
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        try:
            pages = [pg.get_text() for pg in doc]
        finally:
            doc.close()
        return pages, None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def _caption_is_incomplete(caption: str) -> bool:
    c = clean_text_artifacts(caption or "").strip()
    if not c:
        return True
    tail = c[-80:].lower().strip(" .;:,)")
    return bool(
        len(c) < 40
        or tail.endswith(("see", "for details", "for details see", "shown in", "described in"))
        or re.search(r"(?:for details,? see|see)\s*$", c, re.IGNORECASE)
    )



_CAPTION_HARD_STOP_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z]"
    r"|Table\s+\d"
    r"|Fig(?:ure)?\.?\s+\d"
    r"|CRediT authorship"
    r"|Declaration of competing interest"
    r"|Acknowledg(?:e)?ments?"
    r"|Appendix\b"
    r"|References\b"
    r"|Funding\b"
    r"|Data availability"
    r"|Supplementary\b"
    r"|Supporting information"
    r"|usertype\b"
    r"|protocol\b"
    r"|send_\d+"
    r"|role\s+\w+"
    r")",
    re.IGNORECASE,
)

_CAPTION_BODY_START_RE = re.compile(
    r"^\s*(?:In the|For the|The scheme|The first|The second|The last|Then|Next|After|Before|"
    r"As already|In case|In turn|Please note|We |Our |This |These |Those |Also,|Moreover|Furthermore|"
    r"Finally,|The tool|Table \d+ presents|Fig\. \d+ shows)\b",
    re.IGNORECASE,
)


def _safe_sentence_cut(text: str, max_chars: int = 650) -> str:
    t = clean_text_artifacts(text or "")
    if len(t) <= max_chars:
        return t
    # Prefer the last sentence ending before the cap, but keep enough text.
    cut = max(t.rfind(". ", 0, max_chars), t.rfind("); ", 0, max_chars), t.rfind("] ", 0, max_chars))
    if cut < 80:
        cut = max_chars
    return t[:cut + 1].rstrip(" ,;:")


def _sanitize_figure_caption(caption: str, label: str | None = None) -> str:
    """Prevent figure captions from absorbing body text/back matter/code.

    This is intentionally conservative: a slightly shorter caption is much
    safer than a caption containing entire paragraphs, CRediT sections, or code.
    """
    c = clean_text_artifacts(caption or "")
    if not c:
        return ""
    lines = [ln.strip() for ln in c.splitlines() if ln.strip()]
    if not lines:
        return ""

    kept = []
    for i, ln in enumerate(lines):
        if i > 0 and _CAPTION_HARD_STOP_RE.match(ln):
            break
        if i > 0 and kept and _CAPTION_BODY_START_RE.match(ln) and len(" ".join(kept)) > 35:
            break
        if i > 0 and len(" ".join(kept)) > 220 and ln.endswith(".") and _CAPTION_BODY_START_RE.match(ln):
            break
        kept.append(ln)
        if len(" ".join(kept)) > 700:
            break

    c = clean_text_artifacts("\n".join(kept))

    # Hard-stop markers that may occur mid-line after PyMuPDF normalization.
    marker_pat = re.compile(
        r"\b(?:CRediT authorship|Declaration of competing interest|Acknowledg(?:e)?ments?|"
        r"Appendix\s+[A-Z]|References\b|usertype\b|protocol\s+\w+\(|send_\d+)\b",
        re.IGNORECASE,
    )
    m = marker_pat.search(c)
    if m and m.start() > 25:
        c = c[:m.start()].rstrip(" ,;:")

    # If caption clearly contains an explanatory body paragraph, cut before it.
    body_pat = re.compile(
        r"\s+(?:In the|For the|The scheme|The first|Then|Next|After executing|As already mentioned|Please note)\b",
        re.IGNORECASE,
    )
    m = body_pat.search(c, pos=60)
    if m:
        c = c[:m.start()].rstrip(" ,;:")

    return _safe_sentence_cut(c, 650)


def _caption_quality(caption: str) -> tuple[str, list[str]]:
    c = clean_text_artifacts(caption or "")
    reasons: list[str] = []
    if not c:
        return "missing", ["empty_caption"]
    words = re.findall(r"\w+", c)
    if len(words) < 5:
        reasons.append("very_short_caption")
    if len(words) > 250 or len(c) > 1400:
        reasons.append("suspiciously_long_caption")
    if re.search(r"\b(?:In this paper|In this section|The remainder of this paper|CRediT authorship|Declaration of competing interest|References)\b", c, re.I):
        reasons.append("possible_body_text_leakage")
    return ("suspicious" if reasons else "good"), reasons


def _prefixed_name(doi: str, suffix: str) -> str:
    """Filename compatible with DOI-folder naming: 10.1016/x -> 10_1016_x_suffix."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", doi or "paper").strip("_")
    return f"{safe}_{suffix}"

def _figure_label_patterns(label: str) -> list[re.Pattern]:
    lab = clean_text_artifacts(label or "")
    nums = re.findall(r"(?:Figure|Fig\.?|Scheme)\s*([A-Za-z]?\.?\d+[A-Za-z]?|S\d+|A\.?\d+)", lab, re.IGNORECASE)
    if not nums:
        nums = re.findall(r"(S\d+|A\.?\d+|\d+[A-Za-z]?)", lab, re.IGNORECASE)
    pats = []
    for num in nums[:1]:
        num_flex = re.escape(num).replace(r"\.", r"\.?")
        pats.append(re.compile(rf"(?:^|\n)\s*(?:Fig\.?|Figure)\s*{num_flex}\b\.?(.*?)(?=(?:\n\s*(?:Fig\.?|Figure|Table)\s+[A-Za-z]?\.?\d+\b)|(?:\n\s*\d+(?:\.\d+)*\.?\s+[A-Z])|(?:\n\s*[A-Z]\.?.{{0,80}}et al\.?)|\Z)", re.IGNORECASE | re.DOTALL))
    return pats


def _caption_from_raw_pages(fig: dict, raw_pages: list[str]) -> str:
    page = fig.get("page")
    pages_to_check = []
    if page:
        pi = int(page) - 1
        pages_to_check.extend([pi, pi + 1])
    else:
        pages_to_check.extend(range(len(raw_pages or [])))
    for pi in pages_to_check:
        if pi < 0 or pi >= len(raw_pages or []):
            continue
        text = raw_pages[pi] or ""
        for pat in _figure_label_patterns(fig.get("label", "")):
            m = pat.search(text)
            if not m:
                continue
            label_text = re.search(r"(?:Fig\.?|Figure)\s*[A-Za-z]?\.?\d+[A-Za-z]?", m.group(0), re.IGNORECASE)
            prefix = label_text.group(0).strip() if label_text else (fig.get("label") or "Figure")
            cap_body = m.group(1).strip() if m.lastindex else m.group(0).strip()
            cap = clean_text_artifacts(prefix + ". " + cap_body)
            # Remove common footer fragments accidentally captured after captions.
            cap = re.split(r"\n\s*[A-Z](?:\.[A-Z])+.*?\s*/\s*[^/]+\s+\d+", cap)[0].strip()
            cap = _sanitize_figure_caption(cap, fig.get("label"))
            if len(cap) > 20:
                return cap
    return ""


def _enrich_figure_captions_from_raw_pages(figures: list[dict], raw_pages: list[str]) -> int:
    changed = 0
    for fig in figures or []:
        if (fig.get("type") or "figure") == "table":
            continue
        current = _sanitize_figure_caption(fig.get("caption") or "", fig.get("label"))
        raw_cap = _sanitize_figure_caption(_caption_from_raw_pages(fig, raw_pages), fig.get("label"))
        if not raw_cap:
            if current:
                fig["caption"] = current
            continue
        # Prefer raw enrichment only when it is clearly better and not a huge body-text capture.
        if (_caption_is_incomplete(current) and len(raw_cap) >= 25) or (len(raw_cap) > len(current) * 1.15 and len(raw_cap) <= 650):
            fig["caption"] = raw_cap
            fig["caption_source"] = "raw_page_enriched"
            changed += 1
        else:
            fig["caption"] = current or raw_cap
    return changed


def _is_missing_supplementary_figure(fig: dict) -> bool:
    label = clean_text_artifacts(fig.get("label") or "")
    return bool(
        fig.get("image_file_missing")
        and re.search(r"\b(?:Fig\.?|Figure)\s*S\d+\b", label, re.IGNORECASE)
    )


def _make_table_bbox_resolver(table_detections: list[dict] | None):
    dets = [d for d in (table_detections or []) if (d.get("type") or "") == "table"]
    if not dets:
        return None

    def _num_from_caption(cap: str):
        m = re.search(r"\bTable\s+((?:[A-Z]\.?)?\d+[A-Za-z]?|[IVXLC]+)\b", cap or "", re.IGNORECASE)
        return m.group(1).lower() if m else None

    indexed = []
    for d in dets:
        bb = d.get("bounding_box") or {}
        if not all(k in bb for k in ("x", "y", "w", "h")):
            continue
        indexed.append({
            "num": _num_from_caption(d.get("caption", "")),
            "page": d.get("page"),
            "bbox": (bb["x"], bb["y"], bb["w"], bb["h"]),
        })

    def resolver(num, page):
        num_l = (num or "").lower()
        for d in indexed:
            if d["num"] and d["num"] == num_l:
                return (d["page"] or page, d["bbox"])
        for d in indexed:
            if page is not None and d["page"] == page:
                return (d["page"], d["bbox"])
        return None

    return resolver


def _is_related_work_output(section: dict) -> bool:
    if section.get("section_role") == "related_work":
        return True
    try:
        from imrad_classifier import is_related_work
        return is_related_work(section.get("heading", ""))
    except Exception:
        h = (section.get("heading") or "").lower()
        return "related work" in h or "literature review" in h


def _dominant_section(mentions: dict) -> str | None:
    if not mentions:
        return None
    return sorted(mentions.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _asset_quality(item: dict) -> dict:
    quality = item.get("extraction_quality", "good")
    method = item.get("extraction_method", "")
    reasons = item.get("quality_reasons", []) or []

    is_full_page = quality == "page_render" or method == "pymupdf_full_page"
    suspicious = (
        quality in {"header_strip", "failed"}
        or any("duplicate bbox" in str(r).lower() for r in reasons)
    )
    crop_quality = "page_render" if is_full_page else ("suspicious" if suspicious else "good")
    return {
        "crop_quality": crop_quality,
        "vlm_ready": crop_quality == "good",
        "is_full_page_render": bool(is_full_page),
    }


def _figure_placeholder(fig: dict) -> str:
    label = fig.get("label") or fig.get("figure_id") or "FIGURE"
    label = re.sub(r"\s+", " ", str(label)).strip()
    if not re.match(r"(?i)^(fig|figure|scheme|table)\b", label):
        ftype = fig.get("type", "figure")
        if ftype == "scheme":
            label = f"Scheme {label}"
        elif ftype == "table":
            label = f"Table {label}"
        else:
            label = f"Figure {label}"
    return f"[{label} here]"


def _word_sequence_regex(words: list[str]) -> str:
    # Allow spaces/punctuation/newlines between OCR/PDF-extracted words.
    return r"\W+".join(map(re.escape, words))


def _caption_regexes(fig: dict) -> list[re.Pattern]:
    """Build conservative regexes for caption leakage removal."""
    out: list[re.Pattern] = []
    cap = clean_text_artifacts(fig.get("caption", "") or "")
    label = clean_text_artifacts(fig.get("label", "") or "")
    if not cap:
        return out

    # Exact-ish full caption with flexible whitespace and punctuation spacing.
    flex = re.escape(cap)
    flex = flex.replace(r"\ ", r"\s+")
    flex = flex.replace(r"\:", r"\s*:\s*").replace(r"\.", r"\s*\.\s*")
    out.append(re.compile(flex, re.IGNORECASE | re.DOTALL))

    # Label + first/last words. Useful when body text has "Fig. 1 :" but JSON
    # caption has "Fig. 1:".
    words = re.findall(r"[A-Za-z0-9]+", cap)
    if len(words) >= 8:
        first_words = words[:7]
        last_words = words[-4:]
        start = _word_sequence_regex(first_words)
        end = _word_sequence_regex(last_words)
        out.append(re.compile(start + r".{0,1600}?" + end, re.IGNORECASE | re.DOTALL))

    # Optional leading label if caption text was split from the label.
    if label and label.lower() not in cap.lower() and len(words) >= 5:
        first = _word_sequence_regex(words[:6])
        lab = re.escape(label).replace(r"\ ", r"\s+")
        out.append(re.compile(lab + r"\W+" + first + r".{0,1000}?", re.IGNORECASE | re.DOTALL))

    # Caption leakage can appear without the "Fig. N" prefix when GROBID
    # inserts only the description into a section. Match a longer distinctive
    # caption span from the body of the caption.
    if len(words) >= 16:
        for offset in (3, 6, 9):
            if len(words) > offset + 10:
                start = _word_sequence_regex(words[offset:offset + 8])
                end = _word_sequence_regex(words[-5:])
                out.append(re.compile(start + r".{0,2400}?" + end, re.IGNORECASE | re.DOTALL))
    return out


def _replace_caption_leakage_in_text(text: str, figures: list[dict]) -> tuple[str, int]:
    if not text:
        return "", 0
    out = text
    n = 0
    for fig in figures or []:
        if (fig.get("type") or "") == "table":
            continue
        placeholder = _figure_placeholder(fig)
        for rx in _caption_regexes(fig):
            def repl(_m):
                nonlocal n
                n += 1
                return f" {placeholder} "
            out = rx.sub(repl, out, count=1)
    out = clean_text_artifacts(out)
    out = re.sub(r"(?:\n\s*){3,}", "\n\n", out).strip()
    return out, n


def _clean_sections_in_place(sections: list[dict], figures: list[dict]) -> int:
    """Clean text artifacts and remove caption leakage from section fields."""
    total_removed = 0
    for sec in sections:
        for field in ("text", "text_no_tables"):
            txt = sec.get(field) or ""
            cleaned, n = _replace_caption_leakage_in_text(txt, figures)
            sec[field] = cleaned
            total_removed += n
    return total_removed


def _filter_sections_full(sections: list[dict]) -> list[dict]:
    kept = []
    for s in sections:
        body = clean_text_artifacts(s.get("text") or "")
        if not body:
            continue
        kept.append({"order": 0, "heading": s.get("heading", "") or "", "text": body})
    for i, s in enumerate(kept):
        s["order"] = i
    return kept


def _filter_sections_imrad(sections: list[dict]) -> list[dict]:
    kept = []
    for s in sections:
        body = clean_text_artifacts(s.get("text") or s.get("text_no_tables") or "")
        if not body:
            continue
        related_work = _is_related_work_output(s)
        imrad_value = None if related_work else s.get("imrad")
        if imrad_value not in IMRAD_CATEGORIES:
            continue
        entry = {
            "order": 0,
            "heading": s.get("heading", "") or "",
            "imrad": imrad_value,
            "imrad_source": s.get("imrad_source", "unknown"),
            "imrad_confidence": s.get("imrad_confidence", 0.0),
            "imrad_reason": s.get("imrad_reason", ""),
            "text": body,
        }
        if s.get("imrad_secondary"):
            entry["imrad_secondary"] = s.get("imrad_secondary")
            entry["imrad_secondary_source"] = s.get("imrad_secondary_source")
            entry["imrad_secondary_confidence"] = s.get("imrad_secondary_confidence")
            entry["imrad_secondary_reason"] = s.get("imrad_secondary_reason")
        kept.append(entry)
    for i, s in enumerate(kept):
        s["order"] = i
    return kept


def _build_imrad_summary(sections: list[dict], methods_inference_report: dict | None) -> dict:
    found_set = set()
    source_counts = {}
    for s in sections:
        if _is_related_work_output(s):
            continue
        label = s.get("imrad")
        if label in IMRAD_CATEGORIES:
            found_set.add(label)
            src = s.get("imrad_source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        secondary = s.get("imrad_secondary")
        if secondary in IMRAD_CATEGORIES:
            found_set.add(secondary)
            src = s.get("imrad_secondary_source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

    found = [c for c in IMRAD_CATEGORIES if c in found_set]
    missing = [c for c in IMRAD_CATEGORIES if c not in found_set]

    tables_total = sum(int(s.get("table_count", 0) or 0) for s in sections)
    tables_unrecovered = sum(int(s.get("table_unrecovered_count", 0) or 0) for s in sections)

    summary = {
        "complete": len(missing) == 0,
        "found": found,
        "missing": missing,
        "methods_inferred": False,
        "methods_inferred_from": None,
        "classification_version": "v7",
        "parser_output_version": "v13",
        "source_counts": source_counts,
        "tables_total": tables_total,
        "tables_unrecovered": tables_unrecovered,
    }
    if methods_inference_report:
        summary["methods_inferred"] = bool(methods_inference_report.get("inferred", False))
        summary["methods_inferred_from"] = (
            methods_inference_report.get("section_heading")
            if summary["methods_inferred"] else None
        )
        summary["methods_inference_source"] = methods_inference_report.get("source")
        summary["methods_inference_reason"] = methods_inference_report.get("reason")
        if methods_inference_report.get("v4_enrichment") is not None:
            summary["v4_enrichment"] = methods_inference_report.get("v4_enrichment")
    return summary


def _build_fulltext_dict(*, doi, publisher, journal, title, abstract,
                         sections_full, references, extraction_timestamp,
                         quality_flags=None):
    sections_out = list(sections_full)
    if references:
        ref_text = "\n".join(f"[{i + 1}] {r}" for i, r in enumerate(references))
        sections_out.append({"order": len(sections_out), "heading": "References", "text": ref_text})
    parts = []
    for s in sections_out:
        heading = s["heading"]
        text = s["text"]
        parts.append(f"## {heading}\n\n{text}" if heading else text)
    full_text = "\n\n".join(parts)
    return {
        "doi": doi, "publisher": publisher, "journal": journal,
        "title": title, "abstract": clean_text_artifacts(abstract),
        "extraction_tool": "grobid+pdffigures2",
        "parser_output_version": "v13",
        "table_content_in_body": False,
        "figure_caption_content_in_body": False,
        "equation_content_in_body": False,
        "extraction_timestamp": extraction_timestamp,
        "article_structure": {
            "section_count": len(sections_out),
            "first_section_heading": sections_out[0].get("heading", "") if sections_out else "",
            "first_section_page": sections_out[0].get("page") if sections_out else None,
        },
        "quality_flags": quality_flags or {},
        "sections": sections_out, "references": references, "full_text": full_text,
    }


def _build_fulltext_imrad_dict(*, doi, publisher, journal, title, abstract,
                               sections_imrad, extraction_timestamp,
                               imrad_summary, quality_flags=None):
    return {
        "doi": doi, "publisher": publisher, "journal": journal,
        "title": title, "abstract": clean_text_artifacts(abstract),
        "extraction_timestamp": extraction_timestamp,
        "parser_output_version": "v13",
        "table_content_in_body": False,
        "figure_caption_content_in_body": False,
        "equation_content_in_body": False,
        "article_structure": {
            "section_count": len(sections_imrad),
            "first_section_heading": sections_imrad[0].get("heading", "") if sections_imrad else "",
            "first_section_page": sections_imrad[0].get("page") if sections_imrad else None,
        },
        "quality_flags": quality_flags or {},
        "imrad_summary": imrad_summary,
        "sections": sections_imrad,
    }



def _table_sort_key(t: dict):
    num = str(t.get("num") or "")
    m = re.search(r"(\d+)", num)
    n = int(m.group(1)) if m else 10**6
    return (n, str(t.get("num") or ""), int(t.get("page") or 0))

def _build_tables_dict(*, doi, sections: list[dict], extraction_timestamp: str, table_stats: dict | None = None) -> dict:
    table_stats = table_stats or {}
    out = []
    seen = set()
    counter = 0
    for sec in sections:
        sec_imrad = sec.get("imrad")
        sec_heading = sec.get("heading", "") or ""
        for tbl in sec.get("tables", []) or []:
            raw = clean_text_artifacts(tbl.get("raw_text") or "")
            md = clean_text_artifacts(tbl.get("markdown") or raw)
            if not raw and not md and not tbl.get("caption"):
                continue
            key = (str(tbl.get("num") or "").lower(), (raw or md)[:160].lower())
            if key in seen:
                continue
            seen.add(key)
            counter += 1
            table_id = tbl.get("table_id") or f"table{counter:02d}"
            label = tbl.get("label") or (f"Table {tbl.get('num')}" if tbl.get("num") else table_id)
            placeholder = tbl.get("placeholder") or f"[{label} here]"
            out.append({
                "table_id": table_id,
                "label": label,
                "num": tbl.get("num"),
                "caption": clean_text_artifacts(tbl.get("caption") or ""),
                "page": tbl.get("page") or sec.get("page"),
                "section": sec_imrad,
                "section_heading": sec_heading,
                "placeholder": placeholder,
                "source": tbl.get("source", "unknown"),
                "recovered": bool(tbl.get("recovered", True)),
                "ocr_confidence": tbl.get("ocr_confidence"),
                "structure_confidence": tbl.get("structure_confidence"),
                "low_confidence": bool(tbl.get("low_confidence", False)),
                "needs_review": bool(tbl.get("needs_review", False)),
                "quality_reasons": tbl.get("quality_reasons", []) or [],
                "structured_rows": tbl.get("structured_rows", []) or [],
                "structured_markdown": tbl.get("structured_markdown", "") or "",
                "text": raw or md,
                "markdown": (tbl.get("structured_markdown") or md or raw),
                "raw_markdown_text": md or raw,
            })
    out.sort(key=_table_sort_key)
    for i, item in enumerate(out, start=1):
        item["table_id"] = f"table{i:02d}"
    expected_labels = table_stats.get("expected_table_labels") or []
    expected_count = int(table_stats.get("expected_table_count", len(expected_labels)) or 0)
    recovered_count = sum(1 for t in out if t.get("recovered", True))
    unrecovered_count = sum(1 for t in out if not t.get("recovered", True))
    # If expected_table_count is larger than explicit unrecovered records, keep
    # the higher value so quality_report cannot falsely pass missing tables.
    expected_unrecovered = max(0, expected_count - recovered_count) if expected_count else unrecovered_count
    tables_unrecovered = max(unrecovered_count, expected_unrecovered)
    return {
        "doi": doi,
        "extraction_timestamp": extraction_timestamp,
        "parser_output_version": "v13",
        "expected_table_count": expected_count,
        "expected_table_labels": expected_labels,
        "expected_tables_recovered": recovered_count if expected_count else len(out),
        "expected_tables_unrecovered": expected_unrecovered,
        "table_count": len(out),
        "tables_unrecovered": tables_unrecovered,
        "tables": out,
    }


def _build_figures_dict(*, doi, figures_enriched, extraction_timestamp, figures_dir=None):
    figures_out = []
    figures_dir = Path(figures_dir) if figures_dir is not None else None

    for fig in figures_enriched:
        ftype = fig.get("type", "figure")
        if ftype == "table":
            continue
        if _is_missing_supplementary_figure(fig):
            continue
        fig_id = fig.get("figure_id", "")
        final_image_name = f"{fig_id}.png" if fig_id else None
        mentions = fig.get("mentions_by_section", {}) or {}

        quality = fig.get("extraction_quality", "good")
        aq = _asset_quality(fig)
        image_file_missing = False
        if final_image_name and figures_dir is not None:
            image_file_missing = not (figures_dir / final_image_name).exists()
        if image_file_missing:
            quality = "missing_image"
            aq = {"crop_quality": "missing_image", "vlm_ready": False, "is_full_page_render": False}
            reasons = list(fig.get("quality_reasons", []) or [])
            reasons.append("image_file listed in figures.json but missing from figures folder")
        else:
            reasons = fig.get("quality_reasons", [])

        if "assigned_section" in fig:
            assigned_section = fig.get("assigned_section")
            assignment_method = fig.get("assignment_method") or ("mention_based" if mentions else "unassigned")
        else:
            assigned_section = _dominant_section(mentions)
            assignment_method = "mention_based" if mentions else "unassigned"
        possible_sections = fig.get("possible_sections") or []
        if not possible_sections:
            possible_sections = list(mentions.keys())
            if assigned_section and assigned_section not in possible_sections:
                possible_sections.append(assigned_section)

        figures_out.append({
            "figure_id": fig_id,
            "label": fig.get("label", ""),
            "normalized_label": fig.get("normalized_label", ""),
            "display_label": fig.get("display_label") or fig.get("label", ""),
            "is_supplementary_or_appendix": bool(fig.get("is_supplementary_or_appendix", False)),
            "type": ftype,
            "caption": _sanitize_figure_caption(fig.get("caption", ""), fig.get("label")),
            "caption_quality": _caption_quality(_sanitize_figure_caption(fig.get("caption", ""), fig.get("label")))[0],
            "caption_quality_reasons": _caption_quality(_sanitize_figure_caption(fig.get("caption", ""), fig.get("label")))[1],
            "caption_source": fig.get("caption_source", "pdffigures2"),
            "placeholder": _figure_placeholder(fig),
            "image_file": final_image_name, "image_file_missing": image_file_missing,
            "page": fig.get("page"),
            "bounding_box": fig.get("bounding_box"),
            "extraction_method": fig.get("extraction_method", "pdffigures2"),
            "extraction_quality": quality,
            "crop_quality": aq["crop_quality"], "vlm_ready": aq["vlm_ready"],
            "is_full_page_render": aq["is_full_page_render"],
            "quality_reasons": reasons,
            "mentions_by_section": mentions, "mentioned_in_sections": mentions,
            "assigned_section": assigned_section,
            "possible_sections": possible_sections,
            "assignment_method": assignment_method,
            "assignment_evidence": fig.get("assignment_evidence", {}),
        })

    n_figures = sum(1 for f in figures_out if f["type"] != "scheme")
    n_schemes = sum(1 for f in figures_out if f["type"] == "scheme")
    n_good = sum(1 for f in figures_out if f["extraction_quality"] == "good" and not f.get("image_file_missing"))
    n_fallback = sum(1 for f in figures_out if f["extraction_quality"] == "page_render" and not f.get("image_file_missing"))
    n_still_bad = sum(1 for f in figures_out if f["extraction_quality"] not in ("good", "page_render") or f.get("image_file_missing"))

    return {
        "doi": doi, "extraction_timestamp": extraction_timestamp,
        "parser_output_version": "v13",
        "figure_count": len(figures_out),
        "stats": {
            "figures": n_figures, "schemes": n_schemes, "good": n_good,
            "fallback": n_fallback, "still_bad": n_still_bad,
            "page_render": sum(1 for f in figures_out if f["is_full_page_render"]),
            "vlm_ready": sum(1 for f in figures_out if f["vlm_ready"]),
            "unassigned": sum(1 for f in figures_out if f["assignment_method"] == "unassigned"),
            "missing_images": sum(1 for f in figures_out if f.get("image_file_missing")),
        },
        "figures": figures_out,
    }



def _normalize_figure_records(figures: list[dict]) -> int:
    """Preserve real figure labels in ids/files.

    Examples:
      Fig. A.1 -> figure_id fig_A_1, label Figure A.1
      Fig. S8  -> figure_id fig_S8,  label Figure S8
      Figure 3 -> figure_id fig_03,  label Figure 3
    """
    changed = 0
    used: dict[str, int] = {}
    for idx, fig in enumerate(figures or [], start=1):
        if (fig.get("type") or "") == "table":
            continue
        rec = parse_label(fig.get("caption") or fig.get("label") or "")
        if rec and rec.get("kind") in {"figure", "scheme"}:
            new_id = rec["id"]
            new_label = rec["norm"]
            fig["type"] = rec["kind"]
            fig["label"] = new_label
            fig["display_label"] = new_label
            fig["normalized_label"] = rec.get("normalized_label")
            fig["is_supplementary_or_appendix"] = bool(rec.get("prefix"))
        else:
            new_id = fig.get("figure_id") or fig.get("figure_id_hint") or asset_id(fig.get("label"), idx) or f"fig_unlabeled_{idx:02d}"
            fig.setdefault("display_label", fig.get("label") or new_id)
            fig.setdefault("normalized_label", fig.get("label") or new_id)
            fig.setdefault("is_supplementary_or_appendix", False)
        base_id = new_id
        if base_id in used:
            used[base_id] += 1
            new_id = f"{base_id}_{used[base_id]}"
        else:
            used[base_id] = 1
        if fig.get("figure_id") != new_id:
            changed += 1
        fig["figure_id"] = new_id
    return changed

def _render_page_asset(pdf_path, page_1indexed, out_png, dpi: int = 150) -> bool:
    if not pdf_path or not page_1indexed:
        return False
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        try:
            page_i = int(page_1indexed) - 1
            if page_i < 0 or page_i >= doc.page_count:
                return False
            page = doc[page_i]
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(out_png))
            return out_png.exists()
        finally:
            doc.close()
    except Exception:
        return False


def _recover_missing_figure_assets(figures_enriched, figures_source_dir, pdf_path) -> int:
    """Ensure every figure entry has an image file when possible.

    If pdffigures2/TEI gives a metadata entry but the PNG is missing, render the
    figure's page as a conservative fallback so figures.json and figures/ agree.
    """
    if not pdf_path:
        return 0
    figures_source_dir = Path(figures_source_dir)
    recovered = 0
    for idx, fig in enumerate(figures_enriched):
        if (fig.get("type") or "") == "table":
            continue
        src_name = fig.get("image_file")
        if src_name and (figures_source_dir / src_name).exists():
            continue
        page = fig.get("page")
        if not page:
            fig["image_file_missing"] = True
            continue
        fig_id = fig.get("figure_id") or f"recovered_fig_{idx:03d}"
        src_name = src_name or f"{fig_id}_page_render.png"
        out_png = figures_source_dir / src_name
        if _render_page_asset(pdf_path, page, out_png):
            fig["image_file"] = src_name
            fig["extraction_method"] = "missing_asset_page_render"
            fig["extraction_quality"] = "page_render"
            fig["is_full_page_render"] = True
            fig.setdefault("quality_reasons", [])
            fig["quality_reasons"].append("missing image recovered by page render fallback")
            fig.pop("image_file_missing", None)
            recovered += 1
        else:
            fig["image_file_missing"] = True
            fig.setdefault("quality_reasons", [])
            fig["quality_reasons"].append("image file missing and page-render recovery failed")
    return recovered


def _copy_figure_images(figures_enriched, figures_source_dir, figures_dest_dir, pdf_path=None):
    figures_dest_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    missing = []

    def copy_one(fig):
        nonlocal n_copied
        asset_id = fig.get("figure_id", "")
        src_name = fig.get("image_file")
        if not asset_id:
            return
        dst_path = figures_dest_dir / f"{asset_id}.png"
        if src_name:
            src_path = figures_source_dir / src_name
            if src_path.exists():
                try:
                    shutil.copy2(src_path, dst_path)
                    n_copied += 1
                    return
                except OSError as e:
                    missing.append(f"{src_name} (copy failed: {e})")
        # Last safety net: render the page directly into the final figures/ folder.
        page = fig.get("page")
        if pdf_path and page and _render_page_asset(pdf_path, page, dst_path):
            fig["image_file"] = f"{asset_id}.png"
            fig["extraction_method"] = "missing_asset_final_page_render"
            fig["extraction_quality"] = "page_render"
            fig["is_full_page_render"] = True
            fig.pop("image_file_missing", None)
            fig.setdefault("quality_reasons", [])
            fig["quality_reasons"].append("missing image recovered by final page-render fallback")
            n_copied += 1
            return
        missing.append(src_name or f"{asset_id}.png")

    for fig in figures_enriched:
        if (fig.get("type") or "") == "table":
            continue
        copy_one(fig)
    return n_copied, missing


def write_paper_outputs(*, paper_data, figures_enriched, references, doi,
                        publisher, journal, output_dir, figures_source_dir,
                        methods_inference_report=None, pdf_path=None,
                        table_detections=None):
    output_dir = Path(output_dir)
    figures_source_dir = Path(figures_source_dir)
    tmp_dir = output_dir.parent / (output_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        extraction_timestamp = _now_utc_iso()
        title = paper_data.get("title", "") or ""
        abstract = paper_data.get("abstract", "") or ""
        all_sections = paper_data.get("sections", []) or []
        references = references if references is not None else (paper_data.get("references", []) or [])

        raw_pages, raw_err = _read_raw_pages(pdf_path)
        raw_caption_enriched = _enrich_figure_captions_from_raw_pages(figures_enriched, raw_pages)
        figure_labels_normalized = _normalize_figure_records(figures_enriched)

        # Equations: keep display equations out of full text and write them to equations.json.
        equations_dict = build_equations_dict(
            doi=doi, extraction_timestamp=extraction_timestamp, sections=all_sections, raw_pages=raw_pages,
        )
        n_eq_ok = equations_dict.get("equation_count", 0)

        # Tables: collect PyMuPDF/OCR table text into section["tables"], but keep
        # only placeholders in section text.
        bbox_resolver = _make_table_bbox_resolver(table_detections)
        ocr_fn = ocr_engine.ocr_table_region if ocr_engine.ocr_enabled() else None
        try:
            table_stats = inject_tables_into_sections(
                all_sections, raw_pages,
                pdf_path=pdf_path, ocr_fn=ocr_fn, bbox_resolver=bbox_resolver,
            )
        except Exception:
            table_stats = {"injected": 0, "text_tables": 0, "ocr_tables": 0, "unrecovered": 0}
            for s in all_sections:
                if not (s.get("text_no_tables") or "").strip():
                    s["text_no_tables"] = s.get("text", "") or ""
            # Even if table extraction itself fails, still audit expected tables
            # from raw PDF text and add explicit unrecovered placeholders.
            try:
                table_stats.update(ensure_expected_tables_in_sections(all_sections, raw_pages))
            except Exception:
                pass

        # Clean text and remove caption leakage now that figure captions are known.
        captions_removed = _clean_sections_in_place(all_sections, figures_enriched)

        try:
            from imrad_classifier import enrich_imrad_sections
            enrich_imrad_sections(all_sections)
        except Exception:
            pass

        sections_full = _filter_sections_full(all_sections)
        sections_imrad = _filter_sections_imrad(all_sections)
        imrad_summary = _build_imrad_summary(all_sections, methods_inference_report)
        imrad_summary["captions_removed_from_body"] = captions_removed

        quality_flags = {
            "caption_leakage_detected": captions_removed > 0,
            "captions_removed_from_body": captions_removed,
            "low_confidence_tables": 0,  # filled after tables_dict is built
            "full_page_figure_fallbacks": 0,  # filled after figures_dict is built
            "missing_figure_images": 0,  # filled after figures_dict is built
            "bad_figure_crops": 0,
            "noisy_equations": equations_dict.get("stats", {}).get("noisy_equations", 0),
            "repaired_equations": equations_dict.get("stats", {}).get("repaired_equations", 0),
            "unrecovered_equations": equations_dict.get("stats", {}).get("unrecovered_equations", 0),
            "rejected_equations": equations_dict.get("stats", {}).get("rejected_equations", 0),
            "page_null_equations": equations_dict.get("stats", {}).get("page_null_equations", 0),
            "expected_tables": int(table_stats.get("expected_table_count", 0) or 0),
            "expected_table_labels": table_stats.get("expected_table_labels", []) or [],
            "tables_recovered": int(table_stats.get("expected_tables_recovered", 0) or 0),
            "tables_unrecovered": int(table_stats.get("expected_tables_unrecovered", table_stats.get("unrecovered", 0)) or 0),
        }

        fulltext = _build_fulltext_dict(
            doi=doi, publisher=publisher, journal=journal, title=title,
            abstract=abstract, sections_full=sections_full,
            references=references, extraction_timestamp=extraction_timestamp,
            quality_flags=quality_flags,
        )
        if raw_err:
            fulltext["raw_fulltext_error"] = raw_err
        fulltext["raw_fulltext"] = "\n\n".join(raw_pages)
        fulltext["raw_fulltext_pages"] = raw_pages
        with open(tmp_dir / _prefixed_name(doi, "fulltext.json"), "w", encoding="utf-8") as f:
            json.dump(fulltext, f, ensure_ascii=False, indent=2)

        imrad = None
        if sections_imrad:
            imrad = _build_fulltext_imrad_dict(
                doi=doi, publisher=publisher, journal=journal, title=title,
                abstract=abstract, sections_imrad=sections_imrad,
                extraction_timestamp=extraction_timestamp,
                imrad_summary=imrad_summary,
                quality_flags=quality_flags,
            )
            with open(tmp_dir / _prefixed_name(doi, "fulltext_imrad.json"), "w", encoding="utf-8") as f:
                json.dump(imrad, f, ensure_ascii=False, indent=2)

        tables_dict = _build_tables_dict(
            doi=doi, sections=all_sections, extraction_timestamp=extraction_timestamp, table_stats=table_stats,
        )
        quality_flags["low_confidence_tables"] = sum(1 for t in tables_dict.get("tables", []) if t.get("low_confidence"))
        quality_flags["expected_tables"] = int(tables_dict.get("expected_table_count", 0) or 0)
        quality_flags["expected_table_labels"] = tables_dict.get("expected_table_labels", []) or []
        quality_flags["tables_recovered"] = int(tables_dict.get("expected_tables_recovered", tables_dict.get("table_count", 0)) or 0)
        quality_flags["tables_unrecovered"] = int(tables_dict.get("tables_unrecovered", 0) or 0)
        with open(tmp_dir / _prefixed_name(doi, "tables.json"), "w", encoding="utf-8") as f:
            json.dump(tables_dict, f, ensure_ascii=False, indent=2)

        with open(tmp_dir / _prefixed_name(doi, "equations.json"), "w", encoding="utf-8") as f:
            json.dump(equations_dict, f, ensure_ascii=False, indent=2)

        # Recover any figure metadata entries whose image file is missing, then
        # copy assets before writing figures.json so JSON stats match figures/.
        recovered_missing_images = _recover_missing_figure_assets(
            figures_enriched=figures_enriched,
            figures_source_dir=figures_source_dir,
            pdf_path=pdf_path,
        )
        figures_tmp = tmp_dir / "figures"
        n_copied, missing = _copy_figure_images(
            figures_enriched=figures_enriched,
            figures_source_dir=figures_source_dir,
            figures_dest_dir=figures_tmp,
            pdf_path=pdf_path,
        )
        figures_dict = _build_figures_dict(
            doi=doi, figures_enriched=figures_enriched,
            extraction_timestamp=extraction_timestamp,
            figures_dir=figures_tmp,
        )
        quality_flags["full_page_figure_fallbacks"] = figures_dict.get("stats", {}).get("page_render", 0)
        quality_flags["missing_figure_images"] = figures_dict.get("stats", {}).get("missing_images", 0)
        quality_flags["bad_figure_crops"] = figures_dict.get("stats", {}).get("still_bad", 0)
        # refresh quality flags in already-written fulltext JSONs by updating in-memory dicts before final output below
        with open(tmp_dir / _prefixed_name(doi, "figures.json"), "w", encoding="utf-8") as f:
            json.dump(figures_dict, f, ensure_ascii=False, indent=2)

        # Re-write fulltext files once after all quality flags have final counts.
        fulltext["quality_flags"] = quality_flags
        with open(tmp_dir / _prefixed_name(doi, "fulltext.json"), "w", encoding="utf-8") as f:
            json.dump(fulltext, f, ensure_ascii=False, indent=2)
        if sections_imrad:
            imrad["quality_flags"] = quality_flags
            with open(tmp_dir / _prefixed_name(doi, "fulltext_imrad.json"), "w", encoding="utf-8") as f:
                json.dump(imrad, f, ensure_ascii=False, indent=2)

        quality_report = build_quality_report(
            doi=doi, fulltext=fulltext, imrad=imrad,
            figures=figures_dict, tables=tables_dict, equations=equations_dict,
            quality_flags=quality_flags,
        )
        with open(tmp_dir / _prefixed_name(doi, "quality_report.json"), "w", encoding="utf-8") as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_dir.rename(output_dir)

        return {
            "success": True, "error": None,
            "n_sections": len(sections_full),
            "n_imrad": len(sections_imrad),
            "n_references": len(references),
            "n_figures": figures_dict["stats"]["figures"],
            "n_schemes": figures_dict["stats"]["schemes"],
            "n_unassigned_figures": figures_dict["stats"]["unassigned"],
            "tables_total": tables_dict["table_count"],
            "expected_tables": tables_dict.get("expected_table_count", 0),
            "tables_unrecovered": tables_dict["tables_unrecovered"],
            "tables_text": table_stats.get("text_tables", 0),
            "tables_ocr": table_stats.get("ocr_tables", 0),
            "equations_ocr": n_eq_ok,
            "captions_removed_from_body": captions_removed,
            "imrad_complete": imrad_summary["complete"],
            "imrad_found": imrad_summary["found"],
            "methods_inferred": imrad_summary["methods_inferred"],
            "n_images_copied": n_copied,
            "missing_images": missing,
            "figures_missing_asset_recovered": recovered_missing_images,
            "figure_captions_raw_enriched": raw_caption_enriched,
            "figure_labels_normalized": figure_labels_normalized,
            "parser_status": quality_report.get("parser_status"),
        }

    except Exception as e:
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                pass
        return {
            "success": False, "error": f"{type(e).__name__}: {e}",
            "n_sections": 0, "n_imrad": 0,
            "n_references": 0, "n_figures": 0, "n_schemes": 0,
            "n_unassigned_figures": 0, "tables_total": 0,
            "tables_unrecovered": 0, "tables_text": 0, "tables_ocr": 0,
            "equations_ocr": 0, "captions_removed_from_body": 0,
            "imrad_complete": False, "imrad_found": [], "methods_inferred": False,
            "n_images_copied": 0, "missing_images": [],
        }
