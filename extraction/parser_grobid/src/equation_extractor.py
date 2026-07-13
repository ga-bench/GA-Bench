"""
equation_extractor.py (v11)

Production-oriented equation extraction.
Fixes in v11:
- Rejects table/prose false positives.
- Splits one GROBID formula containing multiple printed equation numbers.
- Regenerates unique stable equation IDs after filtering.
- Keeps printed number separate from internal ID.
- Adds latex_like_text and stricter confidence labels.
- Removes rejected equation placeholders from body text.
"""

from __future__ import annotations
import re
from text_cleanup import clean_equation_text, clean_text_artifacts
from equation_quality import (
    equation_confidence,
    extract_printed_number,
    latex_like_from_plain,
    normalize_equation_space,
    split_multiple_numbered_equations,
)
from pdf_text_locator import infer_page_for_text

_EQ_PLACEHOLDER_RE = re.compile(r"\[\[EQN:(?P<id>\d+)\]\]")
_STANDALONE_CONTINUATION_RE = re.compile(r"^\s*[)\]}]+\s*(?:\(?(?P<num>\d{1,3})\)?)?\s*$")


def _placeholder_from_printed(num) -> str:
    return f"[Equation {num} here]" if num is not None else "[Equation here]"


def _replace_placeholders(sec: dict, old_ids: list, replacement: str) -> int:
    count = 0
    for old_id in old_ids:
        if old_id is None:
            continue
        old = f"[[EQN:{old_id}]]"
        for field in ("text", "text_no_tables"):
            if sec.get(field) and old in sec[field]:
                sec[field] = sec[field].replace(old, replacement)
                count += 1
    return count


def _merge_standalone_fragments(eqs: list[dict]) -> tuple[list[dict], int, int]:
    merged: list[dict] = []
    repairs = 0
    dropped = 0
    i = 0
    while i < len(eqs):
        cur = dict(eqs[i] or {})
        raw = normalize_equation_space(clean_text_artifacts(cur.get("raw") or ""))
        cur["raw"] = raw
        cur.setdefault("merged_from_ids", [cur.get("id")])

        if i + 1 < len(eqs):
            nxt = dict(eqs[i + 1] or {})
            nxt_raw = normalize_equation_space(clean_text_artifacts(nxt.get("raw") or ""))
            standalone = bool(_STANDALONE_CONTINUATION_RE.match(nxt_raw))
            needs_close = raw.count("(") > raw.count(")") or raw.rstrip().endswith("(")
            if standalone and (needs_close or len(nxt_raw) <= 4):
                cur["raw"] = normalize_equation_space(raw.rstrip() + " " + nxt_raw.strip())
                cur["merged_from_ids"] = [x for x in [cur.get("id"), nxt.get("id")] if x is not None]
                if not cur.get("coords") and nxt.get("coords"):
                    cur["coords"] = nxt.get("coords")
                if not cur.get("page") and nxt.get("page"):
                    cur["page"] = nxt.get("page")
                repairs += 1
                merged.append(cur)
                i += 2
                continue

        if _STANDALONE_CONTINUATION_RE.match(raw):
            cur["unmerged_fragment"] = True
            dropped += 1
        merged.append(cur)
        i += 1
    return merged, repairs, dropped


def _expand_multi_numbered(eq: dict) -> list[dict]:
    parts = split_multiple_numbered_equations(eq.get("raw") or "")
    if len(parts) <= 1:
        return [eq]
    out: list[dict] = []
    ids = eq.get("merged_from_ids", [eq.get("id")])
    for k, part in enumerate(parts):
        e = dict(eq)
        e["raw"] = part
        e["split_from_multi_numbered_formula"] = True
        # First split consumes original placeholders; later splits do not have a
        # separate TEI placeholder but should still appear in equations.json.
        e["merged_from_ids"] = ids if k == 0 else []
        out.append(e)
    return out


def _section_for_equation(sec: dict) -> tuple[str | None, str, int | None]:
    return sec.get("imrad"), sec.get("heading", "") or "", sec.get("page")


def extract_equations_from_sections(sections: list[dict], raw_pages: list[str] | None = None) -> tuple[list[dict], dict]:
    equations: list[dict] = []
    placeholders_replaced = 0
    repaired_count = 0
    dropped_fragments = 0
    rejected_count = 0
    needs_review_count = 0
    unrecovered = 0
    seen: set[tuple[str, str]] = set()

    for sec in sections or []:
        sec_imrad, sec_heading, sec_page = _section_for_equation(sec)
        raw_eqs = sec.get("equations", []) or []
        merged_eqs, n_repairs, n_dropped = _merge_standalone_fragments(raw_eqs)
        repaired_count += n_repairs
        dropped_fragments += n_dropped

        expanded: list[dict] = []
        for eq in merged_eqs:
            expanded.extend(_expand_multi_numbered(eq))

        for eq in expanded:
            raw = normalize_equation_space(clean_text_artifacts(eq.get("raw") or ""))
            merged_ids = eq.get("merged_from_ids", [eq.get("id")])

            if eq.get("unmerged_fragment"):
                placeholders_replaced += _replace_placeholders(sec, merged_ids, "")
                continue

            clean = normalize_equation_space(clean_equation_text(raw))
            printed_num = extract_printed_number(clean) or extract_printed_number(raw)
            was_repaired = bool(
                (eq.get("merged_from_ids") and len(eq.get("merged_from_ids", [])) > 1)
                or eq.get("split_from_multi_numbered_formula")
            )
            confidence, reasons = equation_confidence(raw, clean, repaired=was_repaired)

            if confidence == "rejected":
                rejected_count += 1
                placeholders_replaced += _replace_placeholders(sec, merged_ids, "")
                continue
            if confidence == "needs_review":
                needs_review_count += 1
            if confidence == "repaired":
                repaired_count += 1
            if confidence == "unrecovered":
                unrecovered += 1

            key = (str(printed_num or ""), clean[:220].lower())
            if key in seen:
                placeholders_replaced += _replace_placeholders(sec, merged_ids, _placeholder_from_printed(printed_num))
                continue
            seen.add(key)

            inferred_page = eq.get("page") or sec_page or infer_page_for_text(clean or raw, raw_pages)
            record = {
                "equation_id": "",  # filled after final filtering
                "label": f"Equation {printed_num}" if printed_num is not None else "Equation",
                "num": printed_num,
                "page": inferred_page,
                "section": sec_imrad,
                "section_heading": sec_heading,
                "placeholder": _placeholder_from_printed(printed_num),
                "raw_text": raw,
                "clean_text": clean if confidence != "unrecovered" else None,
                "latex_like_text": latex_like_from_plain(clean),
                "source": "grobid_formula",
                "confidence": confidence,
                "needs_review": confidence in {"needs_review", "repaired"},
                "quality_reasons": reasons,
                "coords": eq.get("coords") or "",
                "merged_from_ids": merged_ids,
                "number_source": "formula_text" if printed_num is not None else "tei_order",
            }
            if record.get("page") is None:
                record["needs_review"] = True
                record.setdefault("quality_reasons", [])
                if "page_not_recovered" not in record["quality_reasons"]:
                    record["quality_reasons"].append("page_not_recovered")
                if record.get("confidence") == "clean":
                    record["confidence"] = "needs_review"
                    needs_review_count += 1
            equations.append(record)
            placeholders_replaced += _replace_placeholders(sec, merged_ids, record["placeholder"])

        # Safety: remove any orphan formula placeholders that were not accepted.
        for field in ("text", "text_no_tables"):
            txt = sec.get(field) or ""
            def repl(m):
                nonlocal placeholders_replaced
                placeholders_replaced += 1
                return ""
            sec[field] = _EQ_PLACEHOLDER_RE.sub(repl, txt)

    # Stable internal IDs must be unique and independent from printed numbers.
    for i, rec in enumerate(equations, start=1):
        rec["equation_id"] = f"eq{i:03d}"
        if rec.get("num") is None:
            rec["label"] = f"Equation {i}"
            rec["placeholder"] = f"[Equation {i} here]"

    stats = {
        "equation_count": len(equations),
        "placeholders_replaced": placeholders_replaced,
        "repaired_equations": repaired_count,
        "dropped_fragment_equations": dropped_fragments,
        "rejected_equations": rejected_count,
        "needs_review_equations": needs_review_count,
        "noisy_equations": needs_review_count,
        "unrecovered_equations": unrecovered,
        "page_null_equations": sum(1 for e in equations if e.get("page") is None),
    }
    return equations, stats


def build_equations_dict(*, doi: str, extraction_timestamp: str, sections: list[dict], raw_pages: list[str] | None = None) -> dict:
    equations, stats = extract_equations_from_sections(sections, raw_pages=raw_pages)
    return {
        "doi": doi,
        "extraction_timestamp": extraction_timestamp,
        "parser_output_version": "v13",
        "equation_count": len(equations),
        "stats": stats,
        "equations": equations,
    }
