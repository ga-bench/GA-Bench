"""
Figure / table completeness validator (v6).

v6: version string update only. No logic changes from v5 — all validation
logic (label_utils-based canonical key comparison, range/list expansion,
supplemental/appendix key tracking) is unchanged.
"""

from __future__ import annotations
import re

from label_utils import find_caption_anchors, parse_label


MAX_REASONABLE_REFERENCE_NUMBER = 50


_FIG_RANGE_RE = re.compile(
    r"\b(?:Fig(?:s|ures?)?)\.?\s*(\d+)\s*(?:-|\u2013|\u2014|to)\s*(\d+)(?![0-9])",
    re.IGNORECASE,
)
_TAB_RANGE_RE = re.compile(
    r"\b(?:Tab(?:les?)?)\.?\s*(\d+)\s*(?:-|\u2013|\u2014|to)\s*(\d+)(?![0-9])",
    re.IGNORECASE,
)

_LIST_SEP = r"\s*(?:(?:,|;)\s*(?:and\s+)?|\s+and\s+|\s*&\s*)"
_FIG_LIST_RE = re.compile(
    r"\b(?:Fig(?:s|ures?)?)\.?\s*(\d+(?:" + _LIST_SEP + r"\d+)+)(?![0-9])",
    re.IGNORECASE,
)
_TAB_LIST_RE = re.compile(
    r"\b(?:Tab(?:les?)?)\.?\s*(\d+(?:" + _LIST_SEP + r"\d+)+)(?![0-9])",
    re.IGNORECASE,
)


def _keys_from_text(text: str, kind: str) -> set[str]:
    keys: set[str] = set()
    if not text:
        return keys
    for rec in find_caption_anchors(text):
        if rec["kind"] != kind:
            continue
        if not rec["prefix"]:
            try:
                if int(rec["number"]) > MAX_REASONABLE_REFERENCE_NUMBER:
                    continue
            except ValueError:
                pass
        keys.add(rec["key"])

    range_re = _FIG_RANGE_RE if kind == "figure" else (_TAB_RANGE_RE if kind == "table" else None)
    if range_re is not None:
        for m in range_re.finditer(text):
            try:
                lo, hi = int(m.group(1)), int(m.group(2))
            except ValueError:
                continue
            if 0 < lo <= hi <= MAX_REASONABLE_REFERENCE_NUMBER and (hi - lo) <= 30:
                for n in range(lo, hi + 1):
                    keys.add(f"{kind}::{n}")

    list_re = _FIG_LIST_RE if kind == "figure" else (_TAB_LIST_RE if kind == "table" else None)
    if list_re is not None:
        for m in list_re.finditer(text):
            for num_str in re.findall(r"\d+", m.group(1)):
                try:
                    n = int(num_str)
                except ValueError:
                    continue
                if 1 <= n <= MAX_REASONABLE_REFERENCE_NUMBER:
                    keys.add(f"{kind}::{n}")
    return keys


def _extracted_keys(figures: list[dict], kind: str) -> set[str]:
    keys: set[str] = set()
    for fig in figures:
        if (fig.get("type") or "figure") != kind:
            continue
        rec = parse_label(fig.get("label", "") or "")
        if rec and rec["kind"] == kind:
            keys.add(rec["key"])
    return keys


def find_missing_references(
    sections: list[dict],
    extracted_figures: list[dict],
) -> dict:
    body_chunks = []
    for s in sections or []:
        chunk = s.get("text_no_tables") or s.get("text") or ""
        if chunk:
            body_chunks.append(chunk)
    body = "\n\n".join(body_chunks)

    mentioned_figs = _keys_from_text(body, "figure")
    mentioned_tabs = _keys_from_text(body, "table")
    mentioned_schs = _keys_from_text(body, "scheme")

    extracted_figs = _extracted_keys(extracted_figures, "figure")
    extracted_tabs = _extracted_keys(extracted_figures, "table")
    extracted_schs = _extracted_keys(extracted_figures, "scheme")

    return {
        "missing_figures":  sorted(mentioned_figs - extracted_figs),
        "missing_tables":   sorted(mentioned_tabs - extracted_tabs),
        "missing_schemes":  sorted(mentioned_schs - extracted_schs),
        "mentioned_figures": sorted(mentioned_figs),
        "mentioned_tables":  sorted(mentioned_tabs),
        "mentioned_schemes": sorted(mentioned_schs),
        "extracted_figure_keys": sorted(extracted_figs),
        "extracted_table_keys":  sorted(extracted_tabs),
        "extracted_scheme_keys": sorted(extracted_schs),
    }


def _ints_from_keys(keys: list[str]) -> list[int]:
    out: list[int] = []
    for k in keys:
        parts = k.split(":")
        if len(parts) == 3 and parts[1] == "":
            try:
                out.append(int(parts[2]))
            except ValueError:
                pass
    return sorted(set(out))


def missing_table_numbers_int(missing_result: dict) -> list[int]:
    return _ints_from_keys(missing_result.get("missing_tables", []))


def missing_figure_numbers_int(missing_result: dict) -> list[int]:
    return _ints_from_keys(missing_result.get("missing_figures", []))


if __name__ == "__main__":
    fake_sections = [
        {"heading": "Introduction",
         "text_no_tables": "We refer to Fig. 1 and Fig. 3 for context."},
        {"heading": "Methods",
         "text_no_tables": "See Figures 2-4 and Table 1. Scheme 2 illustrates it. "
                           "Supplemental Figure S8 shows the kidney analysis."},
        {"heading": "Results",
         "text_no_tables": "As shown in Figs. 5 and 6. Tables 2, 3, and 5 list values. "
                           "See also Figure A1 in the appendix."},
    ]
    fake_extracted = [
        {"label": "Figure 2", "type": "figure"},
        {"label": "Figure 4", "type": "figure"},
        {"label": "Figure 5", "type": "figure"},
        {"label": "Figure 6", "type": "figure"},
        {"label": "Table 1",  "type": "table"},
        {"label": "Table 2",  "type": "table"},
        {"label": "Scheme 2", "type": "scheme"},
    ]
    result = find_missing_references(fake_sections, fake_extracted)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()
    print("missing table ints:", missing_table_numbers_int(result))
    print("missing figure ints:", missing_figure_numbers_int(result))
    assert "figure::1" in result["missing_figures"]
    assert "figure::3" in result["missing_figures"]
    assert "figure:S:8" in result["missing_figures"]
    assert "figure:A:1" in result["missing_figures"]
    assert "table::3" in result["missing_tables"] and "table::5" in result["missing_tables"]
    print("\nAll assertions passed.")