"""
quality_report.py (v13)

Creates final per-paper quality report for publication-dataset filtering.
v13 adds strict expected-table accounting: if the paper mentions/captions tables
but they are not recovered, tables_status becomes needs_review.
"""

from __future__ import annotations


def _status_from_counts(*, hard_fail: bool = False, review: bool = False) -> str:
    if hard_fail:
        return "failed"
    if review:
        return "needs_review"
    return "pass"


def _table_review_count(tables: dict) -> int:
    return sum(1 for t in tables.get("tables", []) or [] if t.get("needs_review") or t.get("low_confidence"))


def _equation_review_count(equations: dict) -> int:
    return sum(1 for e in equations.get("equations", []) or [] if e.get("needs_review") or e.get("confidence") in {"needs_review", "repaired"})


def build_quality_report(*, doi: str, fulltext: dict, imrad: dict | None,
                         figures: dict, tables: dict, equations: dict,
                         quality_flags: dict) -> dict:
    sections = fulltext.get("sections", []) or []
    imrad_summary = (imrad or {}).get("imrad_summary", {}) if imrad else {}
    fig_stats = figures.get("stats", {}) or {}
    eq_stats = equations.get("stats", {}) or {}

    table_review = _table_review_count(tables)
    equation_review = _equation_review_count(equations)

    expected_table_count = int(tables.get("expected_table_count", quality_flags.get("expected_tables", 0)) or 0)
    expected_table_labels = tables.get("expected_table_labels", quality_flags.get("expected_table_labels", [])) or []
    table_count = int(tables.get("table_count", 0) or 0)
    tables_unrecovered = int(tables.get("tables_unrecovered", quality_flags.get("tables_unrecovered", 0)) or 0)
    low_conf_tables = int(quality_flags.get("low_confidence_tables", 0) or 0)

    # If expected tables exist but no table record exists, this must not pass.
    missing_expected_tables = max(0, expected_table_count - table_count)
    if expected_table_count > 0:
        tables_unrecovered = max(tables_unrecovered, missing_expected_tables)

    text_status = _status_from_counts(hard_fail=len(sections) == 0)
    imrad_status = _status_from_counts(
        hard_fail=False,
        review=not bool(imrad_summary.get("complete", False)),
    )
    figures_status = _status_from_counts(
        hard_fail=False,
        review=(
            int(fig_stats.get("missing_images", 0) or 0) > 0
            or int(fig_stats.get("still_bad", 0) or 0) > 0
            or int(fig_stats.get("page_render", 0) or 0) > 0
        ),
    )
    tables_status = _status_from_counts(
        hard_fail=False,
        review=(
            tables_unrecovered > 0
            or missing_expected_tables > 0
            or low_conf_tables > 0
            or table_review > 0
        ),
    )
    equations_status = _status_from_counts(
        hard_fail=False,
        review=(
            int(eq_stats.get("needs_review_equations", 0) or 0) > 0
            or int(eq_stats.get("unrecovered_equations", 0) or 0) > 0
            or int(eq_stats.get("dropped_fragment_equations", 0) or 0) > 0
            or int(eq_stats.get("page_null_equations", 0) or 0) > 0
            or equation_review > 0
        ),
    )

    component_status = [text_status, imrad_status, figures_status, tables_status, equations_status]
    if "failed" in component_status:
        parser_status = "failed"
    elif "needs_review" in component_status:
        parser_status = "needs_review"
    else:
        parser_status = "pass"

    return {
        "doi": doi,
        "parser_status": parser_status,
        "text_status": text_status,
        "imrad_status": imrad_status,
        "figures_status": figures_status,
        "tables_status": tables_status,
        "equations_status": equations_status,
        "status_meaning": {
            "pass": "component passed automatic checks",
            "needs_review": "component is usable but has flagged quality risks",
            "failed": "component has a serious extraction failure",
        },
        "counts": {
            "sections": len(sections),
            "imrad_sections": len((imrad or {}).get("sections", []) or []),
            "figures": int(fig_stats.get("figures", 0) or 0),
            "schemes": int(fig_stats.get("schemes", 0) or 0),
            "expected_tables": expected_table_count,
            "tables": table_count,
            "tables_needing_review": table_review,
            "equations": int(equations.get("equation_count", 0) or 0),
            "equations_needing_review": equation_review,
        },
        "flags": {
            "caption_leakage_detected": bool(quality_flags.get("caption_leakage_detected", False)),
            "captions_removed_from_body": int(quality_flags.get("captions_removed_from_body", 0) or 0),
            "expected_table_count": expected_table_count,
            "expected_table_labels": expected_table_labels,
            "missing_expected_tables": missing_expected_tables,
            "low_confidence_tables": low_conf_tables,
            "tables_unrecovered": tables_unrecovered,
            "full_page_figure_fallbacks": int(quality_flags.get("full_page_figure_fallbacks", 0) or 0),
            "missing_figure_images": int(quality_flags.get("missing_figure_images", 0) or 0),
            "bad_figure_crops": int(quality_flags.get("bad_figure_crops", 0) or 0),
            "noisy_equations": int(eq_stats.get("noisy_equations", 0) or 0),
            "needs_review_equations": int(eq_stats.get("needs_review_equations", 0) or 0),
            "repaired_equations": int(eq_stats.get("repaired_equations", 0) or 0),
            "rejected_equations": int(eq_stats.get("rejected_equations", 0) or 0),
            "unrecovered_equations": int(eq_stats.get("unrecovered_equations", 0) or 0),
            "dropped_fragment_equations": int(eq_stats.get("dropped_fragment_equations", 0) or 0),
            "page_null_equations": int(eq_stats.get("page_null_equations", 0) or 0),
        },
    }
